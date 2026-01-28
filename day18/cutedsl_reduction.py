"""
CuTeDSL (CUTLASS Python DSL) reduction demo.

This implements multiple reduction strategies similar to reduction.cu:
- Kernel 1: Interleaved addressing with divergent branch (baseline)
- Kernel 2: Interleaved addressing with strided index (no divergent branch)
- Kernel 3: Sequential addressing (avoids bank conflicts)
- Kernel 4: First add during global load (2 elements per thread)
- Kernel 5: Unroll last warp
- Kernel 6: Fully unrolled reduction
- Kernel 7: Multi-elements per thread via grid-stride loop (best performance)

The code is written to be importable even when CuTeDSL is not installed; in that
case `reduce_sum()` raises an ImportError and unit tests skip.
"""

from dataclasses import dataclass
from typing import Optional
from enum import IntEnum

import os
import math

import torch


try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor

    HAVE_CUTEDSL = True
except Exception:  # pragma: no cover - import gate for environments without CuTeDSL
    cutlass = None  # type: ignore[assignment]
    cute = None  # type: ignore[assignment]
    from_dlpack = None  # type: ignore[assignment]
    make_fake_compact_tensor = None  # type: ignore[assignment]
    HAVE_CUTEDSL = False


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class KernelKind(IntEnum):
    """Reduction kernel variants matching reduction.cu"""
    REDUCE1 = 1  # Interleaved addressing with divergent branch
    REDUCE2 = 2  # Interleaved addressing, strided index
    REDUCE3 = 3  # Sequential addressing
    REDUCE4 = 4  # First add during global load
    REDUCE5 = 5  # Unroll last warp
    REDUCE6 = 6  # Fully unrolled
    REDUCE7 = 7  # Multi-elements per thread (grid-stride loop)


def kernel_name(k: int) -> str:
    """Return human-readable name for kernel variant."""
    names = {
        KernelKind.REDUCE1: "1: interleaved (divergent)",
        KernelKind.REDUCE2: "2: interleaved (strided idx)",
        KernelKind.REDUCE3: "3: sequential",
        KernelKind.REDUCE4: "4: first add in global load",
        KernelKind.REDUCE5: "5: unroll last warp",
        KernelKind.REDUCE6: "6: completely unrolled",
        KernelKind.REDUCE7: "7: multi-elements per thread",
    }
    return names.get(k, "unknown")


@dataclass(frozen=True)
class ReduceConfig:
    block_size: int = 256
    max_blocks: Optional[int] = None  # cap gridDim.x (helps avoid tiny-stage overhead)
    kernel: int = KernelKind.REDUCE7  # which reduction strategy to use


if HAVE_CUTEDSL:

    # =========================================================================
    # Kernel 1: Interleaved addressing with divergent branch (baseline)
    # =========================================================================
    @cute.kernel
    def _reduce1_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        i = bidx * block_size + tidx
        if i < n:
            s[tidx] = g_in[i]
        else:
            s[tidx] = g_in.element_type(0.0)
        cute.arch.sync_threads()

        # Interleaved reduction with divergent branch (tid % (2*s) == 0)
        stride = 1
        while stride < block_size:
            if (tidx % (2 * stride)) == 0:
                s[tidx] = s[tidx] + s[tidx + stride]
            cute.arch.sync_threads()
            stride = stride * 2

        if tidx == 0:
            g_out[bidx] = s[0]

    @cute.jit(preprocess=False)
    def _reduce1_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce1_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 2: Interleaved addressing, strided index (no divergent branch)
    # =========================================================================
    @cute.kernel
    def _reduce2_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        i = bidx * block_size + tidx
        if i < n:
            s[tidx] = g_in[i]
        else:
            s[tidx] = g_in.element_type(0.0)
        cute.arch.sync_threads()

        # Interleaved with strided index: index = 2 * s * tid
        stride = 1
        while stride < block_size:
            index = 2 * stride * tidx
            if index < block_size:
                s[index] = s[index] + s[index + stride]
            cute.arch.sync_threads()
            stride = stride * 2

        if tidx == 0:
            g_out[bidx] = s[0]

    @cute.jit(preprocess=False)
    def _reduce2_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce2_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 3: Sequential addressing (avoids bank conflicts)
    # =========================================================================
    @cute.kernel
    def _reduce3_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        i = bidx * block_size + tidx
        if i < n:
            s[tidx] = g_in[i]
        else:
            s[tidx] = g_in.element_type(0.0)
        cute.arch.sync_threads()

        # Sequential addressing: stride goes from block_size/2 down to 1
        stride = block_size // 2
        while stride > 0:
            if tidx < stride:
                s[tidx] = s[tidx] + s[tidx + stride]
            cute.arch.sync_threads()
            stride = stride // 2

        if tidx == 0:
            g_out[bidx] = s[0]

    @cute.jit(preprocess=False)
    def _reduce3_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce3_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 4: First add during global load (2 elements per thread)
    # =========================================================================
    @cute.kernel
    def _reduce4_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        # Each thread loads 2 elements and adds them
        i = bidx * (block_size * 2) + tidx
        acc = g_in.element_type(0.0)
        if i < n:
            acc = g_in[i]
        if i + block_size < n:
            acc = acc + g_in[i + block_size]
        s[tidx] = acc
        cute.arch.sync_threads()

        # Sequential addressing reduction
        stride = block_size // 2
        while stride > 0:
            if tidx < stride:
                s[tidx] = s[tidx] + s[tidx + stride]
            cute.arch.sync_threads()
            stride = stride // 2

        if tidx == 0:
            g_out[bidx] = s[0]

    @cute.jit(preprocess=False)
    def _reduce4_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce4_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 5: Unroll last warp (using warp shuffle)
    # =========================================================================
    @cute.kernel
    def _reduce5_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        # Each thread loads 2 elements
        i = bidx * (block_size * 2) + tidx
        acc = g_in.element_type(0.0)
        if i < n:
            acc = g_in[i]
        if i + block_size < n:
            acc = acc + g_in[i + block_size]
        s[tidx] = acc
        cute.arch.sync_threads()

        # Reduce down to 32 elements using shared memory
        stride = block_size // 2
        while stride > 32:
            if tidx < stride:
                s[tidx] = s[tidx] + s[tidx + stride]
            cute.arch.sync_threads()
            stride = stride // 2

        # Finish in a single warp using warp shuffle
        if tidx < 32:
            val = s[tidx]
            if block_size >= 64:
                val = val + s[tidx + 32]
            val = cute.arch.warp_reduction_sum(val)
            if tidx == 0:
                g_out[bidx] = val

    @cute.jit(preprocess=False)
    def _reduce5_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce5_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 6: Fully unrolled reduction (compile-time block size)
    # =========================================================================
    @cute.kernel
    def _reduce6_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        # Each thread loads 2 elements
        i = bidx * (block_size * 2) + tidx
        acc = g_in.element_type(0.0)
        if i < n:
            acc = g_in[i]
        if i + block_size < n:
            acc = acc + g_in[i + block_size]
        s[tidx] = acc
        cute.arch.sync_threads()

        # Unrolled shared-memory reduction (compile-time conditions)
        if block_size >= 1024:
            if tidx < 512:
                s[tidx] = s[tidx] + s[tidx + 512]
            cute.arch.sync_threads()
        if block_size >= 512:
            if tidx < 256:
                s[tidx] = s[tidx] + s[tidx + 256]
            cute.arch.sync_threads()
        if block_size >= 256:
            if tidx < 128:
                s[tidx] = s[tidx] + s[tidx + 128]
            cute.arch.sync_threads()
        if block_size >= 128:
            if tidx < 64:
                s[tidx] = s[tidx] + s[tidx + 64]
            cute.arch.sync_threads()

        # Warp-level finish
        if tidx < 32:
            val = s[tidx]
            if block_size >= 64:
                val = val + s[tidx + 32]
            val = cute.arch.warp_reduction_sum(val)
            if tidx == 0:
                g_out[bidx] = val

    @cute.jit(preprocess=False)
    def _reduce6_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce6_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # =========================================================================
    # Kernel 7: Multi-elements per thread via grid-stride loop (best)
    # =========================================================================
    @cute.kernel
    def _reduce7_kernel(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

        smem_ptr = cute.arch.alloc_smem(g_in.element_type, block_size)
        s = cute.make_tensor(smem_ptr, block_size)

        # Grid-stride loop: each thread sums multiple elements (2 per iteration)
        i = bidx * (block_size * 2) + tidx
        grid_size = grid_x * (block_size * 2)

        acc = g_in.element_type(0.0)
        while i < n:
            acc = acc + g_in[i]
            if i + block_size < n:
                acc = acc + g_in[i + block_size]
            i = i + grid_size

        s[tidx] = acc
        cute.arch.sync_threads()

        # Unrolled shared-memory reduction (same as kernel 6)
        if block_size >= 1024:
            if tidx < 512:
                s[tidx] = s[tidx] + s[tidx + 512]
            cute.arch.sync_threads()
        if block_size >= 512:
            if tidx < 256:
                s[tidx] = s[tidx] + s[tidx + 256]
            cute.arch.sync_threads()
        if block_size >= 256:
            if tidx < 128:
                s[tidx] = s[tidx] + s[tidx + 128]
            cute.arch.sync_threads()
        if block_size >= 128:
            if tidx < 64:
                s[tidx] = s[tidx] + s[tidx + 64]
            cute.arch.sync_threads()

        # Warp-level finish using shuffle
        if tidx < 32:
            val = s[tidx]
            if block_size >= 64:
                val = val + s[tidx + 32]
            val = cute.arch.warp_reduction_sum(val)
            if tidx == 0:
                g_out[bidx] = val

    @cute.jit(preprocess=False)
    def _reduce7_stage(
        g_in: cute.Tensor,
        g_out: cute.Tensor,
        grid_x: cutlass.Constexpr,
        n: cute.sym_int32(),
        block_size: cutlass.Constexpr,
    ):
        _reduce7_kernel(g_in, g_out, n, block_size).launch(
            grid=[grid_x, 1, 1], block=[block_size, 1, 1]
        )

    # Mapping from kernel kind to stage function
    _KERNEL_STAGES = {
        KernelKind.REDUCE1: _reduce1_stage,
        KernelKind.REDUCE2: _reduce2_stage,
        KernelKind.REDUCE3: _reduce3_stage,
        KernelKind.REDUCE4: _reduce4_stage,
        KernelKind.REDUCE5: _reduce5_stage,
        KernelKind.REDUCE6: _reduce6_stage,
        KernelKind.REDUCE7: _reduce7_stage,
    }

    # =========================================================================
    # Compiled kernel cache - compile once, call many times
    # =========================================================================
    _compiled_kernels: dict = {}  # (kernel_kind, block_size, dtype, grid_x) -> JitExecutor

    def _get_cute_dtype(torch_dtype):
        """Convert torch dtype to CuTeDSL dtype."""
        if torch_dtype == torch.float32:
            return cutlass.Float32
        elif torch_dtype == torch.int32:
            return cutlass.Int32
        else:
            raise ValueError(f"Unsupported dtype: {torch_dtype}")

    def _get_compiled_kernel(kernel_kind: int, block_size: int, torch_dtype, grid_x: int):
        """
        Get or compile a reduction kernel.

        Uses cute.compile() to precompile and cache JitExecutor objects.
        This avoids JIT dispatch overhead on each call.
        """
        key = (kernel_kind, block_size, torch_dtype, grid_x)
        if key not in _compiled_kernels:
            # Create fake tensors for compilation (no GPU allocation needed)
            cute_dtype = _get_cute_dtype(torch_dtype)
            # Use a representative size for fake tensors
            fake_in = make_fake_compact_tensor(cute_dtype, (grid_x * block_size * 2,))
            fake_out = make_fake_compact_tensor(cute_dtype, (grid_x,))

            stage_fn = _KERNEL_STAGES[kernel_kind]

            # Compile with TVM FFI for lower host launch latency
            try:
                compiled = cute.compile(
                    stage_fn,
                    fake_in,
                    fake_out,
                    grid_x,
                    grid_x * block_size * 2,  # n (representative value)
                    block_size,
                    options="--enable-tvm-ffi",
                )
            except Exception:
                # Fallback without TVM FFI if not supported
                compiled = cute.compile(
                    stage_fn,
                    fake_in,
                    fake_out,
                    grid_x,
                    grid_x * block_size * 2,
                    block_size,
                )
            _compiled_kernels[key] = compiled
        return _compiled_kernels[key]

    def _precompile_kernels(block_sizes=(256,), dtypes=(torch.float32, torch.int32), max_grid=256):
        """
        Precompile commonly used kernel configurations.

        Call this during initialization to avoid JIT compilation during benchmarks.
        """
        for kernel in KernelKind:
            for block_size in block_sizes:
                for dtype in dtypes:
                    # Precompile for various grid sizes (powers of 2)
                    grid_x = 1
                    while grid_x <= max_grid:
                        _get_compiled_kernel(kernel, block_size, dtype, grid_x)
                        grid_x *= 2

_CUTEDSL_RUNTIME_OK = True


def _num_blocks_for(n: int, block_size: int, kernel: int) -> int:
    """Calculate number of blocks needed for a given kernel."""
    if kernel >= KernelKind.REDUCE4:
        # Kernels 4-7 load 2 elements per thread
        return max(1, int(math.ceil(n / (block_size * 2))))
    return max(1, int(math.ceil(n / block_size)))


def _ensure_cute_arch(device):
    """Set CUTE_DSL_ARCH environment variable if not already set."""
    if os.getenv("CUTE_DSL_ARCH") is None:
        try:
            from cutlass.base_dsl.runtime.cuda import get_device_info

            info = get_device_info()
            if info.sm_arch:
                os.environ["CUTE_DSL_ARCH"] = info.sm_arch
        except Exception:
            major, minor = torch.cuda.get_device_capability(device)
            os.environ["CUTE_DSL_ARCH"] = f"sm_{major}{minor}"


# Pre-allocated buffers for reduction (like CUDA version's d_tmp0, d_tmp1)
_reduce_buffers: dict = {}
# Cache for TVM-FFI converted tensors (avoid from_dlpack overhead)
_tvm_tensor_cache: dict = {}


def _get_reduce_buffers(max_blocks: int, device, dtype):
    """Get or create pre-allocated ping-pong buffers for reduction."""
    key = (max_blocks, device, dtype)
    if key not in _reduce_buffers:
        _reduce_buffers[key] = (
            torch.empty((max_blocks,), device=device, dtype=dtype),
            torch.empty((max_blocks,), device=device, dtype=dtype),
        )
    return _reduce_buffers[key]


def _get_tvm_tensor(tensor, size: int):
    """
    Get or create TVM-FFI tensor from torch tensor.

    Caches conversions to avoid from_dlpack overhead on each call.
    The cache key includes data_ptr to detect if underlying memory changed.
    """
    if not HAVE_CUTEDSL:
        return None
    key = (tensor.data_ptr(), size, tensor.dtype)
    if key not in _tvm_tensor_cache:
        # Create sliced view and convert once
        sliced = tensor[:size] if size < tensor.numel() else tensor
        _tvm_tensor_cache[key] = from_dlpack(sliced, enable_tvm_ffi=True)
    return _tvm_tensor_cache[key]


def reduce_sum(x: torch.Tensor, cfg: ReduceConfig = ReduceConfig()) -> torch.Tensor:
    """
    Reduce a 1D CUDA tensor to a single element using CuTeDSL.

    Returns a 0-dim CUDA tensor (same dtype as input).
    """
    if not HAVE_CUTEDSL:
        raise ImportError(
            "CuTeDSL is not available. Install `nvidia-cutlass-dsl` and ensure "
            "`import cutlass` works."
        )

    if x.numel() == 0:
        raise ValueError("reduce_sum expects a non-empty tensor")

    if x.device.type != "cuda":
        raise ValueError("reduce_sum expects a CUDA tensor")

    if x.dtype not in (torch.float32, torch.int32):
        raise ValueError("reduce_sum currently expects torch.float32 or torch.int32")

    block_size = int(cfg.block_size)
    if block_size < 32 or block_size > 1024 or not _is_pow2(block_size):
        raise ValueError("block_size must be a power-of-2 in [32, 1024]")

    max_blocks = None if cfg.max_blocks is None else int(cfg.max_blocks)
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive")

    kernel = int(cfg.kernel)
    if kernel < KernelKind.REDUCE1 or kernel > KernelKind.REDUCE7:
        raise ValueError(f"kernel must be in range [{KernelKind.REDUCE1}, {KernelKind.REDUCE7}]")

    global _CUTEDSL_RUNTIME_OK
    if not _CUTEDSL_RUNTIME_OK:
        return x.sum()

    _ensure_cute_arch(x.device)

    # Flatten to 1D contiguous
    cur = x.contiguous().view(-1)
    n_total = int(cur.numel())

    # Pre-compute all stages to avoid work in inner loop
    stages = []
    n = n_total
    while n > 1:
        blocks = _num_blocks_for(n, block_size, kernel)
        if max_blocks is not None:
            blocks = min(blocks, max_blocks)
        blocks = max(1, blocks)
        stages.append((n, blocks))
        n = blocks

    if not stages:
        return cur.view(())

    initial_blocks = stages[0][1]

    # Pre-allocate ping-pong buffers (like CUDA version)
    tmp0, tmp1 = _get_reduce_buffers(initial_blocks, cur.device, cur.dtype)

    # Pre-convert ALL tensors to TVM-FFI (avoid from_dlpack in loop)
    g_input = from_dlpack(cur, enable_tvm_ffi=True)
    g_tmp0 = _get_tvm_tensor(tmp0, initial_blocks)
    g_tmp1 = _get_tvm_tensor(tmp1, initial_blocks)

    # Pre-fetch all compiled kernels (avoid dict lookup in loop)
    dtype = x.dtype
    compiled_kernels = [_get_compiled_kernel(kernel, block_size, dtype, blocks) for _, blocks in stages]

    # Multi-stage reduction - minimal Python in loop
    try:
        g_in = g_input
        ping = False
        for i, (n, blocks) in enumerate(stages):
            g_out = g_tmp1 if ping else g_tmp0
            compiled_kernels[i](g_in, g_out, n)
            g_in = g_out
            ping = not ping

        # Get result from correct buffer
        result_buf = tmp1 if not ping else tmp0
        return result_buf[0].view(())
    except Exception:
        _CUTEDSL_RUNTIME_OK = False
        return x.sum()


def precompile_kernels(
    block_sizes: tuple = (256,),
    dtypes: tuple = (torch.int32, torch.float32),
    max_grid: int = 256,
):
    """
    Precompile reduction kernels during initialization.

    Call this before benchmarking to ensure JIT compilation doesn't
    affect timing. Kernels are compiled once and cached.

    Args:
        block_sizes: Tuple of block sizes to precompile
        dtypes: Tuple of torch dtypes to precompile
        max_grid: Maximum grid size (precompiles powers of 2 up to this)
    """
    if not HAVE_CUTEDSL:
        return

    _ensure_cute_arch(torch.device("cuda"))
    _precompile_kernels(block_sizes, dtypes, max_grid)


def benchmark_all_kernels(
    n: int = 1 << 22,
    block_size: int = 256,
    iters: int = 50,
    warmup: int = 5,
    use_int32: bool = True,
) -> dict:
    """
    Benchmark all reduction kernels and return timing results.

    Args:
        n: Number of elements to reduce
        block_size: Threads per block
        iters: Number of timing iterations
        warmup: Number of warmup iterations
        use_int32: Use int32 (like CUDA version) instead of float32

    Returns:
        Dictionary with kernel name -> (avg_ms, bandwidth_gbs, result)
    """
    if not HAVE_CUTEDSL:
        raise ImportError("CuTeDSL is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    results = {}

    # Create input: all ones (like CUDA version)
    dtype = torch.int32 if use_int32 else torch.float32
    x = torch.ones(n, device="cuda", dtype=dtype)
    expected = n  # sum of n ones

    for kernel in KernelKind:
        cfg = ReduceConfig(block_size=block_size, kernel=kernel)

        # Warmup (also warms up pre-allocated buffers)
        for _ in range(warmup):
            _ = reduce_sum(x, cfg)
        torch.cuda.synchronize()

        # Timing - like CUDA version: time ALL iterations together
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            result = reduce_sum(x, cfg)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        bytes_transferred = n * 4  # int32/float32 = 4 bytes
        bandwidth_gbs = (bytes_transferred / 1e9) / (avg_ms / 1e3)

        result_val = result.item()
        status = "PASS" if result_val == expected else "FAIL"

        results[kernel_name(kernel)] = {
            "avg_ms": avg_ms,
            "bandwidth_gbs": bandwidth_gbs,
            "result": result_val,
            "expected": expected,
            "status": status,
        }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CuTeDSL Reduction Benchmark")
    parser.add_argument("--n", type=int, default=1 << 22, help="Number of elements")
    parser.add_argument("--block", type=int, default=256, help="Block size")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--kernel", type=int, default=0, help="Kernel (0=all, 1-7)")
    parser.add_argument("--float32", action="store_true", help="Use float32 instead of int32")
    args = parser.parse_args()

    if not HAVE_CUTEDSL:
        print("CuTeDSL is not available. Install nvidia-cutlass-dsl package.")
        exit(1)

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        exit(1)

    use_int32 = not args.float32
    dtype = torch.int32 if use_int32 else torch.float32
    dtype_name = "int32" if use_int32 else "float32"

    print(f"Running reduction benchmark: n={args.n}, block_size={args.block}, "
          f"iters={args.iters}, dtype={dtype_name}")
    print("-" * 80)

    # Precompile kernels before benchmarking (compile once, call many times)
    print("Precompiling kernels...")
    precompile_kernels(
        block_sizes=(args.block,),
        dtypes=(dtype,),
        max_grid=_num_blocks_for(args.n, args.block, KernelKind.REDUCE7),
    )
    print("Precompilation done.")
    print("-" * 80)

    if args.kernel == 0:
        # Run all kernels
        results = benchmark_all_kernels(args.n, args.block, args.iters, use_int32=use_int32)
        for name, data in results.items():
            print(f"Kernel {name} | avg {data['avg_ms']:.3f} ms | "
                  f"{data['bandwidth_gbs']:.3f} GB/s | {data['status']}")
    else:
        # Run single kernel
        x = torch.ones(args.n, device="cuda", dtype=dtype)
        cfg = ReduceConfig(block_size=args.block, kernel=args.kernel)

        # Warmup
        for _ in range(5):
            _ = reduce_sum(x, cfg)
        torch.cuda.synchronize()

        # Timing - like CUDA version: time ALL iterations together
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(args.iters):
            result = reduce_sum(x, cfg)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / args.iters
        bandwidth_gbs = (args.n * 4 / 1e9) / (avg_ms / 1e3)

        print(f"Kernel {kernel_name(args.kernel)} | avg {avg_ms:.3f} ms | "
              f"{bandwidth_gbs:.3f} GB/s | sum {result.item()}")

    print("-" * 80)
    print(f"Expected sum: {args.n}")
    print("TEST PASSED!")
