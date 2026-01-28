## Day 18 - Parallel Reduction

This folder implements the classic CUDA reduction optimization steps (as shown in the NVIDIA slides):

- Kernel 1: interleaved addressing + divergent branch
- Kernel 2: interleaved addressing + strided index (removes divergence)
- Kernel 3: sequential addressing
- Kernel 4: first add during global load (2 elements per thread)
- Kernel 5: unroll the last warp
- Kernel 6: completely unrolled (templated)
- Kernel 7: multiple elements per thread (grid-stride loop)

`reduction.cu` contains all kernels plus a CuTe-DSL version:

- Kernel 8 (`--kernel 8`): same idea as Kernel 7, but uses CUTLASS' `cute` DSL
  (`cute::make_rseq` + `cute::for_each`) to generate the unrolled reduction steps.

### Build

```
make
```

To enable the CuTe kernel, build with CUTLASS in your include path. The Makefile defaults to:

- `CUTLASS_DIR=$(HOME)/opensource/cutlass`

Override if needed:

```
make CUTLASS_DIR=/path/to/cutlass
```

### Run

Run all kernels (1..7 and 8 if available):

```
./reduction --n $((1<<22)) --block 256 --iters 50 --kernel 0
```

Run a specific kernel (example: Kernel 6):

```
./reduction --kernel 6
```

Notes:
- The input is all ones, so the expected sum is `N`.
- If no CUDA device is available, the program prints a message and skips the GPU run.

## CuTeDSL (Python) Version + Unit Test

`cutedsl_reduction.py` implements a "kernel 7" style reduction using the CUTLASS
Python DSL (CuTeDSL). The unit tests live in `test_cutedsl_reduction.py`.

Run tests:

```
python3 -m unittest -v test_cutedsl_reduction.py
```

Requirements:
- `torch` with CUDA support
- `nvidia-cutlass-dsl` installed so `import cutlass` works
