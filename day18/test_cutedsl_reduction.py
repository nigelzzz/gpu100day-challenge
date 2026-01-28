import unittest

import torch


class TestCuTeDSLReduction(unittest.TestCase):
    def test_import_and_basic(self):
        try:
            from cutedsl_reduction import reduce_sum, ReduceConfig, HAVE_CUTEDSL
        except Exception as e:
            self.fail(f"Failed to import cutedsl_reduction.py: {e}")

        if not HAVE_CUTEDSL:
            raise unittest.SkipTest("CuTeDSL (nvidia-cutlass-dsl) is not installed")

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for torch in this environment")

        x = torch.arange(1, 1025, device="cuda", dtype=torch.float32)  # sum = 524800
        y = reduce_sum(x, ReduceConfig(block_size=256))
        self.assertTrue(y.is_cuda)
        self.assertEqual(y.ndim, 0)
        torch.testing.assert_close(y, x.sum(), rtol=0, atol=0)

    def test_various_sizes(self):
        from cutedsl_reduction import reduce_sum, ReduceConfig, HAVE_CUTEDSL

        if not HAVE_CUTEDSL:
            raise unittest.SkipTest("CuTeDSL (nvidia-cutlass-dsl) is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for torch in this environment")

        torch.manual_seed(0)
        sizes = [1, 2, 3, 31, 32, 33, 127, 128, 129, 1024, 2047]
        for n in sizes:
            with self.subTest(n=n):
                x = torch.randn(n, device="cuda", dtype=torch.float32)
                y = reduce_sum(x, ReduceConfig(block_size=256, max_blocks=256))
                torch.testing.assert_close(y, x.sum(), rtol=1e-5, atol=1e-5)

    def test_int32_support(self):
        """Test int32 dtype (matching CUDA C++ version)."""
        from cutedsl_reduction import reduce_sum, ReduceConfig, HAVE_CUTEDSL

        if not HAVE_CUTEDSL:
            raise unittest.SkipTest("CuTeDSL (nvidia-cutlass-dsl) is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for torch in this environment")

        # Test with all ones (like CUDA benchmark)
        n = 1024
        x = torch.ones(n, device="cuda", dtype=torch.int32)
        y = reduce_sum(x, ReduceConfig(block_size=256))
        self.assertEqual(y.item(), n)

    def test_all_kernels(self):
        """Test all kernel variants."""
        from cutedsl_reduction import reduce_sum, ReduceConfig, KernelKind, HAVE_CUTEDSL

        if not HAVE_CUTEDSL:
            raise unittest.SkipTest("CuTeDSL (nvidia-cutlass-dsl) is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for torch in this environment")

        n = 2048
        x = torch.ones(n, device="cuda", dtype=torch.int32)

        for kernel in KernelKind:
            with self.subTest(kernel=kernel.name):
                cfg = ReduceConfig(block_size=256, kernel=kernel)
                y = reduce_sum(x, cfg)
                self.assertEqual(y.item(), n)

    def test_invalid_args(self):
        from cutedsl_reduction import reduce_sum, ReduceConfig, HAVE_CUTEDSL

        if not HAVE_CUTEDSL:
            raise unittest.SkipTest("CuTeDSL (nvidia-cutlass-dsl) is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available for torch in this environment")

        x = torch.ones(10, device="cuda", dtype=torch.float32)

        with self.assertRaises(ValueError):
            reduce_sum(x, ReduceConfig(block_size=7))
        with self.assertRaises(ValueError):
            reduce_sum(x, ReduceConfig(block_size=2048))
        with self.assertRaises(ValueError):
            reduce_sum(x, ReduceConfig(max_blocks=0))


if __name__ == "__main__":
    unittest.main()

