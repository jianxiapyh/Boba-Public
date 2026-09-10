"""Numerics, input ownership, streams and large-batch workspace regressions."""
import concurrent.futures
import unittest
from unittest.mock import patch

import torch

from gaussian_splatting.rotation_utils import eigh_3x3


def gram_matrices(count, device="cpu", dtype=torch.float64):
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(count, 3, 3, generator=generator, device=device, dtype=dtype)
    result = x.mT @ x
    if count >= 3:
        result[0] = 0
        result[1] = torch.eye(3, device=device, dtype=dtype)
        result[2] = torch.diag(torch.tensor([0., 0., 2.], device=device, dtype=dtype))
    return result


def assert_eigenpairs(matrices, values, vectors):
    tolerance = 2e-5 if matrices.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        vectors @ torch.diag_embed(values) @ vectors.mT, matrices,
        rtol=tolerance, atol=tolerance,
    )
    torch.testing.assert_close(
        vectors.mT @ vectors, torch.eye(3, device=matrices.device,
                                       dtype=matrices.dtype).expand_as(matrices),
        rtol=tolerance, atol=tolerance,
    )
    assert bool((values[:, 1:] >= values[:, :-1]).all())


class CpuEighTests(unittest.TestCase):
    def test_cpu_noncontiguous_and_explicit_chunking(self):
        a = gram_matrices(137).mT
        w, v = eigh_3x3(a, chunk_size=32)
        assert_eigenpairs(a, w, v)

    def test_empty_and_invalid_inputs(self):
        w, v = eigh_3x3(torch.empty(0, 3, 3))
        self.assertEqual(w.shape, (0, 3))
        self.assertEqual(v.shape, (0, 3, 3))
        with self.assertRaises(ValueError):
            eigh_3x3(torch.empty(3, 3))
        with self.assertRaises(ValueError):
            eigh_3x3(torch.empty(1, 3, 3), chunk_size=0)

    def test_autograd(self):
        a = gram_matrices(5)[3:].requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda a: eigh_3x3(0.5 * (a + a.mT))[0], (a,)))

    def test_failed_chunk_reports_global_index(self):
        solver = torch.linalg.eigh
        with patch("torch.linalg.eigh", side_effect=[
            solver(torch.eye(3).repeat(8, 1, 1)),
            torch.linalg.LinAlgError("Batch element 2: failed to converge"),
        ]):
            with self.assertRaisesRegex(torch.linalg.LinAlgError, "Batch element 10"):
                eigh_3x3(torch.eye(3).repeat(17, 1, 1), chunk_size=8)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class CudaEighTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        eigh_3x3(torch.eye(3, device="cuda").unsqueeze(0))

    def test_fp32_fp64_noncontiguous_input_is_preserved(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = gram_matrices(4097, "cuda", dtype).mT
                before = a.clone()
                w, v = eigh_3x3(a)
                torch.testing.assert_close(a, before, rtol=0, atol=0)
                self.assertNotEqual(a.data_ptr(), v.data_ptr())
                assert_eigenpairs(a, w, v)
                reference = torch.linalg.eigvalsh(a.cpu().double()).to(w)
                torch.testing.assert_close(w, reference, rtol=2e-5, atol=2e-5)

    def test_only_lower_triangle_is_read(self):
        a = gram_matrices(17, "cuda", torch.float32)
        expected = a.clone()
        a[:, 0, 1:] = float("nan")
        a[:, 1, 2] = float("nan")
        assert_eigenpairs(expected, *eigh_3x3(a))

    def test_empty_batch_and_unsupported_dtype(self):
        w, v = eigh_3x3(torch.empty(0, 3, 3, device="cuda"))
        self.assertEqual(w.shape, (0, 3))
        self.assertEqual(v.shape, (0, 3, 3))
        with self.assertRaisesRegex(RuntimeError, "float32 and float64"):
            eigh_3x3(torch.eye(3, device="cuda", dtype=torch.float16).unsqueeze(0))

    def test_cuda_autograd_retains_torch_backward(self):
        a = gram_matrices(5, "cuda")[3:].requires_grad_()
        w, _ = eigh_3x3(a)
        w.sum().backward()
        torch.testing.assert_close(a.grad, torch.eye(3, device="cuda",
                                                   dtype=a.dtype).expand_as(a))

    def test_no_grad_input_does_not_acquire_a_graph(self):
        a = gram_matrices(5, "cuda").requires_grad_()
        with torch.no_grad():
            w, v = eigh_3x3(a)
        self.assertFalse(w.requires_grad)
        self.assertFalse(v.requires_grad)

    def test_concurrent_threads_and_nondefault_streams(self):
        def run(_):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                a = gram_matrices(257, "cuda", torch.float32)
                w, v = eigh_3x3(a)
                assert_eigenpairs(a, w, v)
            stream.synchronize()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, range(4)))

    def test_large_batches_use_bounded_workspace_without_generic_solver(self):
        for count in (4097, 242700, 2427000):
            with self.subTest(count=count):
                a = gram_matrices(count, "cuda", torch.float32)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with patch("torch.linalg.eigh", side_effect=AssertionError("generic CUDA solver")):
                    w, v = eigh_3x3(a)
                peak = torch.cuda.max_memory_allocated() - baseline
                self.assertLess(peak, 1024**3, f"workspace/output peak: {peak} bytes")
                assert_eigenpairs(a, w, v)
                del a, w, v


if __name__ == "__main__":
    unittest.main()
