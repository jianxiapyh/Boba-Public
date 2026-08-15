import unittest
from types import SimpleNamespace

from gaussian_splatting.cuda_linalg import (
    configure_linalg_backend,
    normalize_device_capability,
    validate_cuda_compatibility,
)


def make_torch(cuda_build, capability, expose_backend_api=True):
    calls = []
    backend = SimpleNamespace()
    if expose_backend_api:
        backend.preferred_linalg_library = lambda value: calls.append(value)
    torch_module = SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_build),
        cuda=SimpleNamespace(get_device_capability=lambda: capability),
        backends=SimpleNamespace(cuda=backend),
    )
    return torch_module, calls


class CudaCompatibilityTests(unittest.TestCase):
    def test_pre_capability_12_accepts_cuda12(self):
        self.assertEqual(validate_cuda_compatibility("12.8", (8, 9)), (8, 9))

    def test_capability_12_accepts_cuda13(self):
        self.assertEqual(validate_cuda_compatibility("13.0", (12, 0)), (12, 0))

    def test_capability_12_rejects_cuda12(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA 13"):
            validate_cuda_compatibility("12.8", (12, 0))

    def test_future_capability_reuses_generic_cuda13_gate(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA 13"):
            validate_cuda_compatibility("12.9", (13, 0))

    def test_missing_or_invalid_cuda_build_is_rejected(self):
        for cuda_build in (None, "", "unknown"):
            with self.subTest(cuda_build=cuda_build):
                with self.assertRaisesRegex(RuntimeError, "torch.version.cuda"):
                    validate_cuda_compatibility(cuda_build, (8, 9))

    def test_invalid_device_capability_is_rejected(self):
        for capability in (None, (12,), ("unknown", 0)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(RuntimeError, "device capability"):
                    normalize_device_capability(capability)


class CudaBackendTests(unittest.TestCase):
    def test_cuda12_and_cuda13_both_configure_cusolver(self):
        for cuda_build, capability in (("12.8", (8, 9)), ("13.0", (12, 0))):
            with self.subTest(cuda_build=cuda_build, capability=capability):
                torch_module, calls = make_torch(cuda_build, capability)
                selected = configure_linalg_backend(torch_module)
                self.assertEqual(selected, "cusolver")
                self.assertEqual(calls, ["cusolver"])

    def test_missing_backend_api_is_actionable(self):
        torch_module, _ = make_torch("13.0", (12, 0), expose_backend_api=False)
        with self.assertRaisesRegex(RuntimeError, "preferred_linalg_library"):
            configure_linalg_backend(torch_module)


if __name__ == "__main__":
    unittest.main()
