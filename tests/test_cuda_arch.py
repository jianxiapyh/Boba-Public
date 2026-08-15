import unittest

from env_install.cuda_arch import (
    detect_visible_architectures,
    expected_sm_targets,
    parse_arch_list,
    resolve_architectures,
)


class FakeCuda:
    def __init__(self, capabilities):
        self.capabilities = capabilities

    def device_count(self):
        return len(self.capabilities)

    def get_device_capability(self, device_index):
        return self.capabilities[device_index]


class CudaArchitectureTests(unittest.TestCase):
    def test_explicit_numeric_list_is_sorted_and_deduplicated(self):
        self.assertEqual(
            parse_arch_list("12.0 8.9;12.0+PTX,8.9"),
            ("8.9", "12.0+PTX"),
        )

    def test_non_numeric_architecture_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "numeric capabilities"):
            parse_arch_list("Ada;12.0")

    def test_visible_architectures_are_sorted_and_deduplicated(self):
        torch_module = type("FakeTorch", (), {})()
        torch_module.cuda = FakeCuda([(12, 0), (8, 9), (12, 0)])
        self.assertEqual(
            detect_visible_architectures(torch_module),
            ("8.9", "12.0"),
        )

    def test_explicit_override_does_not_require_visible_gpu(self):
        self.assertEqual(
            resolve_architectures(environ={"TORCH_CUDA_ARCH_LIST": "8.9+PTX"}),
            ("8.9+PTX",),
        )

    def test_headless_build_requires_explicit_override(self):
        torch_module = type("FakeTorch", (), {})()
        torch_module.cuda = FakeCuda([])
        with self.assertRaisesRegex(RuntimeError, "TORCH_CUDA_ARCH_LIST"):
            resolve_architectures(torch_module=torch_module, environ={})

    def test_expected_cubin_names_ignore_ptx_suffix(self):
        self.assertEqual(
            expected_sm_targets(("8.9", "12.0+PTX")),
            ("sm_89", "sm_120"),
        )


if __name__ == "__main__":
    unittest.main()
