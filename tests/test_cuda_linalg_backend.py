import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gaussian_splatting.cuda_linalg import configure_linalg_backend, require_runtime


class RuntimePolicyTests(unittest.TestCase):
    def test_wrong_interpreter_cannot_be_disguised_by_env_metadata(self):
        for prefix in ("/tmp/envs/phystwin", "/tmp/envs/phystwin-cu130", "/tmp/base"):
            with self.subTest(prefix=prefix), patch("sys.prefix", prefix), patch.dict(
                os.environ, {"CONDA_DEFAULT_ENV": "phystwin-cu132"}
            ):
                with self.assertRaisesRegex(RuntimeError, "conda activate phystwin-cu132"):
                    require_runtime(SimpleNamespace(version=SimpleNamespace(cuda="13.2")))

    def test_cuda_132_required_on_every_gpu(self):
        with patch("sys.prefix", "/tmp/envs/phystwin-cu132"):
            for version in (None, "", "invalid", "12.8", "13.0", "13.1"):
                with self.subTest(version=version):
                    with self.assertRaisesRegex(RuntimeError, "CUDA 13.2"):
                        require_runtime(SimpleNamespace(version=SimpleNamespace(cuda=version)))

    def test_actual_interpreter_wins_over_stale_conda_run_metadata(self):
        calls = []
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda="13.2"),
            backends=SimpleNamespace(cuda=SimpleNamespace(
                preferred_linalg_library=calls.append)),
        )
        with patch("sys.prefix", "/tmp/envs/phystwin-cu132"), patch.dict(
            os.environ, {"CONDA_DEFAULT_ENV": "base", "BOBA_LINALG_BACKEND": "MAGMA"}
        ):
            self.assertEqual(configure_linalg_backend(torch), "cusolver")
        self.assertEqual(calls, ["cusolver"])


if __name__ == "__main__":
    unittest.main()
