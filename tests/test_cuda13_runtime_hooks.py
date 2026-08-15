import os
from pathlib import Path
import shlex
import stat
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATE_HOOK = (
    REPO_ROOT
    / "env_install"
    / "conda"
    / "activate.d"
    / "boba-cu130-runtime.sh"
)
DEACTIVATE_HOOK = (
    REPO_ROOT
    / "env_install"
    / "conda"
    / "deactivate.d"
    / "boba-cu130-runtime.sh"
)


class Cuda13RuntimeHookTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.prefix = Path(self.temp_dir.name) / "portable-conda-prefix"
        python_path = self.prefix / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.write_text(
            "#!/bin/sh\n"
            'printf "%s\\n" "$CONDA_PREFIX/lib/python3.10/site-packages"\n',
            encoding="utf-8",
        )
        python_path.chmod(python_path.stat().st_mode | stat.S_IXUSR)

        self.site_packages = self.prefix / "lib" / "python3.10" / "site-packages"
        self.runtime_paths = ":".join(
            (
                str(self.site_packages / "nvidia" / "cu13" / "lib"),
                str(self.site_packages / "torch" / "lib"),
                str(self.prefix / "lib"),
                str(self.prefix / "targets" / "x86_64-linux" / "lib"),
            )
        )

    def run_hooks(self, script, original_library_path=None):
        environment = os.environ.copy()
        environment["CONDA_PREFIX"] = str(self.prefix)
        if original_library_path is None:
            environment.pop("LD_LIBRARY_PATH", None)
        else:
            environment["LD_LIBRARY_PATH"] = original_library_path
        result = subprocess.run(
            ["/bin/sh", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result.stdout.splitlines()

    def test_activation_is_prefix_relative_and_idempotent(self):
        activate = shlex.quote(str(ACTIVATE_HOOK))
        lines = self.run_hooks(
            f'. {activate}\nprintf "%s\\n" "$LD_LIBRARY_PATH"\n'
            f'. {activate}\nprintf "%s\\n" "$LD_LIBRARY_PATH"\n'
        )
        self.assertEqual(lines, [self.runtime_paths, self.runtime_paths])

    def test_existing_library_path_is_appended_then_restored(self):
        activate = shlex.quote(str(ACTIVATE_HOOK))
        deactivate = shlex.quote(str(DEACTIVATE_HOOK))
        original = "/vendor/lib:/custom/lib"
        lines = self.run_hooks(
            f'. {activate}\nprintf "%s\\n" "$LD_LIBRARY_PATH"\n'
            f'. {deactivate}\nprintf "%s\\n" "$LD_LIBRARY_PATH"\n',
            original,
        )
        self.assertEqual(lines, [f"{self.runtime_paths}:{original}", original])

    def test_deactivation_restores_an_unset_variable_as_unset(self):
        activate = shlex.quote(str(ACTIVATE_HOOK))
        deactivate = shlex.quote(str(DEACTIVATE_HOOK))
        lines = self.run_hooks(
            f'. {activate}\n. {deactivate}\n'
            'if [ "${LD_LIBRARY_PATH+x}" = "x" ]; then '
            'printf "set\\n"; else printf "unset\\n"; fi\n'
        )
        self.assertEqual(lines, ["unset"])


if __name__ == "__main__":
    unittest.main()
