import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_version", Path(__file__).with_name("release_version.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseVersionTests(unittest.TestCase):
    def test_first_release(self):
        self.assertEqual("0.1.0", module.select_version([], [], "0.1.0"))

    def test_patch_uses_numeric_order(self):
        self.assertEqual("0.1.11", module.select_version(["v0.1.9", "v0.1.10", "research-v2"], [], "0.1.0"))

    def test_retry_reuses_original_version(self):
        self.assertEqual("0.1.2", module.select_version(["v0.1.2", "v0.1.3"], ["v0.1.2"], "1.0.0"))

    def test_minor_and_major_floor(self):
        for floor in ["0.2.0", "1.0.0"]:
            self.assertEqual(floor, module.select_version(["v0.1.9"], [], floor))

    def test_invalid_versions_and_ambiguous_tags_fail(self):
        for value in ["1.0", "1.0.0-rc.1", "01.2.3", "1.0.0\nmalicious"]:
            with self.assertRaises(ValueError):
                module.parse_version(value)
        with self.assertRaises(ValueError):
            module.select_version([], ["v0.1.0", "v0.1.1"], "0.1.0")

    def test_real_remote_reservation_and_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def git(*args):
                subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
            git("init", "--bare", "remote.git")
            git("clone", "remote.git", "checkout")
            checkout = root / "checkout"
            def local(*args):
                subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True)
            local("config", "user.email", "test@example.invalid")
            local("config", "user.name", "Test")
            (checkout / "release-version.txt").write_text("0.1.0\n")
            local("add", ".")
            local("commit", "-m", "first")
            script = str(Path(__file__).with_name("release_version.py").resolve())
            def reserve():
                import sys
                return subprocess.check_output([sys.executable, script, "--reserve"], cwd=checkout, text=True).strip()
            self.assertEqual("0.1.0", reserve())
            self.assertEqual("0.1.0", reserve())
            local("commit", "--allow-empty", "-m", "second")
            self.assertEqual("0.1.1", reserve())
            self.assertEqual("0.1.1", reserve())


if __name__ == "__main__":
    unittest.main()
