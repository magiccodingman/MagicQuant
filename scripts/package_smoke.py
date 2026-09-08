#!/usr/bin/env python3
"""Install the actual nupkg into a clean local tool directory and test portable startup."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile


def run(args, cwd, success=True, env=None):
    result = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True, timeout=120, env=env)
    if success and result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    if not success and result.returncode == 0:
        raise AssertionError("Command unexpectedly succeeded: " + str(args))
    return result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        for expected in ["README.md", "LICENSE", "icon.png", "THIRD-PARTY-NOTICES.md",
                         "tools/net10.0/any/config.default.yaml", "tools/net10.0/any/Helpers/pip_runner.py",
                         "tools/net10.0/any/MQ.DB.dll"]:
            assert expected in names, f"Missing packaged asset: {expected}"
        for native in ["libe_sqlite3.so", "e_sqlite3.dll", "libduckdb.so", "duckdb.dll"]:
            assert any(n.endswith('/'+native) for n in names), f"Missing native asset: {native}"
        assert any(n.startswith("licenses/") for n in names), "Missing third-party license texts"
        repository = Path(__file__).resolve().parent.parent
        assert archive.read("README.md") == (repository / "README.md").read_bytes(), "Package README differs from the root README"
        assert archive.read("icon.png") == (repository / "assets/icon.png").read_bytes(), "Package icon differs from the repository icon"
        root = ET.fromstring(archive.read("MagicQuant.nuspec"))
        ns = {"n": root.tag.split("}")[0][1:]}
        meta = root.find("n:metadata", ns)
        version = meta.find("n:version", ns).text
        assert meta.find("n:license", ns).text == "AGPL-3.0-only"
        assert meta.find("n:readme", ns).text == "README.md"
        assert meta.find("n:icon", ns).text == "icon.png"
    with tempfile.TemporaryDirectory(prefix="magicquant package smoke ") as temp:
        root = Path(temp)
        feed = root / "feed"
        feed.mkdir()
        import shutil
        shutil.copy2(package, feed / package.name)
        config = root / "nuget.config"
        config.write_text('<configuration><packageSources><clear/><add key="local" value="' + str(feed) + '"/></packageSources></configuration>')
        tool = root / "tool"
        run(["dotnet", "tool", "install", "MagicQuant", "--tool-path", tool, "--version", version,
             "--configfile", config], root, env={**os.environ, "NUGET_PACKAGES": str(root / "packages")})
        command = tool / ("magicquant.exe" if os.name == "nt" else "magicquant")
        work = root / "unrelated working directory"
        work.mkdir()
        assert run([command, "--version"], work).strip().split("+")[0] == version
        for sub in [[], ["pipeline"], ["init-config"], ["initialize-llama-cpp"], ["clone-repository-quants"]]:
            run([command, *sub, "--help"], work)
        assert list(work.iterdir()) == [], "Help created runtime files"
        run([command, "init-config", "--output", "campaign with spaces.yaml"], work)
        campaign = work / "campaign with spaces.yaml"
        original = campaign.read_bytes()
        assert b"scratch_roots:" in original and b"custom_repositories:" in original
        run([command, "init-config", "--output", campaign], work, success=False)
        assert campaign.read_bytes() == original, "init-config overwrote user settings"
        # Fake input metadata is enough for read-only path checks, not conversion.
        model = work / "model with spaces"
        model.mkdir()
        (model / "config.json").write_text('{}')
        (model / "model.safetensors").touch()
        import json
        preflight = work / "preflight.yaml"
        preflight.write_text('paths:\n  model_dir: ' + json.dumps(str(model)) + '\n  magic_quant_root: ' +
                             json.dumps(str(work / 'runtime')) + '\nidentity:\n  architecture_family_name: package-test\n')
        run([command, "pipeline", "--config", preflight, "--check-config", "--strict-config"], work)
        assert not (work / "runtime").exists() and not (model / "MagicQuant").exists()
        assert "not found" in run([command, "pipeline", "--config", "missing.yaml"], work, success=False)
    print(f"Installed package {version}: assets, native libraries, help, config generation, and path preflight passed.")


if __name__ == "__main__":
    main()
