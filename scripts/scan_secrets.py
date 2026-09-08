#!/usr/bin/env python3
"""Run a checksum-pinned Gitleaks build on Git history and the working tree (Linux x64)."""
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = '8.30.1'
SHA256 = '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb'


def main():
    with tempfile.TemporaryDirectory(prefix='mq-secret-scan-') as temp:
        root = Path(temp)
        archive = root / 'gitleaks.tar.gz'
        url = f'https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/gitleaks_{VERSION}_linux_x64.tar.gz'
        urllib.request.urlretrieve(url, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError('Gitleaks checksum mismatch')
        binary = root / 'gitleaks'
        with tarfile.open(archive) as tar:
            binary.write_bytes(tar.extractfile('gitleaks').read())
        binary.chmod(0o700)
        results = []
        for mode, extra in [('git', ['--log-opts=--all']), ('dir', [])]:
            results.append(subprocess.run([str(binary), mode, '.', *extra, '--redact', '--config', '.gitleaks.toml']).returncode)
        if any(results):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
