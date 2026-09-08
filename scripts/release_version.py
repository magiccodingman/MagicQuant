#!/usr/bin/env python3
"""Reserve a release version with an immutable remote Git tag.

Tag creation is the atomic claim: concurrent releases retry collisions, while
reruns of the same commit reuse its reservation even if publication failed.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)\Z")


def parse_version(value):
    match = PATTERN.fullmatch("v" + value)
    if not match:
        raise ValueError("Release version must be a stable major.minor.patch value.")
    return tuple(map(int, match.groups()))


def select_version(tags, current_tags, minimum):
    existing = [t for t in current_tags if PATTERN.fullmatch(t)]
    if len(existing) > 1:
        raise ValueError("Commit has multiple release tags; resolve the ambiguity manually.")
    if existing:
        return existing[0][1:]
    versions = [parse_version(t[1:]) for t in tags if PATTERN.fullmatch(t)]
    version = parse_version(minimum)
    if versions:
        major, minor, patch = max(versions)
        version = max(version, (major, minor, patch + 1))
    return ".".join(map(str, version))


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def reserve(minimum):
    head = git("rev-parse", "HEAD").stdout.strip()
    for _ in range(20):
        git("fetch", "origin", "--tags")
        tags = git("tag", "--list").stdout.splitlines()
        current = git("tag", "--points-at", head).stdout.splitlines()
        version = select_version(tags, current, minimum)
        tag = "v" + version
        if tag in current:
            # A local tag alone is not a successful remote reservation.
            remote = git("ls-remote", "origin", "refs/tags/" + tag).stdout.strip()
            if remote:
                return version
        else:
            git("tag", "-a", tag, "-m", f"MagicQuant {version}; reserved for {head}", head)
        pushed = git("push", "origin", "refs/tags/" + tag, check=False)
        if pushed.returncode == 0:
            return version
        git("tag", "-d", tag)
        # Only a genuine tag race is retryable; auth/network failures must fail.
        if not git("ls-remote", "origin", "refs/tags/" + tag).stdout.strip():
            raise RuntimeError(pushed.stderr)
    raise RuntimeError("Could not reserve a release version after 20 concurrent tag collisions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", action="store_true", help="Create and push an immutable version tag")
    args = parser.parse_args()
    minimum = Path("release-version.txt").read_text().strip()
    parse_version(minimum)
    if args.reserve:
        version = reserve(minimum)
    else:
        version = select_version(git("tag", "--list").stdout.splitlines(),
                                 git("tag", "--points-at", "HEAD").stdout.splitlines(), minimum)
    print(version)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"version={version}\n")


if __name__ == "__main__":
    main()
