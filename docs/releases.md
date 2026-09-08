# NuGet releases

MagicQuant is packaged as a .NET tool, package ID `MagicQuant`, executable `magicquant`. The package embeds the root `README.md` and `assets/icon.png`; package checks verify both files byte-for-byte and validate their NuGet metadata. Keep root README links absolute so the same content works on GitHub and NuGet. `.github/workflows/publish-nuget.yml` publishes when a commit reaches `release`, normally through a merged PR. Pushes directly to `release` also trigger it; use branch protection to require PRs if desired. Manual dispatch is available to retry publication and only runs on `release`.

## Trusted publishing setup

Configure the following on NuGet.org under your account's **Trusted Publishing** settings:

| Field | Value |
| --- | --- |
| Repository owner | `magiccodingman` |
| Repository | `MagicQuant` |
| Workflow filename | `publish-nuget.yml` |
| Environment | `release` |
| Package scope | `MagicQuant` (allow creation for the first release) |

The workflow filename has no `.github/workflows/` prefix in NuGet's policy. The publishing job uses **`environment: release`** and requests `id-token: write`. Create that GitHub environment and restrict its deployment branch to `release`. Add the repository secret **`NUGET_USER`** containing your NuGet profile username, not an email address or API key. Choose the intended package owner when creating the policy.

The workflow uses `NuGet/login@v1` to exchange GitHub OIDC identity for a temporary NuGet key immediately before pushing. No permanent NuGet API key is needed. See [NuGet's official guide](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing). Private-repository policies may need activation within NuGet's stated time window; make sure the source for a public package is accessible to its recipients when launching.

## Automatic versions

The first release is **0.1.0**. Each new release commit increments the greatest reserved stable version's patch number: `0.1.0`, `0.1.1`, `0.1.2`, and so on. You do not edit a version for ordinary patch releases.

`release-version.txt` is a **minimum next version**, not a counter. To release a new minor or major version, raise it in the release PR (for example, `0.2.0` or `1.0.0`). Leaving it unchanged continues patch increments. Only stable `major.minor.patch` versions are supported by the release workflow.

After Linux/Windows validation passes, the workflow reserves the chosen version with an annotated `vX.Y.Z` tag pointing at the exact commit. Tag creation on the remote is the atomic claim; concurrent releases retry a version collision without sharing the same version. Jobs are not canceled merely because another release arrives. Publication order may differ if runs finish at different times.

A retry of the **same commit** reuses its tag/version, even if later releases exist. A failed pack or push can leave a reserved tag; rerun that workflow to finish it. Do not delete or move release tags. A new commit gets a new version. Gaps are acceptable if a failed release is intentionally abandoned.

The release package is built with that version and source revision, installed into a clean tool directory, and tested before publication. `--skip-duplicate` makes retrying an already published version harmless; the first successfully published package remains authoritative. NuGet versions are immutable. A GitHub release record is created after publication.

## Validation and maintenance

PR CI validates Linux and Windows, Debug and Release, with locked restores and warnings as errors. Each job tests a locally packed `0.0.0-ci` package without publishing it. Release validation repeats ordinary and package checks on Linux and Windows before reserving a version. The exact versioned release package is then verified again on Linux before upload.

Package checks inspect bundled config, Python helper, native libraries, license/readme/icon metadata, and tool startup from a directory outside the source tree. They exercise config creation, refusal to overwrite, and read-only path validation. These are not full Windows campaigns or numerical parity tests.

The package uses committed dependencies; the installed llama.cpp/Python runtime is separately managed. Record both for research reproducibility. Required merge checks and environment policies are repository settings, not guaranteed merely by committing a workflow.

## Migration PR merge method

The launch PR connects the original Wiki and Pipeline histories through an unsquashed subtree import, then reorganizes the tree in later commits. **Merge that PR with a merge commit, not squash or rebase**, to retain both histories in `main`. Do not force-push the existing repository history. Future ordinary PRs can use the project's preferred merge policy.
