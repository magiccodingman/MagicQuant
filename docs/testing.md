# Testing and PR checks

## Ordinary checks

```sh
dotnet restore MagicQuant.sln --locked-mode -warnaserror
dotnet build MagicQuant.sln -c Release --no-restore -warnaserror
dotnet test MagicQuant.sln -c Release --no-build
```

Repeat with `-c Debug` when changing startup or compilation-dependent behavior. CI runs both configurations on Linux and Windows. It restores the committed NuGet lock files, treats warnings as errors, runs all ordinary tests, and uploads TRX reports. `MagicQuant.ProcessFixture` is a small offline executable used to test native process exit, full stdout/stderr pipes, literal arguments, and cancellation; it is not a user command.

The suite covers CLI startup/preflight, YAML contracts, managed/output path containment, symlinks on Linux, SQLite/DuckDB identity, native argument construction, log parsing, conversion success markers, concurrency policy, and existing research-policy regressions. It does not establish numerical equivalence for every model or hardware topology. Tests remain serial because the legacy runtime registries are shared.

To update dependencies intentionally, edit package versions, run an unlocked `dotnet restore`, review `packages.lock.json` changes, and rerun the suite. Audit with:

```sh
dotnet list MagicQuant.sln package --vulnerable --include-transitive
```

## Read-only campaign validation

```sh
dotnet run --project src/MagicQuant -c Release -- pipeline \
  --config config.local.yaml --check-config --strict-config
```

This reads YAML and local input/path metadata but does not create a database, initialize tools, clean artifacts, or quantize. `--strict-config` turns unknown/inactive YAML keys into errors; without it, those keys produce warnings. Normal runs perform the same preflight before runtime mutation. This check verifies local paths and model input structure, not available RAM, remote repository existence, tokenizer compatibility with every converter, or numerical quality.

## Opt-in small-model smoke test

The test is explicitly skipped in ordinary CI. On a prepared Linux machine, set these variables to existing resources and a dedicated writable output parent:

```sh
MQ_RUN_MODEL_SMOKE=1 \
MQ_SMOKE_MODEL=/data/models/small-model \
MQ_SMOKE_LLAMA_ROOT=/opt/llama.cpp \
MQ_SMOKE_RUNTIME_ROOT=/data/MagicQuant \
MQ_SMOKE_OUTPUT=/data/test-results/magicquant \
dotnet test tests/MagicQuant.Tests -c Release --filter Category=ModelSmoke
```

The runtime root must contain `MagicQuant-Env` with the converter/gguf dependencies already installed. The test does not install dependencies or download a model. It copies source metadata and links weights into a unique test model directory containing spaces, then exercises native conversion/reuse, Q8 scratch leases, export/reuse, GGUF metadata parity, the native CPU benchmark, and manifest-path writing. It has a 20-minute cancellation deadline. It removes generated GGUFs and input weight links and retains logs plus `smoke-result.json` beneath the output parent. The sample tensor-map JSON is a smoke artifact, not a full clone/release manifest.

This is an IO/toolchain smoke check, not a full discovery campaign or a PPL/KLD parity study. Quantization/selection policy changes still need before/after measurements on representative models.

A manual GitHub Actions workflow is provided for a trusted self-hosted runner labeled `magicquant-smoke`. Review the selected ref before dispatching it. It never runs automatically for an incoming PR, and no runner has been provisioned by this change. Do not route untrusted PR code to a machine containing private model or campaign data.

## Requiring checks before merge

The workflow reports failures; branch protection or a ruleset must require its checks to block merges. Configure `main` to require all four Linux/Windows Debug/Release test jobs after the workflow has run. If merge queues are enabled later, add a `merge_group` workflow trigger as well.

The unified MagicQuant repository is public. At launch preparation, `main` had no branch protection configured. Require the four `test (OS, Configuration)` checks plus `Secret scan` in repository settings if you want failures to block merging. Configure the `release` branch and deployment environment deliberately before publishing; a workflow alone does not prevent bypassing checks.

## Installed-package and release checks

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
dotnet pack src/MagicQuant -c Release --no-restore -p:Version=0.0.0-ci -o artifacts -warnaserror
python3 scripts/package_smoke.py artifacts/MagicQuant.0.0.0-ci.nupkg
```

Use `python` instead of `python3` where appropriate. The package smoke installs only from a temporary local feed, verifies shipped assets and native library presence, exercises CLI help/version and config creation outside the checkout, checks paths containing spaces, and confirms preflight is read-only. It never installs a model or native toolchain. Release-version tests use an isolated local bare Git remote; they never push to GitHub.

PR CI runs these checks on both operating systems. [Release documentation](releases.md) explains the separately gated trusted-publishing workflow.

## Secret checks

The `Secret scan` CI job runs a checksum-pinned Gitleaks binary on full fetched history and the current tree. On Linux x64 run `python3 scripts/scan_secrets.py`. Reports redact candidate credentials. `.gitleaks.toml` retains default detectors and narrowly allows only the exact known tensor-name test fixture; do not suppress whole directories to silence new findings.

A scanner is one check, not proof that every kind of sensitive information is absent. Review changes for private model names, personal paths, datasets, and credentials too. Git history preserves deleted files and author metadata. If a real credential is found, rotate it before planning any history rewrite.
