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

The repository's current private-repository plan returned HTTP 403 when branch protection was queried, explaining that an eligible plan or public visibility is required. The code change cannot override that GitHub restriction. Once supported, enable required checks and verify that a deliberately failing test PR cannot merge. Choosing visibility, billing, and the project software license remains a maintainer decision.
