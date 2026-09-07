# Contributing

Start with the [architecture map](docs/architecture.md), [configuration rules](docs/configuration.md), and the [research wiki](https://github.com/magiccodingman/MagicQuant-Wiki). This repository implements benchmark-driven discovery; the `evolution` name survives as a compatibility alias.

## Local workflow

```sh
dotnet restore MagicQuant-Pipeline.sln --locked-mode
dotnet build MagicQuant-Pipeline.sln -c Debug --no-restore
dotnet test MagicQuant-Pipeline.sln -c Debug --no-build
dotnet build MagicQuant-Pipeline.sln -c Release --no-restore
dotnet test MagicQuant-Pipeline.sln -c Release --no-build
```

CI runs both configurations on Linux and Windows with warnings treated as errors and locked package restores. Use `--filter FullyQualifiedName~YourTestClass` to focus a test run during development. Tests run serially because configuration and runtime registries are global. Source-contract regression tests assume the normal repository/build layout; run the suite from the checkout rather than copying the test DLL elsewhere.

Keep personal settings in an ignored `config.local.yaml` and pass `--config` explicitly. Do not add machine paths or automatic DEBUG campaigns to `Program.cs`. Use IDE run arguments for your campaign. Never commit weights, runtime databases, exported GGUFs, credentials, or local logs.

## Making a change

- Keep commands focused on orchestration; extract cohesive policy or path logic into services when it can be tested independently.
- Use existing baseline identity, effective-state, and path helpers. Do not duplicate database filenames, tensor-slot ordering, or custom-baseline normalization.
- Explain why a non-obvious constraint exists in a comment. Avoid comments that merely restate a method call or retain obsolete blocks of disabled implementation.
- When adding a config option, update the typed model, loader override if needed, commented default YAML, documentation, and a behavior test. Distinguish C# defaults from the distributed YAML profile.
- Test observable behavior: boundary cases, context scoping, cache reuse/invalidation, ranking/tie rules, path resolution, or failure propagation. Avoid tests that only repeat the implementation's constants without exercising a contract.
- Treat persisted IDs, database schemas, manifests, and artifact names as compatibility contracts. Provide an explicit migration plan for changes to them.

For a bug fix, add a regression that fails without the fix. Tests that mutate `Config`, `Cache`, or registries must restore prior state in `finally`; use unique temporary roots. Keep unit tests free of network downloads, sudo, model quantization, and persistent changes to a developer's runtime.

## Hardware integration changes

Changes to conversion, quantization arguments, benchmark scheduling, or numerical selection need a small-model integration check in addition to unit tests. Record model identity, hardware, imatrix, dependency revisions, command/config, observed outputs, and before/after metrics. Use an isolated output directory. Do not claim full quantization parity from a passing unit suite.

For documentation or path refactoring, verify examples against actual help and protect historical path rules. Avoid re-running expensive full campaigns when the changed behavior can be checked directly.

## Pull request expectations

Describe the concrete problem and resulting behavior, relevant compatibility effects, and validation performed. Separate numerical policy changes from mechanical cleanup when possible. Mention untested hardware/platform paths and any remaining compiler warnings. Prefer focused commits that can be reviewed without reconstructing the conversation that led to them.

The maintainer still needs to choose a software license before an open-source release; do not infer one from generated model metadata or dependency licenses.

See [testing and merge checks](docs/testing.md) for the manual small-model workflow,
package lock updates, and required-check setup. [Worked examples](docs/extending.md)
show how to add configuration and test native/process/path changes.
