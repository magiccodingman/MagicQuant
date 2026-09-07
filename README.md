# MagicQuant Pipeline

MagicQuant is a benchmark-driven GGUF evaluation and hybrid-discovery system. It measures standard and external quantization baselines, probes tensor groups, predicts promising combinations, and validates final survivors with real benchmarks. Hybrids earn a place only when their size/fidelity tradeoff is worthwhile.

This repository contains the .NET command-line application. The [MagicQuant research wiki](https://github.com/magiccodingman/MagicQuant-Wiki) explains the methodology and results. Despite the historical `evolution` command name, the current pipeline does **not** perform evolutionary search.

## Build and inspect

Install the .NET 10 SDK, then run from the repository root:

```sh
dotnet restore MagicQuant-Pipeline.sln
dotnet build MagicQuant-Pipeline.sln -c Release
dotnet test MagicQuant-Pipeline.sln -c Release --no-build
dotnet run --project MagicQuant -c Release --no-build -- --help
dotnet run --project MagicQuant -c Release --no-build -- pipeline --help
```

Building, testing, and viewing help do not require model weights or llama.cpp. Running without arguments also shows help, in both Debug and Release.

## Run a model

Real quantization needs a complete local Hugging Face model directory (top-level `.safetensors`, model configuration, and tokenizer assets), llama.cpp, a Python environment, and enough RAM/VRAM and disk space for native GGUFs, baselines, logits, and exports. Hardware requirements depend on the model. Linux with an apt-based distribution is the primary automatic setup path; Windows has setup code but is not covered by the Linux CI job. Automatic macOS setup is not implemented.

1. Copy the distributed tuning profile and edit the paths and model identity:

   ```sh
   cp MagicQuant/config.default.yaml config.local.yaml
   ```

   Set `paths.model_dir` and `identity.architecture_family_name`. Choose a dedicated `output.output_dir` and set `output.output_name_prefix`. Before publishing generated model cards, set `readme.frontmatter` to the source model's actual license and metadata. Use absolute paths for a portable campaign invocation.

2. Prepare dependencies:

   ```sh
   dotnet run --project MagicQuant -c Release --no-build -- initialize-llama-cpp
   ```

   This can download/build llama.cpp, install Python packages, and request sudo for apt packages on Linux. It uses `<user-home>/MagicQuant`. To use existing llama.cpp files, configure **all three** of `paths.llama_root`, `paths.llama_bin`, and `paths.convert_script`, and pass `--config config.local.yaml`. See [setup](docs/setup.md) for Python requirements and custom runtime roots.

3. Start the campaign:

   ```sh
   dotnet run --project MagicQuant -c Release --no-build -- pipeline --config config.local.yaml
   ```

   Review the tensor grouping prompt before allowing learning to continue. The run learns/reuses benchmark truth and exports its selected survivors. Runtime dependency validation may perform setup when using the default environment.

**Use a dedicated export directory:** normal export cleans/rebuilds its contents. `--reuse-existing-final-artifacts` permits reuse only when artifacts match the command's validation rules. Do not point output at your source model directory or another directory containing files you need to keep.

## Commands

| Command | Purpose |
| --- | --- |
| `pipeline` | Full baseline learning, isolation probing, prediction, real validation, and export |
| `evolution` | Backward-compatible alias for `pipeline` |
| `build-hybrids` | Existing entry point for the full pipeline, including export; not an export-only shortcut |
| `clone-repository-quants` | Rebuild configurations from a compatible repository or clone manifest |
| `validate-predictions` | Compare predictions with existing SQLite benchmark truth and export reports |
| `initialize-llama-cpp` | Set up or update native/Python dependencies |

Append `--help` to any command. Arguments after `--` belong to MagicQuant, not `dotnet run`. [Command examples](docs/commands.md) cover cloning and validation.

## Documentation

- [Setup and troubleshooting](docs/setup.md)
- [Configuration and path rules](docs/configuration.md)
- [Commands and workflows](docs/commands.md)
- [Architecture and code map](docs/architecture.md)
- [Storage, caching, and reruns](docs/storage.md)
- [Contributing and tests](CONTRIBUTING.md)
- [Compatibility notes for existing users](docs/migration.md)

The small [example configurations](examples/) demonstrate the required fields. They use C# defaults for omitted settings; they are **not** merged with `config.default.yaml`. Copy the full default file when you want its distributed tuning values.

## Project status

The research pipeline is active software with model- and hardware-dependent integration requirements. Unit/regression tests run without quantizing a model; a passing test suite alone does not establish numerical parity for a full hardware campaign. The repository does not yet contain a software license; the maintainer must choose one before an open-source release. A generated model card's license field does not license this program.
