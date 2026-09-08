# MagicQuant

[![NuGet version](https://img.shields.io/nuget/v/MagicQuant.svg)](https://www.nuget.org/packages/MagicQuant/)
[![NuGet downloads](https://img.shields.io/nuget/dt/MagicQuant.svg)](https://www.nuget.org/packages/MagicQuant/)
[![Build and tests](https://github.com/magiccodingman/MagicQuant/actions/workflows/dotnet.yml/badge.svg)](https://github.com/magiccodingman/MagicQuant/actions/workflows/dotnet.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://github.com/magiccodingman/MagicQuant/blob/main/LICENSE)

**Benchmark-driven GGUF quantization and mixed-precision hybrid discovery for llama.cpp.**

MagicQuant helps answer: **which quantized versions of a model are worth keeping at each size?** It measures baseline quantizations, learns tensor-group assignments, explores hybrid combinations, and validates candidates against size and fidelity criteria. The result is a selected set of GGUF artifacts with supporting measurements, rather than an unranked collection of quantization levels.

It is a .NET command-line application that orchestrates llama.cpp and Python tooling. It does not invent a new quantization format or use evolutionary search. Hybrids must earn their place: a standard baseline can be the better result.

## How it works

1. **Establish baselines.** Read a local source model and measure standard quantization choices. Optionally learn tensor assignments from compatible external GGUFs.
2. **Probe tensor groups.** Measure how changes to groups such as attention, embeddings, and feed-forward tensors affect the model.
3. **Discover hybrids.** Use measured evidence and predictions to explore mixed-precision combinations with promising size/fidelity tradeoffs.
4. **Validate and select.** Measure candidates, reject poor or redundant trades, and export survivors with metadata and local provenance.

KLD and perplexity help evaluate fidelity; throughput and file size provide additional context. The results depend on the model, calibration/evaluation data, configuration, and hardware. A smaller KLD in one campaign is not a universal claim about downstream task quality. Read the [research overview](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/index.md) for the selection policy and its assumptions.

## Support the project

I build and maintain MagicQuant on the side, for free. Developing it and experimenting with quantizations has put a frankly ridiculous amount of terabytes written (TBW) on my drives! My Hugging Face storage is also creeping toward its cap, so there will eventually be more storage to fund. If this project helps you, [supporting the work](https://sayou.biz/support) helps with those costs. Anything helps and is always appreciated. ❤️

## Install and run

**Linux is the tested campaign platform.** Windows has automated build, unit-test, and packaged CLI checks; full Windows quantization campaigns have not been validated. No macOS campaign support is claimed.

Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), then install the CLI from [NuGet](https://www.nuget.org/packages/MagicQuant/):

```bash
dotnet tool install --global MagicQuant
magicquant --version
magicquant init-config --output config.yaml
```

The package becomes available after the first successful release publication; until then use the [source installation instructions](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md#build-from-source).

Edit the generated config for your source model, architecture identity, export destination, and storage. Initialize the external toolchain, validate the config, then start the campaign:

```bash
magicquant initialize-llama-cpp
magicquant pipeline --config ./config.yaml --check-config --strict-config
magicquant pipeline --config ./config.yaml
```

Initialization can download/build llama.cpp and install Python dependencies. NuGet installs MagicQuant, not model weights or a complete GPU toolchain. Follow the [installation guide](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md) for native prerequisites, GPU setup, custom toolchains, and environment paths.

For updates: `dotnet tool update --global MagicQuant`. For reproducible runs, install a particular release with `--version X.Y.Z` and retain your config, model revision, and run provenance.

## Configure a campaign

A minimal example (replace these paths and the architecture identity):

```yaml
paths:
  model_dir: /data/models/my-model
  scratch_roots:
    - /mnt/nvme-a/magicquant-scratch
    - /mnt/nvme-b/magicquant-scratch
identity:
  architecture_family_name: my-model-family
output:
  output_dir: /data/exports/my-model-MagicQuant
  output_name_prefix: MyModel
learning:
  confirm_tensor_group_profile: true
```

Custom YAML uses typed defaults for omitted values; it does not merge with the bundled tuning profile. Start with `init-config` when you want that complete profile. See [configuration](https://github.com/magiccodingman/MagicQuant/blob/main/docs/configuration.md), [examples](https://github.com/magiccodingman/MagicQuant/tree/main/examples), and the [command reference](https://github.com/magiccodingman/MagicQuant/blob/main/docs/commands.md).

**Plan scratch storage early.** Quantization writes and rereads large intermediate models, and storage can be a major throughput limitation. Fast SSD/NVMe scratch disks, especially separate physical devices, can materially improve throughput when IO is the bottleneck. Multiple folders on the same device still share its bandwidth. Allow space for concurrent intermediate artifacts and keep unrelated data out of managed scratch/export directories. See [storage](https://github.com/magiccodingman/MagicQuant/blob/main/docs/storage.md) and [best practices](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md).

## Learning from external quantizations

External providers are optional. MagicQuant can run using its local baseline choices alone, but compatible external tensor assignments can provide valuable additional evidence.

**Unsloth is the maintainer's recommended starting point** for external GGUF baselines. MagicQuant can learn their tensor-group patterns, rebuild a controlled equivalent from your local source model, and benchmark it in your campaign. It does not simply trust an external file's label or score. Choose the exact matching model and revision, and review its license. See the [Unsloth configuration walkthrough](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md#optional-unsloth-baselines) and [research explanation](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Learning-From-Existing-Quantizations.md).

For the same model, prefer linking to the original provider's surviving baselines. For a compatible variant they do not host, cloning can rebuild the full selected set locally. Learning tensor assignments does not automatically reproduce a provider's other processing techniques. See [publishing and cloning guidance](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md#link-upstream-for-the-same-model-build-locally-for-variants).

## Documentation

| Start here | What you will find |
| --- | --- |
| [Installation](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md) | NuGet, native prerequisites, custom environments, source builds |
| [Configuration](https://github.com/magiccodingman/MagicQuant/blob/main/docs/configuration.md) | YAML, overrides, read-only validation, profiles |
| [Commands](https://github.com/magiccodingman/MagicQuant/blob/main/docs/commands.md) | Pipeline, setup, cloning, prediction validation |
| [Best practices](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md) | Scratch disks, Unsloth, reproducibility, first campaigns |
| [Manifests and cloning](https://github.com/magiccodingman/MagicQuant/blob/main/docs/manifests-and-cloning.md) | Publish a reproducible release; rebuild from another MagicQuant repository |
| [Storage](https://github.com/magiccodingman/MagicQuant/blob/main/docs/storage.md) | Persistent data, scratch leases, cache and output ownership |
| [Research](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/index.md) | Measurements, prediction, pruning, hybrid selection |
| [Contributing](https://github.com/magiccodingman/MagicQuant/blob/main/CONTRIBUTING.md) | Development workflow, tests, code boundaries |
| [Releases](https://github.com/magiccodingman/MagicQuant/blob/main/docs/releases.md) | Automatic versions and NuGet trusted publishing |

## Development and history

Application code lives in `src/`, tests in `tests/`, operational guides in `docs/`, and research documentation in `wiki/`. Both the former MagicQuant-Wiki and MagicQuant-Pipeline histories are retained. The `evolution` command remains a compatibility alias for `pipeline`; existing database and artifact contracts are preserved. Historical research remains under `archival/` and is not current setup guidance.

## License

MagicQuant's original code and documentation are licensed under **GNU AGPL version 3 only** (`AGPL-3.0-only`). Commercial use is permitted subject to its terms. Distribution and remote interaction with modified versions carry source-availability obligations; the [license text](https://github.com/magiccodingman/MagicQuant/blob/main/LICENSE) controls the details.

This does not automatically relicense model weights or generated GGUFs. Model, dataset, external-provider, and third-party dependency licenses still apply. See [third-party notices](https://github.com/magiccodingman/MagicQuant/blob/main/THIRD-PARTY-NOTICES.md).
