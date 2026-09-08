# MagicQuant

![MagicQuant](https://raw.githubusercontent.com/magiccodingman/MagicQuant/main/assets/icon.png)

[![NuGet version](https://img.shields.io/nuget/v/MagicQuant.svg)](https://www.nuget.org/packages/MagicQuant/)
[![NuGet downloads](https://img.shields.io/nuget/dt/MagicQuant.svg)](https://www.nuget.org/packages/MagicQuant/)
[![Build and tests](https://github.com/magiccodingman/MagicQuant/actions/workflows/dotnet.yml/badge.svg)](https://github.com/magiccodingman/MagicQuant/actions/workflows/dotnet.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://github.com/magiccodingman/MagicQuant/blob/main/LICENSE)

**Discover better GGUF size/fidelity tradeoffs. Benchmark the results. Share the tensor recipes.**

MagicQuant is a **benchmark-driven LLM quantization, mixed-precision hybrid discovery, and tensor-configuration cloning system for llama.cpp**. It learns from existing quantization strategies, explores combinations across tensor groups, builds promising GGUFs, and measures which ones deserve a place in the final release.

A campaign gives you more than another quantized file: **a measured selection of useful models, an explanation of what survived and why, and manifests that let other people rebuild those configurations.**

[Browse the models](https://huggingface.co/collections/magiccodingman/magic-quant) · [See the results](#magicquant-in-the-wild) · [Install](#install-and-run) · [Clone a release](#clone-the-recipes-onto-your-own-model) · [Read the research](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/index.md)

## Which quants are actually worth keeping?

Q8, Q6, Q5, Q4: familiar names tell you roughly how a model was compressed. They do not tell you whether a particular model has a better trade hiding between those choices—or whether two downloads offer almost the same thing.

MagicQuant investigates that space. It can combine one baseline's attention recipe with another's feed-forward recipe, protect groups that are expensive to damage, and compress groups where the measurements justify it. Candidates earn a place by reducing size and KLD together, delivering an unusually worthwhile fidelity improvement for a small size premium, or beating the expected trade between neighboring choices.

**Every survivor has to earn its slot.** The final selection can contain llama.cpp baselines, configurations learned from Unsloth, and MagicQuant hybrids. The goal is a release where each download represents a meaningful choice.

## Support the project

I build and maintain MagicQuant on the side, for free. Developing it and experimenting with quantizations has put a frankly ridiculous amount of terabytes written (TBW) on my drives! My Hugging Face storage is also creeping toward its cap, so there will eventually be more storage to fund. If this project helps you, [supporting the work](https://sayou.biz/support) helps with those costs. Anything helps and is always appreciated. ❤️

## Nonlinear wins: more fidelity for the extra bytes

A hybrid sitting between Q5 and Q6 is useful when it offers a **better trade than the normal step up**. MagicQuant draws a straight-line size/KLD comparison between neighboring survivors, then checks whether a measured hybrid beats that line. On a graph with file size on the horizontal axis and KLD on the vertical axis, a nonlinear winner sits **below the line**: less divergence than the reference trade at that size.

The original 4B campaign makes this concrete:

| Choice | Size (GB) | Measured KLD ↓ |
| --- | ---: | ---: |
| UD-Q5_K_XL recipe | 2.73 | 0.009839 |
| **MQ-Q5_K_1 hybrid** | **2.88** | **0.006632** |
| LM-Q6_K baseline | 3.08 | 0.004640 |

At 2.88 GB, the straight-line comparison gives about **0.007611 KLD**. The hybrid measures **0.006632—about 12.9% below that line**. Compared with the smaller UD recipe, it buys about **32.6% lower KLD for 5.5% more storage**. Those percentages use the rounded table values.

That is why an intermediate hybrid can earn a download slot. MagicQuant also keeps dominance wins: lower measured KLD at the same or smaller size. Its [selection rules](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Nonlinear-Winners-And-Survivors.md) explain both cases; the [original worked example](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/overview.md#example) preserves the full table and tensor recipes. The UD result is a controlled reconstruction of the external recipe, not a benchmark of the original uploaded artifact.

## MagicQuant in the wild

**[Explore the MagicQuant collection on Hugging Face →](https://huggingface.co/collections/magiccodingman/magic-quant)**

Published releases put the research to work on dense models, mixture-of-experts models, and related fine-tunes. Here are three examples of what that makes possible. Sizes below are decimal GB; KLD measures divergence from each experiment's reference output distributions, with lower values indicating closer agreement on that evaluation.

### Discover useful choices between the usual quant sizes

The [Qwen3.6-35B-A3B release](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-MagicQuant-GGUF) contains **eight MagicQuant hybrids alongside two baseline recipes**. Selected points from its ten-entry survivor manifest show the range:

| Recipe | Source | Size (GB) ↓ | KLD ↓ |
| --- | --- | ---: | ---: |
| LM-Q8_0 | llama.cpp | 36.90 | 0.004654 |
| **MQ-Q6_K_1** | **MagicQuant** | **31.59** | **0.005149** |
| **MQ-Q5_K_1** | **MagicQuant** | **29.19** | **0.005523** |
| **MQ-Q4_K_M_1** | **MagicQuant** | **24.82** | **0.007799** |
| **MQ-IQ3_M_1** | **MagicQuant** | **17.60** | **0.026330** |
| UD-IQ3_S | Unsloth-derived | 13.68 | 0.068376 |

These are measured options at different storage budgets. The full release reaches down to a 9.59 GB hybrid, with a correspondingly larger divergence. The [complete table and pinned evidence](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/showcase.md#discovery-a-35b-moe-release) make that trade visible.

### Take a discovered recipe set to another model

The [Qwen3.6-35B-A3B Uncensored release](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-Uncensored-MagicQuant-GGUF) reuses **all ten configurations** from the original release on a related model by llmfan46, including the external-derived baseline. It rebuilds the artifacts and records fresh measurements rather than repeating the full discovery search.

For example, its cloned **MQ-Q4_K_M_1 is 24.82 GB at 0.007832 KLD**, measured against the new model's reference. The manifest makes the original work reusable; the clone benchmarks record what happened on the new weights. [See the clone evidence](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/showcase.md#cloning-ten-recipes-on-a-related-model), or [clone a release yourself](#clone-the-recipes-onto-your-own-model).

### Explore beyond a conventional starting point

A [specialized Qwen3.8-27B MXFP4 experiment](https://huggingface.co/magiccodingman/Qwen3.8-27B-MXFP4-MagicQuant-GGUF) adapted existing MagicQuant recipes to AMD's already-quantized Quark AWQ MXFP4 model. Its downward-only conversion preserved tensors that were already smaller, producing a **14.98 GB artifact from an 18.89 GB reference—20.70% less storage—with 0.000940 measured KLD**.

Here the reference is the **native MXFP4 GGUF**, so that KLD must not be compared with the other campaigns' numbers. This release used specialized adaptation work beyond ordinary CLI cloning; it illustrates an experiment built around reusable recipes, not an automatic promise for every input format. [Read the method, limits, and publication decisions](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/showcase.md#specialized-research-downward-only-mxfp4-adaptation).

These are published measurements, not new benchmarks run for this README. KLD is one fidelity signal, not a general capability score. External recipes in discovery are rebuilt under controlled local conditions; their results do not establish that an upstream provider's original uploads are better or worse.

## How MagicQuant finds those trades

### Learn the recipes inside real quantizations

Even a baseline with a single name such as Q5_K can use different quantization types across its tensors. MagicQuant reads those actual assignments and organizes them into architecture-aware groups: embeddings, attention, feed-forward layers, output heads, and supported MoE or hybrid-architecture groups.

Standard llama.cpp baselines supply the starting vocabulary. Compatible external GGUFs can add more recipes. **Unsloth is the maintainer's recommended starting point**, and external providers are optional. See [learning from existing quantizations](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Learning-From-Existing-Quantizations.md).

### Measure locally, search intelligently

Trying every combination quickly becomes impractical. MagicQuant first measures isolated group changes, then uses those observations to predict which combinations deserve a real build. Its rank-safe predictor and DuckDB candidate search help focus the benchmark budget on promising dominance, near-baseline, and interior tradeoffs.

When the main search fails to validate a winner, a bounded fallback can revisit isolated evidence for smaller improvements or carefully budgeted protection of sensitive groups. **Predictions choose what to test; actual GGUF benchmarks decide what survives.** Read the [prediction engine guide](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Prediction-Engine.md).

### Investigate interactions and surprising wins

A group recipe that works well in a lightly compressed model may behave differently when the surrounding groups are more compressed. MagicQuant can run controlled context probes, investigate beneficial or harmful interactions, and learn scoped exceptions when a lower-bit choice performs unexpectedly well.

Those probes are bounded so investigating interactions does not recreate an exhaustive search. Aggressive low-fidelity context probing is opt-in. See [context-aware tensor search](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Regime-Aware-Search.md).

### Build a release people can understand

MagicQuant measures candidates, removes dominated or poor trades, and curates the remaining choices into a useful survivor list. It exports GGUFs, a model card, measurements, replacement explanations, and exact tensor maps for cloning.

You can inspect **which assignments produced an artifact, what it replaced, and the measurements behind the decision**. The [manifest guide](https://github.com/magiccodingman/MagicQuant/blob/main/docs/manifests-and-cloning.md) explains what to publish and how others can use it.

## Clone the recipes onto your own model

**A useful quantization recipe should travel with the release.** If someone publishes a MagicQuant model with its clone manifest, you can point the CLI at their Hugging Face repository and rebuild the selected configurations from your own compatible source weights.

This is especially useful for fine-tunes, uncensored variants, and other related models that the original quantization provider does not host. It gives you a practical starting set without repeating the full discovery search. Clone mode rebuilds and benchmarks the manifest entries locally, including both MagicQuant hybrids and external-derived configurations.

After [installing the CLI and preparing the toolchain](#install-and-run), create a config and edit it for your target model, architecture, storage, and any imatrix requirements:

```bash
magicquant init-config --output clone.yaml
magicquant clone-repository-quants \
  --config ./clone.yaml \
  --source-repo magiccodingman/Qwen3.6-35B-A3B-MagicQuant-GGUF
```

This example uses the published 35B recipe set above; choose a source release compatible with your target. `--source-repo` takes a Hugging Face **model repository ID**. You can also supply a local clone JSON file or a pinned raw JSON URL with `--source-json`. The source release needs `magicquant-manifest/magicquant.clone-configs.json`; your local input needs the complete compatible source model.

Publish your own generated manifest folder alongside the release, and other people can do the same with your configurations. See the [publishing and cloning walkthrough](https://github.com/magiccodingman/MagicQuant/blob/main/docs/manifests-and-cloning.md).

Cloning transfers tensor assignments, not proof that a recipe is optimal for different weights. It also does not automatically reproduce a provider's additional transformations, calibration recipes, or custom processing. Fresh discovery is appropriate when model changes or local measurements warrant it.

### Give the original providers their downloads

For the same model, the recommended pipeline default is to **link to an external provider's surviving baselines** rather than re-host copies. Leave `output.export_external_learned_baselines: false`; original creators keep the attribution and downloads they earned.

For a variant they do not host, build those configurations locally. The clone command already rebuilds all manifest entries and does not need that pipeline flag enabled. The [best-practices guide](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md#link-upstream-for-the-same-model-build-locally-for-variants) covers the distinction.

## Put your hardware to work

Discovery involves many large writes and real benchmarks. MagicQuant manages both sides of that workload:

- **Measured GPU scheduling.** It probes viable shared multi-GPU and independent per-GPU execution, then uses candidate size and available batch work to choose a suitable topology. Larger candidates can share GPUs; smaller ready-to-run candidates can use independent workers when measurements favor it.
- **Dedicated scratch storage.** Temporary GGUF work uses scratch leases with one heavy writer per configured root. Fast SSD/NVMe storage on separate physical devices can substantially help when IO is limiting throughput.
- **Reusable evidence.** Persistent model/profile/imatrix identities, cached benchmark truth, and cached hardware plans let compatible later work reuse prior results. Local run records capture settings and available toolchain versions.

As one documented scheduling example, a two-RTX-3090 campaign measured **0.762 jobs/second with independent workers versus 0.529 with shared workers** for the tested sub-crossover workload—about 44% more aggregate throughput. That is a workload-specific result of measuring the topology, not a promised speedup for every machine. Read the [GPU scheduling study](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/GPU-Benchmark-Scheduling.md).

**Plan scratch disks early.** Repeated multi-gigabyte writes can be one of a campaign's biggest bottlenecks. Two folders on the same disk still share its bandwidth; leave enough free space for concurrent artifacts and keep unrelated data outside managed scratch/export directories. Start with [storage](https://github.com/magiccodingman/MagicQuant/blob/main/docs/storage.md) and [best practices](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md).

## Install and run

**Linux is the tested campaign platform.** Windows has automated build, unit-test, and packaged CLI checks; full Windows campaigns have not been validated. Model/architecture support depends on the installed llama.cpp converter and MagicQuant's tensor-group mappings.

Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), then install MagicQuant from [NuGet](https://www.nuget.org/packages/MagicQuant/):

```bash
dotnet tool install --global MagicQuant
magicquant --version
magicquant init-config --output config.yaml
```

Edit the generated YAML for your source model and intended output. This excerpt shows the settings to start with:

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

Prepare the native toolchain, check the configuration, and begin discovery:

```bash
magicquant initialize-llama-cpp
magicquant pipeline --config ./config.yaml --check-config --strict-config
magicquant pipeline --config ./config.yaml
```

Initialization can download/build llama.cpp and install Python dependencies. Follow the [setup guide](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md) for native prerequisites, GPU setup, existing toolchains, and source installation. `init-config` gives you a commented, model-neutral profile and never overwrites an existing file. Keep separate YAML files for different campaigns; `--check-config` validates local settings without starting the work.

A minimal custom YAML uses typed defaults for omitted settings; it does not merge with the bundled tuning profile. The generated full profile, [configuration reference](https://github.com/magiccodingman/MagicQuant/blob/main/docs/configuration.md), and [opt-in external-provider example](https://github.com/magiccodingman/MagicQuant/blob/main/examples/pipeline-external.yaml) explain how to configure a run deliberately.

Update with `dotnet tool update --global MagicQuant`. Install an exact release with `--version X.Y.Z` when you need to keep a campaign's application version fixed.

## Explore MagicQuant

| I want to… | Start here |
| --- | --- |
| Browse published results and their evidence | [Model showcase](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/showcase.md) and [Hugging Face collection](https://huggingface.co/collections/magiccodingman/magic-quant) |
| Understand the results and see the full worked example | [Research overview](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/overview.md) |
| Understand how predictions become measured winners | [Prediction engine](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Prediction-Engine.md) and [survivor selection](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/docs/Nonlinear-Winners-And-Survivors.md) |
| Install and configure my first campaign | [Setup](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md), [configuration](https://github.com/magiccodingman/MagicQuant/blob/main/docs/configuration.md), and [commands](https://github.com/magiccodingman/MagicQuant/blob/main/docs/commands.md) |
| Learn from Unsloth and plan storage | [Best practices](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md) |
| Publish a release or clone someone else's | [Manifests and cloning](https://github.com/magiccodingman/MagicQuant/blob/main/docs/manifests-and-cloning.md) |
| Inspect the research and its assumptions | [Research index](https://github.com/magiccodingman/MagicQuant/blob/main/wiki/index.md) |
| Add a capability or contribute a fix | [Contributor guide](https://github.com/magiccodingman/MagicQuant/blob/main/CONTRIBUTING.md) |

## License

MagicQuant is open source under **GNU AGPL version 3 only** (`AGPL-3.0-only`). Commercial use is permitted subject to its terms; distribution and remote interaction with modified versions carry source-availability obligations. See the [license](https://github.com/magiccodingman/MagicQuant/blob/main/LICENSE).

Model weights, generated GGUFs, datasets, and external dependencies retain their applicable licenses. See [third-party notices](https://github.com/magiccodingman/MagicQuant/blob/main/THIRD-PARTY-NOTICES.md).
