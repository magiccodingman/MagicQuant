# MagicQuant

**Benchmark-driven GGUF quantization and mixed-precision hybrid discovery for llama.cpp.**

MagicQuant measures baselines, learns tensor-group assignments, discovers promising hybrid quantizations, and validates size/fidelity tradeoffs before exporting selected GGUF artifacts. It is a command-line tool, not an evolutionary search algorithm or a new quantization format.

## Install

Install the .NET 10 SDK, then:

```sh
dotnet tool install --global MagicQuant
magicquant init-config --output config.yaml
```

Edit model, architecture, output, and scratch paths, then prepare the native toolchain and run:

```sh
magicquant initialize-llama-cpp
magicquant pipeline --config config.yaml --check-config --strict-config
magicquant pipeline --config config.yaml
```

NuGet does not bundle model weights or a ready-to-use GPU toolchain. Linux is the tested campaign platform; Windows has automated build/unit/package checks but full campaigns remain unvalidated.

Fast, separate physical scratch disks can help substantially when repeated large GGUF writes are the bottleneck. External quantization providers are optional; Unsloth is the maintainer's recommended starting point for compatible tensor-assignment evidence.

- [Project briefing and research](https://github.com/magiccodingman/MagicQuant)
- [Installation and prerequisites](https://github.com/magiccodingman/MagicQuant/blob/main/docs/setup.md)
- [Configuration](https://github.com/magiccodingman/MagicQuant/blob/main/docs/configuration.md)
- [Scratch disks and optional Unsloth learning](https://github.com/magiccodingman/MagicQuant/blob/main/docs/best-practices.md)
- [Support spare-time development and storage costs](https://sayou.biz/support)

MagicQuant is licensed under **AGPL-3.0-only**. Model weights and generated GGUFs retain their applicable licenses. See [license and third-party notices](https://github.com/magiccodingman/MagicQuant/blob/main/THIRD-PARTY-NOTICES.md).
