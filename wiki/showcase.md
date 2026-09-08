# Published MagicQuant models: results and reusable recipes

The [MagicQuant collection](https://huggingface.co/collections/magiccodingman/magic-quant) contains downloadable releases, model cards, and manifests. These examples were checked against pinned published revisions on September 7, 2026. They are existing release measurements; this documentation update did not run new model benchmarks.

Sizes use decimal GB (1,000,000,000 bytes), rounded to two decimals. KLD values describe output-distribution divergence on each release's evaluation. They are not task-accuracy scores, and scores from different reference models or evaluation setups are not directly comparable.

## Discovery: a 35B MoE release

Qwen3.6-35B-A3B's ten-entry selection contains eight MagicQuant hybrids, a llama.cpp baseline, and an Unsloth-derived baseline. The complete survivor list shows both the low-divergence end and the storage savings that require larger fidelity sacrifices.

| Recipe | Size (GB) | KLD |
| --- | ---: | ---: |
| LM-Q8_0 | 36.90 | 0.004654 |
| MQ-Q6_K_1 | 31.59 | 0.005149 |
| MQ-Q5_K_1 | 29.19 | 0.005523 |
| MQ-Q5_K_S_1 | 26.33 | 0.006730 |
| MQ-Q4_K_M_1 | 24.82 | 0.007799 |
| MQ-Q4_K_M_2 | 22.32 | 0.011007 |
| MQ-IQ4_NL_1 | 20.89 | 0.013277 |
| MQ-IQ3_M_1 | 17.60 | 0.026330 |
| UD-IQ3_S | 13.68 | 0.068376 |
| MQ-IQ2_XXS_1 | 9.59 | 0.275130 |

Evidence: [pinned model card](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-MagicQuant-GGUF/resolve/6e1771746ee4c9194fe4e5c8bbe9f301efcfd2d5/README.md), [final survivor measurements](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-MagicQuant-GGUF/resolve/6e1771746ee4c9194fe4e5c8bbe9f301efcfd2d5/magicquant-manifest/magicquant.final-survivors.json), and [exact clone recipes](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-MagicQuant-GGUF/resolve/6e1771746ee4c9194fe4e5c8bbe9f301efcfd2d5/magicquant-manifest/magicquant.clone-configs.json).

The external baseline's measurements come from a local reconstruction of its tensor recipe. The original release links upstream for that download; these figures do not independently benchmark the provider's original artifact.

## Cloning: ten recipes on a related model

The uncensored variant by llmfan46 was rebuilt using the original release's ten configurations, including its external-derived baseline. Fresh clone measurements are listed below. The archived discovery measurements remain evidence about the source campaign; `magicquant.clone-benchmarks.json` supplies results for the new weights.

| Cloned recipe | Size (GB) | KLD against the target reference |
| --- | ---: | ---: |
| LM-Q8_0 | 36.91 | 0.004771 |
| MQ-Q6_K_1 | 31.59 | 0.005383 |
| MQ-Q5_K_1 | 29.19 | 0.006012 |
| MQ-Q5_K_S_1 | 26.33 | 0.007155 |
| MQ-Q4_K_M_1 | 24.82 | 0.007832 |
| MQ-Q4_K_M_2 | 22.32 | 0.010894 |
| MQ-IQ4_NL_1 | 20.89 | 0.013040 |
| MQ-IQ3_M_1 | 17.60 | 0.026825 |
| UD-IQ3_S | 13.68 | 0.068513 |
| MQ-IQ2_XXS_1 | 9.59 | 0.275805 |

Evidence: [pinned clone model card](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-Uncensored-MagicQuant-GGUF/resolve/aeead92caeffa71f045a58ae546e718c87c7111f/README.md) and [fresh clone benchmarks](https://huggingface.co/magiccodingman/Qwen3.6-35B-A3B-Uncensored-MagicQuant-GGUF/resolve/aeead92caeffa71f045a58ae546e718c87c7111f/magicquant-manifest/magicquant.clone-benchmarks.json).

This demonstrates recipe reuse across related weights, not proof that the inherited recipes are optimal or that the two models have equivalent capabilities. Follow the [manifest and cloning guide](../docs/manifests-and-cloning.md) to prepare a compatible target, use `--source-repo`, or pin a manifest with `--source-json`.

## Specialized research: downward-only MXFP4 adaptation

The Qwen3.8-27B MXFP4 release starts from AMD's Quark AWQ MXFP4 checkpoint. It adapts recipes from a separate MagicQuant release using a strict downward-only storage policy: preserve a source tensor when the proposed replacement would not save space. This required specialized conversion work beyond ordinary CLI cloning.

| Artifact | Size (GB) | Storage saved | KLD against native MXFP4 |
| --- | ---: | ---: | ---: |
| Native MXFP4 reference | 18.89 | 0% | 0.000000 |
| MQ-IQ4_XS_1 | 14.98 | 20.70% | 0.000940 |
| MQ-IQ2_XXS_1 | 8.22 | 56.47% | 0.321797 |

The 14.98 GB artifact preserved all 496 native MXFP4 tensors byte-for-byte. The benchmark covers ordinary language logits; it does not establish vision or MTP quality. The reference is already quantized, so these scores do not measure divergence from BF16.

Eleven adapted recipes were tested; ten were published. The dominated `UD-IQ2_XXS` result remains documented but its GGUF was removed. Keeping failed evidence separate from downloadable selections makes the decision auditable.

Evidence: [pinned method and model card](https://huggingface.co/magiccodingman/Qwen3.8-27B-MXFP4-MagicQuant-GGUF/resolve/3ae9728099cc03206096d3ffeeb05e8d24b5f62c/README.md), [artifact metrics and publication status](https://huggingface.co/magiccodingman/Qwen3.8-27B-MXFP4-MagicQuant-GGUF/resolve/3ae9728099cc03206096d3ffeeb05e8d24b5f62c/magicquant-manifest/magicquant.final-survivors.json), and [benchmark scope](https://huggingface.co/magiccodingman/Qwen3.8-27B-MXFP4-MagicQuant-GGUF/resolve/3ae9728099cc03206096d3ffeeb05e8d24b5f62c/magicquant-manifest/magicquant.clone-benchmarks.json). Savings percentages are those reported by the release, before displayed sizes were rounded.

## The original worked example remains available

The [original research overview](overview.md) retains the 4B survivor table, tensor-group recipe breakdown, nonlinear-trade example, methodology diagram, and context and GPU scheduling discussion. The project README draws on that material and links here for newer published results.

Its MQ-Q5_K_1 comparison uses the displayed values: `(2.88 / 2.73 - 1) × 100 ≈ 5.5%` additional storage and `(1 - 0.006632 / 0.009839) × 100 ≈ 32.6%` lower KLD. The [survivor-selection guide](docs/Nonlinear-Winners-And-Survivors.md) explains why a useful interior trade can deserve publication.
