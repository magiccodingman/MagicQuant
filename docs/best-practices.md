# Campaign best practices

## Start with a small, identifiable campaign

Use a complete local source model that the selected llama.cpp converter supports. Keep the exact model revision, architecture/profile identity, imatrix settings, and evaluation data consistent when comparing runs. Begin with a small model and the generated profile before scaling up. Run `magicquant pipeline --config config.yaml --check-config --strict-config` first; this validates input structure and paths, not memory capacity or numerical quality.

## Give scratch IO its own resources

Large intermediate GGUFs make storage throughput a potential bottleneck. Prefer fast local SSD/NVMe scratch storage; separate physical disks can allow independent heavy writers. MagicQuant permits one heavy writer per configured scratch root, so two directories on the same disk do not create independent bandwidth and can increase contention.

```yaml
paths:
  scratch_roots:
    - /mnt/nvme-a/magicquant-scratch
    - /mnt/nvme-b/magicquant-scratch
```

Use existing writable parent locations dedicated to this work. Allow space for several large model artifacts, monitor free space and device throughput, and leave room for durable downloads and exports too. A faster disk helps when IO is limiting; GPU/CPU compute, RAM, and evaluation workload can instead dominate. Avoid fixed speedup expectations. See [storage ownership and cleanup](storage.md).

## Optional Unsloth baselines

The maintainer recommends Unsloth as a primary place to look for external GGUF tensor assignments. These sources are optional, and their value depends on model compatibility and measured results. Start with a repository for the exact source model; a similar name or matching architecture alone is insufficient.

In the generated configuration, edit `baselines.custom_repositories`. The following is a structural example, not a promise that a particular upstream file exists. Replace the model/file placeholders, pin `revision` to the provider commit you inspected, and retain the rest of your campaign configuration:

```yaml
baselines:
  custom_repositories:
    - repo_id: unsloth/YOUR-EXACT-MODEL-GGUF
      revision: PROVIDER_COMMIT_SHA
      enabled: true
      short_source_name: UD
      source_kind: huggingface_gguf_repository
      require_all_includes_to_resolve: true
      validate_tensor_names_against_source_model: true
      includes:
        - file_name: YOUR-EXACT-MODEL-UD-Q4_K_XL.gguf
          baseline_family: Q4_K_M
          quantize_base_name: Q4_K_M
          display_name: UD_Q4_K_XL
          allow_as_learning_baseline: true
          allow_as_combination_carrier: true
          allow_as_explicit_group_candidate: true
```

MagicQuant resolves the specified files, validates tensor-name parity, learns assignments, rebuilds using the local source model, and benchmarks the reconstruction. External downloads are durable cache data, distinct from temporary scratch artifacts. Review provider/model licensing and available disk space before enabling sources. Pure external learned baselines are not exported by default; inspect `output.export_external_learned_baselines` if you need them.

For a provider-free campaign leave `baselines.custom_repositories` empty. See [learning from existing quantizations](../wiki/docs/Learning-From-Existing-Quantizations.md) for the research rationale.

## Link upstream for the same model; build locally for variants

For a release of the **same source model** that an external provider such as Unsloth already hosts, the maintainer recommends leaving this pipeline setting off:

```yaml
output:
  export_external_learned_baselines: false
```

MagicQuant will link external pure-baseline survivors to the provider instead of exporting local copies. This gives the original creator credit and downloads, avoids unnecessary duplicate hosting, and is the friendly default. MagicQuant's comparisons measure locally reconstructed tensor configurations under its own conditions. They do not, by themselves, establish whether the provider's original artifact is better or worse. Finding a useful hybrid or size/fidelity trade is not a reason to claim superiority over an untested upstream release.

For a **different model variant**, such as an uncensored model or another fine-tune, the upstream repository may not host those weights. In that case, build the full selected set locally, including both MagicQuant hybrids and external-derived baseline configurations. If running the pipeline on that variant, enable local external-baseline exports:

```yaml
output:
  export_external_learned_baselines: true
```

The equivalent pipeline switch is `--export-external-learned-baselines`. Retain provider attribution and the applicable licenses even when rebuilding from different weights.

**Clone command distinction:** `clone-repository-quants` already rebuilds every artifact entry in its input clone manifest, including external-derived entries; it does not consult this pipeline export flag. You do not need to enable the flag for that command. A source release can leave external export off and still include those configurations in its clone manifest. Clone mode rebuilds the entries present in that manifest, not every quantization ever offered by the provider.

Cloning is a practical way to reuse a strong set of tensor configurations on a compatible variant without repeating full discovery. In the maintainer's experience, repeating discovery for modest fine-tunes can cost substantial time for little additional improvement. That is a starting assumption, not a guarantee: larger weight changes can shift the useful tradeoffs. Clone mode benchmarks the rebuilt artifacts locally, but does not repeat the full search or prove that inherited choices are optimal. Run discovery again when the model changes substantially, the measurements look poor, or you need stronger evidence for the target model. See the [clone command](commands.md#clone-known-tensor-configurations).

## Limits of tensor-configuration copying

MagicQuant learns quantization assignments for tensors and tensor groups, then rebuilds using the local source weights and its supported toolchain. **It does not automatically reproduce every technique used to create an external artifact.** A provider's extra weight transformations, custom quantization procedures, calibration recipes, or other processing are not reproduced merely because their tensor configuration was learned. Such behavior must be explicitly supported to be reproduced.

Treat external configurations as evidence about useful assignments, not as a byte-for-byte clone of the provider's GGUF or a replication of its entire production process. Keep this distinction clear in release descriptions and benchmark claims. See the [research guide](../wiki/docs/Learning-From-Existing-Quantizations.md).

## Retain enough evidence to reproduce a result

Pin the MagicQuant package version and provider/model revisions. Keep the YAML, imatrix/evaluation data identity, llama.cpp revision, hardware context, and local `Runs/*/run.json` records. Provenance captures available versions and settings; it is not a complete frozen environment or numerical reproducibility guarantee. Remove private paths or credentials before sharing logs.

Use the same measurement conditions for comparisons. Reuse validated caches deliberately; changing runtime roots or tensor profiles can change which evidence is selected. Avoid multiple campaigns in the same process and competing runs in the same model/runtime workspace.

## Platform expectations

Linux campaigns have been exercised, including a small-model conversion/quantization smoke test. Windows CI validates builds, ordinary tests, and tool packaging; end-to-end Windows campaigns remain unvalidated. Report failures with package/toolchain versions, sanitized configuration, and relevant logs.
