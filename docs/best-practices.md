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

## Retain enough evidence to reproduce a result

Pin the MagicQuant package version and provider/model revisions. Keep the YAML, imatrix/evaluation data identity, llama.cpp revision, hardware context, and local `Runs/*/run.json` records. Provenance captures available versions and settings; it is not a complete frozen environment or numerical reproducibility guarantee. Remove private paths or credentials before sharing logs.

Use the same measurement conditions for comparisons. Reuse validated caches deliberately; changing runtime roots or tensor profiles can change which evidence is selected. Avoid multiple campaigns in the same process and competing runs in the same model/runtime workspace.

## Platform expectations

Linux campaigns have been exercised, including a small-model conversion/quantization smoke test. Windows CI validates builds, ordinary tests, and tool packaging; end-to-end Windows campaigns remain unvalidated. Report failures with package/toolchain versions, sanitized configuration, and relevant logs.
