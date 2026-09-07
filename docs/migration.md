# Compatibility notes for existing users

This cleanup retains numerical selection policy, SQLite schemas/migrations, manifest names, and existing output destinations. There is no data migration.

## Startup and names

- Prefer `pipeline`; `evolution` remains a case-insensitive CLI alias. The historical `Evolution` C# class delegates through inheritance to `QuantizationPipeline`.
- Debug builds no longer inject a personal command, model path, or architecture family. With no arguments, both build configurations show help.
- Pass `--config /path/to/your-config.yaml` explicitly for personal campaigns. Debug no longer auto-selects `config.dev.yaml`. Previously tracked developer campaigns were replaced by portable examples; existing local copies still load when selected explicitly.
- Command help runs before initialization. Unknown commands return status 2; caught command/config failures return status 1.
- `initialize-llama-cpp` now honors custom paths from YAML directly, caches the validated paths, and reports invalid custom environments as failures instead of printing an error and returning successfully.

## Removed inactive surfaces

The old evolution/survival knobs and unused sensitivity, brain-layer, collapse-penalty, and MoE-indicator config lists had no active runtime consumers. Their typed properties and inactive CLI overrides were removed. Legacy YAML keys warn (or fail with `--strict-config`); they never tuned the current chooser. Use `prediction`, `candidate_selection`, `anomaly_detection`, and `synergy_detection` for current policy.

Startup no longer enumerates the entire combination universe merely to compare it with a count. That diagnostic helper remains available for explicit development checks. Actual pipeline generation and policy checks remain in place.

## Paths and metadata

Relative-path behavior is preserved, including `Final_Outputs` for pipeline and `FinalOutput` for clone. The shared path services make those differences explicit. Use absolute output paths when sharing configs.

The distributed model-card template no longer asserts `apache-2.0` or a placeholder `base_model`. Set real provenance in your campaign YAML before publishing. Existing explicit frontmatter is still honored.

The documentation now uses the implemented built-in baseline modes (`all`, `selected`, `none`). Older comments referring to `standard_only` and `custom_only` did not match the loader's behavior.

To recover a previously tracked campaign before adopting the new layout, save it as
an ignored local config (replace the revision placeholder):

```sh
git show <pre-cleanup-commit>:MagicQuant/config.dev.yaml > config.local.yaml
```

Then continue with `pipeline --config config.local.yaml` and your explicit model/family
arguments. The change to DEBUG startup does not change values inside that saved YAML.

## Deeper readiness changes

CLI typos/duplicate options and missing values now fail before work; unknown YAML keys warn, and `--strict-config` makes them errors. `--check-config` performs read-only preflight. Unsafe managed/output overlaps and missing model inputs fail before dependency setup or cleanup. Numeric CLI parsing uses an invariant decimal point.

Native processes now share cancellation/log cleanup and use literal argv for benchmark and quantization launches. CPU llama-bench uses `-ngl 0` because the tested native version rejects the historical `-backend cpu` argument. Native conversion and low-level export require completion markers for reuse; partial/canceled builds are removed. Existing higher-level benchmark-size reuse checks remain in place.

.NET package references were updated within the 10.0 patch line to remove the previously reported transitive vulnerabilities. Package lock files are committed. Database schemas and research selection formulas were not changed.
