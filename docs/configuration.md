# Configuration and paths

## Loading and precedence

`--config <file>` selects a YAML file; otherwise the executable loads its adjacent `config.default.yaml`. Debug and Release follow the same rule. `config.dev.yaml` is no longer selected automatically.

The loader deserializes the selected file into `MagicQuantYamlConfig`, whose property initializers supply omitted fields, applies supported CLI overrides, normalizes values, and updates `Config.Current` and `MQ.DB.Cache`. It does **not** merge a custom file with `config.default.yaml`. The distributed YAML intentionally differs from C# defaults for some research tuning settings. Copy that whole file to reproduce its profile.

Unknown CLI options, duplicate options, missing values, and values supplied to presence-only flags are rejected. CLI string options generally override nonblank YAML values. Many boolean switches only enable a feature; use YAML to disable it unless a specific negative CLI switch exists. Use `--name value` or `--name=value`; quote paths with spaces using normal shell quoting.

Unknown or inactive YAML keys produce a warning with their setting path and line number; `--strict-config` rejects them. Compare with the commented default file and `MagicQuant/Configuration/MagicQuantYamlConfig.cs`. CI strictly parses the distributed examples so their keys cannot silently drift.

## Main sections

| Section | Responsibility |
| --- | --- |
| `paths` | Source model, runtime root, llama.cpp, scratch roots, external cache directory name |
| `identity` | Architecture-family name and explicit alias override |
| `flags` | Imatrix, hardware reprobe, high-precision hybrid policy |
| `learning` | Tensor review, safe profile rebucketing, targeted relearn requests |
| `baselines` | Built-in roles and external repository definitions, including optional revision pins |
| `hardware` | Per-GPU usable VRAM limits |
| `imatrix` | Local/remote matrix or dataset source |
| `isolation_pruning` | Isolation gating and damage/tradeoff thresholds |
| `prediction` | Rank-safe KLD fitting and combination memory limits |
| `candidate_selection` | Final candidate windows, fallback attempts, and spacing/tradeoff rules |
| `anomaly_detection`, `synergy_detection` | Bounded contextual probes and evidence adjustments |
| `output` | Export destination, naming, reuse, and external-baseline export policy |
| `readme` | Generated model card title and frontmatter |

Built-in `standard_baselines_mode` is `all`, `selected`, or `none`. `selected` uses the explicit role lists; an empty list enables none for that role. In `all`, those lists do not restrict built-ins. Custom repositories are configured independently. Native anchors may still be required by the runtime. Historical `standard_only`/`custom_only` comments did not describe implemented filtering modes.

## Path contracts

Paths do not expand shell variables or `~` inside YAML. Prefer absolute paths. Relative `--config`, model, runtime, llama.cpp, dataset, and scratch paths resolve against the process working directory, **not** the YAML file's directory.

| Setting / command | Blank default | Relative value |
| --- | --- | --- |
| `paths.magic_quant_root` | `<user-home>/MagicQuant` | Process working directory |
| `paths.model_dir` | Required for model commands | Process working directory |
| Model work directory | `<model>/MagicQuant` | Derived from model path |
| `pipeline` / `build-hybrids` output | `<model>/MagicQuant/Final_Outputs` | Under `<model>/MagicQuant` |
| Clone output | `<model>/MagicQuant/FinalOutput` | Process working directory |
| Prediction validation output via CLI | `<model>/MagicQuant/PredictionValidation` if no YAML output | Process working directory, exact destination |
| Prediction validation with YAML output only | `<output.output_dir>/PredictionValidation` | YAML output resolves against process working directory |

These historical output differences are preserved for existing campaigns. `OutputPathService` owns the rules. An absolute `output_dir` avoids ambiguity.

`external_baseline_cache_dir_name` is intended to be a folder name beneath the model work directory; use a simple name such as `ExternalBaselines`. Scratch roots are separate from durable downloads. They need not be physically distinct disks, but the scheduler's one-heavy-writer-per-root policy assumes you choose them thoughtfully.

## Model metadata and compatibility

`readme.frontmatter` accepts arbitrary scalar/list metadata. Set `license`, `base_model`, and other provenance fields for the actual exported model; no model license is inferred for you.

The old `evolution`, `survival`, sensitivity-group, brain-layer, and collapse-penalty config surfaces had no active consumers and have been removed from the typed configuration. Old YAML containing them is tolerated with warnings unless `--strict-config` is selected; they do not tune the current algorithm. See `candidate_selection` and the research wiki for current selection policy.

## Preflight and cancellation

Normal runs validate local inputs before applying global state, creating runtime directories, installing dependencies, or cleaning artifacts. `--check-config` performs only this check. Non-finite numeric values and null required sections are rejected. Clone preflight requires one source manifest/repository, and model discovery requires explicit architecture-family identity.

Exports cannot contain the source model/runtime root or overlap protected model work directories such as GGUF, Benchmarks, Logs, Runs, and ExternalBaselines. Physical symlink targets are considered. These guards do not make arbitrary existing export contents safe: still choose a dedicated directory.

Ctrl+C requests cooperative cancellation, stops active native work, and returns status 130. A second Ctrl+C requests immediate OS termination. Process/lease cleanup is cooperative; forced termination or power loss can still require stale-artifact cleanup on the next run.
