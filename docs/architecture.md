# Architecture and contributor code map

## Execution flow

`src/MagicQuant/Program.cs` dispatches through `CommandCatalog`. Help returns before runtime initialization. A normal command reads and validates YAML/CLI/input paths before loading `Config.Current` and run state into `MQ.DB.Cache`. It records provenance, cleans stale scratch, checks dependencies, and invokes an `ICommand`. `--check-config` exits before those runtime changes.

`src/MagicQuant/Commands/QuantizationPipeline.cs` coordinates full discovery. `Evolution.cs` preserves the historical C# entry point and the CLI registry keeps `evolution` as an alias. The orchestrator should describe stage order; reusable behavior belongs in services.

1. Validate source model and initialize model-local paths.
2. Obtain model hash; prepare native GGUF and optional projector; review tensor grouping.
3. Resolve architecture family and tensor-group profile; register custom baselines; handle targeted relearn.
4. Acquire imatrix context, rebucket existing tensor truth when appropriate, and load/probe the hardware plan.
5. Establish native benchmark truth and compatibility, then run initial/continuation isolation samples.
6. Apply isolation policy and materialize the remaining candidate space in DuckDB.
7. Fit predictions, investigate contextual evidence, choose candidates, and validate them with real benchmarks.
8. Finalize survivors and write GGUFs, manifests, benchmark summaries, and model cards.

The [research wiki](https://github.com/magiccodingman/MagicQuant) is the source for the mathematical motivation. This guide maps the implementation, not a new algorithm specification.

## Where to change things

| Concern | Main files / services |
| --- | --- |
| CLI routing/help | `Program.cs`, `Commands/CommandCatalog.cs`, command `ShowHelp` methods |
| YAML shape and CLI overrides | `Configuration/MagicQuantYamlConfig.cs`, `MagicQuantYamlLoader.cs`, `config.default.yaml` |
| Baseline identity and roles | `MQ.DB/Models/BaselineQuants.cs`, `BaselineDefinitionResolver`, `HuggingFaceBaselineService` |
| Tensor grouping and profile review | `MQ.DB/tensor_groups.yaml`, `TensorGroupReviewService`, `TensorGroupProfileService`, `TensorGroupRebucketService` |
| Process/tool setup | `InitializeLlamaCpp`, `Helpers/LlamaBuilder`, `Helpers/PythonManager`, `HardwareHelper` |
| Native conversion and quantization | `NativeModelConversionService`, `QuantizationService`, `ExternalBaselineTensorParity`, `CloneManifestTensorMapBuildService` |
| Benchmark execution and GPU planning | `BenchmarkCommands`, `BenchmarkLogParser`, `BenchmarkService`, `BenchmarkGpuPlanning`, `LlamaGpuArgumentBuilder` |
| Isolation sampling and policy | `IsolationPlanningService`, `IsolationOptimizationService`, `Helpers/RuntimeSearchSpace` |
| SQLite measured truth | `MQ.DB/Data/MagicQuantContext.cs`, `MQ.DB/Models/DbModels`, `HybridBenchmarkRepository` |
| DuckDB candidate data | `QuantDatabaseService`, `RemainingCombinationStore`, `CombinationDuckDbSchema` |
| KLD prediction and final selection | `RankSafeKldPredictionService`, `PredictionGuidedHybridSelectionService`, `SmartBaselineTuningFallbackService` |
| Contextual anomaly/synergy evidence | `AnomalyWorkflowService`, `AnomalyRuleRepository`, `AnomalyAdjustedPredictionService` |
| Release artifacts | `HybridArtifactExportService`, `FinalArtifactNamingService`, `FinalReleaseMetadataService`, `ReadmeGenerationService` |
| Native process lifetime | `Runtime/ProcessRunner`, `NativeCommand`, `RunCancellation` |
| Run provenance | `RunProvenanceService` |
| Paths and lifecycle | `ModelArtifactPathService`, `ModelRuntimePathService`, `OutputPathService`, `CombinationDatabasePathService`, `ScratchStorageService` |

## Invariants worth protecting

- **Measured truth and prediction are different.** SQLite stores observations and context. DuckDB is a derived candidate/prediction workspace. Do not turn predictions into benchmark truth or silently substitute a standard family for a missing exact custom baseline measurement.
- **Identity is scoped.** Model hash, architecture family, tensor-group profile, baseline identity, and imatrix context determine which evidence may be reused. Similar display names are not sufficient.
- **Effective tensor assignments matter.** A carrier quant and an explicit group quant can describe the same effective assignment. Use existing resolvers and canonical baseline identities instead of inventing equality rules.
- **Runtime search state is mutable.** `RuntimeSearchSpace` controls the current allowed universe. Do not revive legacy static candidate lists as an authority.
- **Paths are contracts.** Writer and reader use `CombinationDatabasePathService` for the same DuckDB file. `OutputPathService` preserves command-specific destinations. Keep existing filenames, schema IDs, and serialized manifests stable unless a migration is part of the change.
- **Scratch is leased; downloads are durable.** Use `ScratchStorageService` leases for heavy temporary GGUF work, and durable external-baseline paths for reusable downloads. Do not add ad hoc cleanup of model roots.

## Global state and tests

`Config.Current`, `Cache`, and several baseline/search registries are process-wide mutable state. The CLI runs one command per process. Do not run multiple campaigns concurrently inside one process without redesigning those boundaries.

Tests currently disable parallel execution because these globals are shared. Tests that change them must save and restore the prior state in `finally`, use unique temporary directories, and clean up only those directories. Prefer testing a pure policy/path helper when possible. Executable-level CLI tests protect the entry point separately from command implementation tests.

Large benchmark, quantization, and selection services remain candidates for incremental extraction. Extract a cohesive responsibility behind regression tests instead of splitting files by arbitrary line count or changing numerical policy during a readability patch.

`NativeModelConversionService` owns native artifact completion, while `QuantizationConcurrencyPlan` computes CPU/storage limits without IO. `IProcessRunner` permits failure/cancellation tests at that boundary. `RunCancellation` is an async-scoped bridge for legacy service APIs; new APIs should accept explicit cancellation tokens as well. Numerical policy and persisted evidence remain in their existing services.
