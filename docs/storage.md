# Storage, caching, and reruns

The runtime root and model work directory are different things. With default settings:

```text
<user-home>/MagicQuant/
  MagicQuant_SQLite.db       # measured truth, learned mappings, hardware probe state
  llama.cpp/                # shared native checkout/build
  MagicQuant-Env/            # Python environment

<source-model>/
  *.safetensors             # input weights
  config.json               # source metadata and tokenizer assets alongside it
  MagicQuant/
    GGUF/                   # durable native/base artifacts
    Benchmarks/             # measurements, corpora, reference logits
    Logs/Quantization/      # quantization process logs
    ExternalBaselines/      # durable downloaded external GGUFs
    MagicQuant_Combinations_<model>_<imatrix>_<hp>.duckdb
    Final_Outputs/          # pipeline default export directory
    FinalOutput/            # clone default export directory
    PredictionValidation/   # prediction report default
```

Output is configurable and only directories needed by a run are created. Other service-specific files may also appear. Final exports put JSON evidence under `magicquant-manifest/`, including clone configurations, final survivors, replacements, hybrid maps, isolation samples, and bad-trade reports as appropriate to the workflow. `MagicQuantManifestPathService` owns those filenames and links.

## Durable truth

SQLite is initialized/migrated automatically when its context is opened. Existing migrations and stored IDs are compatibility boundaries. Back up the database and related model artifacts while the process is stopped before manual moves or experiments. Do not delete the database as routine troubleshooting: it contains measured evidence that can be expensive to reconstruct.

Changing the runtime root selects another SQLite database. It does not move data or the shared native installer. Changing a tensor-group regex/profile changes the applicable evidence scope; normal rebucketing can copy existing learned mappings into the new profile without erasing the original observations. Targeted relearn settings explicitly delete scoped truth after the program's confirmation step.

## Derived candidate space

DuckDB stores the current allowed combinations and prediction materialization. Both writer and reader resolve the same model/imatrix/high-precision filename through `CombinationDatabasePathService`. The pipeline rebuilds candidate data; the database is not interchangeable with SQLite benchmark evidence.

Do not rename its files or add a scope component on only one side of a writer/reader pair. That can make a populated candidate space appear empty.

## Scratch and cleanup

Configured `paths.scratch_roots` hold `.MagicQuant_tmp` directories for leased heavy writes; blank configuration uses the model-local fallback. The service enforces one heavy writer per root and uses artifact leases and stale-state checks. Separate directory names do not establish separate physical disks.

External baseline downloads are durable and managed separately from transient quantization artifacts. Startup cleanup runs before commands and model-specific cleanup runs after model paths are initialized. Keep personal files outside both managed scratch and dedicated export directories.

## Reproducibility

Retain the exact command, selected YAML, program commit, llama.cpp revision, model source revision/hash, external repository revision pins, imatrix identity/source, hardware plan, and emitted manifests/benchmark reports for a release. Custom repository `revision` can pin a branch, tag, or commit; a commit avoids moving references. Reusing output does not replace recording these inputs.
