# Commands and workflows

Run these examples with the NuGet-installed `magicquant` CLI. Replace paths and identities with your own. For source builds, substitute `dotnet run --project src/MagicQuant -c Release --no-build --` for `magicquant`. `mq` in older help is shorthand, not an installed command.

## Config creation and version

`magicquant init-config --output config.yaml` copies the bundled tuning profile without runtime setup and refuses to overwrite a file. `magicquant --version` prints the application version and available source revision.

## Full discovery pipeline

```sh
magicquant pipeline \
  --config config.local.yaml \
  --model-dir /data/models/my-model \
  --architecture-family my-model-family \
  --output-dir /data/exports/my-model \
  --output-name-prefix MyModel
```

The pipeline converts/loads the native source, reviews tensor groups, resolves identity and baselines, measures isolation samples, predicts useful hybrids, validates them, and exports survivors. The exact runtime choices depend on benchmark truth and YAML policy. `evolution` invokes the same implementation. `build-hybrids` also enters the full pipeline; it is not limited to exporting previously selected files.

`--use-imatrix` enables the configured acquisition/build flow. Provide an appropriate `imatrix` source in YAML. `--recheck-hardware-probe` refreshes execution-plan probing after hardware changes. `--skip-tensor-group-confirm` suppresses the tensor-group prompt for an already reviewed unattended campaign; it does not bypass all other confirmations.

## Clone known tensor configurations

For the complete publisher-to-user walkthrough, including manifest filenames and pointing at another Hugging Face release, see [manifests and cloning](manifests-and-cloning.md).

```sh
magicquant clone-repository-quants \
  --config config.local.yaml \
  --model-dir /data/models/compatible-model \
  --architecture-family my-model-family \
  --source-json /data/releases/source/magicquant-manifest/magicquant.clone-configs.json \
  --output-dir /data/exports/cloned-model
```

Use `--source-repo owner/repo` instead to read a Hugging Face repository. `--source-json` also accepts an HTTP(S) URL. Clone mode rebuilds the tensor configurations and benchmarks them locally; it does not establish that the new model passed full discovery.

Clone mode rebuilds all entries present in the manifest, including configurations originally learned from external providers. It does not use `output.export_external_learned_baselines`, which controls the pipeline's choice between local exports and upstream links. This is useful when cloning to a fine-tuned or uncensored variant that the original provider does not host. Review the [upstream-link versus variant-export guidance](best-practices.md#link-upstream-for-the-same-model-build-locally-for-variants) and [limits of tensor-configuration copying](best-practices.md#limits-of-tensor-configuration-copying).

By default the manifest must match the target tensor inventory. `--allow-missing-manifest-tensors` explicitly allows a strict subset; unmatched target tensors use base quantization. `--missing-manifest-base-quant Q8_0` additionally selects that base quant. Use these only when that compatibility tradeoff is intended.

## Validate predictions against existing measurements

```sh
magicquant validate-predictions \
  --config config.local.yaml \
  --model-dir /data/models/my-model \
  --architecture-family my-model-family \
  --output-dir /data/reports/my-model
```

This writes `prediction_validation_general.csv` and `prediction_validation_general.md`. It uses existing general-category SQLite benchmark truth rather than launching a fresh full discovery campaign. Startup still runs the common dependency validation and stale-artifact cleanup.

For imatrix measurements supply `--imatrix-path /data/imatrix.dat` or `--imatrix-identity-hash <sha256>` to select the exact bucket. Merely enabling `flags.use_imatrix` does not select a validation bucket. Without either option, validation uses no-imatrix truth.

## Rerun and reuse

```sh
magicquant pipeline \
  --config config.local.yaml --reuse-existing-final-artifacts
```

The normal pipeline can reuse scoped measurements, but final export normally cleans/rebuilds output. Artifact reuse is an additional opt-in: pipeline exports require exact filename and benchmark byte-size matches; clone mode also validates its benchmark JSON rows. A file merely existing is not sufficient.

## Help and exit status

No arguments, `help`, `--help`, or `-h` display top-level help. `<command> --help` and `<command> -h` display command help without config loading, cleanup, database access, or dependency installation.

The host returns `0` on normal completion/help, `2` for an unknown command, `130` for cooperative cancellation, and `1` for an exception caught at the command boundary. Services may handle individual candidate failures and continue a campaign, so also inspect the reported sample failures and final artifacts.

Use `--check-config` on a normal command to validate local inputs without running it.
Use `--strict-config` to reject unknown/inactive YAML settings rather than warning.
See [testing](testing.md) for the opt-in real-model workflow.
