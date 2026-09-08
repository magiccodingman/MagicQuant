# Setup and troubleshooting

## Install from NuGet

Linux is the tested campaign platform. Windows CI covers builds, ordinary tests and installed-package startup; full Windows campaigns remain unvalidated. Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) first.

```sh
dotnet tool install --global MagicQuant
magicquant --version
magicquant init-config --output config.yaml
```

NuGet publication begins with the first successful release. If the package is not yet listed, use the source-build route below. The CLI command is `magicquant`; `dotnet add package` is not the installation command for this application.

If your shell cannot find `magicquant`, ensure the .NET tools directory is on `PATH`: `$HOME/.dotnet/tools` on Linux or `%USERPROFILE%\.dotnet\tools` on Windows, then reopen the shell. Use `dotnet tool update --global MagicQuant` to update, `dotnet tool uninstall --global MagicQuant` to remove the tool, or add `--version X.Y.Z` to install an exact version. Removing/updating the tool does not remove model/runtime data.

Edit the generated YAML; set `paths.model_dir`, `identity.architecture_family_name`, `output.output_dir`, and dedicated `paths.scratch_roots`. For a first run:

```sh
magicquant initialize-llama-cpp
magicquant pipeline --config ./config.yaml --check-config --strict-config
magicquant pipeline --config ./config.yaml
```

`init-config` copies the full bundled tuning profile and refuses to overwrite existing files. `--check-config` is read-only. The actual pipeline can perform dependency setup and write model/runtime artifacts. Fast scratch disks are especially valuable for the repeated large GGUF writes; read [best practices](best-practices.md) before a large campaign.

## Build from source

All solution projects target `net10.0`. The solution includes `src/MagicQuant`, `src/MQ.DB`, `tests/MagicQuant.Tests`, and the offline `tests/MagicQuant.ProcessFixture` helper.

```sh
git clone https://github.com/magiccodingman/MagicQuant.git
cd MagicQuant
dotnet restore MagicQuant.sln --locked-mode -warnaserror
dotnet build MagicQuant.sln -c Release --no-restore -warnaserror
dotnet test MagicQuant.sln -c Release --no-build
dotnet run --project src/MagicQuant -c Release --no-build -- init-config --output config.yaml
```

For source execution, replace `magicquant` in the other examples with `dotnet run --project src/MagicQuant -c Release --no-build --`. Build/test commands do not install llama.cpp or Python packages; ordinary tests do not require CUDA or weights. To exercise the actual package locally, see [testing](testing.md).

## Runtime setup

`initialize-llama-cpp` prepares the shared `<user-home>/MagicQuant` installation. On apt-based Linux it checks build tools, CMake, Ninja, Git, Python/venv/pip, and libcurl development files; NVIDIA detection also adds a CUDA toolkit package. Missing packages may trigger sudo. The Python setup installs model conversion and dataset dependencies, PyTorch, and llama-cpp-python. `--update` requests dependency updates and a native rebuild, so record the resulting llama.cpp revision for reproducible research.

The installer uses the configured hardware to choose native build options. macOS automatic setup is unsupported. Windows bootstrap code exists, but validate it on the target machine rather than assuming parity with Linux.

You can bypass automatic native setup by providing an existing environment:

```yaml
paths:
  llama_root: /opt/llama.cpp
  llama_bin: /opt/llama.cpp/build/bin
  convert_script: /opt/llama.cpp/convert_hf_to_gguf.py
```

All three paths are required together. This branch validates their existence and detects hardware; it does not install the Python dependencies. The application expects its Python executable under `<magic_quant_root>/MagicQuant-Env/bin/python` on Linux, or `MagicQuant-Env/python.exe` on Windows. Use the normal initializer first when using the default runtime root. `--validate` and `--verify` are historical setup flags, not read-only dependency checks.

An explicit `paths.magic_quant_root` changes the SQLite/runtime root but **does not relocate the shared installer**. If you isolate that root, provision its expected Python environment as well. For example, on Linux an isolated root can link `MagicQuant-Env` to an already initialized shared environment. Make that choice explicitly; sharing Python still shares installed dependency versions. See [storage](storage.md) before moving existing campaign data.

## Model input

Use a complete source model directory supported by your llama.cpp conversion revision. `pipeline` requires at least one top-level `.safetensors` file. Model config/tokenizer files are also needed by conversion. A directory containing only a downloaded quantized GGUF is not a source model directory.

Architecture-family identity scopes learned data. Choose the intended family deliberately; `--allow-architecture-family-alias-override` bypasses an identity guard and should not be a routine setup flag.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing SDK / unsupported target framework | `dotnet --info`; install a .NET 10 SDK |
| Missing config | Pass `--config` explicitly; the default is next to the executable |
| Missing model directory or safetensors | Set `paths.model_dir` to the complete local source model |
| Partial custom llama.cpp paths | Supply root, binary directory, and converter together |
| Python executable/package failure with isolated root | Check `<magic_quant_root>/MagicQuant-Env` and its installed packages |
| Conversion or unknown tensor failure | Check model support in the actual llama.cpp checkout and review tensor grouping |
| Hardware plan no longer fits | Check GPU visibility and configured VRAM limits; use `--recheck-hardware-probe` after hardware changes |
| Unexpected export location | Check command-specific relative-path rules in [configuration](configuration.md) |
| Prediction reports use the wrong bucket | Use the same model/runtime database and exact imatrix identity as the measured run |
| No cached results reused | Check model hash, architecture family, tensor-group profile, and imatrix scope before considering relearn |

For a useful bug report include the command, sanitized YAML, commit, .NET/OS/native dependency versions, GPU information, and relevant logs. Do not attach model weights or a whole runtime database by default.
