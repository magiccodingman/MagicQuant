# Third-party notices

MagicQuant's original code is AGPL-3.0-only. Dependencies retain their own licenses; they are not relicensed by this repository. The runtime dependency inventory below is taken from the committed application lock file. License texts and bundled notices are retained in [licenses/](licenses/), also included in the tool package.

LibGit2Sharp is MIT; its native libgit2 component is GPL version 2 with an explicit linking exception permitting combinations with other programs. Keep that exception with its license. SQLitePCLRaw is Apache-2.0; SQLite itself is public domain. Other listed managed components use MIT or BSD-2-Clause. Build/test-only dependencies remain governed by their package notices.

| Package | Version | License |
| --- | --- | --- |
| Blake3 | 2.2.0 | BSD-2-Clause |
| DuckDB.NET.Data.Full | 1.4.3 | MIT |
| LibGit2Sharp | 0.31.0 | MIT |
| Spectre.Console | 0.54.0 | MIT |
| System.Management | 10.0.11 | MIT |
| YamlDotNet | 17.0.1 | MIT |
| DuckDB.NET.Bindings.Full | 1.4.3 | MIT |
| LibGit2Sharp.NativeBinaries | 2.0.323 | GPL-2.0 with linking exception |
| Microsoft.Data.Sqlite | 10.0.11 | MIT |
| Microsoft.Data.Sqlite.Core | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore.Abstractions | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore.Analyzers | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore.Relational | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore.Sqlite | 10.0.11 | MIT |
| Microsoft.EntityFrameworkCore.Sqlite.Core | 10.0.11 | MIT |
| Microsoft.Extensions.Caching.Abstractions | 10.0.11 | MIT |
| Microsoft.Extensions.Caching.Memory | 10.0.11 | MIT |
| Microsoft.Extensions.Configuration.Abstractions | 10.0.11 | MIT |
| Microsoft.Extensions.DependencyInjection | 10.0.11 | MIT |
| Microsoft.Extensions.DependencyInjection.Abstractions | 10.0.11 | MIT |
| Microsoft.Extensions.DependencyModel | 10.0.11 | MIT |
| Microsoft.Extensions.Logging | 10.0.11 | MIT |
| Microsoft.Extensions.Logging.Abstractions | 10.0.11 | MIT |
| Microsoft.Extensions.Options | 10.0.11 | MIT |
| Microsoft.Extensions.Primitives | 10.0.11 | MIT |
| SQLitePCLRaw.bundle_e_sqlite3 | 2.1.12 | Apache-2.0 |
| SQLitePCLRaw.core | 2.1.12 | Apache-2.0 |
| SQLitePCLRaw.lib.e_sqlite3 | 2.1.12 | Apache-2.0 |
| SQLitePCLRaw.provider.e_sqlite3 | 2.1.12 | Apache-2.0 |
| System.CodeDom | 10.0.11 | MIT |

The native/Python toolchain (including llama.cpp, Python, PyTorch, llama-cpp-python and downloaded packages) is installed separately and retains its own licenses and notices. Model weights, datasets and external GGUF baselines are also separate works; check their specific terms before downloading or redistributing them. MagicQuant does not grant rights to third-party models or training data.

When updating dependencies, refresh the lock files, review their license metadata and native component notices, and update this inventory. Package source repositories and exact source commits are recorded in their NuGet metadata; retain upstream notices when redistributing binaries.
