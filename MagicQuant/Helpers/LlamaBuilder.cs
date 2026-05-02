using System.Diagnostics;
using System.Runtime.InteropServices;
using LibGit2Sharp;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public class LlamaBuilder
{
    private const string LlamaCppRepositoryUrl = "https://github.com/ggerganov/llama.cpp.git";

    private readonly string _llamaRoot;
    private readonly SystemInfo _sysInfo;

    public LlamaBuilder(string magicRoot, SystemInfo sysInfo)
    {
        _llamaRoot = Path.Combine(magicRoot, MagicConstants.LlamaRepoName);
        _sysInfo = sysInfo;

        Cache.LlamaRoot = _llamaRoot;
        Cache.LlamaBin = Path.Combine(_llamaRoot, "build", "bin");
        Cache.ConvertScript = ResolveConvertScriptPath(_llamaRoot);
    }

    public string GetLlamaBinPath() => Path.Combine(_llamaRoot, "build", "bin");

    public async Task PrepareAndBuildAsync(bool forceRebuild)
    {
        // 1. Validate ALL dependencies before doing anything.
        await DependencyManager.EnsureDependenciesAsync(_sysInfo);

        // 2. Ensure the llama.cpp checkout is real and buildable.
        EnsureLlamaRepository(forceRebuild);

        Cache.LlamaRoot = _llamaRoot;
        Cache.LlamaBin = Path.Combine(_llamaRoot, "build", "bin");
        Cache.ConvertScript = ResolveConvertScriptPath(_llamaRoot);

        ValidateLlamaSourceTreeOrThrow();

        // 3. Setup Build Directory.
        string buildDir = Path.Combine(_llamaRoot, "build");
        EnsureBuildDirectory(buildDir, forceRebuild);

        // 4. Get the CMake Executable (System or Local).
        string cmakeExe = DependencyManager.GetCmakePath() ?? "cmake";

        // 5. Generate Build Files.
        var cmakeArgs = GetOptimalCmakeArgs();
        AnsiConsole.MarkupLine($"[grey]Configuring build with: {Markup.Escape(FormatArgsForDisplay(cmakeArgs))}[/]");

        if (!await RunProcessAsync(cmakeExe, cmakeArgs, buildDir))
            throw new Exception("CMake configuration failed.");

        // 6. Compile.
        AnsiConsole.MarkupLine("[cyan]Compiling llama.cpp (Release Mode)...[/]");

        var buildArgs = new List<string>
        {
            "--build",
            ".",
            "--config",
            "Release",
            "-j",
            Environment.ProcessorCount.ToString()
        };

        if (!await RunProcessAsync(cmakeExe, buildArgs, buildDir))
            throw new Exception("Build failed.");

        AnsiConsole.MarkupLine("[green]✔ Build Success![/]");
    }

    private void EnsureLlamaRepository(bool forceRebuild)
    {
        bool rootExists = Directory.Exists(_llamaRoot);
        bool rootIsValid = IsValidLlamaSourceTree();

        if (forceRebuild && rootExists)
        {
            AnsiConsole.MarkupLine("[yellow]Update requested: removing existing llama.cpp checkout...[/]");
            DeleteDirectoryOrThrow(_llamaRoot, "update was requested");
            rootExists = false;
            rootIsValid = false;
        }

        if (rootExists && !rootIsValid)
        {
            var escapedRoot = Markup.Escape(_llamaRoot);
            AnsiConsole.MarkupLine($"[yellow]Existing llama.cpp directory is invalid or incomplete: {escapedRoot}[/]");
            AnsiConsole.MarkupLine("[grey]Missing CMakeLists.txt or repository metadata. Removing it so MagicQuant can redeploy a clean checkout.[/]");
            DeleteDirectoryOrThrow(_llamaRoot, "existing llama.cpp checkout is invalid/incomplete");
            rootExists = false;
        }

        if (!rootExists)
        {
            CloneLlamaRepository();
            return;
        }

        AnsiConsole.MarkupLine("[grey]Valid llama.cpp repository already exists. Skipping clone.[/]");
    }

    private void CloneLlamaRepository()
    {
        var escapedRoot = Markup.Escape(_llamaRoot);
        AnsiConsole.MarkupLine($"Cloning llama.cpp to [blue]{escapedRoot}[/]...");
        AnsiConsole.MarkupLine("[grey](This includes submodules and may take a moment.)[/]");

        Directory.CreateDirectory(Path.GetDirectoryName(_llamaRoot)!);

        var cloneOptions = new CloneOptions
        {
            RecurseSubmodules = true
        };

        try
        {
            Repository.Clone(LlamaCppRepositoryUrl, _llamaRoot, cloneOptions);
        }
        catch
        {
            if (Directory.Exists(_llamaRoot) && !IsValidLlamaSourceTree())
                DeleteDirectoryOrThrow(_llamaRoot, "clone failed and left a partial checkout");

            throw;
        }

        ValidateLlamaSourceTreeOrThrow();
    }

    private bool IsValidLlamaSourceTree()
    {
        if (!Directory.Exists(_llamaRoot))
            return false;

        // CMakeLists.txt is the non-negotiable build root. The previous bug was
        // caused by trusting Directory.Exists(_llamaRoot) even when this file was gone.
        if (!File.Exists(Path.Combine(_llamaRoot, "CMakeLists.txt")))
            return false;

        // Prefer a real git checkout for auto-managed installs. If a user points at a
        // custom source tree, that path is handled by InitializeLlamaCpp custom args.
        if (!Directory.Exists(Path.Combine(_llamaRoot, ".git")))
            return false;

        return true;
    }

    private void ValidateLlamaSourceTreeOrThrow()
    {
        string cmakeLists = Path.Combine(_llamaRoot, "CMakeLists.txt");
        if (!File.Exists(cmakeLists))
        {
            throw new DirectoryNotFoundException(
                $"llama.cpp checkout is not buildable. Expected CMakeLists.txt at: {cmakeLists}. " +
                "Delete the llama.cpp directory or rerun initialize-llama-cpp --update so MagicQuant can redeploy it.");
        }
    }

    private void EnsureBuildDirectory(string buildDir, bool forceRebuild)
    {
        if (Directory.Exists(buildDir))
        {
            if (forceRebuild)
            {
                DeleteDirectoryOrThrow(buildDir, "clean rebuild requested");
            }
            else if (!BuildCacheMatchesCurrentSource(buildDir))
            {
                AnsiConsole.MarkupLine("[yellow]Existing CMake build cache points at a different or invalid source tree. Recreating build directory...[/]");
                DeleteDirectoryOrThrow(buildDir, "CMake cache does not match the active llama.cpp source tree");
            }
        }

        Directory.CreateDirectory(buildDir);
    }

    private bool BuildCacheMatchesCurrentSource(string buildDir)
    {
        string cacheFile = Path.Combine(buildDir, "CMakeCache.txt");
        if (!File.Exists(cacheFile))
            return true;

        try
        {
            foreach (string line in File.ReadLines(cacheFile))
            {
                if (!line.StartsWith("CMAKE_HOME_DIRECTORY:INTERNAL=", StringComparison.Ordinal))
                    continue;

                string cachedSource = line["CMAKE_HOME_DIRECTORY:INTERNAL=".Length..].Trim();
                if (string.IsNullOrWhiteSpace(cachedSource))
                    return false;

                string normalizedCached = NormalizePath(cachedSource);
                string normalizedCurrent = NormalizePath(_llamaRoot);

                return string.Equals(normalizedCached, normalizedCurrent, RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? StringComparison.OrdinalIgnoreCase
                    : StringComparison.Ordinal);
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private List<string> GetOptimalCmakeArgs()
    {
        var args = new List<string>
        {
            _llamaRoot,
            "-DCMAKE_BUILD_TYPE=Release"
        };

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            args.Add("-G");
            args.Add("Ninja");
        }

        switch (_sysInfo.GpuInfo.FirstOrDefault()?.GpuVendor)
        {
            case GpuVendor.Nvidia:
                args.Add("-DGGML_CUDA=ON");
                args.Add("-DCMAKE_CUDA_ARCHITECTURES=native");
                break;

            case GpuVendor.Amd:
                args.Add("-DGGML_HIPBLAS=ON");
                break;

            case GpuVendor.Intel:
                args.Add("-DGGML_SYCL=ON");
                break;
        }

        return args;
    }

    private async Task<bool> RunProcessAsync(string exe, IReadOnlyList<string> args, string workingDir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            WorkingDirectory = workingDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        foreach (string arg in args)
            psi.ArgumentList.Add(arg);

        using var p = Process.Start(psi);
        if (p == null)
            return false;

        p.OutputDataReceived += (_, e) =>
        {
            if (e.Data != null)
                AnsiConsole.WriteLine(e.Data);
        };

        p.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null)
                AnsiConsole.WriteLine(e.Data);
        };

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        await p.WaitForExitAsync();

        return p.ExitCode == 0;
    }

    private static string ResolveConvertScriptPath(string llamaRoot)
    {
        // llama.cpp has kept this at repo root for the relevant toolchain. Keep a
        // tiny candidate list so a future minor layout change does not poison Cache.
        string[] candidates =
        {
            Path.Combine(llamaRoot, "convert_hf_to_gguf.py"),
            Path.Combine(llamaRoot, "convert.py")
        };

        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    private static void DeleteDirectoryOrThrow(string path, string reason)
    {
        if (!Directory.Exists(path))
            return;

        try
        {
            MakeDirectoryWritable(path);
            Directory.Delete(path, recursive: true);
        }
        catch (Exception ex)
        {
            throw new IOException(
                $"Could not remove directory '{path}' while repairing llama.cpp ({reason}). " +
                "Close any terminals/editors using that path or delete it manually, then rerun initialize-llama-cpp.", ex);
        }
    }

    private static void MakeDirectoryWritable(string root)
    {
        try
        {
            foreach (string file in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
            {
                var attributes = File.GetAttributes(file);
                if ((attributes & FileAttributes.ReadOnly) != 0)
                    File.SetAttributes(file, attributes & ~FileAttributes.ReadOnly);
            }

            foreach (string directory in Directory.EnumerateDirectories(root, "*", SearchOption.AllDirectories))
            {
                var attributes = File.GetAttributes(directory);
                if ((attributes & FileAttributes.ReadOnly) != 0)
                    File.SetAttributes(directory, attributes & ~FileAttributes.ReadOnly);
            }
        }
        catch
        {
            // Best-effort only. Directory.Delete will throw a clearer failure if this mattered.
        }
    }

    private static string NormalizePath(string path)
    {
        return Path.GetFullPath(path)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
    }

    private static string FormatArgsForDisplay(IEnumerable<string> args)
    {
        return string.Join(" ", args.Select(QuoteIfNeeded));
    }

    private static string QuoteIfNeeded(string arg)
    {
        if (string.IsNullOrEmpty(arg))
            return "\"\"";

        return arg.Any(char.IsWhiteSpace)
            ? $"\"{arg.Replace("\"", "\\\"")}\""
            : arg;
    }
}
