using System.Diagnostics;
using System.Runtime.InteropServices;
using LibGit2Sharp;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public class LlamaBuilder
{
    private readonly string _llamaRoot;
    private readonly SystemInfo _sysInfo;

    public LlamaBuilder(string magicRoot, SystemInfo sysInfo)
    {
        _llamaRoot = Path.Combine(magicRoot, MagicConstants.LlamaRepoName);
        _sysInfo = sysInfo;
    }

    public string GetLlamaBinPath() => Path.Combine(_llamaRoot, "build", "bin");

    public async Task PrepareAndBuildAsync(bool forceRebuild)
{
    // 1. Validate ALL dependencies before doing anything
    await DependencyManager.EnsureDependenciesAsync(_sysInfo);

    // 2. Clone / Pull / Update Logic
    // If --update (forceRebuild) is passed, we delete the repo to force a clean clone.
    if (forceRebuild && Directory.Exists(_llamaRoot))
    {
        AnsiConsole.MarkupLine("[yellow]Update requested: Removing old repository...[/]");
        try
        {
            // Recursive delete
            Directory.Delete(_llamaRoot, true); 
        }
        catch (Exception ex)
        {
            // Windows sometimes locks files; warn user but try to proceed or fail
            AnsiConsole.MarkupLine($"[red]Warning: Could not delete old repo: {ex.Message}[/]");
            throw; 
        }
    }

    if (!Directory.Exists(_llamaRoot))
    {
        AnsiConsole.MarkupLine($"Cloning llama.cpp to [blue]{_llamaRoot}[/]...");
        AnsiConsole.MarkupLine("[grey](This includes submodules and may take a moment)[/]");
        
        // Clone with RecurseSubmodules = true matches "git submodule update --init --recursive"
        var cloneOptions = new CloneOptions { RecurseSubmodules = true };
        Repository.Clone("https://github.com/ggerganov/llama.cpp.git", _llamaRoot, cloneOptions);
    }
    else
    {
        AnsiConsole.MarkupLine("[grey]Repository already exists. Skipping clone.[/]");
    }

    // 3. Setup Build Directory
    string buildDir = Path.Combine(_llamaRoot, "build");
    
    // If we just re-cloned, this directory is gone anyway, but if we didn't,
    // and forceRebuild is true (e.g. if deletion failed above but we continue), clean it.
    if (forceRebuild && Directory.Exists(buildDir)) 
    {
        Directory.Delete(buildDir, true);
    }
    Directory.CreateDirectory(buildDir);

    // 4. Get the CMake Executable (System or Local)
    string cmakeExe = DependencyManager.GetCmakePath() ?? "cmake";

    // 5. Generate Build Files
    string cmakeArgs = GetOptimalCmakeArgs();
    AnsiConsole.MarkupLine($"[grey]Configuring build with: {cmakeArgs}[/]");
    
    // Note: We run this inside the 'build' folder
    if (!await RunProcessAsync(cmakeExe, cmakeArgs, buildDir))
        throw new Exception("CMake configuration failed.");

    // 6. Compile
    AnsiConsole.MarkupLine("[cyan]Compiling Llama.cpp (Release Mode)...[/]");
    
    // -j triggers parallel build using all available cores
    string buildCmd = "--build . --config Release -j " + Environment.ProcessorCount;
    
    if (!await RunProcessAsync(cmakeExe, buildCmd, buildDir))
        throw new Exception("Build failed.");

    AnsiConsole.MarkupLine("[green]✔ Build Success![/]");
}

    private string GetOptimalCmakeArgs()
    {
        // Core Args
        var args = new List<string> { "..", "-DCMAKE_BUILD_TYPE=Release" };

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            args.Add("-G Ninja"); 

        // GPU Optimization Logic
        switch (_sysInfo.GpuVendor)
        {
            case GpuVendor.Nvidia:
                args.Add("-DGGML_CUDA=ON");
                // Native = Compiles specifically for the detected card (Perfect optimization)
                args.Add("-DCMAKE_CUDA_ARCHITECTURES=native"); 
                break;

            case GpuVendor.Amd:
                args.Add("-DGGML_HIPBLAS=ON");
                // If on Linux, you might add -DAMDGPU_TARGETS=gfx1100 etc if needed
                // But usually standard HIP build is sufficient
                break;

            case GpuVendor.Intel:
                // Try SYCL (OneAPI) first as it is fastest
                // If OneAPI isn't present (checked in DependencyManager), 
                // you might fallback to Vulkan here: "-DGGML_VULKAN=ON"
                args.Add("-DGGML_SYCL=ON");
                break;

            default:
                // CPU Fallback (ensure AVX is on)
                // CMake usually detects AVX2 automatically
                break;
        }

        return string.Join(" ", args);
    }

    private async Task<bool> RunProcessAsync(string exe, string args, string workingDir)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe, Arguments = args, WorkingDirectory = workingDir,
            RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false
        };
        
        using var p = Process.Start(psi);
        if (p == null) return false;

        // Capture output to show user progress
        p.OutputDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.WriteLine(e.Data); };
        p.ErrorDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.WriteLine(e.Data); };
        
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        await p.WaitForExitAsync();
        
        return p.ExitCode == 0;
    }
}