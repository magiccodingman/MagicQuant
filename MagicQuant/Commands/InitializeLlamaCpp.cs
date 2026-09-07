using MagicQuant.Models;
using MagicQuant.Helpers;
using Spectre.Console;
using System.Runtime.InteropServices;
using System.Diagnostics;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Commands;

public class InitializeLlamaCpp : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            AnsiConsole.MarkupLine("[bold yellow]Command: initialize-llama-cpp[/]");
            AnsiConsole.WriteLine("Initialize llama.cpp and Python dependencies in the shared user MagicQuant directory.");
            AnsiConsole.WriteLine("  --update          Update dependencies and rebuild llama.cpp");
            AnsiConsole.WriteLine("  --llama-root      Existing llama.cpp checkout (requires both paths below)");
            AnsiConsole.WriteLine("  --llama-bin       Existing compiled binaries directory");
            AnsiConsole.WriteLine("  --convert-script  Existing convert_hf_to_gguf.py file");
            AnsiConsole.WriteLine("Without custom paths, setup can download dependencies and request sudo on Linux.");
            AnsiConsole.WriteLine("--validate / --verify retain setup behavior; they are not a read-only check.");
            return;
        }

        // ---------------------------------------------------------
        // 1. Argument Parsing & Path Validation
        // ---------------------------------------------------------
        bool update = args.Any(a => a.Name?.ToLower() == "update");
        
        string? convertScript = args.FirstOrDefault(a => a.Name?.ToLower() == "convert-script")?.Value;
        string? llamaBin = args.FirstOrDefault(a => a.Name?.ToLower() == "llama-bin")?.Value;
        string? llamaRoot = args.FirstOrDefault(a => a.Name?.ToLower() == "llama-root")?.Value;

        convertScript ??= Config.Current.Paths.ConvertScript;
        llamaBin ??= Config.Current.Paths.LlamaBin;
        llamaRoot ??= Config.Current.Paths.LlamaRoot;

        // Custom Path Validation
        if (!string.IsNullOrEmpty(llamaRoot))
        {
            if (string.IsNullOrEmpty(convertScript) || string.IsNullOrEmpty(llamaBin))
            {
                throw new ArgumentException("Custom paths require --llama-root, --llama-bin, AND --convert-script (or their YAML equivalents).");
            }

            // Normalize and Check
            llamaRoot = Path.GetFullPath(llamaRoot);
            llamaBin = Path.GetFullPath(llamaBin);
            convertScript = Path.GetFullPath(convertScript);

            if (!Directory.Exists(llamaRoot) || !Directory.Exists(llamaBin) || !File.Exists(convertScript))
            {
                throw new DirectoryNotFoundException("One or more custom llama.cpp paths do not exist.");
            }

            Cache.LlamaRoot = llamaRoot;
            Cache.LlamaBin = llamaBin;
            Cache.ConvertScript = convertScript;
            AnsiConsole.MarkupLine("[green]✔ Custom Environment Validated.[/]");
            _ = DetectAndCacheSystemInfo();
            return;
        }
        else if (!string.IsNullOrEmpty(convertScript) || !string.IsNullOrEmpty(llamaBin))
        {
             throw new ArgumentException("Partial llama.cpp paths provided. Provide all three custom paths or none.");
        }

        // ---------------------------------------------------------
        // 2. Setup Default Paths
        // ---------------------------------------------------------
        string userHome = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string magicQuantPath = Path.Combine(userHome, MagicConstants.MagicQuantFolder);
        if (!Directory.Exists(magicQuantPath)) Directory.CreateDirectory(magicQuantPath);

        // ---------------------------------------------------------
        // 3. Hardware Detection
        // ---------------------------------------------------------
        var sysInfo = DetectAndCacheSystemInfo();

        // ---------------------------------------------------------
        // 4. Linux System Deps (Sudo Handling)
        // ---------------------------------------------------------
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            var requiredPackages = new List<string> 
            { 
                "build-essential", "cmake", "ninja-build", "git", 
                "python3", "python3-venv", "python3-pip", "libcurl4-openssl-dev" 
            };
            
            if (sysInfo.GpuInfo.FirstOrDefault()?.GpuVendor == GpuVendor.Nvidia) requiredPackages.Add("nvidia-cuda-toolkit");

            // Check if updates are needed
            if (update || !AreLinuxPackagesInstalled(requiredPackages))
            {
                AnsiConsole.MarkupLine("[yellow]System dependencies are missing or update requested.[/]");
                AnsiConsole.MarkupLine("[grey]Sudo permissions are required to install system packages via apt.[/]");
                
                // A. Ask for Sudo permission upfront
                try 
                {
                    await RefreshSudoCredentialsAsync();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException("Sudo access denied or cancelled. Cannot install system dependencies.", ex);
                }

                // B. Run Install WITH sudo
                AnsiConsole.MarkupLine("[cyan]Installing/Updating System Dependencies (sudo apt)...[/]");
                string aptArgs = "install -y " + string.Join(" ", requiredPackages);
                
                // We run 'sudo' directly here
                await RunSimpleProcess("sudo", "apt " + aptArgs);
            }
            else
            {
                AnsiConsole.MarkupLine("[green]✔ System dependencies already installed.[/]");
            }
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            throw new PlatformNotSupportedException("Automatic macOS setup is not implemented. Provide an existing llama.cpp environment.");
        }

        // ---------------------------------------------------------
        // 5. Python Environment Setup (Runs as Normal User)
        // ---------------------------------------------------------
        var pyManager = new PythonManager(magicQuantPath);
        await pyManager.SetupEnvironmentAsync();

        // ---------------------------------------------------------
        // 6. Build Llama.cpp (Runs as Normal User)
        // ---------------------------------------------------------
        // The installer always lives in the user's shared MagicQuant directory, but
        // dependency validation is also invoked inside commands that may use an
        // isolated --magic-quant-root. Do not overwrite that configured runtime root:
        // doing so silently redirects SQLite and other campaign state back to the
        // user's shared installation directory.
        var builder = new LlamaBuilder(magicQuantPath, sysInfo);
        await builder.PrepareAndBuildAsync(update);

        // ---------------------------------------------------------
        // 7. Install Python Libraries (Runs as Normal User)
        // ---------------------------------------------------------
        AnsiConsole.Write(new Rule("[yellow]Installing Python Libraries[/]") { Justification = Justify.Left });

        // Helper to decide if we need to install
        async Task EnsurePackage(string name, string installCmd, Dictionary<string,string>? env = null)
        {
            if (!update)
            {
                string? version = await pyManager.GetInstalledVersionAsync(name);
                if (version != null)
                {
                    AnsiConsole.MarkupLine($"[green]✔ {name} is already installed (v{version}).[/]");
                    return;
                }
            }
            
            AnsiConsole.MarkupLine($"[cyan]Installing {name}...[/]");
            await pyManager.RunPipInstallAsync(installCmd, env);
        }

        // A. Purge Cache (Only on update)
        if (update)
        {
            AnsiConsole.MarkupLine("[grey]Purging pip cache...[/]");
            await pyManager.RunPipInstallAsync("cache purge");
        }

        // B. Install PyTorch (Hardware Specific & Dynamic)
        string torchCmd = "torch torchvision torchaudio";
        bool isNvidia = sysInfo.GpuInfo.FirstOrDefault()?.GpuVendor == GpuVendor.Nvidia;
        
        if (isNvidia)
        {
            double cudaVer = HardwareHelper.GetCudaVersion();
            AnsiConsole.MarkupLine($"[grey]Detected CUDA Version: {cudaVer}[/]");

            if (cudaVer >= 12.0)
            {
                torchCmd += " --index-url https://download.pytorch.org/whl/cu124";
                AnsiConsole.MarkupLine($"[cyan]Targeting PyTorch for CUDA 12.x...[/]");
            }
            else if (cudaVer >= 11.0)
            {
                torchCmd += " --index-url https://download.pytorch.org/whl/cu118";
                AnsiConsole.MarkupLine($"[cyan]Targeting PyTorch for CUDA 11.x...[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[yellow]Warning: Could not detect CUDA version or version is < 11. Installing default PyTorch.[/]");
            }
        }
        else
        {
            AnsiConsole.MarkupLine($"[cyan]Installing Standard PyTorch (CPU/AMD/Intel)...[/]");
        }

        await EnsurePackage("torch", torchCmd);

        // C. Install Core Utilities
        string coreDeps = "gguf tokenizers transformers mistral-common sentencepiece datasets huggingface_hub";
        
        if (!update && await pyManager.GetInstalledVersionAsync("transformers") != null)
        {
             AnsiConsole.MarkupLine("[green]✔ Core utilities (transformers, etc.) are installed.[/]");
        }
        else
        {
             AnsiConsole.MarkupLine("[cyan]Installing Core Utilities...[/]");
             await pyManager.RunPipInstallAsync($"--upgrade --no-cache-dir {coreDeps}");
        }

        // D. Install llama-cpp-python
        var llamaEnv = new Dictionary<string, string>();
        if (isNvidia)
        {
            llamaEnv["CMAKE_ARGS"] = "-DGGML_CUDA=on";
            llamaEnv["FORCE_CMAKE"] = "1";
        }
        else if (sysInfo.GpuInfo.FirstOrDefault()?.GpuVendor == GpuVendor.Amd)
        {
            llamaEnv["CMAKE_ARGS"] = "-DGGML_HIPBLAS=on";
            llamaEnv["FORCE_CMAKE"] = "1";
        }

        await EnsurePackage("llama-cpp-python", 
            "--upgrade --force-reinstall --no-cache-dir llama-cpp-python", 
            llamaEnv);

        AnsiConsole.MarkupLine("[bold green]Initialization Complete![/]");
        AnsiConsole.MarkupLine($"Llama Binaries: [grey]{builder.GetLlamaBinPath()}[/]");
    }

    private static SystemInfo DetectAndCacheSystemInfo()
    {
        var sysInfo = HardwareHelper.GetSystemInfo();
        Cache.SysInfo = sysInfo;

        AnsiConsole.Write(new Rule("[yellow]System Detection[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine(
            $"Detected GPU: [green]{sysInfo.GpuInfo.FirstOrDefault()?.GpuVendor}[/] " +
            $"([blue]{sysInfo.GpuInfo.FirstOrDefault()?.GpuName}[/] - {sysInfo.GpuInfo.Sum(x => x.VramGb):F1} GB)");
        AnsiConsole.MarkupLine($"Detected RAM: [blue]{sysInfo.RamGb:F1} GB[/]");

        return sysInfo;
    }

    // --- Helpers ---

    private static async Task RunSimpleProcess(string exe, string args)
    {
        var startInfo = new ProcessStartInfo(exe, args) { UseShellExecute = false, CreateNoWindow = true };
        using var p = Process.Start(startInfo)
            ?? throw new InvalidOperationException($"Could not start {exe}.");
        await p.WaitForExitAsync();
        if (p.ExitCode != 0)
            throw new InvalidOperationException($"{exe} failed with exit code {p.ExitCode}.");
    }

    private async Task RefreshSudoCredentialsAsync()
    {
        // "sudo -v" updates the user's cached credentials.
        // It will prompt for a password if necessary in the standard input.
        var psi = new ProcessStartInfo
        {
            FileName = "sudo",
            Arguments = "-v",
            UseShellExecute = false // Required to handle password prompt
        };
        
        var p = Process.Start(psi);
        await p!.WaitForExitAsync();
        
        if (p.ExitCode != 0)
        {
            throw new Exception("Sudo access denied.");
        }
    }

    private bool AreLinuxPackagesInstalled(List<string> packages)
    {
        // dpkg-query check
        foreach (var pkg in packages)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "dpkg-query",
                    Arguments = $"-W -f='${{Status}}' {pkg}",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using var p = Process.Start(psi);
                string output = p?.StandardOutput.ReadToEnd() ?? "";
                p?.WaitForExit();

                if (!output.Contains("install ok installed"))
                {
                    return false; // Found a missing package
                }
            }
            catch
            {
                return false; // Command failed, assume missing
            }
        }
        return true;
    }
}
