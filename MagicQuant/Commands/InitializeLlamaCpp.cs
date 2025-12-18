using MagicQuant.Models;
using MagicQuant.Helpers;
using Spectre.Console;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace MagicQuant.Commands;

public class InitializeLlamaCpp : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        // ---------------------------------------------------------
        // 1. Argument Parsing & Path Validation
        // ---------------------------------------------------------
        bool validate = args.Any(a => a.Name?.ToLower() == "validate" || a.Name?.ToLower() == "verify");
        bool update = args.Any(a => a.Name?.ToLower() == "update");
        
        string? convertScript = args.FirstOrDefault(a => a.Name?.ToLower() == "convert-script")?.Value;
        string? llamaBin = args.FirstOrDefault(a => a.Name?.ToLower() == "llama-bin")?.Value;
        string? llamaRoot = args.FirstOrDefault(a => a.Name?.ToLower() == "llama-root")?.Value;

        // Custom Path Validation
        if (!string.IsNullOrEmpty(llamaRoot))
        {
            if (string.IsNullOrEmpty(convertScript) || string.IsNullOrEmpty(llamaBin))
            {
                AnsiConsole.MarkupLine("[red]Error: If you provide custom paths, you must provide --llama-root, --llama-bin, AND --convert-script[/]");
                return;
            }

            // Normalize and Check
            llamaRoot = Path.GetFullPath(llamaRoot);
            llamaBin = Path.GetFullPath(llamaBin);
            convertScript = Path.GetFullPath(convertScript);

            if (!Directory.Exists(llamaRoot) || !Directory.Exists(llamaBin) || !File.Exists(convertScript))
            {
                AnsiConsole.MarkupLine("[red]Error: One or more provided custom paths do not exist.[/]");
                return;
            }

            AnsiConsole.MarkupLine("[green]✔ Custom Environment Validated.[/]");
            return;
        }
        else if (!string.IsNullOrEmpty(convertScript) || !string.IsNullOrEmpty(llamaBin))
        {
             AnsiConsole.MarkupLine("[red]Error: Partial paths provided. Provide ALL custom paths or NONE to use defaults.[/]");
             return;
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
        var sysInfo = HardwareHelper.GetSystemInfo();
        AnsiConsole.Write(new Rule("[yellow]System Detection[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Detected GPU: [green]{sysInfo.GpuVendor}[/] ([blue]{sysInfo.GpuName}[/] - {sysInfo.VramGb:F1} GB)");
        AnsiConsole.MarkupLine($"Detected RAM: [blue]{sysInfo.RamGb:F1} GB[/]");

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
            
            if (sysInfo.GpuVendor == GpuVendor.Nvidia) requiredPackages.Add("nvidia-cuda-toolkit");

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
                catch
                {
                    AnsiConsole.MarkupLine("[red]Error: Sudo access denied or cancelled. Cannot install system dependencies.[/]");
                    return;
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
            AnsiConsole.MarkupLine("[red]Error: MacOS support coming soon.[/]");
            return;
        }

        // ---------------------------------------------------------
        // 5. Python Environment Setup (Runs as Normal User)
        // ---------------------------------------------------------
        var pyManager = new PythonManager(magicQuantPath);
        await pyManager.SetupEnvironmentAsync();

        // ---------------------------------------------------------
        // 6. Build Llama.cpp (Runs as Normal User)
        // ---------------------------------------------------------
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
        bool isNvidia = sysInfo.GpuVendor == GpuVendor.Nvidia;
        
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
        string coreDeps = "gguf tokenizers transformers mistral-common sentencepiece datasets";
        
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
        else if (sysInfo.GpuVendor == GpuVendor.Amd)
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

    // --- Helpers ---

    private static async Task RunSimpleProcess(string exe, string args)
    {
        var startInfo = new ProcessStartInfo(exe, args) { UseShellExecute = false, CreateNoWindow = true };
        var p = Process.Start(startInfo);
        await p!.WaitForExitAsync();
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