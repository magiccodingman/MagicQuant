using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public class PythonManager
{
    private readonly string _basePath;
    private readonly string _envPath;

    public PythonManager(string basePath)
    {
        _basePath = basePath;
        _envPath = Path.Combine(basePath, MagicConstants.EnvName);
    }

    public async Task<string?> GetInstalledVersionAsync(string packageName)
    {
        // We use a tiny python script to check importlib.metadata
        // This is instant compared to pip
        string script = $"import importlib.metadata; " +
                        $"try: print(importlib.metadata.version('{packageName}')); " +
                        $"except: print('NONE')";

        string python = GetPythonExecutable();
        string exe, args;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            exe = "cmd.exe";
            args = $"/c \"{python}\" -c \"{script}\"";
        }
        else
        {
            exe = python;
            args = $"-c \"{script}\"";
        }

        // Run without printing output to console
        var psi = new ProcessStartInfo
        {
            FileName = exe, Arguments = args,
            RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true
        };

        using var proc = Process.Start(psi);
        string output = await proc!.StandardOutput.ReadToEndAsync();
        await proc.WaitForExitAsync();

        string version = output.Trim();
        return version == "NONE" ? null : version;
    }
    
    public string GetPythonExecutable()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return Path.Combine(_envPath, "python.exe");
        
        return Path.Combine(_envPath, "bin", "python3");
    }

    public async Task SetupEnvironmentAsync()
    {
        AnsiConsole.MarkupLine("[cyan]Configuring Python Environment...[/]");

        if (CheckSuccessMarker()) 
        {
            AnsiConsole.MarkupLine("[green]✔ Python Environment is ready.[/]");
            return;
        }

        // Clean slate if corrupt
        if (Directory.Exists(_envPath)) Directory.Delete(_envPath, true);
        Directory.CreateDirectory(_envPath);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            await SetupWindowsEmbedAsync();
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            await SetupLinuxVenvAsync();
        }

        // Install Pip Runner logic
        await SetupPipRunnerAsync();
        
        WriteSuccessMarker();
    }

    private async Task SetupWindowsEmbedAsync()
    {
        string zipPath = Path.Combine(_basePath, MagicConstants.WinPythonZip);
        
        // Download
        if (!File.Exists(zipPath))
        {
            using var client = new HttpClient();
            AnsiConsole.MarkupLine($"Downloading Python Embeddable from [blue]{MagicConstants.WinPythonUrl}[/]");
            var data = await client.GetByteArrayAsync(MagicConstants.WinPythonUrl);
            await File.WriteAllBytesAsync(zipPath, data);
        }

        // Extract
        AnsiConsole.MarkupLine("Extracting Python...");
        ZipFile.ExtractToDirectory(zipPath, _envPath);
        
        // Cleanup Zip
        File.Delete(zipPath);

        // Modify .pth file to allow importing site-packages (Crucial for pip)
        string pthFile = Directory.GetFiles(_envPath, "*._pth").FirstOrDefault();
        if (pthFile != null)
        {
            var lines = await File.ReadAllLinesAsync(pthFile);
            var newLines = lines.Select(l => l.Trim() == "#import site" ? "import site" : l).ToList();
            await File.WriteAllLinesAsync(pthFile, newLines);
        }
    }

    private async Task SetupLinuxVenvAsync()
    {
        AnsiConsole.MarkupLine("Creating venv...");
        // FIX 1: Added _basePath as working dir, and null for env vars
        await RunShellCommand("python3", $"-m venv \"{_envPath}\"", _basePath, null);
    }

    private async Task SetupPipRunnerAsync()
    {
        // Copy pip_runner.py from Helpers to Env
        string source = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Helpers", "pip_runner.py");
        string dest = Path.Combine(_envPath, "pip_runner.py");

        if (File.Exists(source))
        {
            File.Copy(source, dest, true);
            AnsiConsole.MarkupLine("Copied pip_runner.py.");
        }
        else
        {
             AnsiConsole.MarkupLine("[yellow]Warning: pip_runner.py not found in Helpers.[/]");
        }

        // Upgrade Pip using the runner or standard module
        string python = GetPythonExecutable();
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
             // FIX 2: Added null for env vars (4th arg)
             await RunShellCommand("cmd.exe", $"/c \"{python}\" pip_runner.py install --upgrade pip setuptools wheel", _envPath, null);
        }
        else
        {
             // FIX 3: Added null for env vars (4th arg)
             await RunShellCommand(python, "-m pip install --upgrade pip setuptools wheel", _basePath, null);
        }
    }

    public async Task RunPipInstallAsync(string args, Dictionary<string, string>? envVars = null)
    {
        string python = GetPythonExecutable();
        string exe, finalArgs;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            exe = "cmd.exe";
            finalArgs = $"/c \"{python}\" pip_runner.py install {args}";
        }
        else
        {
            exe = python;
            finalArgs = $"-m pip install {args}";
        }

        await RunShellCommand(exe, finalArgs, _envPath, envVars);
    }

    private bool CheckSuccessMarker() => File.Exists(Path.Combine(_envPath, MagicConstants.SuccessJson));
    private void WriteSuccessMarker() => File.WriteAllText(Path.Combine(_envPath, MagicConstants.SuccessJson), "{\"status\":\"success\"}");

    // The Method Signature causing the issue
    private async Task RunShellCommand(string exe, string args, string workingDir, Dictionary<string, string>? envVars = null)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
    
        if (!string.IsNullOrEmpty(workingDir)) psi.WorkingDirectory = workingDir;

        if (envVars != null)
        {
            foreach (var kvp in envVars)
            {
                psi.EnvironmentVariables[kvp.Key] = kvp.Value;
            }
        }
        
        using var proc = Process.Start(psi);
        if (proc == null) return;

        proc.OutputDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.MarkupLine($"[grey]{Markup.Escape(e.Data)}[/]"); };
        proc.ErrorDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.MarkupLine($"[red]{Markup.Escape(e.Data)}[/]"); };
        
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        await proc.WaitForExitAsync();
    }
}