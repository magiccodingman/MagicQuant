using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using MQ.DB;
using MQ.DB.Models;
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
        // NOTE:
        // - Empty stdout is treated as NOT installed
        // - stderr is captured
        // - Python errors fail fast instead of lying

        string script =
            $"import importlib.metadata, sys\n" +
            $"try:\n" +
            $"    print(importlib.metadata.version('{packageName}'))\n" +
            $"except Exception:\n" +
            $"    print('NONE')\n";

        var psi = new ProcessStartInfo(GetPythonExecutable());
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add(script);
        var result = await new MagicQuant.Runtime.ProcessRunner().RunAsync(psi);
        if (!result.Success)
            throw new InvalidOperationException($"Python package check failed for '{packageName}'.\n{result.StdErr}");
        string stdout = result.StdOut;

        string version = stdout.Trim();

        // CRITICAL FIX:
        // Empty output MUST be treated as not installed
        if (string.IsNullOrEmpty(version) || version == "NONE")
            return null;

        return version;
    }


    public string GetPythonExecutable()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return Path.Combine(_envPath, "python.exe");

        return Path.Combine(_envPath, "bin", "python");
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
        string? pthFile = Directory.GetFiles(_envPath, "*._pth").FirstOrDefault();
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

        string python = GetPythonExecutable();
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            await RunShellCommand("cmd.exe", $"/c \"{python}\" pip_runner.py install --upgrade pip setuptools wheel",
                _envPath, null);
        }
        else
        {
            await RunShellCommand(python, "-m pip install --upgrade pip setuptools wheel", _basePath, null);
        }
    }

    public async Task RunPipAsync(string pipArgs, Dictionary<string, string>? envVars = null)
    {
        string python = GetPythonExecutable();
        string exe, finalArgs;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            exe = "cmd.exe";
            finalArgs = $"/c \"{python}\" pip_runner.py {pipArgs}";
            await RunShellCommand(exe, finalArgs, _envPath, envVars);
        }
        else
        {
            exe = python;
            finalArgs = $"-m pip {pipArgs}";
            await RunShellCommand(exe, finalArgs, _envPath, envVars);
        }
    }

    public Task RunPipInstallAsync(string installArgs, Dictionary<string, string>? envVars = null)
        => RunPipAsync($"install {installArgs}", envVars);


    private bool CheckSuccessMarker() => File.Exists(Path.Combine(_envPath, MagicConstants.SuccessJson));

    private void WriteSuccessMarker() =>
        File.WriteAllText(Path.Combine(_envPath, MagicConstants.SuccessJson), "{\"status\":\"success\"}");

    public Task RunPythonScriptAsync(string scriptPath, string args = "", Dictionary<string, string>? envVars = null)
    {
        return RunShellCommand(GetPythonExecutable(), $"\"{scriptPath}\" {args}", _envPath, envVars);
    }

    public async Task RunPythonScriptAsync(string scriptPath, IReadOnlyList<string> args,
        Dictionary<string, string>? envVars = null, CancellationToken ct = default)
    {
        var start = new MagicQuant.Runtime.NativeCommand(GetPythonExecutable(), [scriptPath, .. args]).CreateStartInfo(envVars);
        start.WorkingDirectory = _envPath;
        var result = await new MagicQuant.Runtime.ProcessRunner().RunAsync(start,
            onLine: (line, _) => AnsiConsole.WriteLine(line), ct: ct);
        if (!result.Success) throw new InvalidOperationException($"Python script '{scriptPath}' failed (exit {result.ExitCode}). {result.StdErr}");
    }

    // Legacy string arguments are retained here; ProcessRunner owns native lifetime.
    private async Task RunShellCommand(string exe, string args, string workingDir,
        Dictionary<string, string>? envVars = null)
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

        if (!string.IsNullOrEmpty(workingDir))
            psi.WorkingDirectory = workingDir;

        if (envVars != null)
            foreach (var kvp in envVars)
                psi.Environment[kvp.Key] = kvp.Value;

        var result = await new MagicQuant.Runtime.ProcessRunner().RunAsync(psi,
            onLine: (line, error) => AnsiConsole.MarkupLine($"[{(error ? "red" : "grey")}]{Markup.Escape(line)}[/]"));
        if (!result.Success)
            throw new InvalidOperationException($"Command failed (exit {result.ExitCode}): {exe} {args}");
    }
}
