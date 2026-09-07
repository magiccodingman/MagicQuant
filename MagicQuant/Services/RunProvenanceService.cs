using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Runtime;
using MQ.DB;

namespace MagicQuant.Services;

/// <summary>
/// Records local campaign inputs and completion independently of export cleanup.
/// It is not published into model cards: config/argv can contain private paths or URLs.
/// </summary>
public sealed class RunProvenanceService
{
    private readonly string _path;
    private readonly Dictionary<string, object?> _record;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public RunProvenanceService(string command, string[] args, MagicQuantYamlLoader.LoadedConfiguration loaded)
    {
        string root = string.IsNullOrWhiteSpace(loaded.Settings.Paths.ModelDir)
            ? loaded.Settings.Paths.MagicQuantRoot!
            : Path.Combine(Path.GetFullPath(loaded.Settings.Paths.ModelDir), "MagicQuant");
        string runId = $"{DateTime.UtcNow:yyyyMMddTHHmmssfffZ}-{Guid.NewGuid():N}";
        _path = Path.Combine(root, "Runs", runId, "run.json");
        _record = new()
        {
            ["schemaVersion"] = 1,
            ["runId"] = runId,
            ["command"] = command,
            ["arguments"] = args,
            ["startedUtc"] = DateTimeOffset.UtcNow,
            ["status"] = "running",
            ["programVersion"] = typeof(RunProvenanceService).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion,
            ["dotnetVersion"] = Environment.Version.ToString(),
            ["operatingSystem"] = RuntimeInformation.OSDescription,
            ["configPath"] = loaded.Path,
            ["configSha256"] = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(loaded.Path))).ToLowerInvariant(),
            // Serialize now: dynamic custom-baseline registration must not rewrite the input snapshot.
            ["configuration"] = JsonSerializer.SerializeToElement(loaded.Settings, JsonOptions)
        };
        Write();
    }

    public string ManifestPath => _path;

    public async Task CaptureToolchainAsync()
    {
        _record["llamaRoot"] = Cache.LlamaRoot;
        _record["llamaBin"] = Cache.LlamaBin;
        _record["llamaRevision"] = await TryReadToolAsync("git", ["-C", Cache.LlamaRoot ?? "", "rev-parse", "HEAD"]);
        string python = new PythonManager(Cache.MagicQuantDirectory!).GetPythonExecutable();
        _record["pythonExecutable"] = python;
        _record["pythonVersion"] = await TryReadToolAsync(python, ["--version"]);
        _record["pythonPackages"] = await TryReadToolAsync(python, ["-m", "pip", "freeze"]);
        Write();
    }

    public void Complete(string status, string? error = null)
    {
        _record["status"] = status;
        _record["completedUtc"] = DateTimeOffset.UtcNow;
        _record["error"] = error;
        _record["modelId"] = Cache.CurrentModelId;
        _record["architectureFamily"] = Cache.CurrentArchitectureFamilyName;
        _record["tensorGroupProfile"] = Cache.CurrentTensorGroupProfileFingerprintHash;
        _record["imatrixIdentity"] = Cache.ActiveImatrixIdentityHash;
        _record["outputDirectory"] = Cache.OutputDirectory;
        Write();
    }

    private void Write()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        string temporary = _path + ".tmp";
        try
        {
            File.WriteAllText(temporary, JsonSerializer.Serialize(_record, JsonOptions));
            File.Move(temporary, _path, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private static async Task<string?> TryReadToolAsync(string executable, string[] args)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try
        {
            var result = await new ProcessRunner().RunAsync(new NativeCommand(executable, args).CreateStartInfo(), ct: timeout.Token);
            return result.Success ? result.CombinedOutput.Trim() : null;
        }
        catch (OperationCanceledException) when (!RunCancellation.Token.IsCancellationRequested) { return null; }
        catch (System.ComponentModel.Win32Exception) { return null; }
    }
}
