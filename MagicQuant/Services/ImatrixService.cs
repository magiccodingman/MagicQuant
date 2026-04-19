using System.Security.Cryptography;
using System.Text.Json;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class ImatrixService
{
    private readonly JsonSerializerOptions _json = new()
    {
        WriteIndented = true
    };

    public async Task<ImatrixEnsureResult> EnsureImatrixAsync(ImatrixRequest request, CancellationToken ct = default)
    {
        if (!request.UseImatrix)
        {
            Cache.IsImatrixAvailable = false;
            Cache.ActiveImatrixPath = null;
            RuntimeSearchSpace.SetImatrixAvailability(false);
            return new ImatrixEnsureResult { Enabled = false, Available = false };
        }

        ValidateRequest(request, out var sourceKind, out var sourceIdentity);

        string imatrixDir = Path.Combine(request.MagicQuantDirectory, "imatrix");
        Directory.CreateDirectory(imatrixDir);

        string datPath = Path.Combine(imatrixDir, "imatrix.dat");
        string successPath = Path.Combine(imatrixDir, "imatrix.success.json");
        string metadataPath = Path.Combine(imatrixDir, "imatrix.metadata.json");
        string buildLogPath = Path.Combine(imatrixDir, "imatrix.build.log");

        if (request.ForceRebuild)
            await CleanupArtifactsAsync(datPath, successPath, metadataPath, buildLogPath);

        bool shouldRebuild = await ShouldRebuildAsync(request, sourceKind, datPath, successPath, metadataPath);

        if (shouldRebuild)
        {
            await CleanupArtifactsAsync(datPath, successPath, metadataPath, buildLogPath);
            await AcquireImatrixAsync(request, sourceKind, sourceIdentity, datPath, metadataPath, successPath, buildLogPath, ct);

            Cache.IsImatrixAvailable = true;
            Cache.ActiveImatrixPath = datPath;
            RuntimeSearchSpace.SetImatrixAvailability(true);

            return new ImatrixEnsureResult
            {
                Enabled = true,
                Available = true,
                Rebuilt = true,
                CanonicalImatrixPath = datPath,
                SourceKind = sourceKind
            };
        }

        Cache.IsImatrixAvailable = true;
        Cache.ActiveImatrixPath = datPath;
        RuntimeSearchSpace.SetImatrixAvailability(true);

        return new ImatrixEnsureResult
        {
            Enabled = true,
            Available = true,
            Rebuilt = false,
            CanonicalImatrixPath = datPath,
            SourceKind = sourceKind
        };
    }

    public bool ShouldUseImatrixForQuant(HybridQuant quant)
    {
        if (!Cache.UseImatrix || !Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixPath))
            return false;

        if (quant.BaseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId)
            return false;

        string baseName = quant.BaseQuant.Names.IsDefaultOrEmpty
            ? string.Empty
            : quant.BaseQuant.Names[0];

        return !baseName.Equals("BF16", StringComparison.OrdinalIgnoreCase) &&
               !baseName.Equals("F16", StringComparison.OrdinalIgnoreCase) &&
               !baseName.Equals("F32", StringComparison.OrdinalIgnoreCase);
    }

    public string GetCanonicalImatrixPath()
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new InvalidOperationException("Cache.ModelMagicQuantDirectory is not set.");

        return Path.Combine(Cache.ModelMagicQuantDirectory, "imatrix", "imatrix.dat");
    }

    private static void ValidateRequest(ImatrixRequest request, out ImatrixSourceKind sourceKind, out string sourceIdentity)
    {
        int activeModes = 0;

        bool urlMode = !string.IsNullOrWhiteSpace(request.ImatrixUrl);
        bool hfMode = !string.IsNullOrWhiteSpace(request.DatasetRepo);
        bool localMode = !string.IsNullOrWhiteSpace(request.LocalDatasetFile);

        if (urlMode) activeModes++;
        if (hfMode) activeModes++;
        if (localMode) activeModes++;

        if (activeModes != 1)
        {
            throw new InvalidOperationException(
                "When --use-imatrix is true, exactly one source mode must be provided: --imatrix-url OR --imatrix-dataset-repo OR --imatrix-dataset-local-file.");
        }

        if (urlMode)
        {
            if (!Uri.TryCreate(request.ImatrixUrl, UriKind.Absolute, out var parsed) ||
                (parsed.Scheme != Uri.UriSchemeHttps && parsed.Scheme != Uri.UriSchemeHttp))
            {
                throw new InvalidOperationException("--imatrix-url must be a valid http/https URL.");
            }

            sourceKind = ImatrixSourceKind.Https;
            sourceIdentity = request.ImatrixUrl!;
            return;
        }

        if (hfMode)
        {
            if (string.IsNullOrWhiteSpace(request.DatasetSplit))
                throw new InvalidOperationException("--imatrix-dataset-split is required with --imatrix-dataset-repo.");

            sourceKind = ImatrixSourceKind.HfDataset;
            sourceIdentity = $"{request.DatasetRepo}:{request.DatasetConfig ?? "default"}:{request.DatasetSplit}";
            return;
        }

        if (string.IsNullOrWhiteSpace(request.LocalDatasetFile))
            throw new InvalidOperationException("--imatrix-dataset-local-file cannot be empty.");

        string ext = Path.GetExtension(request.LocalDatasetFile).ToLowerInvariant();
        if (ext is not ".json" and not ".jsonl")
        {
            throw new InvalidOperationException(
                $"Local dataset file mode supports only .json/.jsonl in MVP. Got '{ext}'. If this is a YAML recipe, add explicit recipe parsing support or use a raw JSON/JSONL corpus file.");
        }

        sourceKind = ImatrixSourceKind.LocalDatasetFile;
        sourceIdentity = Path.GetFullPath(request.LocalDatasetFile);
    }

    private async Task<bool> ShouldRebuildAsync(
        ImatrixRequest request,
        ImatrixSourceKind sourceKind,
        string datPath,
        string successPath,
        string metadataPath)
    {
        bool hasDat = File.Exists(datPath);
        bool hasSuccess = File.Exists(successPath);

        if (hasDat && !hasSuccess)
            return true;

        if (!hasDat && hasSuccess)
            return true;

        if (!hasDat || !hasSuccess || !File.Exists(metadataPath))
            return true;

        var metadata = JsonSerializer.Deserialize<ImatrixMetadataSidecar>(await File.ReadAllTextAsync(metadataPath), _json);
        if (metadata == null)
            return true;

        return !MetadataMatchesRequest(metadata, request, sourceKind);
    }

    private static bool MetadataMatchesRequest(ImatrixMetadataSidecar metadata, ImatrixRequest request, ImatrixSourceKind sourceKind)
    {
        if (!string.Equals(metadata.SourceKind, ToSidecarSourceKind(sourceKind), StringComparison.OrdinalIgnoreCase))
            return false;

        return sourceKind switch
        {
            ImatrixSourceKind.Https => string.Equals(metadata.OriginalUrl, request.ImatrixUrl, StringComparison.Ordinal),
            ImatrixSourceKind.HfDataset =>
                string.Equals(metadata.DatasetRepo, request.DatasetRepo, StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetConfig, request.DatasetConfig, StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetSplit, request.DatasetSplit, StringComparison.Ordinal),
            ImatrixSourceKind.LocalDatasetFile =>
                string.Equals(metadata.LocalDatasetFile, Path.GetFullPath(request.LocalDatasetFile!), StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetSplit, request.DatasetSplit, StringComparison.Ordinal),
            _ => false
        };
    }

    private async Task AcquireImatrixAsync(
        ImatrixRequest request,
        ImatrixSourceKind sourceKind,
        string sourceIdentity,
        string datPath,
        string metadataPath,
        string successPath,
        string buildLogPath,
        CancellationToken ct)
    {
        switch (sourceKind)
        {
            case ImatrixSourceKind.Https:
                await AcquireFromHttpsAsync(request, datPath, buildLogPath, ct);
                break;
            case ImatrixSourceKind.LocalDatasetFile:
                await BuildFromLocalDatasetAsync(request, datPath, buildLogPath, ct);
                break;
            case ImatrixSourceKind.HfDataset:
                await BuildFromHfDatasetAsync(request, datPath, buildLogPath, ct);
                break;
            default:
                throw new InvalidOperationException($"Unknown imatrix source kind '{sourceKind}'.");
        }

        var metadata = new ImatrixMetadataSidecar
        {
            SourceKind = ToSidecarSourceKind(sourceKind),
            OriginalUrl = request.ImatrixUrl,
            OriginalDownloadName = request.ImatrixUrl == null ? null : Path.GetFileName(new Uri(request.ImatrixUrl).AbsolutePath),
            DatasetRepo = request.DatasetRepo,
            DatasetConfig = request.DatasetConfig,
            DatasetSplit = request.DatasetSplit,
            LocalDatasetFile = string.IsNullOrWhiteSpace(request.LocalDatasetFile) ? null : Path.GetFullPath(request.LocalDatasetFile)
        };

        await File.WriteAllTextAsync(metadataPath, JsonSerializer.Serialize(metadata, _json), ct);

        var fileInfo = new FileInfo(datPath);
        if (!fileInfo.Exists || fileInfo.Length == 0)
            throw new InvalidOperationException("Imatrix acquisition completed but canonical imatrix.dat is missing or empty.");

        var success = new ImatrixSuccessSidecar
        {
            CompletedUtc = DateTime.UtcNow,
            CanonicalPath = datPath,
            SourceKind = ToSidecarSourceKind(sourceKind),
            SourceIdentity = sourceIdentity,
            Split = request.DatasetSplit,
            Config = request.DatasetConfig,
            Sha256 = await ComputeSha256Async(datPath, ct),
            FileSizeBytes = fileInfo.Length
        };

        await File.WriteAllTextAsync(successPath, JsonSerializer.Serialize(success, _json), ct);
    }

    private static async Task CleanupArtifactsAsync(params string[] paths)
    {
        foreach (var path in paths)
        {
            if (File.Exists(path))
                await HardDeleteHelper.DeleteFileIfExistsAsync(path);
        }
    }

    private static string ToSidecarSourceKind(ImatrixSourceKind kind) => kind switch
    {
        ImatrixSourceKind.Https => "https",
        ImatrixSourceKind.HfDataset => "hf_dataset",
        ImatrixSourceKind.LocalDatasetFile => "local_dataset_file",
        _ => "unknown"
    };

    private static async Task<string> ComputeSha256Async(string path, CancellationToken ct)
    {
        await using var stream = File.OpenRead(path);
        var hash = await SHA256.HashDataAsync(stream, ct);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static async Task AcquireFromHttpsAsync(ImatrixRequest request, string datPath, string buildLogPath, CancellationToken ct)
    {
        string tempPath = datPath + ".source.tmp";

        using var client = new HttpClient();
        await using (var sourceStream = await client.GetStreamAsync(request.ImatrixUrl!, ct))
        await using (var destinationStream = File.Create(tempPath))
        {
            await sourceStream.CopyToAsync(destinationStream, ct);
        }

        var tempInfo = new FileInfo(tempPath);
        if (!tempInfo.Exists || tempInfo.Length == 0)
            throw new InvalidOperationException("Downloaded imatrix file is empty.");

        if (File.Exists(datPath))
            File.Delete(datPath);

        File.Move(tempPath, datPath);
        await File.WriteAllTextAsync(buildLogPath, $"Downloaded from {request.ImatrixUrl} at {DateTime.UtcNow:O}{Environment.NewLine}", ct);

        AnsiConsole.MarkupLine($"[green]Imatrix downloaded and normalized:[/] {Markup.Escape(datPath)}");
    }

    private async Task BuildFromLocalDatasetAsync(ImatrixRequest request, string datPath, string buildLogPath, CancellationToken ct)
    {
        string datasetPath = Path.GetFullPath(request.LocalDatasetFile!);
        if (!File.Exists(datasetPath))
            throw new FileNotFoundException($"Local dataset file not found: {datasetPath}");

        await BuildImatrixFromDatasetTextAsync(datasetPath, datPath, buildLogPath, ct);
    }

    private async Task BuildFromHfDatasetAsync(ImatrixRequest request, string datPath, string buildLogPath, CancellationToken ct)
    {
        string tempJsonl = Path.Combine(Path.GetDirectoryName(datPath)!, "hf-dataset.export.jsonl");
        string python = ResolvePythonExecutableOrThrow();
        string scriptPath = Path.Combine(Path.GetDirectoryName(datPath)!, "build_hf_imatrix_dataset.py");

        string script = """
import json
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--config', required=False)
parser.add_argument('--out', required=True)
args = parser.parse_args()

if args.config:
    ds = load_dataset(args.repo, args.config, split=args.split)
else:
    ds = load_dataset(args.repo, split=args.split)

with open(args.out, 'w', encoding='utf-8') as f:
    for row in ds:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
""";

        await File.WriteAllTextAsync(scriptPath, script, ct);

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = python,
            Arguments =
                $"\"{scriptPath}\" --repo \"{request.DatasetRepo}\" --split \"{request.DatasetSplit}\" " +
                (string.IsNullOrWhiteSpace(request.DatasetConfig) ? string.Empty : $"--config \"{request.DatasetConfig}\" ") +
                $"--out \"{tempJsonl}\"",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false
        };

        using var p = System.Diagnostics.Process.Start(psi)
                      ?? throw new InvalidOperationException("Failed to start Python process for HF dataset export.");

        string stdout = await p.StandardOutput.ReadToEndAsync();
        string stderr = await p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync(ct);

        await File.WriteAllTextAsync(buildLogPath, stdout + Environment.NewLine + stderr, ct);

        if (p.ExitCode != 0)
            throw new InvalidOperationException("Failed to export HF dataset for imatrix generation. See imatrix.build.log.");

        await BuildImatrixFromDatasetTextAsync(tempJsonl, datPath, buildLogPath, ct);
    }

    private static async Task BuildImatrixFromDatasetTextAsync(string datasetPath, string datPath, string buildLogPath, CancellationToken ct)
    {
        string llamaBin = Cache.LlamaBin ?? throw new InvalidOperationException("Cache.LlamaBin not set.");
        string binaryName = OperatingSystem.IsWindows() ? "llama-imatrix.exe" : "llama-imatrix";
        string imatrixBin = Path.Combine(llamaBin, binaryName);

        if (!File.Exists(imatrixBin))
            throw new InvalidOperationException($"Missing {binaryName}. Cannot build imatrix from dataset sources.");

        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        string torchType = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        string baseModelPath = Path.Combine(Cache.ModelMagicQuantDirectory!, "GGUF", $"{modelName}-{torchType}.gguf");

        if (!File.Exists(baseModelPath))
            throw new InvalidOperationException($"Base model GGUF is required before dataset-based imatrix build. Missing: {baseModelPath}");

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = imatrixBin,
            Arguments = $"-m \"{baseModelPath}\" -f \"{datasetPath}\" -o \"{datPath}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        using var p = System.Diagnostics.Process.Start(psi)
                      ?? throw new InvalidOperationException("Failed to start llama-imatrix process.");

        string stdout = await p.StandardOutput.ReadToEndAsync();
        string stderr = await p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync(ct);

        await File.AppendAllTextAsync(buildLogPath, stdout + Environment.NewLine + stderr, ct);

        if (p.ExitCode != 0)
            throw new InvalidOperationException("llama-imatrix failed. See imatrix.build.log.");
    }

    private static string ResolvePythonExecutableOrThrow()
    {
        if (string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            throw new InvalidOperationException("Cache.MagicQuantDirectory is not set.");

        var py = new PythonManager(Cache.MagicQuantDirectory);
        string pythonExe = py.GetPythonExecutable();

        if (!File.Exists(pythonExe))
            throw new InvalidOperationException(
                $"Python environment is missing or broken at '{pythonExe}'. Re-run initialize-llama-cpp.");

        return pythonExe;
    }
}
