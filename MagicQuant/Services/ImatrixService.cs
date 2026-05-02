using System.Security.Cryptography;
using System.Text.Json;
using System.Text;
using System.Text.RegularExpressions;
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
        AnsiConsole.MarkupLine("[grey]Imatrix: starting ensure flow...[/]");

        if (!request.UseImatrix)
        {
            AnsiConsole.MarkupLine("[grey]Imatrix: disabled by --use-imatrix flag (false).[/]");
            Cache.IsImatrixAvailable = false;
            Cache.ActiveImatrixPath = null;
            Cache.ActiveImatrixIdentityHash = null;
            RuntimeSearchSpace.SetImatrixAvailability(false);
            return new ImatrixEnsureResult { Enabled = false, Available = false };
        }

        ValidateRequest(request, out var sourceKind, out var sourceIdentity);
        AnsiConsole.MarkupLine(
            $"[grey]Imatrix: validated source mode:[/] [cyan]{Markup.Escape(ToSidecarSourceKind(sourceKind))}[/]");

        string imatrixDir = Path.Combine(request.MagicQuantDirectory, "imatrix");
        Directory.CreateDirectory(imatrixDir);
        AnsiConsole.MarkupLine($"[grey]Imatrix: using directory:[/] [cyan]{Markup.Escape(imatrixDir)}[/]");

        string datPath = Path.Combine(imatrixDir, "imatrix.dat");
        string successPath = Path.Combine(imatrixDir, "imatrix.success.json");
        string metadataPath = Path.Combine(imatrixDir, "imatrix.metadata.json");
        string buildLogPath = Path.Combine(imatrixDir, "imatrix.build.log");

        if (request.ForceRebuild)
        {
            AnsiConsole.MarkupLine("[yellow]Imatrix: force rebuild enabled, cleaning prior canonical artifacts...[/]");
            await CleanupArtifactsAsync(datPath, successPath, metadataPath, buildLogPath);
        }

        bool shouldRebuild = await ShouldRebuildAsync(request, sourceKind, datPath, successPath, metadataPath);

        if (shouldRebuild)
        {
            AnsiConsole.MarkupLine("[grey]Imatrix: canonical artifacts missing/stale/mismatched; rebuilding now...[/]");
            await CleanupArtifactsAsync(datPath, successPath, metadataPath, buildLogPath);
            await AcquireImatrixAsync(request, sourceKind, sourceIdentity, datPath, metadataPath, successPath, buildLogPath, ct);

            Cache.IsImatrixAvailable = true;
            Cache.ActiveImatrixPath = datPath;
            Cache.ActiveImatrixIdentityHash = await ComputeSha256Async(datPath, ct);
            RuntimeSearchSpace.SetImatrixAvailability(true);
            AnsiConsole.MarkupLine($"[green]Imatrix: ready (rebuilt).[/] [grey]{Markup.Escape(datPath)}[/]");

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
        Cache.ActiveImatrixIdentityHash = await ComputeSha256Async(datPath, ct);
        RuntimeSearchSpace.SetImatrixAvailability(true);
        AnsiConsole.MarkupLine($"[green]Imatrix: ready (reused existing trusted artifact).[/] [grey]{Markup.Escape(datPath)}[/]");

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

        var baseScheme = quant.BaseQuant.DefaultTensorScheme;
        if (baseScheme != null && TensorWeightScheme.IsNativePrecisionScheme(baseScheme))
            return false;

        return true;
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

        string effectiveSplit = GetEffectiveSplitForMetadata(request, sourceKind);

        return sourceKind switch
        {
            ImatrixSourceKind.Https => string.Equals(metadata.OriginalUrl, request.ImatrixUrl, StringComparison.Ordinal),
            ImatrixSourceKind.HfDataset =>
                string.Equals(metadata.DatasetRepo, request.DatasetRepo, StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetConfig, request.DatasetConfig, StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetSplit, effectiveSplit, StringComparison.Ordinal),
            ImatrixSourceKind.LocalDatasetFile =>
                string.Equals(metadata.LocalDatasetFile, Path.GetFullPath(request.LocalDatasetFile!), StringComparison.Ordinal) &&
                string.Equals(metadata.DatasetSplit, effectiveSplit, StringComparison.Ordinal),
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
        AnsiConsole.MarkupLine(
            $"[grey]Imatrix: acquiring from source:[/] [cyan]{Markup.Escape(ToSidecarSourceKind(sourceKind))}[/]");
        string effectiveSplit = GetEffectiveSplitForMetadata(request, sourceKind);

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
            DatasetSplit = effectiveSplit,
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
            Split = effectiveSplit,
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

    private static string GetEffectiveSplitForMetadata(ImatrixRequest request, ImatrixSourceKind sourceKind)
    {
        if (sourceKind == ImatrixSourceKind.LocalDatasetFile)
            return string.IsNullOrWhiteSpace(request.DatasetSplit) ? "<fallback-recursive>" : request.DatasetSplit.Trim();

        return request.DatasetSplit ?? string.Empty;
    }

    private static async Task<string> ComputeSha256Async(string path, CancellationToken ct)
    {
        await using var stream = File.OpenRead(path);
        var hash = await SHA256.HashDataAsync(stream, ct);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static async Task AcquireFromHttpsAsync(ImatrixRequest request, string datPath, string buildLogPath, CancellationToken ct)
    {
        string tempPath = datPath + ".source.tmp";
        AnsiConsole.MarkupLine($"[grey]Imatrix: downloading from URL:[/] [cyan]{Markup.Escape(request.ImatrixUrl ?? string.Empty)}[/]");

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

        AnsiConsole.MarkupLine($"[grey]Imatrix: building from local dataset file:[/] [cyan]{Markup.Escape(datasetPath)}[/]");
        string exportedCorpusPath = Path.Combine(Path.GetDirectoryName(datPath)!, "local-dataset.export.txt");
        string? splitPropertyPath = string.IsNullOrWhiteSpace(request.DatasetSplit) ? null : request.DatasetSplit!.Trim();

        if (splitPropertyPath != null)
        {
            AnsiConsole.MarkupLine(
                $"[grey]Imatrix: local dataset split/property =[/] [cyan]{Markup.Escape(splitPropertyPath)}[/]");
            AnsiConsole.MarkupLine(
                $"[grey]Imatrix: extracting property/path[/] [cyan]{Markup.Escape(splitPropertyPath)}[/] [grey]from local JSON rows.[/]");
        }
        else
        {
            AnsiConsole.MarkupLine(
                "[yellow]Imatrix warning:[/] no local split/property provided; using explicit fallback recursive text extraction mode.");
        }

        var exportSummary = await ExportLocalDatasetToCorpusAsync(
            datasetPath,
            exportedCorpusPath,
            splitPropertyPath,
            buildLogPath,
            ct);

        if (exportSummary.StructuredRows > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Imatrix warning:[/] input appears to be structured JSON rows " +
                $"([cyan]{exportSummary.StructuredRows}[/]/[cyan]{exportSummary.TotalRows}[/]). " +
                "Flattening extracted text into temporary corpus before llama-imatrix.");
        }

        AnsiConsole.MarkupLine(
            $"[grey]Imatrix: local dataset export complete.[/] rows=[cyan]{exportSummary.TotalRows}[/], " +
            $"structured=[cyan]{exportSummary.StructuredRows}[/], missing_split=[cyan]{exportSummary.RowsMissingSplitProperty}[/], " +
            $"extracted_text_blocks=[cyan]{exportSummary.ExtractedTextBlocks}[/], " +
            $"corpus=[cyan]{Markup.Escape(exportedCorpusPath)}[/]");

        await BuildImatrixFromDatasetTextAsync(exportedCorpusPath, datPath, buildLogPath, ct);
    }

    private async Task BuildFromHfDatasetAsync(ImatrixRequest request, string datPath, string buildLogPath, CancellationToken ct)
    {
        string tempJsonl = Path.Combine(Path.GetDirectoryName(datPath)!, "hf-dataset.export.jsonl");
        string python = ResolvePythonExecutableOrThrow();
        string scriptPath = Path.Combine(Path.GetDirectoryName(datPath)!, "build_hf_imatrix_dataset.py");
        AnsiConsole.MarkupLine(
            $"[grey]Imatrix: exporting Hugging Face dataset[/] [cyan]{Markup.Escape(request.DatasetRepo ?? string.Empty)}[/]" +
            $"[grey] split=[/][cyan]{Markup.Escape(request.DatasetSplit ?? string.Empty)}[/]");

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
        AnsiConsole.MarkupLine(
            $"[grey]Imatrix: invoking llama-imatrix build from dataset:[/] [cyan]{Markup.Escape(datasetPath)}[/]");
        AnsiConsole.MarkupLine($"[grey]Imatrix: streaming llama-imatrix output to:[/] [cyan]{Markup.Escape(buildLogPath)}[/]");
        if (File.Exists(datasetPath))
        {
            long corpusSizeBytes = new FileInfo(datasetPath).Length;
            AnsiConsole.MarkupLine(
                $"[grey]Imatrix: exported corpus ready.[/] [cyan]size={Markup.Escape(FormatBytes(corpusSizeBytes))}[/]");
        }

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

        /*var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = imatrixBin,
            Arguments =
                $"--no-mmap " +
                $"-m \"{baseModelPath}\" " +
                $"-f \"{datasetPath}\" " +
                $"-o \"{datPath}\" ",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };*/

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = imatrixBin,
            Arguments =
                $"--no-mmap " +
                //$"-ngl 45 " +
                //$"--tensor-split 19,22 " +
                $"-m \"{baseModelPath}\" " +
                $"-f \"{datasetPath}\" " +
                $"-o \"{datPath}\" " +
               // $"-b 128 " +
               // $"-ub 64 " +
                $"-fa off",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        psi.Environment["GGML_CUDA_DISABLE_GRAPHS"] = "1";
        
        string launchedCommand = $"\"{imatrixBin}\" {psi.Arguments}";
        
        AnsiConsole.MarkupLine($"[grey]Imatrix: launching command:[/] [cyan]{Markup.Escape(launchedCommand)}[/]");

        using var p = System.Diagnostics.Process.Start(psi)
                      ?? throw new InvalidOperationException("Failed to start llama-imatrix process.");

        await using var buildLog = new StreamWriter(buildLogPath, append: true, Encoding.UTF8);
        await buildLog.WriteLineAsync($"[{DateTime.UtcNow:O}] Launch: {launchedCommand}");
        await buildLog.FlushAsync();

        using var writeLock = new SemaphoreSlim(1, 1);
        var startedUtc = DateTime.UtcNow;
        var maxRuntime = TimeSpan.FromHours(168);
        int outputLineCount = 0;
        bool datDetected = false;
        long lastDatSize = -1;

        Task stdoutTask = PumpProcessStreamAsync(p.StandardOutput, "stdout", buildLog, writeLock, line =>
        {
            outputLineCount++;
            if (Cache.VerboseProcessOutput)
                AnsiConsole.MarkupLine($"[grey]llama-imatrix stdout:[/] {Markup.Escape(line)}");
        }, ct);

        Task stderrTask = PumpProcessStreamAsync(p.StandardError, "stderr", buildLog, writeLock, line =>
        {
            outputLineCount++;
            if (Cache.VerboseProcessOutput)
                AnsiConsole.MarkupLine($"[grey]llama-imatrix stderr:[/] {Markup.Escape(line)}");
        }, ct);

        while (!p.HasExited)
        {
            await Task.Delay(TimeSpan.FromSeconds(30), ct);
            var elapsed = DateTime.UtcNow - startedUtc;
            bool datExists = File.Exists(datPath);
            string datSizeText = "n/a";
            if (datExists)
            {
                long datSize = new FileInfo(datPath).Length;
                datSizeText = FormatBytes(datSize);

                if (!datDetected)
                {
                    datDetected = true;
                    lastDatSize = datSize;
                    AnsiConsole.MarkupLine($"[green]Imatrix: output file detected:[/] [cyan]{Markup.Escape(datPath)}[/]");
                    AnsiConsole.MarkupLine($"[green]Imatrix: output file size now[/] [cyan]{Markup.Escape(datSizeText)}[/]");
                }
                else if (datSize != lastDatSize)
                {
                    lastDatSize = datSize;
                    AnsiConsole.MarkupLine($"[grey]Imatrix: output file size now[/] [cyan]{Markup.Escape(datSizeText)}[/]");
                }
            }

            string corpusSizeText = File.Exists(datasetPath) ? FormatBytes(new FileInfo(datasetPath).Length) : "n/a";

            if (elapsed > maxRuntime)
            {
                await buildLog.WriteLineAsync($"[{DateTime.UtcNow:O}] Timeout after {elapsed}. Killing llama-imatrix.");
                await buildLog.FlushAsync();
                p.Kill(entireProcessTree: true);
                throw new TimeoutException(
                    $"llama-imatrix exceeded safeguard runtime of {maxRuntime}. Process was terminated. See imatrix.build.log.");
            }

            AnsiConsole.MarkupLine(
                $"[grey]Imatrix: llama-imatrix still running... elapsed[/] [cyan]{elapsed:hh\\:mm\\:ss}[/]" +
                $"[grey], output lines[/] [cyan]{outputLineCount}[/]" +
                $"[grey], dat_exists[/] [cyan]{datExists}[/]" +
                $"[grey], dat_size[/] [cyan]{Markup.Escape(datSizeText)}[/]" +
                $"[grey], corpus_size[/] [cyan]{Markup.Escape(corpusSizeText)}[/]" +
                $"[grey], log[/] [cyan]{Markup.Escape(buildLogPath)}[/]");
        }

        await Task.WhenAll(stdoutTask, stderrTask);
        await p.WaitForExitAsync(ct);
        await buildLog.FlushAsync();

        if (p.ExitCode != 0)
            throw new InvalidOperationException("llama-imatrix failed. See imatrix.build.log.");

        AnsiConsole.MarkupLine($"[green]Imatrix: llama-imatrix completed successfully.[/] [grey]exit={p.ExitCode}[/]");
    }

    private static async Task<LocalDatasetExportSummary> ExportLocalDatasetToCorpusAsync(
        string datasetPath,
        string exportedCorpusPath,
        string? splitPropertyPath,
        string buildLogPath,
        CancellationToken ct)
    {
        string ext = Path.GetExtension(datasetPath).ToLowerInvariant();
        if (ext is not ".json" and not ".jsonl")
            throw new InvalidOperationException($"Unsupported local dataset extension '{ext}'.");

        int totalRows = 0;
        int structuredRows = 0;
        int extractedTextBlocks = 0;
        int rowsWithMissingSplitProperty = 0;
        bool usingSplitProperty = !string.IsNullOrWhiteSpace(splitPropertyPath);

        await using var writer = new StreamWriter(exportedCorpusPath, false, Encoding.UTF8);

        if (ext == ".jsonl")
        {
            using var reader = new StreamReader(datasetPath, Encoding.UTF8);
            while (!reader.EndOfStream)
            {
                ct.ThrowIfCancellationRequested();
                string? line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                totalRows++;
                string trimmed = line.TrimStart();
                bool looksStructured = trimmed.StartsWith("{", StringComparison.Ordinal) || trimmed.StartsWith("[", StringComparison.Ordinal);
                if (looksStructured)
                    structuredRows++;

                bool rowMissingRequestedSplit;
                foreach (string text in ExtractCorpusTextFromJsonPayload(line, splitPropertyPath, out rowMissingRequestedSplit))
                {
                    await writer.WriteLineAsync(text);
                    await writer.WriteLineAsync();
                    extractedTextBlocks++;
                }

                if (rowMissingRequestedSplit)
                    rowsWithMissingSplitProperty++;
            }
        }
        else
        {
            string json = await File.ReadAllTextAsync(datasetPath, ct);
            using var doc = JsonDocument.Parse(json);

            if (doc.RootElement.ValueKind is JsonValueKind.Object or JsonValueKind.Array)
                structuredRows = 1;

            if (doc.RootElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var row in doc.RootElement.EnumerateArray())
                {
                    totalRows++;
                    bool rowMissingRequestedSplit;
                    foreach (string text in ExtractCorpusTextFromElement(row, splitPropertyPath, out rowMissingRequestedSplit))
                    {
                        await writer.WriteLineAsync(text);
                        await writer.WriteLineAsync();
                        extractedTextBlocks++;
                    }

                    if (rowMissingRequestedSplit)
                        rowsWithMissingSplitProperty++;
                }
            }
            else
            {
                totalRows = 1;
                bool rowMissingRequestedSplit;
                foreach (string text in ExtractCorpusTextFromElement(doc.RootElement, splitPropertyPath, out rowMissingRequestedSplit))
                {
                    await writer.WriteLineAsync(text);
                    await writer.WriteLineAsync();
                    extractedTextBlocks++;
                }

                if (rowMissingRequestedSplit)
                    rowsWithMissingSplitProperty++;
            }
        }

        await writer.FlushAsync();

        if (usingSplitProperty && extractedTextBlocks == 0)
        {
            throw new InvalidOperationException(
                $"Local dataset split/property '{splitPropertyPath}' was requested but no text could be extracted from '{datasetPath}'. " +
                "Verify the property/path exists in your JSON rows.");
        }

        if (extractedTextBlocks == 0)
            throw new InvalidOperationException(
                $"No usable text content was extracted from local dataset file '{datasetPath}'.");

        await File.AppendAllTextAsync(
            buildLogPath,
            $"[{DateTime.UtcNow:O}] Local dataset export: src={datasetPath}, out={exportedCorpusPath}, " +
            $"split_property={(splitPropertyPath ?? "<fallback-recursive>")}, rows={totalRows}, structured_rows={structuredRows}, " +
            $"rows_missing_split={rowsWithMissingSplitProperty}, extracted_text_blocks={extractedTextBlocks}{Environment.NewLine}",
            ct);

        return new LocalDatasetExportSummary(totalRows, structuredRows, extractedTextBlocks, rowsWithMissingSplitProperty);
    }

    private static IEnumerable<string> ExtractCorpusTextFromJsonPayload(string payload, string? splitPropertyPath, out bool missingRequestedSplit)
    {
        missingRequestedSplit = false;
        try
        {
            using var doc = JsonDocument.Parse(payload);
            return ExtractCorpusTextFromElement(doc.RootElement, splitPropertyPath, out missingRequestedSplit).ToList();
        }
        catch (JsonException)
        {
            if (splitPropertyPath == null && LooksLikeUsefulText(payload))
                return new[] { payload.Trim() };

            if (splitPropertyPath != null)
                missingRequestedSplit = true;

            return Array.Empty<string>();
        }
    }

    private static IEnumerable<string> ExtractCorpusTextFromElement(JsonElement element, string? splitPropertyPath, out bool missingRequestedSplit)
    {
        missingRequestedSplit = false;

        if (!string.IsNullOrWhiteSpace(splitPropertyPath))
        {
            if (!TryResolveJsonPath(element, splitPropertyPath!, out JsonElement resolved))
            {
                missingRequestedSplit = true;
                return Array.Empty<string>();
            }

            var pathTexts = new List<string>();
            CollectText(resolved, pathTexts);
            return NormalizeDistinct(pathTexts);
        }

        var texts = new List<string>();
        CollectText(element, texts);
        return NormalizeDistinct(texts);
    }

    private static IEnumerable<string> NormalizeDistinct(List<string> texts)
    {
        // de-dup while preserving order
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var text in texts)
        {
            string normalized = Regex.Replace(text.Trim(), "\\s+", " ");
            if (normalized.Length == 0)
                continue;

            if (seen.Add(normalized))
                yield return normalized;
        }
    }

    private static bool TryResolveJsonPath(JsonElement row, string splitPropertyPath, out JsonElement resolved)
    {
        resolved = row;
        foreach (string rawSegment in splitPropertyPath.Split('.', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (resolved.ValueKind == JsonValueKind.Object)
            {
                if (!TryGetPropertyCaseInsensitive(resolved, rawSegment, out resolved))
                    return false;
                continue;
            }

            if (resolved.ValueKind == JsonValueKind.Array)
            {
                if (int.TryParse(rawSegment, out int index))
                {
                    if (index < 0 || index >= resolved.GetArrayLength())
                        return false;

                    resolved = resolved[index];
                    continue;
                }

                // if segment points to a property on each array element, gather all hits
                var hits = new List<JsonElement>();
                foreach (var item in resolved.EnumerateArray())
                {
                    if (item.ValueKind == JsonValueKind.Object && TryGetPropertyCaseInsensitive(item, rawSegment, out JsonElement value))
                        hits.Add(value);
                }

                if (hits.Count == 0)
                    return false;

                using var hitsDoc = JsonDocument.Parse(JsonSerializer.Serialize(hits));
                resolved = hitsDoc.RootElement.Clone();
                continue;
            }

            return false;
        }

        return true;
    }

    private static bool TryGetPropertyCaseInsensitive(JsonElement obj, string name, out JsonElement value)
    {
        foreach (var prop in obj.EnumerateObject())
        {
            if (string.Equals(prop.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                value = prop.Value;
                return true;
            }
        }

        value = default;
        return false;
    }

    private static void CollectText(JsonElement element, List<string> sink)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.String:
                string value = element.GetString() ?? string.Empty;
                if (LooksLikeUsefulText(value))
                    sink.Add(value);
                break;
            case JsonValueKind.Array:
                foreach (var item in element.EnumerateArray())
                    CollectText(item, sink);
                break;
            case JsonValueKind.Object:
                foreach (var prop in element.EnumerateObject())
                {
                    if (prop.Value.ValueKind == JsonValueKind.String)
                    {
                        string text = prop.Value.GetString() ?? string.Empty;
                        if (IsLikelyTextFieldName(prop.Name) || LooksLikeUsefulText(text))
                            sink.Add(text);
                    }
                    else
                    {
                        CollectText(prop.Value, sink);
                    }
                }
                break;
        }
    }

    private static bool LooksLikeUsefulText(string value)
    {
        string trimmed = value.Trim();
        if (trimmed.Length < 4)
            return false;

        bool hasLetter = trimmed.Any(char.IsLetter);
        bool hasWordBreak = trimmed.Contains(' ') || trimmed.Contains('\t') || trimmed.Contains('\n');
        return hasLetter && (hasWordBreak || trimmed.Length >= 20);
    }

    private static bool IsLikelyTextFieldName(string fieldName) =>
        fieldName.Equals("text", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("content", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("prompt", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("completion", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("response", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("instruction", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("input", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("output", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("question", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("answer", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("value", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("body", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("message", StringComparison.OrdinalIgnoreCase) ||
        fieldName.Equals("messages", StringComparison.OrdinalIgnoreCase);

    private static async Task PumpProcessStreamAsync(
        StreamReader reader,
        string label,
        StreamWriter buildLog,
        SemaphoreSlim writeLock,
        Action<string> onLine,
        CancellationToken ct)
    {
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            string? line = await reader.ReadLineAsync(ct);
            if (line == null)
                break;

            onLine(line);
            await writeLock.WaitAsync(ct);
            try
            {
                await buildLog.WriteLineAsync($"[llama-imatrix {label}] {line}");
                await buildLog.FlushAsync();
            }
            finally
            {
                writeLock.Release();
            }
        }
    }

    private static string FormatBytes(long sizeBytes)
    {
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        double size = sizeBytes;
        int unit = 0;
        while (size >= 1024 && unit < units.Length - 1)
        {
            size /= 1024;
            unit++;
        }

        return $"{size:0.0} {units[unit]}";
    }

    private sealed record LocalDatasetExportSummary(int TotalRows, int StructuredRows, int ExtractedTextBlocks, int RowsMissingSplitProperty);

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