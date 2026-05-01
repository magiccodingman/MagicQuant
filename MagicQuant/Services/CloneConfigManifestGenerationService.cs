using System.Diagnostics;
using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class CloneConfigManifestGenerationService
{
    public const string FileName = "magicquant.clone-configs.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly QuantizationService _quantizationService;
    private readonly HybridBenchmarkRepository _benchmarkRepository = new();
    private readonly FinalArtifactNamingService _namingService = new();

    public CloneConfigManifestGenerationService(QuantizationService quantizationService)
    {
        _quantizationService = quantizationService;
    }

    public async Task<string> GenerateAsync(
        string outputDirectory,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        BenchmarkSnapshotRecord? pplReference = null,
        string? sourceRepository = null,
        string? sourceJson = null,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);

        var manifest = new MagicQuantCloneManifest
        {
            SchemaVersion = 1,
            GeneratedUtc = DateTime.UtcNow,
            Generator = "MagicQuant",
            SourceRepository = sourceRepository,
            SourceJson = sourceJson,
            SourceModelId = Cache.CurrentModelId,
            SourceArchitectureFamily = Cache.CurrentArchitectureFamilyName,
            Notes = "Exact GGUF tensor quantization map for repository clone/reproducibility mode. This file is not a proof that another cloned model went through the full MagicQuant evolution pipeline. External reference finalists use persisted SQLite learned tensor truth when no local final GGUF was exported."
        };

        double? referencePpl = ResolveReferencePpl(pplReference, exportedArtifacts.Select(x => x.Snapshot));

        var orderedArtifacts = exportedArtifacts
            .OrderBy(x => x.Snapshot.Kld)
            .ThenBy(x => x.Snapshot.SizeBytes)
            .ThenBy(x => x.DisplayName, StringComparer.Ordinal)
            .ToList();

        WriteCloneLog($"Starting clone manifest generation for {orderedArtifacts.Count:N0} finalist artifacts.");

        int index = 0;
        foreach (var artifact in orderedArtifacts)
        {
            ct.ThrowIfCancellationRequested();
            index++;

            var artifactSw = Stopwatch.StartNew();
            WriteCloneLog(
                $"[{index:N0}/{orderedArtifacts.Count:N0}] Resolving tensor map for '{artifact.DisplayName}' " +
                $"provider='{artifact.ProviderName}' family='{artifact.BaselineFamily}' externalReference={artifact.IsExternalReference} externalPureBaseline={artifact.Snapshot.IsExternalPureBaseline} hybrid={artifact.Snapshot.IsHybrid}.");

            CloneTensorMapResolution resolution;
            try
            {
                resolution = await ResolveTensorTypesForCloneAsync(artifact, ct);
            }
            catch (Exception ex)
            {
                artifactSw.Stop();
                WriteCloneLog(
                    $"[{index:N0}/{orderedArtifacts.Count:N0}] FAILED resolving tensor map for '{artifact.DisplayName}' after {FormatDuration(artifactSw.Elapsed)}: {ex.GetType().Name}: {ex.Message}",
                    isError: true);
                throw;
            }

            artifactSw.Stop();
            WriteCloneLog(
                $"[{index:N0}/{orderedArtifacts.Count:N0}] Resolved '{artifact.DisplayName}' via {resolution.SourceDescription} in {FormatDuration(artifactSw.Elapsed)}; tensors={resolution.TensorTypes.Count:N0}.");

            manifest.Artifacts.Add(new MagicQuantCloneArtifact
            {
                FileName = ResolveManifestFileName(artifact),
                DisplayName = artifact.DisplayName,
                ShortName = _namingService.ToShortDisplayName(artifact.DisplayName),
                Provider = artifact.ProviderName,
                QuantFamily = artifact.BaselineFamily,
                BaseQuant = ResolveBaseQuantName(artifact),
                IsHybrid = artifact.Snapshot.IsHybrid,
                UsedImatrix = Cache.UseImatrix && Cache.IsImatrixAvailable,
                SourceKld = artifact.Snapshot.Kld,
                SourcePpl = artifact.Snapshot.Ppl,
                SourcePplDeltaPercent = FinalReleaseMetadataService.CalculatePplDeltaPercent(artifact.Snapshot.Ppl, referencePpl),
                SourceSizeBytes = artifact.ActualSizeBytes ?? artifact.ExpectedSizeBytes,
                SourceSizeGB = ToGBNumber(artifact.ActualSizeBytes ?? artifact.ExpectedSizeBytes),
                SourceSizeGiB = ToGiBNumber(artifact.ActualSizeBytes ?? artifact.ExpectedSizeBytes),
                TensorTypes = resolution.TensorTypes.ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal)
            });
        }

        string path = Path.Combine(outputDirectory, FileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(manifest, JsonOptions), ct);
        WriteCloneLog($"Clone configuration JSON generated: {path} | artifacts={manifest.Artifacts.Count:N0}");
        return path;
    }

    private async Task<CloneTensorMapResolution> ResolveTensorTypesForCloneAsync(
        ExportedArtifactRecord artifact,
        CancellationToken ct)
    {
        // Critical: an external reference means the final output directory intentionally does not contain
        // a local GGUF for this finalist. Do not re-download the upstream GGUF here. The exact learned
        // tensor truth was already captured in SQLite during learning/benchmarking, and that is the correct
        // source for clone reproducibility metadata.
        if (artifact.IsExternalReference)
        {
            var learned = await LoadExternalPureBaselineTensorTruthAsync(artifact, ct);
            return new CloneTensorMapResolution(learned, "SQLite learned tensor truth for external reference");
        }

        if (!string.IsNullOrWhiteSpace(artifact.FullPath) && File.Exists(artifact.FullPath))
        {
            var tensorTypes = await _quantizationService.ReadExactTensorTypesAsync(artifact.FullPath, ct);
            return new CloneTensorMapResolution(
                tensorTypes.ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal),
                $"local final GGUF '{artifact.FullPath}'");
        }

        if (!string.IsNullOrWhiteSpace(artifact.Snapshot.OutputModelPath) && File.Exists(artifact.Snapshot.OutputModelPath))
        {
            var tensorTypes = await _quantizationService.ReadExactTensorTypesAsync(artifact.Snapshot.OutputModelPath, ct);
            return new CloneTensorMapResolution(
                tensorTypes.ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal),
                $"existing benchmark GGUF '{artifact.Snapshot.OutputModelPath}'");
        }

        if (artifact.Snapshot.IsExternalPureBaseline && !artifact.Snapshot.IsHybrid)
        {
            var learned = await LoadExternalPureBaselineTensorTruthAsync(artifact, ct);
            return new CloneTensorMapResolution(learned, "SQLite learned tensor truth fallback for external pure baseline");
        }

        throw new InvalidOperationException(
            $"Cannot generate clone tensor map for '{artifact.DisplayName}'. No local final GGUF exists at '{artifact.FullPath ?? "<null>"}', " +
            $"no existing benchmark GGUF exists at '{artifact.Snapshot.OutputModelPath ?? "<null>"}', and the artifact is not an external pure baseline with persisted learned tensor truth.");
    }

    private async Task<Dictionary<string, string>> LoadExternalPureBaselineTensorTruthAsync(
        ExportedArtifactRecord artifact,
        CancellationToken ct)
    {
        var baseline = artifact.Snapshot.Quant.BaseQuant;

        if (!baseline.IsExternalRepositoryBaseline && !artifact.Snapshot.IsExternalPureBaseline)
        {
            throw new InvalidOperationException(
                $"Artifact '{artifact.DisplayName}' was marked as an external reference, but its base quant '{baseline.Names[0]}' is not an external repository baseline and the snapshot is not marked as an external pure baseline.");
        }

        if (string.IsNullOrWhiteSpace(baseline.CanonicalKey))
        {
            throw new InvalidOperationException(
                $"External baseline artifact '{artifact.DisplayName}' has no canonical baseline key, so SQLite learned tensor truth cannot be loaded.");
        }

        WriteCloneLog(
            $"Loading SQLite learned tensor truth for external baseline '{baseline.Names[0]}' canonicalKey='{baseline.CanonicalKey}' preferredScheme='{baseline.DefaultTensorScheme?.Names[0] ?? "<none>"}'.");

        var strict = await _benchmarkRepository.LoadLearnedTensorMappingsAsync(
            canonicalBaselineKey: baseline.CanonicalKey,
            groupId: null,
            preferredSourceScheme: baseline.DefaultTensorScheme,
            allowDominantFallback: false,
            ct: ct);

        if (strict.Count > 0)
            return strict;

        WriteCloneLog(
            $"Strict SQLite learned tensor truth lookup returned 0 rows for '{baseline.Names[0]}'. Trying dominant-scheme fallback for legacy/mixed rows.",
            isWarning: true);

        var fallback = await _benchmarkRepository.LoadLearnedTensorMappingsAsync(
            canonicalBaselineKey: baseline.CanonicalKey,
            groupId: null,
            preferredSourceScheme: baseline.DefaultTensorScheme,
            allowDominantFallback: true,
            ct: ct);

        if (fallback.Count > 0)
            return fallback;

        throw new InvalidOperationException(
            $"No SQLite learned tensor truth exists for external baseline '{baseline.Names[0]}' canonicalKey='{baseline.CanonicalKey}'. " +
            "Final clone manifest generation will not re-download external GGUFs. Re-run the external baseline learning/benchmark stage for this model/context so the tensor truth is present in SQLite.");
    }

    private static double ToGBNumber(ulong bytes) => bytes / 1000d / 1000d / 1000d;
    private static double ToGiBNumber(ulong bytes) => bytes / 1024d / 1024d / 1024d;

    private static string ResolveBaseQuantName(ExportedArtifactRecord artifact)
    {
        var quant = artifact.Snapshot.Quant;
        if (!string.IsNullOrWhiteSpace(quant.BaseQuant.QuantizeBaseArgumentName))
            return quant.BaseQuant.QuantizeBaseArgumentName;

        if (!quant.BaseQuant.Names.IsDefaultOrEmpty)
            return quant.BaseQuant.Names[0];

        return "Q8_0";
    }

    private static string ResolveManifestFileName(ExportedArtifactRecord artifact)
    {
        if (!string.IsNullOrWhiteSpace(artifact.FileName))
            return artifact.FileName;

        if (!string.IsNullOrWhiteSpace(artifact.FullPath))
            return Path.GetFileName(artifact.FullPath);

        if (!string.IsNullOrWhiteSpace(artifact.Snapshot.OutputModelPath))
            return Path.GetFileName(artifact.Snapshot.OutputModelPath);

        string? targetName = TryGetFileNameFromDownloadTarget(artifact.DownloadTarget);
        if (!string.IsNullOrWhiteSpace(targetName))
            return targetName;

        return ToSafeGgufFileName(artifact.DisplayName);
    }

    private static string? TryGetFileNameFromDownloadTarget(string? downloadTarget)
    {
        if (string.IsNullOrWhiteSpace(downloadTarget))
            return null;

        string value = downloadTarget.Trim();
        int queryIndex = value.IndexOf('?', StringComparison.Ordinal);
        if (queryIndex >= 0)
            value = value[..queryIndex];

        value = value.TrimEnd('/');
        string fileName = Path.GetFileName(value.Replace('\\', '/'));
        return string.IsNullOrWhiteSpace(fileName) ? null : fileName;
    }

    private static string ToSafeGgufFileName(string value)
    {
        string safe = new string((value ?? string.Empty)
            .Select(ch => char.IsLetterOrDigit(ch) || ch is '.' or '_' or '-' ? ch : '_')
            .ToArray())
            .Trim('_', '.', '-');

        if (string.IsNullOrWhiteSpace(safe))
            safe = "external-baseline";

        return safe.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? safe
            : $"{safe}.gguf";
    }

    private static double? ResolveReferencePpl(
        BenchmarkSnapshotRecord? pplReference,
        IEnumerable<BenchmarkSnapshotRecord> snapshots)
    {
        if (pplReference is { Ppl: > 0d })
            return pplReference.Ppl;

        return snapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .FirstOrDefault()
            ?.Ppl;
    }

    private static string FormatDuration(TimeSpan value) => value.ToString(@"hh\:mm\:ss");

    private static void WriteCloneLog(string message, bool isWarning = false, bool isError = false)
    {
        string color = isError ? "red" : isWarning ? "yellow" : "grey";
        string line = $"[{DateTime.Now:HH:mm:ss}] Clone manifest: {message}";
        AnsiConsole.MarkupLine($"[{color}]{Markup.Escape(line)}[/]");
    }

    private sealed class CloneTensorMapResolution
    {
        public CloneTensorMapResolution(Dictionary<string, string> tensorTypes, string sourceDescription)
        {
            TensorTypes = tensorTypes;
            SourceDescription = sourceDescription;
        }

        public Dictionary<string, string> TensorTypes { get; }
        public string SourceDescription { get; }
    }
}
