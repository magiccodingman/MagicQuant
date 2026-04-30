using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
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
            Notes = "Exact GGUF tensor quantization map for repository clone/reproducibility mode. This file is not a proof that another cloned model went through the full MagicQuant evolution pipeline."
        };

        double? referencePpl = ResolveReferencePpl(pplReference, exportedArtifacts.Select(x => x.Snapshot));

        foreach (var artifact in exportedArtifacts
                     .Where(x => !x.IsExternalReference)
                     .Where(x => !string.IsNullOrWhiteSpace(x.FullPath))
                     .OrderBy(x => x.Snapshot.Kld)
                     .ThenBy(x => x.Snapshot.SizeBytes))
        {
            ct.ThrowIfCancellationRequested();

            string fullPath = artifact.FullPath!;
            if (!File.Exists(fullPath))
            {
                AnsiConsole.MarkupLine($"[yellow]Skipping clone config for missing artifact:[/] {Markup.Escape(fullPath)}");
                continue;
            }

            var tensorTypes = await _quantizationService.ReadExactTensorTypesAsync(fullPath, ct);

            manifest.Artifacts.Add(new MagicQuantCloneArtifact
            {
                FileName = artifact.FileName ?? Path.GetFileName(fullPath),
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
                TensorTypes = tensorTypes.ToDictionary(x => x.Key, x => x.Value, StringComparer.Ordinal)
            });
        }

        string path = Path.Combine(outputDirectory, FileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(manifest, JsonOptions), ct);
        AnsiConsole.MarkupLine($"[green]Clone configuration JSON generated:[/] {Markup.Escape(path)}");
        return path;
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
}
