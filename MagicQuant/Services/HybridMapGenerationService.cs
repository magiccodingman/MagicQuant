using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class HybridMapGenerationService
{
    public async Task<string> GenerateAsync(
        string outputDirectory,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);
        string path = Path.Combine(outputDirectory, "magicquant.hybrid-map.json");

        var entries = exportedArtifacts
            .Where(x => !x.IsExternalReference)
            .Where(x => x.Snapshot.IsHybrid)
            .Select(x => new HybridMapEntry
            {
                ExportedFileName = x.FileName ?? string.Empty,
                DisplayName = x.DisplayName,
                ProviderSource = x.ProviderName,
                BaselineFamily = x.BaselineFamily,
                OriginalReferenceBaseline = x.Snapshot.Quant.BaseQuant.Names[0],
                TensorGroups = x.Snapshot.Quant.Tensors.ToDictionary(
                    t => t.TGroup.Name,
                    t => t.OverrideMode == HybridTensorOverrideMode.ExactTensorScheme
                        ? t.ExactTensorScheme!.Names[0]
                        : t.CandidateBaseline!.Names[0],
                    StringComparer.Ordinal),
                EffectiveQuantStateKey = x.EffectiveState?.EffectiveStateKey ?? string.Empty,
                HasUnknownMappings = x.EffectiveState?.HasUnknownMappings ?? false,
                Warnings = x.EffectiveState?.Warnings.ToList() ?? new List<string>(),
                UsedImatrix = Cache.UseImatrix && Cache.IsImatrixAvailable,
                ExpectedSizeBytes = x.ExpectedSizeBytes,
                ActualSizeBytes = x.ActualSizeBytes,
                OriginalExternalSource = HybridBenchmarkRepository.BuildExternalRepositoryUrl(x.Snapshot.Quant.BaseQuant)
            })
            .ToList();

        var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json, ct);
        AnsiConsole.MarkupLine($"[green]Hybrid map JSON generated:[/] {Markup.Escape(path)}");
        return path;
    }
}
