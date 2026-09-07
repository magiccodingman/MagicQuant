using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class FinalReleaseMetadataService
{
    public const string FinalSurvivorsFileName = MagicQuantManifestPathService.FinalSurvivorsFileName;
    public const string ReplacementsFileName = MagicQuantManifestPathService.ReplacementsFileName;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly FinalArtifactNamingService _namingService = new();

    public async Task GenerateAsync(
        string outputDirectory,
        IReadOnlyCollection<ExportedArtifactRecord> exportedArtifacts,
        IReadOnlyCollection<BaselineEliminationRecord> eliminations,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        BenchmarkSnapshotRecord? pplReference = null,
        CancellationToken ct = default)
    {
        string manifestDirectory = MagicQuantManifestPathService.EnsureManifestDirectory(outputDirectory);

        double? referencePpl = ResolveReferencePpl(pplReference, pureBaselineSnapshots, exportedArtifacts.Select(x => x.Snapshot).ToList());
        var namingContext = _namingService.CreateContext(pureBaselineSnapshots);
        var exportedByKey = exportedArtifacts
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Snapshot.Config), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.First(), StringComparer.Ordinal);

        var replacementMap = BuildReplacementMap(eliminations);

        string finalPath = Path.Combine(manifestDirectory, FinalSurvivorsFileName);
        var survivors = exportedArtifacts
            .OrderBy(x => x.Snapshot.Kld)
            .ThenBy(x => x.Snapshot.SizeBytes)
            .Select(x => ToSurvivorJson(x, referencePpl, replacementMap, namingContext))
            .ToList();
        await File.WriteAllTextAsync(finalPath, JsonSerializer.Serialize(survivors, JsonOptions), ct);

        string replacementsPath = Path.Combine(manifestDirectory, ReplacementsFileName);
        var replacements = eliminations
            .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{x.Reason}")
            .OrderBy(x => x.Eliminated.Kld)
            .ThenBy(x => x.Eliminated.SizeBytes)
            .Select(x => ToReplacementJson(x, exportedByKey, namingContext, referencePpl))
            .ToList();
        await File.WriteAllTextAsync(replacementsPath, JsonSerializer.Serialize(replacements, JsonOptions), ct);

        AnsiConsole.MarkupLine($"[green]Final survivor metrics JSON generated:[/] {Markup.Escape(finalPath)}");
        AnsiConsole.MarkupLine($"[green]Replacement detail JSON generated:[/] {Markup.Escape(replacementsPath)}");
    }

    private object ToSurvivorJson(
        ExportedArtifactRecord artifact,
        double? referencePpl,
        IReadOnlyDictionary<string, List<BaselineEliminationRecord>> replacementMap,
        FinalArtifactNamingContext namingContext)
    {
        string key = TensorConfigIdentity.ToKey(artifact.Snapshot.Config);
        var replacements = ResolveTransitiveReplacements(key, replacementMap)
            .Select(x => new
            {
                key = TensorConfigIdentity.ToKey(x.Eliminated.Config),
                shortName = ToSnapshotShortName(x.Eliminated, null, namingContext),
                internalDisplayName = x.Eliminated.DisplayName,
                kld = x.Eliminated.Kld,
                ppl = x.Eliminated.Ppl,
                pplDeltaPercent = CalculatePplDeltaPercent(x.Eliminated.Ppl, referencePpl),
                sizeBytes = x.Eliminated.SizeBytes,
                sizeGB = ToGBNumber(x.Eliminated.SizeBytes),
                sizeGiB = ToGiBNumber(x.Eliminated.SizeBytes),
                reasonCode = FinalArtifactNamingService.ReasonCode(x.Reason),
                reason = x.Reason
            })
            .ToList();

        return new
        {
            key,
            fileName = artifact.IsExternalReference
                ? EnsureGgufExtension(artifact.DisplayName)
                : artifact.FileName,
            displayName = artifact.DisplayName,
            shortName = _namingService.ToPublicArtifactShortName(
                artifact.DisplayName,
                artifact.FileName,
                artifact.ProviderName,
                artifact.BaselineFamily,
                artifact.Snapshot,
                namingContext),
            provider = artifact.ProviderName,
            quantFamily = artifact.BaselineFamily,
            isHybrid = artifact.Snapshot.IsHybrid,
            isExternalPureBaseline = artifact.Snapshot.IsExternalPureBaseline,
            isExternalRebuiltBaseline = artifact.Snapshot.IsExternalRebuiltBaseline,
            isMaterializedTensorMapped = artifact.Snapshot.IsMaterializedTensorMapped,
            isExternalReference = artifact.IsExternalReference,
            downloadTarget = artifact.DownloadTarget,
            kld = artifact.Snapshot.Kld,
            ppl = artifact.Snapshot.Ppl,
            pplDeltaPercent = CalculatePplDeltaPercent(artifact.Snapshot.Ppl, referencePpl),
            sizeBytes = artifact.Snapshot.SizeBytes,
            sizeGB = ToGBNumber(artifact.Snapshot.SizeBytes),
            sizeGiB = ToGiBNumber(artifact.Snapshot.SizeBytes),
            expectedSizeBytes = artifact.ExpectedSizeBytes,
            actualSizeBytes = artifact.ActualSizeBytes,
            usedImatrix = Cache.UseImatrix && Cache.IsImatrixAvailable,
            replacedArtifacts = replacements
        };
    }

    private object ToReplacementJson(
        BaselineEliminationRecord row,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext,
        double? referencePpl)
    {
        string reasonCode = FinalArtifactNamingService.ReasonCode(row.Reason);
        double kldDelta = row.Eliminated.Kld - row.Eliminator.Kld;
        long sizeDeltaBytes = (long)row.Eliminated.SizeBytes - (long)row.Eliminator.SizeBytes;
        double? pplDeltaPercentRemoved = CalculatePplDeltaPercent(row.Eliminated.Ppl, referencePpl);
        double? pplDeltaPercentWinner = CalculatePplDeltaPercent(row.Eliminator.Ppl, referencePpl);
        double? pplDeltaPercentImprovement = pplDeltaPercentRemoved.HasValue && pplDeltaPercentWinner.HasValue
            ? pplDeltaPercentRemoved.Value - pplDeltaPercentWinner.Value
            : null;

        return new
        {
            reasonCode,
            reasonDescription = FinalArtifactNamingService.ReasonDescription(reasonCode),
            rawReason = row.Reason,
            removed = ToReplacementSideJson(row.Eliminated, exportedByKey, namingContext, referencePpl),
            winner = ToReplacementSideJson(row.Eliminator, exportedByKey, namingContext, referencePpl),
            deltas = new
            {
                kld = kldDelta,
                sizeBytes = sizeDeltaBytes,
                sizeGB = ToGBNumber(sizeDeltaBytes),
                sizeGiB = ToGiBNumber(sizeDeltaBytes),
                removedPplDeltaPercent = pplDeltaPercentRemoved,
                winnerPplDeltaPercent = pplDeltaPercentWinner,
                pplDeltaPercentImprovement = pplDeltaPercentImprovement
            }
        };
    }

    private object ToReplacementSideJson(
        BenchmarkSnapshotRecord snapshot,
        IReadOnlyDictionary<string, ExportedArtifactRecord> exportedByKey,
        FinalArtifactNamingContext namingContext,
        double? referencePpl)
    {
        string key = TensorConfigIdentity.ToKey(snapshot.Config);
        string displayName;
        string fileName;
        string shortName;
        string provider;
        string quantFamily;

        if (exportedByKey.TryGetValue(key, out var artifact))
        {
            displayName = artifact.DisplayName;
            shortName = _namingService.ToPublicArtifactShortName(
                artifact.DisplayName,
                artifact.FileName,
                artifact.ProviderName,
                artifact.BaselineFamily,
                artifact.Snapshot,
                namingContext);
            fileName = artifact.IsExternalReference ? EnsureGgufExtension(artifact.DisplayName) : artifact.FileName ?? EnsureGgufExtension(artifact.DisplayName);
            provider = artifact.ProviderName;
            quantFamily = artifact.BaselineFamily;
        }
        else
        {
            displayName = _namingService.BuildDisplayLabel(snapshot, namingContext);
            provider = snapshot.IsHybrid ? "MagicQuant" : HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false);
            quantFamily = snapshot.BaselineFamily;
            shortName = _namingService.ToPublicArtifactShortName(
                displayName,
                null,
                provider,
                quantFamily,
                snapshot,
                namingContext);
            fileName = EnsureGgufExtension(displayName);
        }

        return new
        {
            key,
            fileName,
            displayName,
            shortName,
            provider,
            quantFamily,
            isHybrid = snapshot.IsHybrid,
            isExternalPureBaseline = snapshot.IsExternalPureBaseline,
            isExternalRebuiltBaseline = snapshot.IsExternalRebuiltBaseline,
            isMaterializedTensorMapped = snapshot.IsMaterializedTensorMapped,
            kld = snapshot.Kld,
            ppl = snapshot.Ppl,
            pplDeltaPercent = CalculatePplDeltaPercent(snapshot.Ppl, referencePpl),
            sizeBytes = snapshot.SizeBytes,
            sizeGB = ToGBNumber(snapshot.SizeBytes),
            sizeGiB = ToGiBNumber(snapshot.SizeBytes)
        };
    }

    public static IReadOnlyDictionary<string, List<BaselineEliminationRecord>> BuildReplacementMap(
        IReadOnlyCollection<BaselineEliminationRecord> eliminations)
    {
        return eliminations
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Eliminator.Config), StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.Ordinal);
    }

    public static IReadOnlyList<BaselineEliminationRecord> ResolveTransitiveReplacements(
        string finalWinnerKey,
        IReadOnlyDictionary<string, List<BaselineEliminationRecord>> replacementMap)
    {
        var output = new List<BaselineEliminationRecord>();
        var visited = new HashSet<string>(StringComparer.Ordinal);

        void Visit(string winnerKey)
        {
            if (!visited.Add(winnerKey))
                return;

            if (!replacementMap.TryGetValue(winnerKey, out var direct))
                return;

            foreach (var row in direct)
            {
                output.Add(row);
                Visit(TensorConfigIdentity.ToKey(row.Eliminated.Config));
            }
        }

        Visit(finalWinnerKey);

        return output
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Eliminated.Config), StringComparer.Ordinal)
            .ToList();
    }

    private static double? ResolveReferencePpl(
        BenchmarkSnapshotRecord? pplReference,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        IReadOnlyCollection<BenchmarkSnapshotRecord> snapshots)
    {
        if (pplReference is { Ppl: > 0d })
            return pplReference.Ppl;

        var bestPure = pureBaselineSnapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .ThenByDescending(x => x.SizeBytes)
            .FirstOrDefault();

        if (bestPure != null)
            return bestPure.Ppl;

        return snapshots
            .Where(x => x.Ppl > 0d)
            .OrderBy(x => x.Kld)
            .FirstOrDefault()
            ?.Ppl;
    }

    public static double? CalculatePplDeltaPercent(double ppl, double? referencePpl)
    {
        if (referencePpl is null or <= 0d || ppl <= 0d)
            return null;

        return ((ppl - referencePpl.Value) / referencePpl.Value) * 100d;
    }

    private static string EnsureGgufExtension(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        return value.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? value
            : value + ".gguf";
    }

    private string ToSnapshotShortName(
        BenchmarkSnapshotRecord snapshot,
        ExportedArtifactRecord? artifact,
        FinalArtifactNamingContext namingContext)
    {
        if (artifact != null)
        {
            return _namingService.ToPublicArtifactShortName(
                artifact.DisplayName,
                artifact.FileName,
                artifact.ProviderName,
                artifact.BaselineFamily,
                artifact.Snapshot,
                namingContext);
        }

        string provider = snapshot.IsHybrid ? "MagicQuant" : HybridBenchmarkRepository.ResolveProviderName(snapshot.Quant, exportNaming: false);
        string display = _namingService.BuildDisplayLabel(snapshot, namingContext);
        return _namingService.ToPublicArtifactShortName(display, null, provider, snapshot.BaselineFamily, snapshot, namingContext);
    }

    private static double ToGBNumber(ulong bytes) => bytes / 1000d / 1000d / 1000d;
    private static double ToGiBNumber(ulong bytes) => bytes / 1024d / 1024d / 1024d;
    private static double ToGBNumber(long bytes) => bytes / 1000d / 1000d / 1000d;
    private static double ToGiBNumber(long bytes) => bytes / 1024d / 1024d / 1024d;
}