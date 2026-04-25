using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class SelectionDiagnosticsLogService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public async Task WriteAsync(
        IReadOnlyCollection<BenchmarkSnapshotRecord> benchmarkOverview,
        IReadOnlyCollection<CandidateValidationResult> validationFailures,
        CancellationToken ct = default)
    {
        string directory = ResolveGgufDirectory();
        Directory.CreateDirectory(directory);

        string overviewPath = Path.Combine(directory, "magicquant-benchmark-overview.json");
        string missesPath = Path.Combine(directory, "magicquant-selection-validation-misses.json");

        var overview = benchmarkOverview
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .Select(ToSnapshotLog)
            .ToList();

        var misses = validationFailures
            .Where(x => !x.Accepted)
            .Select(ToFailureLog)
            .ToList();

        await File.WriteAllTextAsync(overviewPath, JsonSerializer.Serialize(overview, JsonOptions), ct);
        await File.WriteAllTextAsync(missesPath, JsonSerializer.Serialize(misses, JsonOptions), ct);

        AnsiConsole.MarkupLine($"[green]Benchmark overview log:[/] {Markup.Escape(overviewPath)}");
        AnsiConsole.MarkupLine($"[green]Prediction miss log:[/] {Markup.Escape(missesPath)}");
    }

    private static object ToSnapshotLog(BenchmarkSnapshotRecord snap)
    {
        return new
        {
            key = TensorConfigIdentity.ToKey(snap.Config),
            displayName = snap.DisplayName,
            provider = snap.ProviderName,
            baselineFamily = snap.BaselineFamily,
            isHybrid = snap.IsHybrid,
            isExternalPureBaseline = snap.IsExternalPureBaseline,
            sizeBytes = snap.SizeBytes,
            sizeGiB = ToGb(snap.SizeBytes),
            kld = snap.Kld,
            ppl = snap.Ppl,
            outputModelPath = snap.OutputModelPath,
            externalRepositoryUrl = snap.ExternalRepositoryUrl
        };
    }

    private static object ToFailureLog(CandidateValidationResult failure)
    {
        var c = failure.Candidate;
        var snap = failure.Snapshot;

        double? actualLine = null;
        double? actualGainOverLine = null;
        long? sizeMissBytes = null;
        double? kldMiss = null;

        if (snap != null)
        {
            actualLine = InterpolateKldLine(snap.SizeBytes, c.HigherDamageAnchor, c.LowerDamageAnchor);
            actualGainOverLine = actualLine.Value - snap.Kld;

            if (snap.SizeBytes < c.WindowMinSizeBytes)
                sizeMissBytes = (long)c.WindowMinSizeBytes - (long)snap.SizeBytes;
            else if (snap.SizeBytes > c.WindowMaxSizeBytes)
                sizeMissBytes = (long)snap.SizeBytes - (long)c.WindowMaxSizeBytes;
            else
                sizeMissBytes = 0;

            kldMiss = snap.Kld - actualLine.Value;
        }

        return new
        {
            reason = c.Reason.ToString(),
            attemptOrder = c.AttemptOrder,
            windowLabel = c.WindowLabel,
            candidateKey = TensorConfigIdentity.ToKey(c.Prediction.Config),
            candidateInternalName = HybridBenchmarkRepository.BuildDisplayName(c.Prediction.Quant),
            predicted = new
            {
                sizeBytes = c.Prediction.PredictedSizeBytes,
                sizeGiB = ToGb(c.Prediction.PredictedSizeBytes),
                kld = c.Prediction.PredictedKld,
                lineKldAtPredictedSize = c.LinearExpectedKld,
                gainOverLine = c.PredictedGainOverLine
            },
            actual = snap == null
                ? null
                : new
                {
                    displayName = snap.DisplayName,
                    sizeBytes = snap.SizeBytes,
                    sizeGiB = ToGb(snap.SizeBytes),
                    kld = snap.Kld,
                    ppl = snap.Ppl,
                    lineKldAtActualSize = actualLine,
                    gainOverLine = actualGainOverLine,
                    sizeMissBytes,
                    kldMiss
                },
            anchors = new
            {
                higherDamageSmaller = ToAnchorLog(c.HigherDamageAnchor),
                lowerDamageLarger = ToAnchorLog(c.LowerDamageAnchor)
            },
            accepted = failure.Accepted,
            message = failure.Message
        };
    }

    private static object ToAnchorLog(BenchmarkSnapshotRecord anchor)
    {
        return new
        {
            key = TensorConfigIdentity.ToKey(anchor.Config),
            displayName = anchor.DisplayName,
            sizeBytes = anchor.SizeBytes,
            sizeGiB = ToGb(anchor.SizeBytes),
            kld = anchor.Kld,
            ppl = anchor.Ppl,
            bitRange = anchor.Quant.BaseQuant.BitRange,
            quantizeBase = anchor.Quant.BaseQuant.QuantizeBaseArgumentName
        };
    }

    private static double InterpolateKldLine(
        ulong candidateSize,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger)
    {
        ulong smallSize = higherDamageSmaller.SizeBytes;
        ulong largeSize = lowerDamageLarger.SizeBytes;

        if (largeSize <= smallSize)
            return Math.Min(higherDamageSmaller.Kld, lowerDamageLarger.Kld);

        double t = Math.Clamp((candidateSize - smallSize) / (double)(largeSize - smallSize), 0d, 1d);
        return higherDamageSmaller.Kld + ((lowerDamageLarger.Kld - higherDamageSmaller.Kld) * t);
    }

    private static string ResolveGgufDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Path.Combine(Cache.ModelMagicQuantDirectory!, "GGUF");

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Path.Combine(Cache.MagicQuantDirectory!, "GGUF");

        return Path.Combine(Directory.GetCurrentDirectory(), "GGUF");
    }

    private static string ToGb(ulong bytes) => (bytes / 1024d / 1024d / 1024d).ToString("0.00");
}
