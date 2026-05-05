using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
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
            isExternalRebuiltBaseline = snap.IsExternalRebuiltBaseline,
            isMaterializedTensorMapped = snap.IsMaterializedTensorMapped,
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
        bool? actualInsideSizeWindow = null;
        bool? actualBeatLine = null;

        if (snap != null)
        {
            actualLine = InterpolateKldLine(snap.SizeBytes, c.HigherDamageAnchor, c.LowerDamageAnchor);
            actualGainOverLine = actualLine.Value - snap.Kld;
            actualInsideSizeWindow = snap.SizeBytes >= c.WindowMinSizeBytes && snap.SizeBytes <= c.WindowMaxSizeBytes;
            actualBeatLine = snap.Kld + Config.SelectionMinimumKldImprovementEpsilon < actualLine.Value;

            if (snap.SizeBytes < c.WindowMinSizeBytes)
                sizeMissBytes = (long)c.WindowMinSizeBytes - (long)snap.SizeBytes;
            else if (snap.SizeBytes > c.WindowMaxSizeBytes)
                sizeMissBytes = (long)snap.SizeBytes - (long)c.WindowMaxSizeBytes;
            else
                sizeMissBytes = 0;

            kldMiss = snap.Kld + Config.SelectionMinimumKldImprovementEpsilon - actualLine.Value;
        }

        return new
        {
            reason = c.Reason.ToString(),
            attemptOrder = c.AttemptOrder,
            attemptLimit = c.CandidateAttemptLimit,
            windowLabel = c.WindowLabel,
            phaseWindowIndex = c.PhaseWindowIndex,
            phaseWindowCount = c.PhaseWindowCount,
            candidateKey = TensorConfigIdentity.ToKey(c.Prediction.Config),
            candidateInternalName = HybridBenchmarkRepository.BuildDisplayName(c.Prediction.Quant),
            bitSpace = DescribeBitSpace(c.Prediction.Config),
            overrideSummary = DescribeOverrides(c.Prediction.Config),
            baseQuant = c.Prediction.Quant.BaseQuant.Names[0],
            baseBitRange = c.Prediction.Quant.BaseQuant.BitRange,
            failureCode = failure.FailureCode,
            predicted = new
            {
                sizeBytes = c.Prediction.PredictedSizeBytes,
                sizeGiB = ToGb(c.Prediction.PredictedSizeBytes),
                kld = c.Prediction.PredictedKld,
                lineKldAtPredictedSize = c.LinearExpectedKld,
                gainOverLine = c.PredictedGainOverLine,
                confidence = c.Prediction.PredictionConfidence,
                rank = c.Prediction.PredictedRank
            },
            selectionContext = new
            {
                candidatePoolSize = c.CandidatePoolSize,
                windowCandidateCount = c.WindowCandidateCount,
                lineBeatingCandidateCount = c.LineBeatingCandidateCount,
                fetchedCandidateCount = c.FetchedCandidateCount,
                candidatesAfterBrutalityCount = c.CandidatesAfterBrutalityCount,
                candidateAttemptLimit = c.CandidateAttemptLimit,
                rawSelectionRank = c.RawSelectionRank,
                diversityMode = c.DiversityMode,
                candidateTheoryFamilyKey = c.CandidateTheoryFamilyKey,
                candidateTheoryFamilyDisplay = c.CandidateTheoryFamilyDisplay,
                candidateTheoryFamilyRank = c.CandidateTheoryFamilyRank,
                candidateTheoryFamilyMemberRank = c.CandidateTheoryFamilyMemberRank,
                notes = c.CandidateSelectionNotes
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
                    insideSizeWindow = actualInsideSizeWindow,
                    beatLine = actualBeatLine,
                    sizeMissBytes,
                    kldMiss,
                    positiveKldShortfall = kldMiss.HasValue ? (double?)Math.Max(0d, kldMiss.Value) : null
                },
            anchors = new
            {
                higherDamageSmaller = ToAnchorLog(c.HigherDamageAnchor),
                lowerDamageLarger = ToAnchorLog(c.LowerDamageAnchor)
            },
            acceptancePolicy = new
            {
                minimumKldImprovementEpsilon = Config.SelectionMinimumKldImprovementEpsilon,
                windowMinSizeBytes = c.WindowMinSizeBytes,
                windowMaxSizeBytes = c.WindowMaxSizeBytes,
                mustBeatLineByEpsilon = true
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
            provider = anchor.ProviderName,
            baselineFamily = anchor.BaselineFamily,
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


    private static string DescribeBitSpace(TensorConfig config)
    {
        var baseQuant = BaselineQuants.FromId(config.BaseQuant);
        var overrides = DescribeOverrides(config);
        return string.IsNullOrWhiteSpace(overrides)
            ? $"base={baseQuant.Names[0]}({baseQuant.BitRange}b); overrides=inherit-all"
            : $"base={baseQuant.Names[0]}({baseQuant.BitRange}b); overrides={overrides}";
    }

    private static string DescribeOverrides(TensorConfig config)
    {
        var parts = new List<string>();
        AddOverride(parts, "E", config.Embeddings);
        AddOverride(parts, "H", config.LmHead);
        AddOverride(parts, "Q", config.AttnQ);
        AddOverride(parts, "K", config.AttnKV);
        AddOverride(parts, "O", config.AttnOutput);
        AddOverride(parts, "U", config.FfnUpGate);
        AddOverride(parts, "D", config.FfnDown);
        AddOverride(parts, "X", config.MoeExperts);
        AddOverride(parts, "R", config.MoeRouter);
        return string.Join(", ", parts);
    }

    private static void AddOverride(List<string> parts, string groupToken, byte storedSlot)
    {
        if (BaselineQuants.IsNullTensorConfigGroupSlot(storedSlot))
            return;

        var baseline = BaselineQuants.DecodeTensorConfigGroupSlotToBaseline(storedSlot);
        parts.Add($"{groupToken}:{baseline.Names[0]}({baseline.BitRange}b)");
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