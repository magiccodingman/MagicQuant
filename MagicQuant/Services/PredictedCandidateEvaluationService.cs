using MagicQuant.Models;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class PredictedTradeComparisonPolicy
{
    public int Compare(PredictedCandidateEvaluation left, PredictedCandidateEvaluation right)
    {
        if (!IsKldClose(left, right))
            return left.PredictedKldCost.CompareTo(right.PredictedKldCost);

        double sizeDeltaPercent = PercentDifference(left.PredictedSizeBytes, right.PredictedSizeBytes);
        if (sizeDeltaPercent >= Config.SurvivalMeaningfulSizeBiasPercent && left.PredictedSizeBytes != right.PredictedSizeBytes)
            return left.PredictedSizeBytes.CompareTo(right.PredictedSizeBytes);

        double pplDelta = Math.Abs(left.PredictedPplCost - right.PredictedPplCost);
        if (pplDelta >= Config.SurvivalPplLargeDifferencePercent)
            return left.PredictedPplCost.CompareTo(right.PredictedPplCost);

        return left.CompositeScore.CompareTo(right.CompositeScore);
    }

    public bool Dominates(PredictedCandidateEvaluation better, PredictedCandidateEvaluation worse)
    {
        bool sizeOk = better.PredictedSizeBytes <= worse.PredictedSizeBytes;
        bool kldBetter = better.PredictedKldCost <= worse.PredictedKldCost;
        bool pplNotWorse = better.PredictedPplCost <= worse.PredictedPplCost + 1e-9;

        bool strict = better.PredictedSizeBytes < worse.PredictedSizeBytes ||
                      better.PredictedKldCost < worse.PredictedKldCost ||
                      better.PredictedPplCost < worse.PredictedPplCost;

        return sizeOk && kldBetter && pplNotWorse && strict;
    }

    private static bool IsKldClose(PredictedCandidateEvaluation left, PredictedCandidateEvaluation right)
    {
        double diff = Math.Abs(left.PredictedKldCost - right.PredictedKldCost);
        double absolute = Config.SurvivalKldCloseCallAbsoluteEpsilon;
        double relative = Math.Min(left.PredictedKldCost, right.PredictedKldCost) * Config.SurvivalKldCloseCallRelativeFraction;
        return diff <= Math.Max(absolute, relative);
    }

    private static double PercentDifference(ulong left, ulong right)
    {
        if (left == 0 || right == 0)
            return 0;

        double min = Math.Min(left, right);
        double max = Math.Max(left, right);
        return ((max - min) / min) * 100d;
    }
}

public sealed class PredictedCandidateEvaluationService
{
    private readonly HybridBenchmarkRepository _repository;
    private readonly EffectiveCandidateStateResolverService _effectiveResolver;

    public PredictedCandidateEvaluationService(
        HybridBenchmarkRepository repository,
        EffectiveCandidateStateResolverService effectiveResolver)
    {
        _repository = repository;
        _effectiveResolver = effectiveResolver;
    }

    public async Task<List<PredictedCandidateEvaluation>> EvaluateAsync(
        IReadOnlyCollection<TensorConfig> configs,
        CancellationToken ct = default)
    {
        var result = new List<PredictedCandidateEvaluation>(configs.Count);

        var pureSnapshots = (await _repository.LoadPureBaselineSnapshotsAsync(ct))
            .GroupBy(x => x.BaselineFamily, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.OrderBy(x => x.SizeBytes).First(), StringComparer.Ordinal);

        var pureQ8 = pureSnapshots.TryGetValue(BaselineQuants.Q8_0.Names[0], out var q8Snap) ? q8Snap : null;
        if (pureQ8 == null)
            throw new InvalidOperationException("Prediction requires a learned pure Q8_0 benchmark anchor.");

        var isolationCache = new Dictionary<string, BenchmarkSnapshotRecord?>(StringComparer.Ordinal);

        foreach (var config in configs)
        {
            var quant = (HybridQuant)config;
            var effective = await _effectiveResolver.ResolveAsync(config, ct);
            var notes = new List<string>(effective.Warnings);

            BenchmarkSnapshotRecord baselineAnchor = pureSnapshots.TryGetValue(quant.BaseQuant.Names[0], out var baselineSnap)
                ? baselineSnap
                : pureQ8;

            ulong predictedSize = baselineAnchor.SizeBytes;
            double predictedKld = baselineAnchor.Kld;
            double predictedPpl = baselineAnchor.Ppl;

            foreach (var tensor in quant.Tensors)
            {
                tensor.ValidateOrThrow();

                string isolationKey = $"{tensor.TGroup.UniqueId}:{tensor.OverrideMode}:{tensor.CandidateBaseline?.CanonicalKey}:{tensor.ExactTensorScheme?.Names[0]}";
                if (!isolationCache.TryGetValue(isolationKey, out var isolation))
                {
                    var isolationQuant = HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0);

                    if (tensor.OverrideMode == HybridTensorOverrideMode.ExactTensorScheme)
                        isolationQuant.SetExactOverride(tensor.TGroup, tensor.ExactTensorScheme!);
                    else
                        isolationQuant.SetLearnedCandidateOverride(tensor.TGroup, tensor.CandidateBaseline!);

                    isolation = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)isolationQuant, ct);
                    isolationCache[isolationKey] = isolation;
                }

                if (isolation == null)
                {
                    notes.Add($"Isolation benchmark missing for group '{tensor.TGroup.Name}'. Applied conservative penalty.");
                    predictedKld += 0.005d;
                    predictedPpl += 0.25d;
                    continue;
                }

                long sizeDelta = (long)isolation.SizeBytes - (long)pureQ8.SizeBytes;
                if (sizeDelta >= 0)
                    predictedSize += (ulong)sizeDelta;
                else
                    predictedSize = predictedSize > (ulong)(-sizeDelta) ? predictedSize - (ulong)(-sizeDelta) : 0;

                predictedKld += Math.Max(0d, isolation.Kld - pureQ8.Kld);
                predictedPpl += Math.Max(0d, isolation.Ppl - pureQ8.Ppl);
            }

            if (Config.ManualMaxPredictedSizeBytes > 0 && predictedSize > Config.ManualMaxPredictedSizeBytes)
                notes.Add($"Predicted size {predictedSize:N0} bytes exceeds configured manual ceiling {Config.ManualMaxPredictedSizeBytes:N0} bytes.");

            double sizeGb = predictedSize / 1024d / 1024d / 1024d;
            double composite = (predictedKld * 10000d) +
                               (predictedPpl * Config.SurvivalTradeScorePplWeight) +
                               (sizeGb / Math.Max(0.01d, Config.SurvivalTradeScoreSizeBiasWeight));

            result.Add(new PredictedCandidateEvaluation
            {
                Config = config,
                Quant = quant,
                PredictedSizeBytes = predictedSize,
                PredictedKldCost = predictedKld,
                PredictedPplCost = predictedPpl,
                CompositeScore = composite,
                EffectiveStateKey = effective.EffectiveStateKey,
                HasUnknownMappings = effective.HasUnknownMappings,
                BaseBitRange = quant.BaseQuant.BitRange,
                IsPureBaseline = TensorConfigIdentity.IsPureBaseline(config),
                Notes = notes
            });
        }

        AnsiConsole.MarkupLine($"[grey]Prediction evaluation completed for[/] [cyan]{result.Count:N0}[/] [grey]remaining combinations.[/]");
        return result;
    }
}
