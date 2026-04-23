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

        var pureSnapshots = await _repository.LoadPureBaselineSnapshotsAsync(ct);
        var pureByBaselineId = pureSnapshots
            .GroupBy(x => x.Quant.BaseQuant.UniqueId)
            .ToDictionary(g => g.Key, g => g.OrderBy(x => x.SizeBytes).ThenBy(x => x.Kld).First());

        if (!pureByBaselineId.TryGetValue(BaselineQuants.Q8_0.UniqueId, out var pureQ8))
            throw new InvalidOperationException("Prediction requires a learned pure Q8_0 benchmark anchor.");

        var isolationCache = new Dictionary<string, BenchmarkSnapshotRecord?>(StringComparer.Ordinal);

        foreach (var config in configs)
        {
            var quant = (HybridQuant)config;
            var effective = await _effectiveResolver.ResolveAsync(config, ct);
            var notes = new List<string>(effective.Warnings);

            byte normalizedBaseId = NormalizeBaselineIdForIsolation(quant.BaseQuant.UniqueId);
            BenchmarkSnapshotRecord baselineAnchor = pureByBaselineId.TryGetValue(quant.BaseQuant.UniqueId, out var baselineSnap)
                ? baselineSnap
                : pureByBaselineId.TryGetValue(normalizedBaseId, out var normalizedSnap)
                    ? normalizedSnap
                    : pureQ8;

            ulong predictedSize = baselineAnchor.SizeBytes;
            double predictedKld = baselineAnchor.Kld;
            double predictedPpl = baselineAnchor.Ppl;

            foreach (var tensor in quant.Tensors)
            {
                tensor.ValidateOrThrow();

                BenchmarkSnapshotRecord? targetIsolation = await LoadIsolationSnapshotAsync(tensor, isolationCache, ct);
                if (targetIsolation == null)
                {
                    notes.Add($"Isolation benchmark missing for target override on group '{tensor.TGroup.Name}'. Applied conservative penalty.");
                    predictedKld += 0.005d;
                    predictedPpl += 0.25d;
                    continue;
                }

                BenchmarkSnapshotRecord? baseIsolation = await LoadBaseIsolationSnapshotAsync(quant.BaseQuant, tensor.TGroup, isolationCache, ct);
                if (baseIsolation == null)
                {
                    notes.Add($"Base-family isolation benchmark missing for '{quant.BaseQuant.Names[0]}' on group '{tensor.TGroup.Name}'. Using target isolation without relative improvement credit.");
                    baseIsolation = targetIsolation;
                }

                long sizeDelta = (long)targetIsolation.SizeBytes - (long)baseIsolation.SizeBytes;
                if (sizeDelta >= 0)
                    predictedSize += (ulong)sizeDelta;
                else
                    predictedSize = predictedSize > (ulong)(-sizeDelta) ? predictedSize - (ulong)(-sizeDelta) : 0;

                predictedKld += (targetIsolation.Kld - baseIsolation.Kld);
                predictedPpl += (targetIsolation.Ppl - baseIsolation.Ppl);

                ApplyProportionalTradeWeighting(baseIsolation, targetIsolation, ref predictedKld, ref predictedPpl);
            }

            if (predictedKld < 0d)
                predictedKld = 0d;

            if (Config.ManualMaxPredictedSizeBytes > 0 && predictedSize > Config.ManualMaxPredictedSizeBytes)
                notes.Add($"Predicted size {predictedSize:N0} bytes exceeds configured manual ceiling {Config.ManualMaxPredictedSizeBytes:N0} bytes.");

            double sizeGb = predictedSize / 1024d / 1024d / 1024d;
            double composite = (predictedKld * 10000d) +
                               (Math.Max(0d, predictedPpl) * Config.SurvivalTradeScorePplWeight) +
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

        PrintPredictionDiagnostics(result);
        return result;
    }

    private async Task<BenchmarkSnapshotRecord?> LoadBaseIsolationSnapshotAsync(
        BaselineQuants baseQuant,
        TensorGroup group,
        Dictionary<string, BenchmarkSnapshotRecord?> cache,
        CancellationToken ct)
    {
        byte normalizedId = NormalizeBaselineIdForIsolation(baseQuant.UniqueId);
        var normalizedBaseline = BaselineQuants.FromId(normalizedId);
        return await LoadLearnedCandidateIsolationAsync(group, normalizedBaseline, cache, ct);
    }

    private async Task<BenchmarkSnapshotRecord?> LoadIsolationSnapshotAsync(
        HybridTensor tensor,
        Dictionary<string, BenchmarkSnapshotRecord?> cache,
        CancellationToken ct)
    {
        return tensor.OverrideMode switch
        {
            HybridTensorOverrideMode.ExactTensorScheme => await LoadExactIsolationAsync(tensor.TGroup, tensor.ExactTensorScheme!, cache, ct),
            _ => await LoadLearnedCandidateIsolationAsync(tensor.TGroup, tensor.CandidateBaseline!, cache, ct)
        };
    }

    private async Task<BenchmarkSnapshotRecord?> LoadLearnedCandidateIsolationAsync(
        TensorGroup group,
        BaselineQuants baseline,
        Dictionary<string, BenchmarkSnapshotRecord?> cache,
        CancellationToken ct)
    {
        string cacheKey = $"learned:{group.UniqueId}:{baseline.CanonicalKey}";
        if (cache.TryGetValue(cacheKey, out var existing))
            return existing;

        var isolationQuant = HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0);
        isolationQuant.SetLearnedCandidateOverride(group, baseline);

        var snapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)isolationQuant, ct);
        cache[cacheKey] = snapshot;
        return snapshot;
    }

    private async Task<BenchmarkSnapshotRecord?> LoadExactIsolationAsync(
        TensorGroup group,
        TensorWeightScheme exactScheme,
        Dictionary<string, BenchmarkSnapshotRecord?> cache,
        CancellationToken ct)
    {
        string cacheKey = $"exact:{group.UniqueId}:{exactScheme.Names[0]}";
        if (cache.TryGetValue(cacheKey, out var existing))
            return existing;

        var isolationQuant = HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0);
        isolationQuant.SetExactOverride(group, exactScheme);

        var snapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)isolationQuant, ct);
        cache[cacheKey] = snapshot;
        return snapshot;
    }

    private static byte NormalizeBaselineIdForIsolation(byte baselineId)
    {
        var baseline = BaselineQuants.FromId(baselineId);
        if (!baseline.IsExternalRepositoryBaseline)
            return baselineId;

        var builtIn = BaselineQuants.ResolveBuiltInStandardBaseline(baseline.QuantizeBaseArgumentName)
                      ?? BaselineQuants.ResolveBuiltInStandardBaseline(baseline.Names[0]);

        return builtIn?.UniqueId ?? baselineId;
    }

    private static void ApplyProportionalTradeWeighting(
        BenchmarkSnapshotRecord baseIsolation,
        BenchmarkSnapshotRecord targetIsolation,
        ref double predictedKld,
        ref double predictedPpl)
    {
        if (targetIsolation.SizeBytes >= baseIsolation.SizeBytes)
            return;

        double savingsPercent = ((double)baseIsolation.SizeBytes - targetIsolation.SizeBytes) / baseIsolation.SizeBytes * 100d;
        if (savingsPercent <= 0d)
            return;

        double kldDelta = targetIsolation.Kld - baseIsolation.Kld;
        double pplDelta = targetIsolation.Ppl - baseIsolation.Ppl;

        if (kldDelta <= 0d && pplDelta <= 0d)
            return;

        double damage = Math.Max(0d, kldDelta) * 1000d + Math.Max(0d, pplDelta);
        double damagePerSavings = damage / Math.Max(0.10d, savingsPercent);

        if (damagePerSavings <= 1.0d)
            return;

        double multiplier = Math.Min(2.75d, 1.0d + ((damagePerSavings - 1.0d) * 0.20d));
        predictedKld += Math.Max(0d, kldDelta) * (multiplier - 1.0d);
        predictedPpl += Math.Max(0d, pplDelta) * (multiplier - 1.0d);
    }

    private static void PrintPredictionDiagnostics(IReadOnlyList<PredictedCandidateEvaluation> evaluations)
    {
        int pureCount = evaluations.Count(x => x.IsPureBaseline);
        int hybridCount = evaluations.Count - pureCount;
        AnsiConsole.MarkupLine($"[grey]Prediction composition:[/] [cyan]{pureCount:N0}[/] [grey]pure[/] / [cyan]{hybridCount:N0}[/] [grey]hybrid[/]");

        if (evaluations.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]Prediction evaluation completed for[/] [cyan]0[/] [grey]remaining combinations.[/]");
            return;
        }

        ulong min = evaluations.Min(x => x.PredictedSizeBytes);
        ulong max = evaluations.Max(x => x.PredictedSizeBytes);
        AnsiConsole.MarkupLine($"[grey]Prediction size spread:[/] [cyan]{ToGb(min):F2}[/] [grey]GB ..[/] [cyan]{ToGb(max):F2}[/] [grey]GB[/]");

        foreach (var byBitRange in evaluations.GroupBy(x => x.BaseBitRange).OrderBy(x => x.Key))
        {
            ulong bitMin = byBitRange.Min(x => x.PredictedSizeBytes);
            ulong bitMax = byBitRange.Max(x => x.PredictedSizeBytes);
            AnsiConsole.MarkupLine(
                $"[grey]Base BitRange {byBitRange.Key} prediction spread:[/] [cyan]{byBitRange.Count():N0}[/] [grey]candidate(s),[/] [cyan]{ToGb(bitMin):F2}[/] [grey]GB ..[/] [cyan]{ToGb(bitMax):F2}[/] [grey]GB[/]");

            foreach (var sample in byBitRange.OrderBy(x => x.PredictedSizeBytes).ThenBy(x => x.PredictedKldCost).Take(3))
            {
                AnsiConsole.MarkupLine(
                    $"  [grey]- sample:[/] {Markup.Escape(sample.Quant.BaseQuant.Names[0])} [grey]| predicted[/] [cyan]{ToGb(sample.PredictedSizeBytes):F2}[/] [grey]GB | KLD[/] [cyan]{sample.PredictedKldCost:G6}[/] [grey]| PPL[/] [cyan]{sample.PredictedPplCost:F4}[/]");
            }
        }

        if (min == max && evaluations.Select(x => x.BaseBitRange).Distinct().Count() > 1)
        {
            AnsiConsole.MarkupLine("[yellow]Prediction diagnostic warning:[/] all candidates resolved to the same predicted size even though multiple base BitRanges remain. This usually means the relative size predictor is still collapsing too aggressively.[/]");
        }

        AnsiConsole.MarkupLine($"[grey]Prediction evaluation completed for[/] [cyan]{evaluations.Count:N0}[/] [grey]remaining combinations.[/]");
    }

    private static double ToGb(ulong bytes) => bytes / 1024d / 1024d / 1024d;
}
