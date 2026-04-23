using MagicQuant.Models;

namespace MagicQuant.Services;

public sealed class BucketLocalPruningResult
{
    public IReadOnlyList<PredictedCandidateEvaluation> Survivors { get; init; } = Array.Empty<PredictedCandidateEvaluation>();
    public IReadOnlyList<BucketPruneDiagnostics> Diagnostics { get; init; } = Array.Empty<BucketPruneDiagnostics>();
}

public sealed class BucketLocalPruningService
{
    private readonly PredictedTradeComparisonPolicy _policy = new();

    public BucketLocalPruningResult Prune(BitRangeBucketBuildResult buildResult)
    {
        var diagnostics = new List<BucketPruneDiagnostics>();
        var survivors = new List<PredictedCandidateEvaluation>();

        foreach (var bucket in buildResult.Buckets)
        {
            var incoming = buildResult.BucketedCandidates
                .Where(x => x.Bucket.Key == bucket.Key)
                .Select(x => x.Evaluation)
                .ToList();

            var diag = new BucketPruneDiagnostics
            {
                BucketKey = bucket.Key,
                LowerAnchorSizeBytes = bucket.LowerAnchorSizeBytes,
                UpperAnchorSizeBytes = bucket.UpperAnchorSizeBytes,
                IncomingCount = incoming.Count
            };

            if (incoming.Count == 0)
            {
                diag.KeptCount = 0;
                diagnostics.Add(diag);
                continue;
            }

            var deduped = incoming
                .GroupBy(x => x.EffectiveStateKey, StringComparer.Ordinal)
                .Select(g =>
                {
                    var ordered = g.OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare)).ToList();
                    int removed = ordered.Count - 1;
                    if (removed > 0)
                        diag.CountReason("effective-duplicate");
                    return ordered[0];
                })
                .ToList();

            var dominancePruned = new List<PredictedCandidateEvaluation>(deduped);

            for (int i = dominancePruned.Count - 1; i >= 0; i--)
            {
                var current = dominancePruned[i];
                bool dominated = dominancePruned
                    .Where((_, index) => index != i)
                    .Any(other => _policy.Dominates(other, current));

                if (dominated)
                {
                    dominancePruned.RemoveAt(i);
                    diag.CountReason("dominance");
                }
            }

            var orderedByPracticalTrade = dominancePruned
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .ToList();

            var keptAfterPractical = new List<PredictedCandidateEvaluation>();
            if (orderedByPracticalTrade.Count > 0)
            {
                var best = orderedByPracticalTrade[0];
                foreach (var candidate in orderedByPracticalTrade)
                {
                    bool sameNeighborhood =
                        Math.Abs(candidate.PredictedKldCost - best.PredictedKldCost) <= Math.Max(Config.SurvivalKldCloseCallAbsoluteEpsilon, best.PredictedKldCost * Config.SurvivalKldCloseCallRelativeFraction) &&
                        Math.Abs(candidate.PredictedPplCost - best.PredictedPplCost) <= Config.SurvivalPplLargeDifferencePercent &&
                        PercentDifference(candidate.PredictedSizeBytes, best.PredictedSizeBytes) < Config.SurvivalMeaningfulSizeBiasPercent;

                    bool obviouslyJunk = candidate.CompositeScore > best.CompositeScore * 1.65d && sameNeighborhood;

                    if (obviouslyJunk)
                    {
                        diag.CountReason("practical-trade");
                        continue;
                    }

                    keptAfterPractical.Add(candidate);
                }
            }

            var capped = keptAfterPractical
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .Take(Config.MaxSelectedChoicesPerBucket)
                .ToList();

            int capRemoved = keptAfterPractical.Count - capped.Count;
            if (capRemoved > 0)
                diag.CountReason("bucket-cap");

            diag.KeptCount = capped.Count;
            diag.RemovedCount = diag.IncomingCount - diag.KeptCount;
            survivors.AddRange(capped);
            diagnostics.Add(diag);
        }

        foreach (var candidate in buildResult.UnbucketedCandidates)
            survivors.Add(candidate);

        survivors = survivors
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .ToList();

        return new BucketLocalPruningResult
        {
            Survivors = survivors,
            Diagnostics = diagnostics
        };
    }

    private static double PercentDifference(ulong left, ulong right)
    {
        if (left == 0 || right == 0)
            return 0d;

        double min = Math.Min(left, right);
        double max = Math.Max(left, right);
        return ((max - min) / min) * 100d;
    }
}
