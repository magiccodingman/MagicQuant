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

    public BucketLocalPruningResult Prune(
        BitRangeBucketBuildResult buildResult,
        IReadOnlyCollection<BenchmarkSnapshotRecord>? pureBaselines = null)
    {
        var diagnostics = new List<BucketPruneDiagnostics>();
        var survivors = new List<PredictedCandidateEvaluation>();

        foreach (var bucket in buildResult.Buckets)
        {
            var incoming = buildResult.BucketedCandidates
                .Where(x => x.Bucket.Key == bucket.Key)
                .Select(x => x.Evaluation)
                .ToList();

            var bucketPureBaselines = GetPureBaselinesForBucket(bucket, pureBaselines);

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

            // Hybrids that beat a pure baseline in this bucket are bonus keeps and do not count
            // against the configured bucket cap.
            var bonusHybrids = keptAfterPractical
                .Where(x => !x.IsPureBaseline)
                .Where(x => BeatsAnyPureBaseline(x, bucketPureBaselines))
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .ToList();

            if (bonusHybrids.Count > 0)
                AddReasonCount(diag, "bonus-hybrid-kept", bonusHybrids.Count);

            var regularPool = keptAfterPractical
                .Where(x => !bonusHybrids.Any(b => TensorConfigIdentity.ToKey(b.Config) == TensorConfigIdentity.ToKey(x.Config)))
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .ToList();

            var viableRegular = new List<PredictedCandidateEvaluation>();
            var shadowedRegular = new List<PredictedCandidateEvaluation>();

            foreach (var candidate in regularPool)
            {
                if (IsShadowedByPureBaseline(candidate, bucketPureBaselines))
                    shadowedRegular.Add(candidate);
                else
                    viableRegular.Add(candidate);
            }

            var capped = viableRegular
                .Take(Config.MaxSelectedChoicesPerBucket)
                .ToList();

            // Refill from shadowed candidates only if we still have open slots in this bucket.
            if (capped.Count < Config.MaxSelectedChoicesPerBucket)
            {
                capped.AddRange(
                    shadowedRegular
                        .Take(Config.MaxSelectedChoicesPerBucket - capped.Count));
            }
            else if (shadowedRegular.Count > 0)
            {
                AddReasonCount(diag, "pure-baseline-shadowed", shadowedRegular.Count);
            }

            int capRemoved = Math.Max(0, viableRegular.Count - Math.Min(viableRegular.Count, Config.MaxSelectedChoicesPerBucket));
            if (capRemoved > 0)
                AddReasonCount(diag, "bucket-cap", capRemoved);

            var finalBucketSurvivors = bonusHybrids
                .Concat(capped)
                .GroupBy(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal)
                .Select(g => g.First())
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .ToList();

            diag.KeptCount = finalBucketSurvivors.Count;
            diag.RemovedCount = Math.Max(0, diag.IncomingCount - diag.KeptCount);
            survivors.AddRange(finalBucketSurvivors);
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

    private static List<BenchmarkSnapshotRecord> GetPureBaselinesForBucket(
        BitRangeBucketDefinition bucket,
        IReadOnlyCollection<BenchmarkSnapshotRecord>? pureBaselines)
    {
        if (pureBaselines == null || pureBaselines.Count == 0)
            return new List<BenchmarkSnapshotRecord>();

        ulong lowerBound = bucket.LowerAnchorSizeBytes;
        ulong upperBound = bucket.UpperAnchorSizeBytes;

        return pureBaselines
            .Where(x => x.SizeBytes >= lowerBound && x.SizeBytes <= upperBound)
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ToList();
    }

    private static bool BeatsAnyPureBaseline(
        PredictedCandidateEvaluation candidate,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselines)
    {
        foreach (var baseline in pureBaselines)
        {
            bool sameOrSmaller = candidate.PredictedSizeBytes <= baseline.SizeBytes;
            bool betterKld = candidate.PredictedKldCost + 1e-9 < baseline.Kld;
            bool betterPpl = candidate.PredictedPplCost + 1e-9 < baseline.Ppl;

            if (sameOrSmaller && (betterKld || betterPpl))
                return true;
        }

        return false;
    }

    private static bool IsShadowedByPureBaseline(
        PredictedCandidateEvaluation candidate,
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselines)
    {
        foreach (var baseline in pureBaselines)
        {
            bool sameOrSmaller = baseline.SizeBytes <= candidate.PredictedSizeBytes;
            bool kldNoWorse = baseline.Kld <= candidate.PredictedKldCost + 1e-9;
            bool pplNoWorse = baseline.Ppl <= candidate.PredictedPplCost + 1e-9;
            bool strict = baseline.SizeBytes < candidate.PredictedSizeBytes ||
                          baseline.Kld + 1e-9 < candidate.PredictedKldCost ||
                          baseline.Ppl + 1e-9 < candidate.PredictedPplCost;

            if (sameOrSmaller && kldNoWorse && pplNoWorse && strict)
                return true;
        }

        return false;
    }

    private static void AddReasonCount(BucketPruneDiagnostics diag, string reason, int count)
    {
        for (int i = 0; i < count; i++)
            diag.CountReason(reason);
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
