using MagicQuant.Models;
using MQ.DB.Models;

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
                        AddReasonCount(diag, "effective-duplicate", removed);
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
                        Math.Abs(candidate.PredictedKldCost - best.PredictedKldCost) <= Math.Max(
                            Config.SurvivalKldCloseCallAbsoluteEpsilon,
                            best.PredictedKldCost * Config.SurvivalKldCloseCallRelativeFraction) &&
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

            var bucketPureBaselines = FilterPureBaselinesForBucket(bucket, pureBaselines);

            // Keep truly special hybrids as bonus survivors. They do not consume the bucket cap.
            var bonusHybrids = keptAfterPractical
                .Where(x => !x.IsPureBaseline)
                .Where(x => bucketPureBaselines.Count == 0 || BeatsAnyPureBaseline(x, bucketPureBaselines))
                .Where(x => !IsShadowedByPureBaseline(x, bucketPureBaselines))
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
                .ToList();

            if (bonusHybrids.Count > 0)
            {
                survivors.AddRange(bonusHybrids);
                diag.Notes.Add($"bonus-hybrid-kept={bonusHybrids.Count:N0}");
            }

            var remainingPool = keptAfterPractical
                .Where(x => bonusHybrids.All(b => TensorConfigIdentity.ToKey(b.Config) != TensorConfigIdentity.ToKey(x.Config)))
                .ToList();

            int shadowedRemoved = 0;
            if (bucketPureBaselines.Count > 0)
            {
                remainingPool = remainingPool
                    .Where(x =>
                    {
                        bool shadowed = IsShadowedByPureBaseline(x, bucketPureBaselines);
                        if (shadowed)
                            shadowedRemoved++;
                        return !shadowed;
                    })
                    .ToList();

                if (shadowedRemoved > 0)
                    AddReasonCount(diag, "pure-baseline-shadowed", shadowedRemoved);
            }

            int bucketBudget = Math.Max(0, Config.MaxSelectedChoicesPerBucket);
            var capped = SelectDiversifiedBySizeBands(remainingPool, bucket, bucketBudget);

            int capRemoved = Math.Max(0, remainingPool.Count - capped.Count);
            if (capRemoved > 0)
                AddReasonCount(diag, "bucket-cap", capRemoved);

            survivors.AddRange(capped);

            diag.KeptCount = bonusHybrids.Count + capped.Count;
            diag.RemovedCount = Math.Max(0, diag.IncomingCount - diag.KeptCount);
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

    private List<PredictedCandidateEvaluation> SelectDiversifiedBySizeBands(
        IReadOnlyList<PredictedCandidateEvaluation> candidates,
        BitRangeBucketDefinition bucket,
        int budget)
    {
        if (budget <= 0 || candidates.Count == 0)
            return new List<PredictedCandidateEvaluation>();

        var ordered = candidates
            .OrderBy(x => x.PredictedSizeBytes)
            .ThenBy(x => x.PredictedKldCost)
            .ThenBy(x => x.PredictedPplCost)
            .ThenBy(x => x.CompositeScore)
            .ToList();

        // First pass: pull a Pareto-ish size frontier so obvious smaller-size wins get first dibs.
        var frontier = new List<PredictedCandidateEvaluation>();
        double bestKldSeen = double.PositiveInfinity;
        foreach (var candidate in ordered)
        {
            bool materiallyBetterKld = candidate.PredictedKldCost + Config.SurvivalKldCloseCallAbsoluteEpsilon < bestKldSeen;
            bool meaningfullySmallerThanLast = frontier.Count == 0 ||
                                               PercentDifference(candidate.PredictedSizeBytes, frontier[^1].PredictedSizeBytes) >= Config.SurvivalMeaningfulSizeBiasPercent;

            if (materiallyBetterKld || meaningfullySmallerThanLast)
            {
                frontier.Add(candidate);
                if (candidate.PredictedKldCost < bestKldSeen)
                    bestKldSeen = candidate.PredictedKldCost;
            }
        }

        var selected = new List<PredictedCandidateEvaluation>();
        ulong lower = bucket.LowerAnchorSizeBytes;
        ulong upper = bucket.UpperAnchorSizeBytes > lower ? bucket.UpperAnchorSizeBytes : lower + 1UL;
        double span = Math.Max(1d, upper - lower);

        // Second pass: reserve one slot per size band so we do not just take the five most Q8-adjacent items.
        var bands = new Dictionary<int, List<PredictedCandidateEvaluation>>();
        foreach (var candidate in frontier)
        {
            double normalized = Math.Clamp((candidate.PredictedSizeBytes - lower) / span, 0d, 0.999999d);
            int band = Math.Min(budget - 1, (int)Math.Floor(normalized * budget));
            if (!bands.TryGetValue(band, out var list))
            {
                list = new List<PredictedCandidateEvaluation>();
                bands[band] = list;
            }

            list.Add(candidate);
        }

        foreach (var band in bands.OrderBy(x => x.Key))
        {
            var best = band.Value
                .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                .First();

            if (selected.All(x => TensorConfigIdentity.ToKey(x.Config) != TensorConfigIdentity.ToKey(best.Config)))
                selected.Add(best);

            if (selected.Count >= budget)
                return selected;
        }

        // Final fill: if we still have room, backfill from the overall frontier, then the full pool.
        foreach (var candidate in frontier
                     .OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))
                     .Concat(ordered.OrderBy(x => x, Comparer<PredictedCandidateEvaluation>.Create(_policy.Compare))))
        {
            if (selected.Count >= budget)
                break;

            if (selected.Any(x => TensorConfigIdentity.ToKey(x.Config) == TensorConfigIdentity.ToKey(candidate.Config)))
                continue;

            selected.Add(candidate);
        }

        return selected;
    }

    private static List<BenchmarkSnapshotRecord> FilterPureBaselinesForBucket(
        BitRangeBucketDefinition bucket,
        IReadOnlyCollection<BenchmarkSnapshotRecord>? pureBaselines)
    {
        if (pureBaselines == null || pureBaselines.Count == 0)
            return new List<BenchmarkSnapshotRecord>();

        ulong lower = bucket.LowerAnchorSizeBytes;
        ulong upper = bucket.UpperAnchorSizeBytes;

        var inBucket = pureBaselines
            .Where(x => x.SizeBytes >= lower && x.SizeBytes <= upper)
            .ToList();

        return inBucket.Count > 0 ? inBucket : pureBaselines.ToList();
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
