using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB.Models;

namespace MagicQuant.Services;

public sealed class FinalRealBenchmarkEliminationService
{
    public FinalRealEliminationResult Eliminate(IReadOnlyCollection<BenchmarkSnapshotRecord> snapshots)
    {
        var uniqueByConfig = snapshots
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ThenBy(x => x.Ppl)
            .ThenBy(x => x.DisplayName, StringComparer.Ordinal)
            .ToList();

        var eliminated = new List<BenchmarkSnapshotRecord>();
        var collapsed = CollapseEquivalentTruths(uniqueByConfig, eliminated);

        var survivors = new List<BenchmarkSnapshotRecord>();

        for (int i = 0; i < collapsed.Count; i++)
        {
            var current = collapsed[i];
            bool dominated = collapsed
                .Where((_, index) => index != i)
                .Any(other => Dominates(other, current));

            if (dominated)
                eliminated.Add(current);
            else
                survivors.Add(current);
        }

        return new FinalRealEliminationResult
        {
            Survivors = survivors
                .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
                .OrderBy(x => x.Kld)
                .ThenBy(x => x.SizeBytes)
                .ThenBy(x => x.Ppl)
                .ThenByDescending(x => x.Quant.BaseQuant.BitRange)
                .ThenByDescending(x => x.Quant.BaseQuant.ExplicitCandidateSortOrder)
                .ThenBy(x => x.IsHybrid)
                .ThenBy(x => x.IsExternalPureBaseline)
                .ThenBy(x => x.DisplayName, StringComparer.Ordinal)
                .ToList(),
            Eliminated = eliminated
                .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
                .ToList()
        };
    }

    private static List<BenchmarkSnapshotRecord> CollapseEquivalentTruths(
        IReadOnlyList<BenchmarkSnapshotRecord> ordered,
        List<BenchmarkSnapshotRecord> eliminated)
    {
        var kept = new List<BenchmarkSnapshotRecord>();
        var used = new bool[ordered.Count];

        for (int i = 0; i < ordered.Count; i++)
        {
            if (used[i])
                continue;

            var seed = ordered[i];
            var tied = new List<BenchmarkSnapshotRecord> { seed };
            used[i] = true;

            for (int j = i + 1; j < ordered.Count; j++)
            {
                if (used[j])
                    continue;

                if (!AreEquivalentTruths(seed, ordered[j]))
                    continue;

                tied.Add(ordered[j]);
                used[j] = true;
            }

            if (tied.Count == 1)
            {
                kept.Add(seed);
                continue;
            }

            var representative = tied
                .OrderByDescending(GetSafetyRank)
                .ThenBy(x => x.IsHybrid)
                .ThenBy(x => x.IsExternalPureBaseline)
                .ThenBy(x => x.ProviderName, StringComparer.Ordinal)
                .ThenBy(x => x.DisplayName, StringComparer.Ordinal)
                .First();

            kept.Add(representative);

            foreach (var loser in tied)
            {
                if (!ReferenceEquals(loser, representative))
                    eliminated.Add(loser);
            }
        }

        return kept;
    }

    private static bool Dominates(BenchmarkSnapshotRecord better, BenchmarkSnapshotRecord worse)
    {
        bool sameOrSmaller = better.SizeBytes <= worse.SizeBytes;
        bool strictlyBetterKld = better.Kld + IsolationPruningConfig.FloatingPointEpsilon < worse.Kld;
        bool strictlyBetterPpl = better.Ppl + IsolationPruningConfig.FloatingPointEpsilon < worse.Ppl;
        return sameOrSmaller && strictlyBetterKld && strictlyBetterPpl;
    }

    private static bool AreEquivalentTruths(BenchmarkSnapshotRecord left, BenchmarkSnapshotRecord right)
    {
        if (left.SizeBytes != right.SizeBytes)
            return false;

        return Math.Abs(left.Kld - right.Kld) <= IsolationPruningConfig.FloatingPointEpsilon &&
               Math.Abs(left.Ppl - right.Ppl) <= IsolationPruningConfig.FloatingPointEpsilon;
    }

    private static int GetSafetyRank(BenchmarkSnapshotRecord snapshot)
    {
        var baseline = snapshot.Quant.BaseQuant;

        // Prefer the safest / most default representative when multiple rows have identical truth.
        // 1) Higher BitRange is safer.
        // 2) Higher ExplicitCandidateSortOrder wins ties inside the same BitRange.
        // 3) Pure baseline beats hybrid when the measured truth is identical.
        // 4) Internal/non-external beats external pure reference when still tied.
        int rank = baseline.BitRange * 10_000;
        rank += baseline.ExplicitCandidateSortOrder * 10;
        rank += snapshot.IsHybrid ? 0 : 2;
        rank += snapshot.IsExternalPureBaseline ? 0 : 1;
        return rank;
    }
}
