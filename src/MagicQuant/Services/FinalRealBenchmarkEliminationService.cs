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

                if (!EquivalentTruthSelectionHelper.AreEquivalentTruths(
                        seed.SizeBytes,
                        seed.Kld,
                        seed.Ppl,
                        ordered[j].SizeBytes,
                        ordered[j].Kld,
                        ordered[j].Ppl))
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
                .OrderByDescending(x => EquivalentTruthSelectionHelper.GetBaselineSafetyRank(
                    x.Quant.BaseQuant,
                    isHybrid: x.IsHybrid,
                    isExternalPureBaseline: x.IsExternalPureBaseline))
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

        // Final dominance intentionally follows the new survival rule:
        // size must be same-or-smaller and KLD must be lower. PPL remains displayed
        // and available for manual judgment, but it no longer prevents a KLD/size win.
        return sameOrSmaller && strictlyBetterKld;
    }
}