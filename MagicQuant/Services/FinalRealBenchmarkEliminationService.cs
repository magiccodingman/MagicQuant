using MagicQuant.Models;

namespace MagicQuant.Services;

public sealed class FinalRealBenchmarkEliminationService
{
    public FinalRealEliminationResult Eliminate(IReadOnlyCollection<BenchmarkSnapshotRecord> snapshots)
    {
        var ordered = snapshots
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ThenBy(x => x.Ppl)
            .ToList();

        var survivors = new List<BenchmarkSnapshotRecord>();
        var eliminated = new List<BenchmarkSnapshotRecord>();

        for (int i = 0; i < ordered.Count; i++)
        {
            var current = ordered[i];
            bool dominated = ordered
                .Where((_, index) => index != i)
                .Any(other =>
                    other.SizeBytes <= current.SizeBytes &&
                    other.Kld < current.Kld &&
                    other.Ppl < current.Ppl);

            if (dominated)
                eliminated.Add(current);
            else
                survivors.Add(current);
        }

        return new FinalRealEliminationResult
        {
            Survivors = survivors
                .OrderBy(x => x.Kld)
                .ThenBy(x => x.SizeBytes)
                .ThenBy(x => x.Ppl)
                .ToList(),
            Eliminated = eliminated
        };
    }
}
