using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class BitRangeBucketBuildResult
{
    public IReadOnlyList<BitRangeBucketDefinition> Buckets { get; init; } = Array.Empty<BitRangeBucketDefinition>();
    public IReadOnlyList<BucketedCandidate> BucketedCandidates { get; init; } = Array.Empty<BucketedCandidate>();
    public IReadOnlyList<PredictedCandidateEvaluation> UnbucketedCandidates { get; init; } = Array.Empty<PredictedCandidateEvaluation>();
}

public sealed class BitRangeBucketBuilderService
{
    private readonly HybridBenchmarkRepository _repository;

    public BitRangeBucketBuilderService(HybridBenchmarkRepository repository)
    {
        _repository = repository;
    }

    public async Task<BitRangeBucketBuildResult> BuildAsync(
        IReadOnlyCollection<PredictedCandidateEvaluation> candidates,
        CancellationToken ct = default)
    {
        var pureBaselines = await _repository.LoadPureBaselineSnapshotsAsync(ct);
        var baselineAnchors = pureBaselines
            .GroupBy(x => x.Quant.BaseQuant.BitRange)
            .Select(g => new
            {
                BitRange = g.Key,
                Snapshot = g.OrderBy(x => x.SizeBytes).ThenBy(x => x.Kld).First()
            })
            .OrderBy(x => x.BitRange)
            .ToList();

        var buckets = new List<BitRangeBucketDefinition>();

        for (int i = 0; i < baselineAnchors.Count - 1; i++)
        {
            var lower = baselineAnchors[i];
            var upper = baselineAnchors[i + 1];

            buckets.Add(new BitRangeBucketDefinition
            {
                Key = $"{lower.BitRange}->{upper.BitRange}",
                LowerBitRange = lower.BitRange,
                UpperBitRange = upper.BitRange,
                LowerAnchorSizeBytes = lower.Snapshot.SizeBytes,
                UpperAnchorSizeBytes = upper.Snapshot.SizeBytes
            });
        }

        if (buckets.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No usable BitRange buckets could be built. Survival will fall back to global predicted sorting if required.[/]");
            return new BitRangeBucketBuildResult
            {
                Buckets = Array.Empty<BitRangeBucketDefinition>(),
                BucketedCandidates = Array.Empty<BucketedCandidate>(),
                UnbucketedCandidates = candidates.ToList()
            };
        }

        var bucketed = new List<BucketedCandidate>();
        var unbucketed = new List<PredictedCandidateEvaluation>();

        foreach (var candidate in candidates)
        {
            BitRangeBucketDefinition? selected = null;

            foreach (var bucket in buckets)
            {
                ulong lowerBound = bucket == buckets[0] ? 0UL : bucket.LowerAnchorSizeBytes;
                ulong upperBound = bucket.UpperAnchorSizeBytes;

                if (candidate.PredictedSizeBytes >= lowerBound && candidate.PredictedSizeBytes <= upperBound)
                {
                    selected = bucket;
                    break;
                }
            }

            if (selected == null && candidate.PredictedSizeBytes > buckets[^1].UpperAnchorSizeBytes)
                selected = buckets[^1];

            if (selected == null)
            {
                unbucketed.Add(candidate);
                continue;
            }

            bucketed.Add(new BucketedCandidate
            {
                Evaluation = candidate,
                Bucket = selected
            });
        }

        return new BitRangeBucketBuildResult
        {
            Buckets = buckets,
            BucketedCandidates = bucketed,
            UnbucketedCandidates = unbucketed
        };
    }
}
