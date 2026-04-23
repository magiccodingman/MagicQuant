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

        if (candidates.Count > 0)
        {
            ulong minPredicted = candidates.Min(x => x.PredictedSizeBytes);
            ulong maxPredicted = candidates.Max(x => x.PredictedSizeBytes);
            AnsiConsole.MarkupLine($"[grey]Predicted size spread:[/] [cyan]{FormatBytes(minPredicted)}[/] [grey]..[/] [cyan]{FormatBytes(maxPredicted)}[/]");

            foreach (var baseBitRange in candidates.GroupBy(x => x.BaseBitRange).OrderBy(x => x.Key))
            {
                AnsiConsole.MarkupLine(
                    $"[grey]Predicted candidates using base BitRange {baseBitRange.Key}:[/] [cyan]{baseBitRange.Count():N0}[/]");
            }
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

        foreach (var bucket in buckets)
        {
            AnsiConsole.MarkupLine(
                $"[grey]BitRange bucket {bucket.Key} -> lower_anchor={bucket.LowerAnchorSizeBytes:N0} upper_anchor={bucket.UpperAnchorSizeBytes:N0}[/]");
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

        int populatedBucketCount = 0;
        foreach (var bucket in buckets)
        {
            int count = bucketed.Count(x => x.Bucket.Key == bucket.Key);
            if (count > 0)
                populatedBucketCount++;

            AnsiConsole.MarkupLine(
                $"[grey]Bucket assignment {bucket.Key}:[/] [cyan]{count:N0}[/] [grey]candidate(s)[/]");
        }

        if (unbucketed.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Unbucketed predicted candidates:[/] [cyan]{unbucketed.Count:N0}[/]");
        }

        if (candidates.Count > 0 && populatedBucketCount <= 1)
        {
            AnsiConsole.MarkupLine(
                "[bold yellow]Bucket diagnostic warning:[/] [grey]Only one BitRange bucket received predicted candidates. This usually means carrier pruning or predicted-size anchoring collapsed the search into one neighborhood.[/]");
        }

        return new BitRangeBucketBuildResult
        {
            Buckets = buckets,
            BucketedCandidates = bucketed,
            UnbucketedCandidates = unbucketed
        };
    }

    private static string FormatBytes(ulong bytes)
    {
        double gb = bytes / 1024d / 1024d / 1024d;
        return $"{gb:F2} GB ({bytes:N0} bytes)";
    }
}
