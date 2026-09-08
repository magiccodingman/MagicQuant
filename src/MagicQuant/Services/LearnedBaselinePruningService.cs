using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MagicQuant.Services;

public sealed class LearnedBaselinePruningResult
{
    public int GroupCandidateEliminations { get; set; }
    public int BaselinesSkippedWithoutLearnedRows { get; set; }
    public List<string> Notes { get; } = new();
}

public sealed class LearnedBaselineCoverageStatus
{
    public bool HasAnyLearnedRows { get; set; }
    public bool SafeToApplyBeforeStartup { get; set; }
    public int ExpectedCandidateGroupPairs { get; set; }
    public int PresentCandidateGroupPairs { get; set; }
    public List<string> MissingPairs { get; } = new();
}

public sealed class LearnedBaselinePruningService
{
    public Task<LearnedBaselineCoverageStatus> GetCoverageStatusAsync(CancellationToken ct = default)
    {
        var status = new LearnedBaselineCoverageStatus
        {
            HasAnyLearnedRows = false,
            SafeToApplyBeforeStartup = false,
            ExpectedCandidateGroupPairs = 0,
            PresentCandidateGroupPairs = 0
        };

        status.MissingPairs.Add("Learned-baseline early pruning is disabled.");
        return Task.FromResult(status);
    }

    public Task<LearnedBaselinePruningResult> AnalyzeAndApplyAsync(CancellationToken ct = default)
    {
        var result = new LearnedBaselinePruningResult();
        result.Notes.Add("Learned-baseline early pruning is disabled. No candidates were removed from the search space.");
        return Task.FromResult(result);
    }
}