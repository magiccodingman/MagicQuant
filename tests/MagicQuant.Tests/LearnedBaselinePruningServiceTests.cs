using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public class LearnedBaselinePruningServiceTests
{
    [Fact]
    public async Task CoverageStatus_ReportsEarlyPruningDisabled()
    {
        var service = new LearnedBaselinePruningService();

        var status = await service.GetCoverageStatusAsync();

        Assert.False(status.HasAnyLearnedRows);
        Assert.False(status.SafeToApplyBeforeStartup);
        Assert.Equal(0, status.ExpectedCandidateGroupPairs);
        Assert.Equal(0, status.PresentCandidateGroupPairs);
        Assert.Contains(status.MissingPairs, x => x.Contains("disabled", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task AnalyzeAndApply_DoesNotPruneWhileFeatureIsDisabled()
    {
        var service = new LearnedBaselinePruningService();

        var result = await service.AnalyzeAndApplyAsync();

        Assert.Equal(0, result.GroupCandidateEliminations);
        Assert.Equal(0, result.BaselinesSkippedWithoutLearnedRows);
        Assert.Contains(result.Notes, x => x.Contains("disabled", StringComparison.OrdinalIgnoreCase));
    }
}
