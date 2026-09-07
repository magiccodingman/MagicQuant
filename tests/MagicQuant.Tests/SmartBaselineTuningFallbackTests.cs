using MagicQuant.Services;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public sealed class SmartBaselineTuningFallbackTests
{
    [Fact]
    public void LearningOnlyBaselineWithMissingIsolationCoverageIsNotTunable()
    {
        var eligible = SmartBaselineTuningFallbackService.IsEligibleForSmartFallbackTuning(
            BaselineQuants.Q6_K,
            [BaselineQuants.Q8_0],
            hasCompleteIsolationCoverage: false);

        Assert.False(eligible);
    }

    [Fact]
    public void ExplicitCandidateWithMissingIsolationCoverageRemainsAHardFailurePath()
    {
        var eligible = SmartBaselineTuningFallbackService.IsEligibleForSmartFallbackTuning(
            BaselineQuants.Q6_K,
            [BaselineQuants.Q8_0, BaselineQuants.Q6_K],
            hasCompleteIsolationCoverage: false);

        Assert.True(eligible);
    }

    [Fact]
    public void HistoricalCompleteCoverageAllowsTuningWithoutCurrentExplicitRole()
    {
        var eligible = SmartBaselineTuningFallbackService.IsEligibleForSmartFallbackTuning(
            BaselineQuants.Q6_K,
            [BaselineQuants.Q8_0],
            hasCompleteIsolationCoverage: true);

        Assert.True(eligible);
    }
}
