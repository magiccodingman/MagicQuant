using MagicQuant.Helpers;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public class BaselineCandidatePolicyTests
{
    [Fact]
    public void NoImatrix_GroupCandidates_ExcludeRequiresImatrix()
    {
        var candidates = BaselineQuants.GetGroupCombinationCandidates(hasUsableImatrix: false, allowHighPrecisionHybrids: true);
        Assert.DoesNotContain(candidates, x => x.RequiresImatrix);
    }

    [Fact]
    public void ImatrixEnabled_GroupCandidates_IncludeI3XXS()
    {
        var candidates = BaselineQuants.GetGroupCombinationCandidates(hasUsableImatrix: true, allowHighPrecisionHybrids: true);
        Assert.Contains(candidates, x => x.UniqueId == BaselineQuants.IQ3_XXS.UniqueId);
    }

    [Fact]
    public void ImatrixEnabled_PureBaselines_IncludeIq2Xxs()
    {
        var baselines = BaselineQuants.GetPureBaselineCandidates(hasUsableImatrix: true);
        Assert.Contains(baselines, x => x.UniqueId == BaselineQuants.IQ2_XXS.UniqueId);
    }

    [Fact]
    public void ExplicitCandidateExhaustion_UsesQ8FallbackPolicy()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(true);

        RuntimeSearchSpace.BanAllExplicitCombinationCandidatesForGroup(TReg.AttnQ);

        Assert.True(RuntimeSearchSpace.IsGroupExplicitCandidateBanned(TReg.AttnQ));
        Assert.Equal(BaselineQuants.Q8_0.UniqueId, BaselineQuants.GetDefaultExplicitFallbackBaseline().UniqueId);
    }

    [Fact]
    public void HighPrecisionCandidatesRemainInReasoningUniverse_UntilLatePruneStage()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(true);

        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(BaselineQuants.Q8_0);
        var attnQIndex = TReg.All.OrderBy(x => x.UniqueId).ToList().FindIndex(x => x.UniqueId == TReg.AttnQ.UniqueId);

        Assert.Contains(BaselineQuants.BF16_Hybrid.UniqueId, allowed[attnQIndex]);
    }
}
