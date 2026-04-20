using MagicQuant.Helpers;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public class BaselineCandidatePolicyTests
{
    [Fact]
    public void GetPureBaselineCandidates_NoImatrix_ReturnsExactlyIq4Xs()
    {
        var ids = BaselineQuants.GetPureBaselineCandidates(hasUsableImatrix: false)
            .Select(x => x.UniqueId)
            .ToArray();

        Assert.Equal([BaselineQuants.IQ4_XS.UniqueId], ids);
    }

    [Fact]
    public void GetCombinationCarrierBaselines_NoImatrix_ReturnsExactlySixExpectedBaselines()
    {
        var ids = BaselineQuants.GetCombinationCarrierBaselines(hasUsableImatrix: false)
            .Select(x => x.UniqueId)
            .ToArray();

        Assert.Equal(
        [
            BaselineQuants.Q8_0.UniqueId,
            BaselineQuants.Q6_K.UniqueId,
            BaselineQuants.Q5_K.UniqueId,
            BaselineQuants.Q4_K_M.UniqueId,
            BaselineQuants.IQ4_NL.UniqueId,
            BaselineQuants.IQ4_XS.UniqueId
        ], ids);
    }

    [Fact]
    public void GetGroupCombinationCandidates_NoImatrixNoHighPrecision_ReturnsExactlySixExpectedBaselines()
    {
        var ids = BaselineQuants.GetGroupCombinationCandidates(hasUsableImatrix: false, allowHighPrecisionHybrids: false)
            .Select(x => x.UniqueId)
            .ToArray();

        Assert.Equal(
        [
            BaselineQuants.Q8_0.UniqueId,
            BaselineQuants.Q6_K.UniqueId,
            BaselineQuants.Q5_K.UniqueId,
            BaselineQuants.Q4_K_M.UniqueId,
            BaselineQuants.IQ4_NL.UniqueId,
            BaselineQuants.IQ4_XS.UniqueId
        ], ids);

        Assert.DoesNotContain(BaselineQuants.IQ3_S.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.IQ3_XS.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.IQ3_XXS.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.IQ2_S.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.IQ2_XS.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.IQ2_XXS.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.BF16_Hybrid.UniqueId, ids);
        Assert.DoesNotContain(BaselineQuants.F16_Hybrid.UniqueId, ids);
    }

    [Fact]
    public void RuntimeSearchSpace_GetActiveCombinationBaselines_ReturnsExactlySixExpectedBaselines()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(false);

        var ids = RuntimeSearchSpace.GetActiveCombinationBaselines()
            .Select(x => x.UniqueId)
            .ToArray();

        Assert.Equal(
        [
            BaselineQuants.Q8_0.UniqueId,
            BaselineQuants.Q6_K.UniqueId,
            BaselineQuants.Q5_K.UniqueId,
            BaselineQuants.Q4_K_M.UniqueId,
            BaselineQuants.IQ4_NL.UniqueId,
            BaselineQuants.IQ4_XS.UniqueId
        ], ids);
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
    public void CandidateBanAuthority_DrivesAllowedCandidateSet()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(true);

        RuntimeSearchSpace.BanCombinationCandidateForGroup(TReg.AttnQ, BaselineQuants.Q6_K);
        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(BaselineQuants.Q8_0);
        var attnQIndex = TReg.All.OrderBy(x => x.UniqueId).ToList().FindIndex(x => x.UniqueId == TReg.AttnQ.UniqueId);

        Assert.DoesNotContain(BaselineQuants.Q6_K.UniqueId, allowed[attnQIndex]);
    }

    [Fact]
    public void ComboLogic_WhenHighPrecisionDisabled_DoesNotInjectBf16OrF16()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(true);
        RuntimeSearchSpace.AllowHighPrecisionHybrids = false;

        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(BaselineQuants.Q8_0);
        var attnQIndex = TReg.All.OrderBy(x => x.UniqueId).ToList().FindIndex(x => x.UniqueId == TReg.AttnQ.UniqueId);

        Assert.DoesNotContain(BaselineQuants.BF16_Hybrid.UniqueId, allowed[attnQIndex]);
        Assert.DoesNotContain(BaselineQuants.F16_Hybrid.UniqueId, allowed[attnQIndex]);
    }
}
