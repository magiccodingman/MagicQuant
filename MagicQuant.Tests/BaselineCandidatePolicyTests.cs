using MagicQuant.Helpers;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public class BaselineCandidatePolicyTests
{
    [Fact]
    public void Iq1Families_AreRegisteredAsImatrixLearningAndExplicitCandidates()
    {
        Assert.Equal((byte)1, BaselineQuants.IQ1_S.BitRange);
        Assert.Equal((byte)1, BaselineQuants.IQ1_M.BitRange);
        Assert.True(BaselineQuants.IQ1_S.RequiresImatrix);
        Assert.True(BaselineQuants.IQ1_M.RequiresImatrix);
        Assert.True(BaselineQuants.IQ1_S.IsLearningBaseline);
        Assert.True(BaselineQuants.IQ1_M.IsLearningBaseline);
        Assert.True(BaselineQuants.IQ1_S.IsExplicitGroupCombinationCandidate);
        Assert.True(BaselineQuants.IQ1_M.IsExplicitGroupCombinationCandidate);
        Assert.False(BaselineQuants.IQ1_S.IsCombinationCarrierCandidate);
        Assert.False(BaselineQuants.IQ1_M.IsCombinationCarrierCandidate);
        Assert.Same(BaselineQuants.IQ1_S, BaselineQuants.ResolveBuiltInStandardBaseline("IQ1_S"));
        Assert.Same(BaselineQuants.IQ1_M, BaselineQuants.ResolveBuiltInStandardBaseline("IQ1_M"));
        Assert.Equal("IQ1_S", TensorWeightScheme.FromId(TensorWeightScheme.IQ1_S.UniqueId).Names[0]);
        Assert.Equal("IQ1_M", TensorWeightScheme.FromId(TensorWeightScheme.IQ1_M.UniqueId).Names[0]);
    }

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

    [Fact]
    public void ComboLogic_UsesCandidateLevelBannedGroups()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(false);

        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(BaselineQuants.Q8_0);
        var moeRouterIndex = TReg.All.OrderBy(x => x.UniqueId).ToList().FindIndex(x => x.UniqueId == TReg.MoeRouter.UniqueId);

        Assert.DoesNotContain(BaselineQuants.Q5_K.UniqueId, allowed[moeRouterIndex]);
        Assert.DoesNotContain(BaselineQuants.IQ4_NL.UniqueId, allowed[moeRouterIndex]);
        Assert.DoesNotContain(BaselineQuants.IQ4_XS.UniqueId, allowed[moeRouterIndex]);
    }
}
