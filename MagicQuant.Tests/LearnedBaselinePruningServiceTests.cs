using MagicQuant.Helpers;
using MagicQuant.Services;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public class LearnedBaselinePruningServiceTests
{
    [Fact]
    public void Embeddings_LearnedBaselinePruning_OnlyAllowsQ6KAndBansOtherBaselinesWhenFinalTypeMapsToQ6K()
    {
        RuntimeSearchSpace.ResetForNewModel();
        RuntimeSearchSpace.SetImatrixAvailability(true);

        var result = new LearnedBaselinePruningResult();

        var learnedRows = new List<LearnedBaselinePruningService.LearnedRow>
        {
            new(BaselineQuants.Q6_K.UniqueId, TensorWeightScheme.Q6_K.UniqueId, TReg.Embeddings.UniqueId, "Q6_K"),
            new(BaselineQuants.Q5_K.UniqueId, TensorWeightScheme.Q5_K.UniqueId, TReg.Embeddings.UniqueId, "Q6_K"),
            new(BaselineQuants.Q4_K_M.UniqueId, TensorWeightScheme.Q4_K.UniqueId, TReg.Embeddings.UniqueId, "Q6_K"),
            new(BaselineQuants.IQ4_NL.UniqueId, TensorWeightScheme.IQ4_NL.UniqueId, TReg.Embeddings.UniqueId, "Q6_K"),
            new(BaselineQuants.IQ4_XS.UniqueId, TensorWeightScheme.IQ4_XS.UniqueId, TReg.Embeddings.UniqueId, "Q6_K")
        };

        var unused = new HashSet<byte>();

        LearnedBaselinePruningService.ApplyLearnedBaselinePruning(
            learnedRows,
            aiModelHashId: 1,
            aiModelHashUniqueHash: "regression-model-hash",
            unusedGroupIds: unused,
            result: result);

        Assert.False(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(TReg.Embeddings, TensorWeightScheme.Q6_K));
        Assert.True(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(TReg.Embeddings, TensorWeightScheme.Q5_K));
        Assert.True(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(TReg.Embeddings, TensorWeightScheme.Q4_K));
        Assert.True(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(TReg.Embeddings, TensorWeightScheme.IQ4_NL));
        Assert.True(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(TReg.Embeddings, TensorWeightScheme.IQ4_XS));

        Assert.Contains(result.Notes, x => x.Contains("group=embeddings") && x.Contains("candidate=Q6_K") && x.Contains("decision=ALLOW"));
        Assert.Contains(result.Notes, x => x.Contains("group=embeddings") && x.Contains("candidate=Q5_K") && x.Contains("decision=BAN"));
        Assert.Contains(result.Notes, x => x.Contains("group=embeddings") && x.Contains("candidate=Q4_K") && x.Contains("decision=BAN"));
        Assert.Contains(result.Notes, x => x.Contains("group=embeddings") && x.Contains("candidate=IQ4_NL") && x.Contains("decision=BAN"));
        Assert.Contains(result.Notes, x => x.Contains("group=embeddings") && x.Contains("candidate=IQ4_XS") && x.Contains("decision=BAN"));
    }
}
