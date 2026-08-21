using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

[Collection(AnomalyContextScopeCollection.Name)]
public sealed class SynergyTransferPlanningTests
{
    [Fact]
    public void ControlledTransfer_ChangesOneGroupInsideExplicitLowFidelityBlanket()
    {
        var priorUnusedGroups = Cache.UnusedTensorGroups.ToList();

        try
        {
            Cache.UnusedTensorGroups.Clear();

            bool built = AnomalyWorkflowService.TryBuildControlledTransferConfig(
                BaselineQuants.IQ3_S.UniqueId,
                [(TReg.Embeddings.UniqueId, BaselineQuants.IQ4_NL.UniqueId)],
                out var reference,
                out var probe,
                out var changed);

            Assert.True(built);
            var movement = new QuantFidelityComparerService();
            Assert.All(movement.ActiveGroups, group =>
                Assert.Equal(BaselineQuants.IQ3_S.UniqueId, movement.EffectiveQuantId(reference, group)));
            Assert.Equal(BaselineQuants.IQ4_NL.UniqueId, movement.EffectiveQuantId(probe, TReg.Embeddings));
            Assert.All(movement.ActiveGroups.Where(x => x.UniqueId != TReg.Embeddings.UniqueId), group =>
                Assert.Equal(BaselineQuants.IQ3_S.UniqueId, movement.EffectiveQuantId(probe, group)));

            var groupChange = Assert.Single(changed);
            Assert.Equal(TReg.Embeddings.UniqueId, groupChange.Group.UniqueId);
            Assert.Equal(QuantMovementKind.Upgrade, groupChange.Movement);
        }
        finally
        {
            Cache.UnusedTensorGroups.Clear();
            Cache.UnusedTensorGroups.AddRange(priorUnusedGroups);
        }
    }

    [Fact]
    public void ControlledTransfer_SkipsTemplateStateEqualToBlanket()
    {
        bool built = AnomalyWorkflowService.TryBuildControlledTransferConfig(
            BaselineQuants.Q4_K_M.UniqueId,
            [(TReg.Embeddings.UniqueId, BaselineQuants.Q4_K_M.UniqueId)],
            out _,
            out _,
            out var changed);

        Assert.False(built);
        Assert.Empty(changed);
    }
}
