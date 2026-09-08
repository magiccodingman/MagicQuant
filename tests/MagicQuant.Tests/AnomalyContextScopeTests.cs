using MagicQuant.Configuration;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Xunit;

namespace MagicQuant.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class AnomalyContextScopeCollection
{
    public const string Name = "Anomaly context scope";
}

[Collection(AnomalyContextScopeCollection.Name)]
public sealed class AnomalyContextScopeTests
{
    [Fact]
    public void BuildContextFidelityPredicate_BoundsOnlyNonRuleGroups()
    {
        var priorConfig = Config.Current;
        var priorUnusedGroups = Cache.UnusedTensorGroups.ToList();

        try
        {
            var config = MagicQuantYamlConfig.CreateDefault();
            config.SynergyDetection.ContextScopedRuleApplicationEnabled = true;
            config.SynergyDetection.MaxNonRuleGroupContextMismatches = 1;
            Config.Load(config);
            Cache.UnusedTensorGroups.Clear();

            var rule = new AnomalyInteractionRule
            {
                ReferenceQuantId = BaselineQuants.Q8_0.UniqueId,
                ReferenceContextKey = $"{TReg.LmHead.UniqueId}:{BaselineQuants.Q4_K_M.UniqueId}",
                GroupStates =
                [
                    new AnomalyInteractionRuleGroupState
                    {
                        TensorGroupId = TReg.Embeddings.UniqueId,
                        ReferenceQuantId = BaselineQuants.Q8_0.UniqueId,
                        CandidateQuantId = BaselineQuants.Q6_K.UniqueId
                    }
                ]
            };

            string predicate = AnomalyAdjustedPredictionService.BuildContextFidelityPredicate(rule, "c");

            Assert.Contains("c.LmHead", predicate, StringComparison.Ordinal);
            Assert.Contains($"= {BaselineQuants.Q4_K_M.UniqueId} THEN 0", predicate, StringComparison.Ordinal);
            Assert.Contains("c.AttnQ", predicate, StringComparison.Ordinal);
            Assert.DoesNotContain("c.Embeddings", predicate, StringComparison.Ordinal);
            Assert.EndsWith("<= 1)", predicate, StringComparison.Ordinal);
        }
        finally
        {
            Config.Load(priorConfig);
            Cache.UnusedTensorGroups.Clear();
            Cache.UnusedTensorGroups.AddRange(priorUnusedGroups);
        }
    }

    [Fact]
    public void BuildRuleCandidateWhere_UsesEffectiveContextInsteadOfCarrierIdentity()
    {
        var priorConfig = Config.Current;
        var priorUnusedGroups = Cache.UnusedTensorGroups.ToList();

        try
        {
            var config = MagicQuantYamlConfig.CreateDefault();
            config.SynergyDetection.ContextScopedRuleApplicationEnabled = true;
            config.SynergyDetection.MaxNonRuleGroupContextMismatches = 0;
            Config.Load(config);
            Cache.UnusedTensorGroups.Clear();

            var rule = new AnomalyInteractionRule
            {
                ReferenceQuantId = BaselineQuants.Q4_K_M.UniqueId,
                ReferenceContextKey = string.Join("|", TReg.All.Select(x => $"{x.UniqueId}:{BaselineQuants.Q4_K_M.UniqueId}")),
                GroupStates =
                [
                    new AnomalyInteractionRuleGroupState
                    {
                        TensorGroupId = TReg.Embeddings.UniqueId,
                        ReferenceQuantId = BaselineQuants.Q4_K_M.UniqueId,
                        CandidateQuantId = BaselineQuants.IQ4_NL.UniqueId
                    }
                ]
            };

            string predicate = AnomalyAdjustedPredictionService.BuildRuleCandidateWhere(rule, "c");

            Assert.DoesNotContain($"c.BaseQuant = {BaselineQuants.Q4_K_M.UniqueId}", predicate, StringComparison.Ordinal);
            Assert.Contains("c.Embeddings", predicate, StringComparison.Ordinal);
            Assert.Contains("c.LmHead", predicate, StringComparison.Ordinal);
            Assert.Contains($"= {BaselineQuants.Q4_K_M.UniqueId} THEN 0", predicate, StringComparison.Ordinal);
        }
        finally
        {
            Config.Load(priorConfig);
            Cache.UnusedTensorGroups.Clear();
            Cache.UnusedTensorGroups.AddRange(priorUnusedGroups);
        }
    }

    [Fact]
    public void BuildContextFidelityPredicate_ReturnsEmptyWhenDisabled()
    {
        var priorConfig = Config.Current;

        try
        {
            var config = MagicQuantYamlConfig.CreateDefault();
            config.SynergyDetection.ContextScopedRuleApplicationEnabled = false;
            Config.Load(config);

            var rule = new AnomalyInteractionRule
            {
                ReferenceQuantId = BaselineQuants.Q8_0.UniqueId,
                GroupStates =
                [
                    new AnomalyInteractionRuleGroupState
                    {
                        TensorGroupId = TReg.Embeddings.UniqueId,
                        ReferenceQuantId = BaselineQuants.Q8_0.UniqueId,
                        CandidateQuantId = BaselineQuants.Q6_K.UniqueId
                    }
                ]
            };

            Assert.Empty(AnomalyAdjustedPredictionService.BuildContextFidelityPredicate(rule, "c"));
        }
        finally
        {
            Config.Load(priorConfig);
        }
    }

    [Fact]
    public void RuleSuppressionKey_DistinguishesEffectiveSurroundingContext()
    {
        var priorUnusedGroups = Cache.UnusedTensorGroups.ToList();

        try
        {
            Cache.UnusedTensorGroups.Clear();
            var movement = new QuantFidelityComparerService();
            var repository = new AnomalyRuleRepository(movement);
            var q8Context = movement.CreateActivatedContextBlanket(BaselineQuants.Q8_0.UniqueId);
            var q4PassengerContext = movement.WithStoredSlot(
                q8Context,
                TReg.LmHead,
                BaselineQuants.EncodeTensorConfigGroupSlot(BaselineQuants.Q4_K_M));
            var changed = new List<AnomalyChangedGroup>
            {
                new()
                {
                    Group = TReg.Embeddings,
                    ReferenceQuantId = BaselineQuants.Q8_0.UniqueId,
                    CandidateQuantId = BaselineQuants.Q6_K.UniqueId,
                    ReferenceStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlot(BaselineQuants.Q8_0),
                    CandidateStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlot(BaselineQuants.Q6_K),
                    Movement = QuantMovementKind.Downgrade
                }
            };

            string q8Key = repository.BuildRuleSuppressionKey(q8Context, changed);
            string q4PassengerKey = repository.BuildRuleSuppressionKey(q4PassengerContext, changed);

            Assert.NotEqual(q8Key, q4PassengerKey);
            Assert.Contains($"{TReg.LmHead.UniqueId}:{BaselineQuants.Q4_K_M.UniqueId}", q4PassengerKey, StringComparison.Ordinal);
        }
        finally
        {
            Cache.UnusedTensorGroups.Clear();
            Cache.UnusedTensorGroups.AddRange(priorUnusedGroups);
        }
    }
}
