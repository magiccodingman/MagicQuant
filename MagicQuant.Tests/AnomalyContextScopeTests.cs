using MagicQuant.Configuration;
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
            config.SynergyDetection.MaxNonRuleGroupsBelowReferenceTier = 1;
            Config.Load(config);
            Cache.UnusedTensorGroups.Clear();

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

            string predicate = AnomalyAdjustedPredictionService.BuildContextFidelityPredicate(rule, "c");

            Assert.Contains("c.LmHead", predicate, StringComparison.Ordinal);
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
}
