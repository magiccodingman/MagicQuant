using MagicQuant.Configuration;
using Xunit;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Tests;

public sealed class SynergyTransferConfigTests
{
    [Fact]
    public void ContextStrata_DeserializeControlledBlanketQuantLists()
    {
        const string yaml = """
            synergy_detection:
              transfer_probe_context_strata:
                high_fidelity_reference_quants: [Q6_K, Q5_K]
                mid_fidelity_reference_quants: [Q4_K_M]
                low_fidelity_reference_quants: [IQ3_S]
                low_fidelity_enabled: true
              exploratory_context_pair_enabled: true
              max_exploratory_context_pairs_per_run: 11
              exploratory_pair_bit_ranges: [3, 4]
              exploratory_pair_context_strata: [mid-fidelity, low-fidelity]
              context_scoped_rule_application_enabled: true
              max_non_rule_group_context_mismatches: 2
            """;

        var config = new DeserializerBuilder()
            .IgnoreUnmatchedProperties()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build()
            .Deserialize<MagicQuantYamlConfig>(yaml);

        Assert.Equal(["Q6_K", "Q5_K"], config.SynergyDetection.TransferProbeContextStrata.HighFidelityReferenceQuants);
        Assert.Equal(["Q4_K_M"], config.SynergyDetection.TransferProbeContextStrata.MidFidelityReferenceQuants);
        Assert.Equal(["IQ3_S"], config.SynergyDetection.TransferProbeContextStrata.LowFidelityReferenceQuants);
        Assert.True(config.SynergyDetection.TransferProbeContextStrata.LowFidelityEnabled);
        Assert.True(config.SynergyDetection.ExploratoryContextPairEnabled);
        Assert.Equal(11, config.SynergyDetection.MaxExploratoryContextPairsPerRun);
        Assert.Equal([3, 4], config.SynergyDetection.ExploratoryPairBitRanges);
        Assert.Equal(["mid-fidelity", "low-fidelity"], config.SynergyDetection.ExploratoryPairContextStrata);
        Assert.True(config.SynergyDetection.ContextScopedRuleApplicationEnabled);
        Assert.Equal(2, config.SynergyDetection.MaxNonRuleGroupContextMismatches);
    }
}
