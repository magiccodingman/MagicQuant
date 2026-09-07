using MagicQuant.Configuration;
using MagicQuant.Models;
using Xunit;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Tests;

public sealed class ConfigurationContractTests
{
    [Fact]
    public void Default_config_selection_does_not_depend_on_build_configuration()
    {
        Assert.Equal(Path.Combine(AppContext.BaseDirectory, "config.default.yaml"), MagicQuantYamlLoader.ResolveConfigPath([]));
    }

    [Fact]
    public void Explicit_config_path_is_relative_to_working_directory()
    {
        Assert.Equal(Path.GetFullPath("configs/my campaign.yaml"), MagicQuantYamlLoader.ResolveConfigPath(
            [new CliArg { Name = "CONFIG", Value = "configs/my campaign.yaml" }]));
    }

    [Theory]
    [InlineData("src/MagicQuant/config.default.yaml")]
    [InlineData("examples/pipeline.yaml")]
    [InlineData("examples/clone.yaml")]
    public void Distributed_configs_have_no_unknown_keys(string relativePath)
    {
        string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
        var config = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build()
            .Deserialize<MagicQuantYamlConfig>(File.ReadAllText(Path.Combine(root, relativePath)));
        Assert.NotNull(config.Paths);
        Assert.NotNull(config.CandidateSelection);
    }
    [Fact]
    public void Legacy_inactive_yaml_remains_compatible_with_current_selection_settings()
    {
        const string yaml = """
            evolution:
              max_survival_rounds: 100
            survival:
              max_selected_choices_per_bucket: 50
            brain_layers: [embeddings]
            candidate_selection:
              max_fallback_attempts_per_anchor: 7
            """;
        var config = new DeserializerBuilder()
            .IgnoreUnmatchedProperties()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build()
            .Deserialize<MagicQuantYamlConfig>(yaml);
        Assert.Equal(7, config.CandidateSelection.MaxFallbackAttemptsPerAnchor);
    }

}
