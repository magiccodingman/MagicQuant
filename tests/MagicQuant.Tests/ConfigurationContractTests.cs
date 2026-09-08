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
    [InlineData("examples/pipeline-external.yaml")]
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
    [Theory]
    [InlineData("README.md")]
    [InlineData("docs/best-practices.md")]
    public void Onboarding_yaml_examples_match_the_configuration_contract(string relativePath)
    {
        string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
        string markdown = File.ReadAllText(Path.Combine(root, relativePath));
        var snippets = System.Text.RegularExpressions.Regex.Matches(markdown, @"```yaml\r?\n(.*?)```",
            System.Text.RegularExpressions.RegexOptions.Singleline);
        Assert.NotEmpty(snippets);
        foreach (System.Text.RegularExpressions.Match snippet in snippets)
        {
            string yaml = snippet.Groups[1].Value;
            Assert.Empty(YamlConfigurationDiagnostics.Inspect(yaml));
            var config = new DeserializerBuilder().WithNamingConvention(UnderscoredNamingConvention.Instance)
                .Build().Deserialize<MagicQuantYamlConfig>(yaml);
            ConfigurationShapeValidator.Validate(config);
        }
    }

    [Fact]
    public void Starter_and_typed_defaults_do_not_select_a_model_or_external_provider()
    {
        var yaml = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "config.default.yaml"));
        var starter = new DeserializerBuilder().WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build().Deserialize<MagicQuantYamlConfig>(yaml);
        Assert.DoesNotContain("Qwen", yaml, StringComparison.OrdinalIgnoreCase);
        foreach (var config in new[] { starter, MagicQuantYamlConfig.CreateDefault() })
        {
            Assert.True(string.IsNullOrWhiteSpace(config.Paths.ModelDir));
            Assert.True(string.IsNullOrWhiteSpace(config.Paths.MagicQuantRoot));
            Assert.True(string.IsNullOrWhiteSpace(config.Identity.ArchitectureFamilyName));
            Assert.True(string.IsNullOrWhiteSpace(config.Readme.TitleModelNameOverride));
            Assert.True(string.IsNullOrWhiteSpace(config.Imatrix.DatasetRepo));
            Assert.Empty(config.Paths.ScratchRoots);
            Assert.Empty(config.Hardware.GpuMemoryLimitsGb);
            Assert.Empty(config.Baselines.CustomRepositories);
            Assert.False(config.Output.ExportExternalLearnedBaselines);
            Assert.False(config.Learning.ForceRelearnArchitectureFamily);
            Assert.True(config.Learning.ConfirmTensorGroupProfile);
            Assert.Equal(new[] { "Q6_K", "Q5_K" }, config.AnomalyDetection.ConfirmedAnomalyExpansion.AllowedCandidateQuants);
            Assert.False(config.Readme.Frontmatter.ContainsKey("license"));
            Assert.False(config.Readme.Frontmatter.ContainsKey("base_model"));
        }
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
