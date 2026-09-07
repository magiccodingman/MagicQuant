using MagicQuant.Configuration;
using Xunit;

namespace MagicQuant.Tests;

public sealed class YamlDiagnosticsTests
{
    [Fact]
    public void Unknown_nested_and_sequence_keys_report_the_full_setting_path()
    {
        var warnings = YamlConfigurationDiagnostics.Inspect("""
            paths:
              modle_dir: /models/test
            baselines:
              custom_repositories:
                - repo_id: owner/model
                  revison: main
            """);
        Assert.Contains(warnings, x => x.Contains("paths.modle_dir"));
        Assert.Contains(warnings, x => x.Contains("baselines.custom_repositories[0].revison"));
    }

    [Fact]
    public void Frontmatter_and_gpu_dictionary_keys_are_not_schema_properties()
    {
        Assert.Empty(YamlConfigurationDiagnostics.Inspect("""
            readme:
              frontmatter:
                arbitrary_metadata: [hello, world]
            hardware:
              gpu_memory_limits_gb:
                0: 12
            """));
    }

    [Fact]
    public void Removed_destructive_key_is_rejected_but_comments_are_not_options()
    {
        Assert.Throws<InvalidOperationException>(() => YamlConfigurationDiagnostics.Inspect("flags:\n  force_relearn_baseline_tensor_mappings: true"));
        Assert.Empty(YamlConfigurationDiagnostics.Inspect("# force_relearn_baseline_tensor_mappings was removed\npaths: {}"));
    }

    [Fact]
    public void Invalid_shape_and_nonfinite_numbers_fail_with_setting_names()
    {
        var config = MagicQuantYamlConfig.CreateDefault();
        config.Prediction = null!;
        Assert.Contains("prediction", Assert.Throws<InvalidOperationException>(() => ConfigurationShapeValidator.Validate(config)).Message);
        config.Prediction = new RuntimePredictionConfig { DefaultBitStressThreshold = double.NaN };
        Assert.Contains("default_bit_stress_threshold", Assert.Throws<InvalidOperationException>(() => ConfigurationShapeValidator.Validate(config)).Message);
    }
}
