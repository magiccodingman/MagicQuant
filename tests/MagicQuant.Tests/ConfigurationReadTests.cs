using System.Globalization;
using MagicQuant.Configuration;
using MagicQuant.Models;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ConfigurationReadTests
{
    [Fact]
    public void Read_does_not_mutate_runtime_state_and_cli_decimal_is_culture_independent()
    {
        string file = Path.GetTempFileName();
        var oldCulture = CultureInfo.CurrentCulture;
        string? oldRoot = Cache.MagicQuantDirectory;
        var oldConfig = Config.Current;
        try
        {
            File.WriteAllText(file, "prediction:\n  default_bit_stress_threshold: 5.0\n");
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
            var loaded = MagicQuantYamlLoader.Read([
                new CliArg { Name = "config", Value = file },
                new CliArg { Name = "prediction-default-bit-stress-threshold", Value = "7.5" }
            ]);
            Assert.Equal(7.5, loaded.Settings.Prediction.DefaultBitStressThreshold);
            Assert.Same(oldConfig, Config.Current);
            Assert.Equal(oldRoot, Cache.MagicQuantDirectory);
        }
        finally { CultureInfo.CurrentCulture = oldCulture; File.Delete(file); }
    }

    [Fact]
    public void Strict_config_rejects_typos_that_normal_mode_reports()
    {
        string file = Path.GetTempFileName();
        try
        {
            File.WriteAllText(file, "paths:\n  modle_dir: /model\n");
            var arg = new CliArg { Name = "config", Value = file };
            Assert.Single(MagicQuantYamlLoader.Read([arg]).Warnings);
            Assert.Throws<InvalidOperationException>(() => MagicQuantYamlLoader.Read([arg, new CliArg { Name = "strict-config", Value = "" }]));
        }
        finally { File.Delete(file); }
    }
}
