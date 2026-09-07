using System.Text.Json;
using MagicQuant.Configuration;
using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class RunProvenanceTests
{
    [Theory]
    [InlineData("completed")]
    [InlineData("failed")]
    [InlineData("canceled")]
    public void Records_immutable_inputs_and_terminal_status_atomically(string status)
    {
        string root = Path.Combine(Path.GetTempPath(), $"mq-provenance-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            string configPath = Path.Combine(root, "config.yaml");
            File.WriteAllText(configPath, "paths: {}");
            var config = MagicQuantYamlConfig.CreateDefault();
            config.Paths.MagicQuantRoot = root;
            var provenance = new RunProvenanceService("initialize-llama-cpp", ["initialize-llama-cpp"], new(configPath, config, []));
            config.Output.OutputNamePrefix = "changed-after-start";
            provenance.Complete(status);
            using var json = JsonDocument.Parse(File.ReadAllText(provenance.ManifestPath));
            Assert.Equal(status, json.RootElement.GetProperty("status").GetString());
            Assert.Equal("Model", json.RootElement.GetProperty("configuration").GetProperty("Output").GetProperty("OutputNamePrefix").GetString());
            Assert.Equal(64, json.RootElement.GetProperty("configSha256").GetString()!.Length);
            Assert.False(File.Exists(provenance.ManifestPath + ".tmp"));
        }
        finally { Directory.Delete(root, true); }
    }
}
