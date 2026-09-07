using MagicQuant.Helpers;
using Xunit;

namespace MagicQuant.Tests;

public class CliArgumentParsingTests
{
    [Fact]
    public void ArgvParser_PreservesHyphensInOrdinaryShellValues()
    {
        var parsed = CliHelpers.ParseArguments([
            "--config", "/repo/MagicQuant-Pipeline/config.dev.yaml",
            "--architecture-family", "Qwen3.8-27B",
            "--recheck-hardware-probe"
        ]);

        Assert.Equal("/repo/MagicQuant-Pipeline/config.dev.yaml", parsed[0].Value);
        Assert.Equal("Qwen3.8-27B", parsed[1].Value);
        Assert.Equal(string.Empty, parsed[2].Value);
    }

    [Fact]
    public void ArgvParser_SupportsEqualsAndLegacyLiteralQuotes()
    {
        var parsed = CliHelpers.ParseArguments([
            "--model-dir=/models/Qwen-27B",
            "--output-dir", "\"/models/Agent-Run\""
        ]);

        Assert.Equal("/models/Qwen-27B", parsed[0].Value);
        Assert.Equal("/models/Agent-Run", parsed[1].Value);
    }
}
