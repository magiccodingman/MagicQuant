using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public class LlamaGpuArgumentBuilderTests
{
    private static readonly IReadOnlyDictionary<int, double> Limits =
        new Dictionary<int, double>
        {
            [0] = 19,
            [1] = 23
        };

    [Fact]
    public void CommonCli_UsesCommaSeparatedMembers()
    {
        string args = LlamaGpuArgumentBuilder.BuildTensorSplitArgs(
            [0, 1], Limits, LlamaGpuTool.CommonCli);

        Assert.Equal(" --tensor-split 19,23", args);
    }

    [Fact]
    public void LlamaBench_UsesSlashSeparatedMembers()
    {
        string args = LlamaGpuArgumentBuilder.BuildTensorSplitArgs(
            [0, 1], Limits, LlamaGpuTool.LlamaBench);

        Assert.Equal(" --tensor-split 19/23", args);
    }

    [Fact]
    public void SingleGpu_DoesNotEmitTensorSplit()
    {
        string args = LlamaGpuArgumentBuilder.BuildTensorSplitArgs(
            [1], Limits, LlamaGpuTool.CommonCli);

        Assert.Equal(string.Empty, args);
    }

    [Fact]
    public void MissingConfiguredLimit_IsRejected()
    {
        var error = Assert.Throws<InvalidOperationException>(() =>
            LlamaGpuArgumentBuilder.BuildTensorSplitArgs(
                [0, 2], Limits, LlamaGpuTool.CommonCli));

        Assert.Contains("2", error.Message, StringComparison.Ordinal);
    }
}
