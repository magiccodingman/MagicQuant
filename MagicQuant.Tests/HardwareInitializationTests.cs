using MagicQuant.Commands;
using MagicQuant.Models;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class HardwareInitializationCollection
{
    public const string Name = "Hardware initialization";
}

[Collection(HardwareInitializationCollection.Name)]
public sealed class HardwareInitializationTests
{
    [Fact]
    public async Task Custom_environment_validation_populates_system_info()
    {
        string testRoot = Path.Combine(
            Path.GetTempPath(),
            $"magicquant-hardware-init-{Guid.NewGuid():N}");
        string llamaRoot = Path.Combine(testRoot, "llama.cpp");
        string llamaBin = Path.Combine(llamaRoot, "build", "bin");
        string convertScript = Path.Combine(llamaRoot, "convert_hf_to_gguf.py");
        var previous = Cache.SysInfo;

        try
        {
            Directory.CreateDirectory(llamaBin);
            await File.WriteAllTextAsync(convertScript, "# test");
            Cache.SysInfo = null;

            await new InitializeLlamaCpp().Run(
            [
                new CliArg { Name = "validate", Value = string.Empty },
                new CliArg { Name = "llama-root", Value = llamaRoot },
                new CliArg { Name = "llama-bin", Value = llamaBin },
                new CliArg { Name = "convert-script", Value = convertScript }
            ]);

            Assert.NotNull(Cache.SysInfo);
            Assert.True(Cache.SysInfo.ThreadCount > 0);
            Assert.True(Cache.SysInfo.RamGb > 0);
        }
        finally
        {
            Cache.SysInfo = previous;
            if (Directory.Exists(testRoot))
                Directory.Delete(testRoot, recursive: true);
        }
    }
}
