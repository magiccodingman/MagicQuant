using MagicQuant.Commands;
using MagicQuant.Configuration;
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
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Custom_environment_validation_populates_system_info(bool useYamlPaths)
    {
        string testRoot = Path.Combine(
            Path.GetTempPath(),
            $"magicquant-hardware-init-{Guid.NewGuid():N}");
        string llamaRoot = Path.Combine(testRoot, "llama.cpp");
        string llamaBin = Path.Combine(llamaRoot, "build", "bin");
        string convertScript = Path.Combine(llamaRoot, "convert_hf_to_gguf.py");
        var previous = Cache.SysInfo;
        var previousPaths = (Cache.LlamaRoot, Cache.LlamaBin, Cache.ConvertScript);
        var previousConfig = Config.Current;

        try
        {
            Directory.CreateDirectory(llamaBin);
            await File.WriteAllTextAsync(convertScript, "# test");
            Cache.SysInfo = null;

            var config = MagicQuantYamlConfig.CreateDefault();
            Config.Load(config);
            List<CliArg> args = [new() { Name = "validate", Value = string.Empty }];
            if (useYamlPaths)
            {
                config.Paths.LlamaRoot = llamaRoot;
                config.Paths.LlamaBin = llamaBin;
                config.Paths.ConvertScript = convertScript;
            }
            else
            {
                args.AddRange([
                    new CliArg { Name = "llama-root", Value = llamaRoot },
                    new CliArg { Name = "llama-bin", Value = llamaBin },
                    new CliArg { Name = "convert-script", Value = convertScript }
                ]);
            }
            await new InitializeLlamaCpp().Run(args);

            Assert.Equal(llamaRoot, Cache.LlamaRoot);
            Assert.Equal(llamaBin, Cache.LlamaBin);
            Assert.Equal(convertScript, Cache.ConvertScript);

            Assert.NotNull(Cache.SysInfo);
            Assert.True(Cache.SysInfo.ThreadCount > 0);
            Assert.True(Cache.SysInfo.RamGb > 0);
        }
        finally
        {
            Cache.SysInfo = previous;
            (Cache.LlamaRoot, Cache.LlamaBin, Cache.ConvertScript) = previousPaths;
            Config.Load(previousConfig);
            if (Directory.Exists(testRoot))
                Directory.Delete(testRoot, recursive: true);
        }
    }
    [Fact]
    public async Task Partial_custom_paths_fail_before_setup()
    {
        var previous = Config.Current;
        try
        {
            Config.Load(MagicQuantYamlConfig.CreateDefault());
            await Assert.ThrowsAsync<ArgumentException>(() => new InitializeLlamaCpp().Run(
                [new CliArg { Name = "llama-root", Value = "/missing/llama.cpp" }]));
            await Assert.ThrowsAsync<ArgumentException>(() => new InitializeLlamaCpp().Run(
                [new CliArg { Name = "llama-bin", Value = "/missing/bin" }]));
        }
        finally
        {
            Config.Load(previous);
        }
    }

}
