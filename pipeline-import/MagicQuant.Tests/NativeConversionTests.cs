using System.Diagnostics;
using MagicQuant.Helpers;
using MagicQuant.Runtime;
using MagicQuant.Services;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

public sealed class NativeConversionTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(7)]
    [InlineData(-1)]
    public async Task Only_completed_conversion_is_reusable(int exitCode)
    {
        string root = Path.Combine(Path.GetTempPath(), $"mq-conversion-{Guid.NewGuid():N}");
        var old = (Cache.ModelDirectory, Cache.ModelMagicQuantDirectory, Cache.ConvertScript, Cache.TorchType);
        try
        {
            Cache.ModelDirectory = Path.Combine(root, "source with spaces");
            Cache.ModelMagicQuantDirectory = Path.Combine(Cache.ModelDirectory, "MagicQuant");
            Cache.ConvertScript = Path.Combine(root, "converter with spaces.py");
            Cache.TorchType = Cache.MainTorchType.BF16;
            var paths = new ModelArtifactPathService();
            Directory.CreateDirectory(paths.GgufDir);
            var runner = new ConverterStub(exitCode);
            var converter = new NativeModelConversionService(paths, new PythonManager(root), runner);
            string native = paths.GetNativeBaseGgufPath();
            if (exitCode == 0)
            {
                Assert.Equal(native, await converter.EnsureAsync());
                Assert.Equal(native, await converter.EnsureAsync());
                Assert.Equal(1, runner.Calls);
                // A stale marker cannot make a truncated artifact look complete.
                File.WriteAllText(native, "");
                await converter.EnsureAsync();
                Assert.Equal(2, runner.Calls);
            }
            else
            {
                await Assert.ThrowsAnyAsync<Exception>(() => converter.EnsureAsync());
                Assert.False(File.Exists(native));
                Assert.False(File.Exists(native + ".success.json"));
            }
        }
        finally
        {
            (Cache.ModelDirectory, Cache.ModelMagicQuantDirectory, Cache.ConvertScript, Cache.TorchType) = old;
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    private sealed class ConverterStub(int exitCode) : IProcessRunner
    {
        public int Calls { get; private set; }
        public Task<ProcessResult> RunAsync(ProcessStartInfo start, string? logPath = null, Action<string, bool>? onLine = null, CancellationToken ct = default)
        {
            Calls++;
            Assert.Equal(Cache.ConvertScript, start.ArgumentList[0]);
            Assert.Equal(Cache.ModelDirectory, start.ArgumentList[1]);
            int outputIndex = start.ArgumentList.IndexOf("--outfile") + 1;
            File.WriteAllText(start.ArgumentList[outputIndex], "GGUF fixture");
            if (exitCode == -1) throw new OperationCanceledException();
            return Task.FromResult(new ProcessResult(exitCode, "", ""));
        }
    }
}
