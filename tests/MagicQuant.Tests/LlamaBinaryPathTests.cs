using MQ.DB;
using MQ.DB.Models;
using Xunit;

namespace MagicQuant.Tests;

public sealed class LlamaBinaryPathTests
{
    [Fact]
    public void Explicit_binary_directory_wins_and_checkout_root_is_a_real_fallback()
    {
        string? old = Cache.LlamaBin;
        try
        {
            string suffix = OperatingSystem.IsWindows() ? ".exe" : "";
            Cache.LlamaBin = Path.Combine(Path.GetTempPath(), "custom bin");
            Assert.Equal(Path.Combine(Cache.LlamaBin, "llama-bench" + suffix), new LlamaBinaries("ignored").Bench);
            Cache.LlamaBin = null;
            string root = Path.Combine(Path.GetTempPath(), "llama");
            Assert.Equal(Path.Combine(root, "build", "bin", "llama-cli" + suffix), new LlamaBinaries(root).Cli);
            Assert.Throws<InvalidOperationException>(() => new LlamaBinaries(null));
        }
        finally { Cache.LlamaBin = old; }
    }
}
