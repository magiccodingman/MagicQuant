using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class BenchmarkContractTests
{
    [Fact]
    public void Cpu_and_gpu_benchmark_arguments_preserve_runtime_policy()
    {
        var cpu = BenchmarkCommands.Bench("bench", "/models/a b.gguf", false, 99, "");
        Assert.Equal(["-m", "/models/a b.gguf", "-p", "8", "-t", "16", "-ngl", "0", "-o", "md"], cpu.Arguments);
        var gpu = BenchmarkCommands.Perplexity("ppl", "model", "corpus", true, 32, " --tensor-split 19,23", "logits", true);
        Assert.Equal(["-m", "model", "-ngl", "32", "--tensor-split", "19,23", "-t", "4", "-c", "2048", "--file", "corpus", "--kl-divergence-base", "logits", "--kl-divergence"], gpu.Arguments);
        Assert.DoesNotContain("--kl-divergence", BenchmarkCommands.Perplexity("ppl", "model", "corpus", false, 32, "", "logits", false).Arguments);
    }

    [Theory]
    [InlineData("PPL = 12.5 +/- 0.2\nMean KLD: 1.2e-3", 12.5, 0.0012)]
    [InlineData("\u001b[32mMean PPL(Q) : 8.1 ± 0.1\u001b[0m\nKL-divergence = 0.02", 8.1, 0.02)]
    public void Parses_plain_and_ansi_scientific_notation_logs(string content, double ppl, double kld)
    {
        WithLog(content, path =>
        {
            var metrics = BenchmarkLogParser.ParsePerplexity(path, false);
            Assert.Equal(ppl, metrics.Ppl);
            Assert.Equal(kld, metrics.Kld);
        });
    }

    [Fact]
    public void Missing_kld_is_allowed_only_for_native_reference_logs()
    {
        WithLog("PPL = 12.5 +/- 0.2", path =>
        {
            Assert.Null(BenchmarkLogParser.ParsePerplexity(path, true).Kld);
            Assert.Throws<InvalidOperationException>(() => BenchmarkLogParser.ParsePerplexity(path, false));
        });
        WithLog("process failed before measurement", path =>
            Assert.Throws<InvalidOperationException>(() => BenchmarkLogParser.ParsePerplexity(path, true)));
    }

    [Fact]
    public void Parses_benchmark_table_by_column_name()
    {
        WithLog("| test | backend | t/s | ngl |\n|---|---|---|---|\n| pp8 | CPU | 123.45 ± 0.1 | 0 |", path =>
        {
            var metrics = BenchmarkLogParser.ParseLlamaBench(path);
            Assert.Equal(123.45, metrics.Tps);
            Assert.Equal("CPU", metrics.Backend);
            Assert.Equal("pp8", metrics.Test);
        });
    }

    private static void WithLog(string text, Action<string> test)
    {
        string path = Path.GetTempFileName();
        try { File.WriteAllText(path, text); test(path); }
        finally { File.Delete(path); }
    }
}
