using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class HuggingFaceBaselineCacheTests
{
    [Fact]
    public void MatchingGgufLengthAndTimestamp_IsReusable()
    {
        using var files = new TemporaryFiles();
        File.WriteAllBytes(files.Source, "GGUF-source-payload"u8.ToArray());
        File.Copy(files.Source, files.Destination);
        File.SetLastWriteTimeUtc(files.Destination, File.GetLastWriteTimeUtc(files.Source));

        Assert.True(HuggingFaceBaselineService.CanReuseDownloadedFile(files.Source, files.Destination));
    }

    [Fact]
    public void TruncatedDestination_IsNotReusable()
    {
        using var files = new TemporaryFiles();
        File.WriteAllBytes(files.Source, "GGUF-complete-payload"u8.ToArray());
        File.WriteAllBytes(files.Destination, "GGUF-partial"u8.ToArray());
        File.SetLastWriteTimeUtc(files.Destination, File.GetLastWriteTimeUtc(files.Source));

        Assert.False(HuggingFaceBaselineService.CanReuseDownloadedFile(files.Source, files.Destination));
    }

    [Fact]
    public void SameSizeButDifferentTimestamp_IsNotReusable()
    {
        using var files = new TemporaryFiles();
        File.WriteAllBytes(files.Source, "GGUF-source-payload"u8.ToArray());
        File.Copy(files.Source, files.Destination);
        File.SetLastWriteTimeUtc(files.Destination, File.GetLastWriteTimeUtc(files.Source).AddSeconds(-1));

        Assert.False(HuggingFaceBaselineService.CanReuseDownloadedFile(files.Source, files.Destination));
    }

    [Fact]
    public void InvalidGgufMagic_IsNotReusable()
    {
        using var files = new TemporaryFiles();
        File.WriteAllBytes(files.Source, "NOPE-source-payload"u8.ToArray());
        File.Copy(files.Source, files.Destination);
        File.SetLastWriteTimeUtc(files.Destination, File.GetLastWriteTimeUtc(files.Source));

        Assert.False(HuggingFaceBaselineService.CanReuseDownloadedFile(files.Source, files.Destination));
    }

    private sealed class TemporaryFiles : IDisposable
    {
        private readonly string _directory = Path.Combine(
            Path.GetTempPath(), "mq-hf-cache-test-" + Guid.NewGuid().ToString("N"));

        public TemporaryFiles()
        {
            Directory.CreateDirectory(_directory);
        }

        public string Source => Path.Combine(_directory, "source.gguf");
        public string Destination => Path.Combine(_directory, "destination.gguf");

        public void Dispose()
        {
            if (Directory.Exists(_directory))
                Directory.Delete(_directory, recursive: true);
        }
    }
}
