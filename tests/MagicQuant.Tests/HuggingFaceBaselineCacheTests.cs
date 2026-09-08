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

    [Fact]
    public void StagingCleanupPath_MustRemainInsideDestinationDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "mq-hf-path-test", Guid.NewGuid().ToString("N"));
        string cache = Path.Combine(root, "ExternalBaselines");

        Assert.True(HuggingFaceBaselineService.IsPathInsideDirectory(
            Path.Combine(cache, "source.gguf"), cache));
        Assert.False(HuggingFaceBaselineService.IsPathInsideDirectory(
            Path.Combine(root, "outside.gguf"), cache));
    }

    [Fact]
    public void StagingCleanupPath_ResolvesSymlinkedParentDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "mq-hf-symlink-test-" + Guid.NewGuid().ToString("N"));
        string physical = Path.Combine(root, "physical-model");
        string alias = Path.Combine(root, "model-alias");
        string cache = Path.Combine(physical, "MagicQuant", "ExternalBaselines");

        try
        {
            Directory.CreateDirectory(cache);
            Directory.CreateSymbolicLink(alias, physical);

            string downloadedPath = Path.Combine(cache, "source.gguf");
            File.WriteAllBytes(downloadedPath, "GGUF-source-payload"u8.ToArray());

            string aliasedCache = Path.Combine(alias, "MagicQuant", "ExternalBaselines");
            Assert.True(HuggingFaceBaselineService.IsPathInsideDirectory(downloadedPath, aliasedCache));
            Assert.True(HuggingFaceBaselineService.PathsReferToSameLocation(
                downloadedPath,
                Path.Combine(aliasedCache, "source.gguf")));
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
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
