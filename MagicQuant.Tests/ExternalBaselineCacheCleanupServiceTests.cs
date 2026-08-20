using MagicQuant.Services;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ExternalBaselineCacheCleanupServiceTests
{
    [Fact]
    public async Task CleanupStaleArtifactsAsync_HardDeletesCacheTree()
    {
        string temp = Path.Combine(Path.GetTempPath(), "mq-external-cache-test-" + Guid.NewGuid().ToString("N"));
        string modelRoot = Path.Combine(temp, "MagicQuant");
        string cacheRoot = Path.Combine(modelRoot, "ExternalBaselines");

        Directory.CreateDirectory(Path.Combine(cacheRoot, ".cache", "huggingface"));
        await File.WriteAllTextAsync(Path.Combine(cacheRoot, "baseline.gguf"), "GGUF");
        await File.WriteAllTextAsync(Path.Combine(cacheRoot, ".cache", "huggingface", "metadata"), "x");

        string? priorModelMagicQuantDirectory = Cache.ModelMagicQuantDirectory;
        string? priorExternalBaselineCacheDirectory = Cache.ExternalBaselineCacheDirectory;

        try
        {
            Cache.ModelMagicQuantDirectory = modelRoot;
            Cache.ExternalBaselineCacheDirectory = cacheRoot;

            bool cleaned = await new ExternalBaselineCacheCleanupService().CleanupStaleArtifactsAsync();

            Assert.True(cleaned);
            Assert.False(Directory.Exists(cacheRoot));
        }
        finally
        {
            Cache.ModelMagicQuantDirectory = priorModelMagicQuantDirectory;
            Cache.ExternalBaselineCacheDirectory = priorExternalBaselineCacheDirectory;

            if (Directory.Exists(temp))
                Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public void ValidateCleanupRoot_RejectsPathOutsideModelWorkDirectory()
    {
        string temp = Path.Combine(Path.GetTempPath(), "mq-external-cache-test-" + Guid.NewGuid().ToString("N"));
        string modelRoot = Path.Combine(temp, "model", "MagicQuant");
        string outsideRoot = Path.Combine(temp, "outside");

        var ex = Assert.Throws<InvalidOperationException>(() =>
            ExternalBaselineCacheCleanupService.ValidateCleanupRoot(outsideRoot, modelRoot));

        Assert.Contains("unsafe external baseline cache path", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ValidateCleanupRoot_RejectsModelWorkDirectoryItself()
    {
        string modelRoot = Path.Combine(Path.GetTempPath(), "mq-external-cache-test-" + Guid.NewGuid().ToString("N"));

        Assert.Throws<InvalidOperationException>(() =>
            ExternalBaselineCacheCleanupService.ValidateCleanupRoot(modelRoot, modelRoot));
    }
}
