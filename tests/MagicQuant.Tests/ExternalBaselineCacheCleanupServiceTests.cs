using MagicQuant.Services;
using MagicQuant.Configuration;
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
        var priorConfig = Config.Current;

        try
        {
            Config.Load(MagicQuantYamlConfig.CreateDefault());
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
            Config.Load(priorConfig);

            if (Directory.Exists(temp))
                Directory.Delete(temp, recursive: true);
        }
    }

    [Fact]
    public async Task CleanupStaleArtifactsAsync_PreservesCompletedAndInProgressResumableDownloads()
    {
        string temp = Path.Combine(Path.GetTempPath(), "mq-external-cache-resume-test-" + Guid.NewGuid().ToString("N"));
        string modelRoot = Path.Combine(temp, "MagicQuant");
        string cacheRoot = Path.Combine(modelRoot, "ExternalBaselines");
        string hubCache = Path.Combine(cacheRoot, ".cache", "huggingface");
        string completed = Path.Combine(cacheRoot, "baseline.gguf");
        string incomplete = Path.Combine(hubCache, "baseline.gguf.incomplete");
        string transient = Path.Combine(cacheRoot, "baseline.gguf.partial.interrupted");

        Directory.CreateDirectory(hubCache);
        await File.WriteAllTextAsync(completed, "GGUF-complete");
        await File.WriteAllTextAsync(incomplete, "partial-download");
        await File.WriteAllTextAsync(transient, "partial-copy");

        string? priorModelMagicQuantDirectory = Cache.ModelMagicQuantDirectory;
        string? priorExternalBaselineCacheDirectory = Cache.ExternalBaselineCacheDirectory;
        var priorConfig = Config.Current;

        try
        {
            var config = MagicQuantYamlConfig.CreateDefault();
            config.Baselines.CustomRepositories.Add(new CustomBaselineRepositoryConfig
            {
                RepoId = "owner/model",
                Enabled = true,
                ResumeOrRetryDownloads = true
            });
            Config.Load(config);
            Cache.ModelMagicQuantDirectory = modelRoot;
            Cache.ExternalBaselineCacheDirectory = cacheRoot;

            bool cleaned = await new ExternalBaselineCacheCleanupService().CleanupStaleArtifactsAsync();

            Assert.True(cleaned);
            Assert.True(File.Exists(completed));
            Assert.True(File.Exists(incomplete));
            Assert.False(File.Exists(transient));
        }
        finally
        {
            Cache.ModelMagicQuantDirectory = priorModelMagicQuantDirectory;
            Cache.ExternalBaselineCacheDirectory = priorExternalBaselineCacheDirectory;
            Config.Load(priorConfig);

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
