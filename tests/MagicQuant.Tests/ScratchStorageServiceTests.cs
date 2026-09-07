using MagicQuant.Services;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ScratchStorageServiceTests
{
    [Fact]
    public async Task EmptyScratchRoots_FallsBackToSingleWriterCapacity()
    {
        var temp = Path.Combine(Path.GetTempPath(), "mq-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);

        Cache.MagicQuantDirectory = temp;
        Cache.ModelDirectory = temp;
        Cache.ModelMagicQuantDirectory = Path.Combine(temp, "MagicQuant");
        Cache.CurrentModelId = "test-model";
        Cache.ScratchRoots = new List<string>();

        var paths = new ModelArtifactPathService();
        var service = new ScratchStorageService(paths);

        Assert.Equal(1, service.WriterCapacity);

        await service.CleanupStaleScratchArtifactsAsync();
    }

    [Fact]
    public async Task PreserveOutput_PreventsLeaseDeletion()
    {
        var temp = Path.Combine(Path.GetTempPath(), "mq-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);

        Cache.MagicQuantDirectory = temp;
        Cache.ModelDirectory = temp;
        Cache.ModelMagicQuantDirectory = Path.Combine(temp, "MagicQuant");
        Cache.CurrentModelId = "test-model";
        Cache.ScratchRoots = new List<string> { temp };

        var paths = new ModelArtifactPathService();
        var service = new ScratchStorageService(paths);

        var lease = await service.AcquireAsync(ScratchArtifactKind.Other, "artifact");
        await File.WriteAllTextAsync(lease.GgufPath, "x");
        lease.PreserveOutput();
        await lease.DisposeAsync();

        Assert.True(Directory.Exists(lease.LeaseDirectory));

        await service.CleanupStaleScratchArtifactsAsync();
    }
}
