using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;
using Xunit;

namespace MagicQuant.Tests;

public class ImatrixIdentityServiceTests
{
    [Fact]
    public async Task EnsureActiveImatrixIdentityHash_IsStableForSameArtifact()
    {
        string temp = Path.GetTempFileName();
        await File.WriteAllTextAsync(temp, "imatrix-test-content");

        Cache.IsImatrixAvailable = true;
        Cache.ActiveImatrixPath = temp;
        Cache.ActiveImatrixIdentityHash = null;

        var first = await ImatrixIdentityService.EnsureActiveImatrixIdentityHashAsync();
        Cache.ActiveImatrixIdentityHash = null;
        var second = await ImatrixIdentityService.EnsureActiveImatrixIdentityHashAsync();

        Assert.False(string.IsNullOrWhiteSpace(first));
        Assert.Equal(first, second);

        File.Delete(temp);
    }

    [Fact]
    public async Task ResolveCurrentImatrixDefinitionId_SameModelSameImatrix_ReusesSameRow()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "mq-imatrix-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        Cache.MagicQuantDirectory = tempRoot;
        Cache.CurrentModelId = "model-test-same";

        string tempImatrix = Path.Combine(tempRoot, "imatrix.dat");
        await File.WriteAllTextAsync(tempImatrix, "same-imatrix");

        Cache.IsImatrixAvailable = true;
        Cache.ActiveImatrixPath = tempImatrix;
        Cache.ActiveImatrixIdentityHash = null;

        await using var db = new MagicQuantContext();
        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId);
        if (model == null)
        {
            model = new AiModelHash { UniqueHash = Cache.CurrentModelId };
            db.AiModelHashes.Add(model);
            await db.SaveChangesAsync();
        }

        var first = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, model.Id, createIfMissing: true);
        var second = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, model.Id, createIfMissing: true);

        Assert.NotNull(first);
        Assert.Equal(first, second);
    }

    [Fact]
    public async Task ResolveCurrentImatrixDefinitionId_NoImatrix_ReturnsNull()
    {
        string tempRoot = Path.Combine(Path.GetTempPath(), "mq-imatrix-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempRoot);

        Cache.MagicQuantDirectory = tempRoot;
        Cache.CurrentModelId = "model-test-none";
        Cache.IsImatrixAvailable = false;
        Cache.ActiveImatrixPath = null;
        Cache.ActiveImatrixIdentityHash = null;

        await using var db = new MagicQuantContext();
        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId);
        if (model == null)
        {
            model = new AiModelHash { UniqueHash = Cache.CurrentModelId };
            db.AiModelHashes.Add(model);
            await db.SaveChangesAsync();
        }

        var id = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, model.Id, createIfMissing: false);
        Assert.Null(id);
    }
}
