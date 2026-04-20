using MagicQuant.Services;
using MQ.DB;
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
}
