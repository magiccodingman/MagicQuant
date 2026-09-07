using MagicQuant.Helpers;
using MagicQuant.Services;
using MQ.DB;
using Xunit;

namespace MagicQuant.Tests;

public sealed class CombinationDatabasePathTests
{
    [Theory]
    [InlineData(false, null, false, "no-imatrix_hp-off")]
    [InlineData(true, "abc123", false, "abc123_hp-off")]
    [InlineData(true, null, true, "imatrix-unknown_hp-on")]
    public void Reader_path_preserves_context_filename(bool imatrix, string? hash, bool highPrecision, string suffix)
    {
        var previous = (Cache.ModelMagicQuantDirectory, Cache.MagicQuantDirectory, Cache.CurrentModelId,
            Cache.IsImatrixAvailable, Cache.ActiveImatrixIdentityHash, RuntimeSearchSpace.AllowHighPrecisionHybrids);
        try
        {
            Cache.ModelMagicQuantDirectory = Path.Combine(Path.GetTempPath(), "model", "MagicQuant");
            Cache.MagicQuantDirectory = Path.Combine(Path.GetTempPath(), "shared");
            Cache.CurrentModelId = "model123";
            Cache.IsImatrixAvailable = imatrix;
            Cache.ActiveImatrixIdentityHash = hash;
            RuntimeSearchSpace.AllowHighPrecisionHybrids = highPrecision;
            string expected = Path.Combine(Cache.ModelMagicQuantDirectory, $"MagicQuant_Combinations_model123_{suffix}.duckdb");
            Assert.Equal(expected, CombinationDatabasePathService.GetPath());
            Assert.Equal(expected, new RemainingCombinationStore().GetDatabaseFilePath());
            Cache.ModelMagicQuantDirectory = null;
            Assert.Equal(Cache.MagicQuantDirectory, CombinationDatabasePathService.GetDirectory());
            Cache.MagicQuantDirectory = null;
            Assert.Throws<InvalidOperationException>(() => CombinationDatabasePathService.GetPath());
        }
        finally
        {
            (Cache.ModelMagicQuantDirectory, Cache.MagicQuantDirectory, Cache.CurrentModelId,
                Cache.IsImatrixAvailable, Cache.ActiveImatrixIdentityHash, RuntimeSearchSpace.AllowHighPrecisionHybrids) = previous;
        }
    }
}
