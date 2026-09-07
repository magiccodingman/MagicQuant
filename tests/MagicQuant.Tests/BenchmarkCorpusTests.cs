using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public class BenchmarkCorpusTests
{
    [Fact]
    public void DatasetIds_AreCanonicalNamespacedIds()
    {
        Assert.Equal("Salesforce/wikitext", BenchmarkService.GeneralPplDatasetId);
        Assert.Equal("openai/gsm8k", BenchmarkService.MathPplDatasetId);
    }

    [Fact]
    public void IsPplCorpusUsable_RequiresTheFullCharacterTarget()
    {
        string path = Path.GetTempFileName();

        try
        {
            File.WriteAllText(path, new string('x', 31));
            Assert.False(BenchmarkService.IsPplCorpusUsable(path, tokenTarget: 8));

            File.AppendAllText(path, "x");
            Assert.True(BenchmarkService.IsPplCorpusUsable(path, tokenTarget: 8));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
