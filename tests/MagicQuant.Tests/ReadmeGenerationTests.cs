using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ReadmeGenerationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Exported_readmes_link_to_the_canonical_project(bool clone)
    {
        string root = Path.Combine(Path.GetTempPath(), "mq-readme-" + Guid.NewGuid().ToString("N"));
        try
        {
            var service = new ReadmeGenerationService();
            string file = clone
                ? await service.GenerateCloneAsync(root, "test", "owner/model", true, [])
                : await service.GenerateAsync(root, "test", [], []);
            string readme = await File.ReadAllTextAsync(file);
            Assert.Contains("[MagicQuant](https://github.com/magiccodingman/MagicQuant)", readme);
            Assert.DoesNotContain("magicquant-wiki", readme, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
