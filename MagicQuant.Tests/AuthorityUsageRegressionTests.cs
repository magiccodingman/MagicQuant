using Xunit;

namespace MagicQuant.Tests;

public class AuthorityUsageRegressionTests
{
    [Fact]
    public void ComboGenerationPaths_DoNotUseLegacyAllAllowedHybridQuantsAuthority()
    {
        string repositoryRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../.."));
        var files = new[]
        {
            Path.Combine(repositoryRoot, "MagicQuant", "Helpers", "ComboLogic.cs"),
            Path.Combine(repositoryRoot, "MagicQuant", "Helpers", "TensorConfigGenerator.cs"),
            Path.Combine(repositoryRoot, "MagicQuant", "Services", "IsolationOptimizationService.cs")
        };

        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            Assert.DoesNotContain("All_Allowed_Hybrid_Quants", text);
        }
    }
}
