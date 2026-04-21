using Xunit;

namespace MagicQuant.Tests;

public class AuthorityUsageRegressionTests
{
    [Fact]
    public void ComboGenerationPaths_DoNotUseLegacyAllAllowedHybridQuantsAuthority()
    {
        var files = new[]
        {
            Path.Combine("..", "MagicQuant", "Helpers", "ComboLogic.cs"),
            Path.Combine("..", "MagicQuant", "Helpers", "TensorConfigGenerator.cs"),
            Path.Combine("..", "MagicQuant", "Services", "IsolationOptimizationService.cs")
        };

        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            Assert.DoesNotContain("All_Allowed_Hybrid_Quants", text);
        }
    }
}
