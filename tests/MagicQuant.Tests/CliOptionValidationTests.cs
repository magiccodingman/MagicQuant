using MagicQuant.Configuration;
using MagicQuant.Models;
using Xunit;

namespace MagicQuant.Tests;

public sealed class CliOptionValidationTests
{
    [Theory]
    [InlineData("modle-dir", "/model")]
    [InlineData("model-dir", "")]
    [InlineData("use-imatrix", "false")]
    [InlineData("config", "")]
    [InlineData("prediction-minimum-fit-rows", "one")]
    [InlineData("prediction-minimum-fit-rows", "1")]
    [InlineData("prediction-default-bit-stress-threshold", "NaN")]
    [InlineData("selection-interior-window-fractions", "0.3,broken")]
    [InlineData("selection-diversify-validation-candidates", "maybe")]
    public void Rejects_typos_missing_values_and_ambiguous_flags(string name, string value)
    {
        Assert.Throws<ArgumentException>(() => CliOptionValidator.Validate([new CliArg { Name = name, Value = value }]));
    }

    [Fact]
    public void Duplicate_options_are_not_silently_resolved_by_order()
    {
        Assert.Throws<ArgumentException>(() => CliOptionValidator.Validate(
            [new CliArg { Name = "model-dir", Value = "one" }, new CliArg { Name = "MODEL-DIR", Value = "two" }]));
    }
}
