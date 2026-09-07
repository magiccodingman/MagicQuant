using System.Reflection;
using MagicQuant.Helpers;
using Xunit;

namespace MagicQuant.Tests;

public class RuntimeSearchSpaceObsoleteContractTests
{
    [Fact]
    public void AllowHighPrecisionHybrids_IsNotObsolete()
    {
        var prop = typeof(RuntimeSearchSpace).GetProperty(nameof(RuntimeSearchSpace.AllowHighPrecisionHybrids), BindingFlags.Public | BindingFlags.Static);
        Assert.NotNull(prop);
        Assert.Empty(prop!.GetCustomAttributes(typeof(ObsoleteAttribute), inherit: false));

        RuntimeSearchSpace.AllowHighPrecisionHybrids = true;
        Assert.True(RuntimeSearchSpace.AllowHighPrecisionHybrids);
    }

    [Fact]
    public void LegacySchemeWrappers_AreObsolete()
    {
        var wrappers = new[]
        {
            nameof(RuntimeSearchSpace.BanSchemeForGroup),
            nameof(RuntimeSearchSpace.BanSchemeForGroupByLearnedBaselineAbsence),
            nameof(RuntimeSearchSpace.BanAllExplicitTensorSchemesForGroup),
            nameof(RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup),
            nameof(RuntimeSearchSpace.IsGroupExplicitQuantBanned)
        };

        foreach (var methodName in wrappers)
        {
            var method = typeof(RuntimeSearchSpace).GetMethod(methodName, BindingFlags.Public | BindingFlags.Static);
            Assert.NotNull(method);
            Assert.NotEmpty(method!.GetCustomAttributes(typeof(ObsoleteAttribute), inherit: false));
        }
    }
}
