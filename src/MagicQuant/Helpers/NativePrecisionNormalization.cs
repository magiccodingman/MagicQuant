using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class NativePrecisionNormalization
{
    public static string NormalizeLearnedFinalQuantTypeForApplication(string? observedFinalQuantType)
    {
        if (string.IsNullOrWhiteSpace(observedFinalQuantType))
            return string.Empty;

        var canonical = Canonicalize(observedFinalQuantType);
        var native = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        // Keep raw F32 as-is. Do not silently collapse F32 to BF16/F16.
        if (canonical == "F32")
            return TensorWeightScheme.F32.Names[0];

        // Treat F16/BF16 as "native high precision kept" when replaying
        // learned behavior into the active source model.
        if (canonical == "F16" || canonical == "BF16")
            return native.Names[0];

        return observedFinalQuantType.Trim();
    }

    public static IReadOnlyCollection<byte> ResolveSchemeIdsForLearnedFinalQuantType(string? observedFinalQuantType)
    {
        var result = new HashSet<byte>();

        if (string.IsNullOrWhiteSpace(observedFinalQuantType))
            return result;

        var canonical = Canonicalize(observedFinalQuantType);

        // Keep F32 exact if observed.
        if (canonical == "F32")
        {
            result.Add(TensorWeightScheme.F32.UniqueId);
            return result;
        }

        // Treat F16/BF16 as native high precision for pruning/application logic.
        if (canonical == "F16" || canonical == "BF16")
        {
            result.Add(TensorWeightScheme.GetCurrentNativePrecisionScheme().UniqueId);
            return result;
        }

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.Names.IsDefaultOrEmpty)
                continue;

            if (scheme.Names.Any(x => Canonicalize(x) == canonical))
                result.Add(scheme.UniqueId);
        }

        return result;
    }

    private static string Canonicalize(string value)
    {
        return value
            .Trim()
            .Replace("-", "_")
            .Replace(" ", string.Empty)
            .ToUpperInvariant();
    }
}
