using System.Collections.Immutable;

namespace MQ.DB.Models;

public record BaselineQuants(
    byte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    ImmutableArray<TensorWeightScheme> TensorWeightSchemes,
    bool IsPureBaselineCandidate = true,
    bool IsCombinationCarrierCandidate = true,
    bool IsExplicitGroupCombinationCandidate = true,
    bool IsHighPrecisionExplicitCandidate = false)
{
    public const byte NativeSourceUniqueId = 250;

    public TensorWeightScheme? DefaultTensorScheme =>
        TensorWeightSchemes.IsDefaultOrEmpty ? null : TensorWeightSchemes[0];

    public static readonly BaselineQuants Q8_0 =
        new(0, false, ["Q8_0"], [TensorWeightScheme.Q8_0], IsCombinationCarrierCandidate: true);

    public static readonly BaselineQuants Q6_K =
        new(1, false, ["Q6_K"], [TensorWeightScheme.Q6_K]);

    public static readonly BaselineQuants Q5_K =
        new(2, false, ["Q5_K"], [TensorWeightScheme.Q5_K]);

    public static readonly BaselineQuants Q4_K_M =
        new(3, false, ["Q4_K_M"], [TensorWeightScheme.Q4_K]);

    public static readonly BaselineQuants IQ4_NL =
        new(5, false, ["IQ4_NL"], [TensorWeightScheme.IQ4_NL]);

    public static readonly BaselineQuants IQ4_XS =
        new(6, false, ["IQ4_XS"], [TensorWeightScheme.IQ4_XS]);

    public static readonly BaselineQuants IQ3_S =
        new(7, true, ["IQ3_S"], [TensorWeightScheme.IQ3_S]);

    public static readonly BaselineQuants IQ3_XS =
        new(8, true, ["IQ3_XS"], [TensorWeightScheme.IQ3_XS]);

    public static readonly BaselineQuants IQ3_XXS =
        new(9, true, ["IQ3_XXS"], [TensorWeightScheme.IQ3_XXS]);

    public static readonly BaselineQuants IQ2_S =
        new(10, true, ["IQ2_S"], [TensorWeightScheme.IQ2_S]);

    public static readonly BaselineQuants IQ2_XS =
        new(11, true, ["IQ2_XS"], [TensorWeightScheme.IQ2_XS]);

    public static readonly BaselineQuants IQ2_XXS =
        new(12, true, ["IQ2_XXS"], [TensorWeightScheme.IQ2_XXS]);

    public static readonly BaselineQuants BF16_Hybrid =
        new(201, false, ["BF16"], [TensorWeightScheme.BF16],
            IsCombinationCarrierCandidate: false,
            IsHighPrecisionExplicitCandidate: true);

    public static readonly BaselineQuants F16_Hybrid =
        new(202, false, ["F16"], [TensorWeightScheme.F16],
            IsCombinationCarrierCandidate: false,
            IsHighPrecisionExplicitCandidate: true);

    public static readonly ImmutableArray<BaselineQuants> All =
    [
        Q8_0,
        Q6_K,
        Q5_K,
        Q4_K_M,
        IQ4_NL,
        IQ4_XS,
        IQ3_S,
        IQ3_XS,
        IQ3_XXS,
        IQ2_S,
        IQ2_XS,
        IQ2_XXS,
        BF16_Hybrid,
        F16_Hybrid
    ];

    public static BaselineQuants GetNativeQuant()
    {
        var nativeScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        return new(
            NativeSourceUniqueId,
            false,
            [nativeScheme.Names[0]],
            [nativeScheme],
            IsCombinationCarrierCandidate: false,
            IsExplicitGroupCombinationCandidate: false,
            IsHighPrecisionExplicitCandidate: true);
    }

    // Compatibility alias for older code paths.
    public static BaselineQuants GetBF16Quant() => GetNativeQuant();

    public static IReadOnlyList<BaselineQuants> GetAllRecognizedBaselines() =>
        All.OrderBy(x => x.UniqueId).ToList();

    public static IReadOnlyList<BaselineQuants> GetPureBaselineCandidates(bool hasUsableImatrix) =>
        All.Where(x => x.IsPureBaselineCandidate)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetCombinationCarrierBaselines(bool hasUsableImatrix) =>
        All.Where(x => x.IsCombinationCarrierCandidate)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetGroupCombinationCandidates(bool hasUsableImatrix, bool allowHighPrecisionHybrids) =>
        All.Where(x => x.IsExplicitGroupCombinationCandidate)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .Where(x => allowHighPrecisionHybrids || !x.IsHighPrecisionExplicitCandidate)
            .OrderBy(x => x.UniqueId)
            .ToList();

    public static BaselineQuants GetDefaultExplicitFallbackBaseline() => Q8_0;

    public static void ValidateIntegrityOrThrow()
    {
        var invalidBaselines = All
            .Where(x => x.TensorWeightSchemes.IsDefaultOrEmpty)
            .Select(x => x.Names.IsDefaultOrEmpty ? $"id:{x.UniqueId}" : x.Names[0])
            .ToList();

        if (invalidBaselines.Count > 0)
        {
            throw new InvalidOperationException(
                "Every BaselineQuants entry must define at least one TensorWeightScheme. Missing for: " +
                string.Join(", ", invalidBaselines));
        }

        var duplicateSchemeIds = All
            .SelectMany(x => x.TensorWeightSchemes.Select(s => new { Baseline = x, Scheme = s }))
            .GroupBy(x => x.Scheme.UniqueId)
            .Where(g => g.Count() > 1)
            .Where(g => g.Key != TensorWeightScheme.BF16.UniqueId && g.Key != TensorWeightScheme.F16.UniqueId)
            .Select(g => g.Key)
            .ToList();

        if (duplicateSchemeIds.Count > 0)
        {
            var duplicateNames = duplicateSchemeIds
                .Select(id => TensorWeightScheme.All.First(s => s.UniqueId == id).Names[0]);

            throw new InvalidOperationException(
                "TensorWeightScheme associations must be unique across BaselineQuants entries. Duplicates: " +
                string.Join(", ", duplicateNames));
        }
    }

    public static BaselineQuants FromId(byte id)
    {
        if (id == NativeSourceUniqueId)
            return GetNativeQuant();

        var found = All.FirstOrDefault(x => x.UniqueId == id);
        if (found == null)
            throw new InvalidOperationException($"Unknown baseline quant id '{id}'.");

        return found;
    }

    public static BaselineQuants FromTensorSchemeId(byte schemeId)
    {
        var found = All.FirstOrDefault(x => x.TensorWeightSchemes.Any(s => s.UniqueId == schemeId));
        if (found == null)
            throw new InvalidOperationException($"Unknown tensor scheme id '{schemeId}' for baseline conversion.");

        return found;
    }
}
