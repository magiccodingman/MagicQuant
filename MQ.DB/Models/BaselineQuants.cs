using System.Collections.Immutable;
using MQ.DB;

namespace MQ.DB.Models;

public record BaselineQuants(
    byte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    TensorWeightScheme PrimaryTensorWeightScheme,
    ImmutableArray<TensorWeightScheme> LearnedMatchTensorWeightSchemes,
    ImmutableArray<byte> BannedGroupIds,
    bool IsLearningBaseline,
    bool IsCombinationCarrierCandidate,
    bool IsExplicitGroupCombinationCandidate,
    bool IsHighPrecisionExactAlias,
    int ExplicitCandidateSortOrder = int.MaxValue)
{
    public const byte NativeSourceUniqueId = 250;

    public TensorWeightScheme? DefaultTensorScheme => PrimaryTensorWeightScheme;

    // Compatibility aliases retained for older call sites.
    public ImmutableArray<TensorWeightScheme> TensorWeightSchemes => LearnedMatchTensorWeightSchemes;
    public bool IsPureBaselineCandidate => IsLearningBaseline;
    public bool IsHighPrecisionExplicitCandidate => IsHighPrecisionExactAlias;

    private static BaselineQuants Create(
        byte uniqueId,
        bool requiresImatrix,
        string name,
        TensorWeightScheme primaryTensorWeightScheme,
        ImmutableArray<TensorWeightScheme> learnedMatchTensorWeightSchemes,
        ImmutableArray<byte> bannedGroupIds,
        bool isLearningBaseline,
        bool isCombinationCarrierCandidate,
        bool isExplicitGroupCombinationCandidate,
        bool isHighPrecisionExactAlias,
        int explicitCandidateSortOrder = int.MaxValue)
    {
        return new BaselineQuants(
            uniqueId,
            requiresImatrix,
            [name],
            primaryTensorWeightScheme,
            learnedMatchTensorWeightSchemes,
            bannedGroupIds,
            isLearningBaseline,
            isCombinationCarrierCandidate,
            isExplicitGroupCombinationCandidate,
            isHighPrecisionExactAlias,
            explicitCandidateSortOrder);
    }

    public static readonly BaselineQuants Q8_0 =
        Create(0, false, "Q8_0", TensorWeightScheme.Q8_0, [TensorWeightScheme.Q8_0], [],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 11);

    public static readonly BaselineQuants Q6_K =
        Create(1, false, "Q6_K", TensorWeightScheme.Q6_K, [TensorWeightScheme.Q6_K], [],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 10);

    public static readonly BaselineQuants Q5_K =
        Create(2, false, "Q5_K", TensorWeightScheme.Q5_K, [TensorWeightScheme.Q5_K], [TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 9);

    public static readonly BaselineQuants Q4_K_M =
        Create(3, false, "Q4_K_M", TensorWeightScheme.Q4_K, [TensorWeightScheme.Q4_K], [TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 8);

    public static readonly BaselineQuants IQ4_NL =
        Create(5, false, "IQ4_NL", TensorWeightScheme.IQ4_NL, [TensorWeightScheme.IQ4_NL], [TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 7);

    public static readonly BaselineQuants IQ4_XS =
        Create(6, false, "IQ4_XS", TensorWeightScheme.IQ4_XS, [TensorWeightScheme.IQ4_XS], [TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: true,
            isExplicitGroupCombinationCandidate: true,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 6);

    public static readonly BaselineQuants IQ3_S =
        Create(7, true, "IQ3_S", TensorWeightScheme.IQ3_S, [TensorWeightScheme.IQ3_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 5);

    public static readonly BaselineQuants IQ3_XS =
        Create(8, true, "IQ3_XS", TensorWeightScheme.IQ3_XS, [TensorWeightScheme.IQ3_XS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 4);

    public static readonly BaselineQuants IQ3_XXS =
        Create(9, true, "IQ3_XXS", TensorWeightScheme.IQ3_XXS, [TensorWeightScheme.IQ3_XXS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 3);

    public static readonly BaselineQuants IQ2_S =
        Create(10, true, "IQ2_S", TensorWeightScheme.IQ2_S, [TensorWeightScheme.IQ2_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 2);

    public static readonly BaselineQuants IQ2_XS =
        Create(11, true, "IQ2_XS", TensorWeightScheme.IQ2_XS, [TensorWeightScheme.IQ2_XS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 1);

    public static readonly BaselineQuants IQ2_XXS =
        Create(12, true, "IQ2_XXS", TensorWeightScheme.IQ2_XXS, [TensorWeightScheme.IQ2_XXS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId, TReg.AttnKV.UniqueId],
            isLearningBaseline: true,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: false,
            explicitCandidateSortOrder: 0);

    // These are exact/native override aliases. They are NOT learned baseline identities.
    public static readonly BaselineQuants BF16_Hybrid =
        Create(201, false, "BF16", TensorWeightScheme.BF16, [TensorWeightScheme.BF16], [],
            isLearningBaseline: false,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: true);

    public static readonly BaselineQuants F16_Hybrid =
        Create(202, false, "F16", TensorWeightScheme.F16, [TensorWeightScheme.F16], [],
            isLearningBaseline: false,
            isCombinationCarrierCandidate: false,
            isExplicitGroupCombinationCandidate: false,
            isHighPrecisionExactAlias: true);

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

        return new BaselineQuants(
            NativeSourceUniqueId,
            false,
            [nativeScheme.Names[0]],
            nativeScheme,
            [nativeScheme],
            [],
            IsLearningBaseline: false,
            IsCombinationCarrierCandidate: false,
            IsExplicitGroupCombinationCandidate: false,
            IsHighPrecisionExactAlias: true,
            ExplicitCandidateSortOrder: int.MaxValue);
    }

    public static BaselineQuants GetBF16Quant() => GetNativeQuant();

    public static IReadOnlyList<BaselineQuants> GetAllRecognizedBaselines() =>
        All.OrderBy(x => x.UniqueId).ToList();

    public static IReadOnlyList<BaselineQuants> GetLearningBaselines(bool hasUsableImatrix) =>
        All.Where(x => x.IsLearningBaseline)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetPureBaselineCandidates(bool hasUsableImatrix) =>
        GetLearningBaselines(hasUsableImatrix);

    public static IReadOnlyList<BaselineQuants> GetCombinationCarrierBaselines(bool hasUsableImatrix) =>
        All.Where(x => x.IsCombinationCarrierCandidate)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetGroupCombinationCandidates(bool hasUsableImatrix, bool allowHighPrecisionHybrids) =>
        All.Where(x => x.IsExplicitGroupCombinationCandidate)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetGroupCombinationCandidatesSmallestFirst(bool hasUsableImatrix, bool allowHighPrecisionHybrids) =>
        GetGroupCombinationCandidates(hasUsableImatrix, allowHighPrecisionHybrids)
            .OrderBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetExactHighPrecisionAliases(bool allowHighPrecisionHybrids)
    {
        if (!allowHighPrecisionHybrids)
            return Array.Empty<BaselineQuants>();

        return All.Where(x => x.IsHighPrecisionExactAlias)
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public const byte TensorConfigNullSlotValue = 0;

    public static BaselineQuants GetDefaultExplicitFallbackBaseline() => Q8_0;

    public static bool IsNullTensorConfigGroupSlot(byte storedValue) => storedValue == TensorConfigNullSlotValue;

    public static byte EncodeTensorConfigGroupSlot(BaselineQuants baseline) =>
        EncodeTensorConfigGroupSlotBaselineId(baseline.UniqueId);

    public static byte EncodeTensorConfigGroupSlot(TensorWeightScheme exactScheme) =>
        EncodeTensorConfigGroupSlotBaselineId(GetExactOverrideStorageId(exactScheme));

    public static byte EncodeTensorConfigGroupSlotBaselineId(byte baselineId)
    {
        if (baselineId == byte.MaxValue)
            throw new InvalidOperationException("Baseline id 255 cannot be encoded into a tensor-config group slot.");

        return checked((byte)(baselineId + 1));
    }

    public static byte DecodeTensorConfigGroupSlotToBaselineId(byte storedValue)
    {
        if (IsNullTensorConfigGroupSlot(storedValue))
            throw new InvalidOperationException("Tensor-config group slot 0 represents NULL and cannot be decoded as a baseline id.");

        return checked((byte)(storedValue - 1));
    }

    public static BaselineQuants DecodeTensorConfigGroupSlotToBaseline(byte storedValue) =>
        FromId(DecodeTensorConfigGroupSlotToBaselineId(storedValue));

    public static bool IsNativeExactAlias(BaselineQuants baseline) => IsNativeExactAlias(baseline.UniqueId);

    public static bool IsNativeExactAlias(byte baselineId)
    {
        return baselineId == NativeSourceUniqueId ||
               baselineId == BF16_Hybrid.UniqueId ||
               baselineId == F16_Hybrid.UniqueId;
    }

    public static byte CanonicalLearningBaselineId(BaselineQuants baseline) => CanonicalLearningBaselineId(baseline.UniqueId);

    public static byte CanonicalLearningBaselineId(byte baselineId)
    {
        return IsNativeExactAlias(baselineId)
            ? NativeSourceUniqueId
            : baselineId;
    }

    public static byte GetExactOverrideStorageId(TensorWeightScheme scheme)
    {
        if (scheme.UniqueId == TensorWeightScheme.BF16.UniqueId)
            return BF16_Hybrid.UniqueId;

        if (scheme.UniqueId == TensorWeightScheme.F16.UniqueId)
            return F16_Hybrid.UniqueId;

        if (scheme.UniqueId == TensorWeightScheme.GetCurrentNativePrecisionScheme().UniqueId)
            return NativeSourceUniqueId;

        throw new InvalidOperationException(
            $"Tensor scheme '{scheme.Names[0]}' does not have a supported exact-override storage baseline id.");
    }

    public static TensorWeightScheme ResolveExactOverrideScheme(byte baselineId)
    {
        return baselineId switch
        {
            NativeSourceUniqueId => TensorWeightScheme.GetCurrentNativePrecisionScheme(),
            201 => TensorWeightScheme.BF16,
            202 => TensorWeightScheme.F16,
            _ => throw new InvalidOperationException($"Baseline id '{baselineId}' is not an exact override alias.")
        };
    }

    public static void ValidateIntegrityOrThrow()
    {
        var invalidBaselines = All
            .Where(x => x.LearnedMatchTensorWeightSchemes.IsDefaultOrEmpty)
            .Select(x => x.Names.IsDefaultOrEmpty ? $"id:{x.UniqueId}" : x.Names[0])
            .ToList();

        if (invalidBaselines.Count > 0)
        {
            throw new InvalidOperationException(
                "Every BaselineQuants entry must define at least one TensorWeightScheme. Missing for: " +
                string.Join(", ", invalidBaselines));
        }

        var duplicateSchemeIds = All
            .Where(x => !x.IsHighPrecisionExactAlias)
            .SelectMany(x => x.LearnedMatchTensorWeightSchemes.Select(s => new { Baseline = x, Scheme = s }))
            .GroupBy(x => x.Scheme.UniqueId)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateSchemeIds.Count > 0)
        {
            var duplicateNames = duplicateSchemeIds
                .Select(id => TensorWeightScheme.All.First(s => s.UniqueId == id).Names[0]);

            throw new InvalidOperationException(
                "TensorWeightScheme associations must be unique across learned BaselineQuants entries. Duplicates: " +
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
        if (schemeId == TensorWeightScheme.BF16.UniqueId)
            return BF16_Hybrid;

        if (schemeId == TensorWeightScheme.F16.UniqueId)
            return F16_Hybrid;

        if (schemeId == TensorWeightScheme.GetCurrentNativePrecisionScheme().UniqueId &&
            schemeId == TensorWeightScheme.F32.UniqueId)
            return GetNativeQuant();

        var found = All.FirstOrDefault(x =>
            x.PrimaryTensorWeightScheme.UniqueId == schemeId ||
            x.LearnedMatchTensorWeightSchemes.Any(s => s.UniqueId == schemeId));
        if (found == null)
            throw new InvalidOperationException($"Unknown tensor scheme id '{schemeId}' for baseline conversion.");

        return found;
    }
}
