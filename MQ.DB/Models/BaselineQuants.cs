using System.Collections.Immutable;
using MQ.DB;

namespace MQ.DB.Models;

public record BaselineQuants(
    byte UniqueId,
    bool RequiresImatrix,
    ImmutableArray<string> Names,
    string QuantizeBaseArgumentName,
    TensorWeightScheme PrimaryTensorWeightScheme,
    ImmutableArray<TensorWeightScheme> LearnedMatchTensorWeightSchemes,
    ImmutableArray<byte> BannedGroupIds,
    bool IsLearningBaseline,
    bool IsCombinationCarrierCandidate,
    bool IsExplicitGroupCombinationCandidate,
    bool IsHighPrecisionExactAlias,
    byte BitRange,
    bool IsCustomBaseline = false,
    string CanonicalKey = "",
    string SourceKind = "standard",
    string? SourceOwner = null,
    string? SourceRepository = null,
    string? SourceFileName = null,
    string? ShortSourceName = null,
    int ExplicitCandidateSortOrder = int.MaxValue)
{
    public const byte NativeSourceUniqueId = 250;
    private const byte FirstDynamicCustomBaselineId = 100;

    private static readonly object DynamicLock = new();
    private static readonly List<BaselineQuants> DynamicCustomBaselines = new();
    private static HashSet<byte>? EnabledStandardLearningBaselineIds;
    private static HashSet<byte>? EnabledStandardCombinationCarrierIds;
    private static HashSet<byte>? EnabledStandardExplicitCandidateIds;

    public TensorWeightScheme? DefaultTensorScheme => PrimaryTensorWeightScheme;
    public ImmutableArray<TensorWeightScheme> TensorWeightSchemes => LearnedMatchTensorWeightSchemes;
    public bool IsPureBaselineCandidate => IsLearningBaseline;
    public bool IsHighPrecisionExplicitCandidate => IsHighPrecisionExactAlias;

    public bool IsExternalRepositoryBaseline =>
        IsCustomBaseline &&
        !string.IsNullOrWhiteSpace(SourceRepository) &&
        !string.IsNullOrWhiteSpace(SourceFileName);

    private static BaselineQuants Create(
        byte uniqueId,
        bool requiresImatrix,
        string name,
        string quantizeBaseArgumentName,
        TensorWeightScheme primaryTensorWeightScheme,
        ImmutableArray<TensorWeightScheme> learnedMatchTensorWeightSchemes,
        ImmutableArray<byte> bannedGroupIds,
        bool isLearningBaseline,
        bool isCombinationCarrierCandidate,
        bool isExplicitGroupCombinationCandidate,
        bool isHighPrecisionExactAlias,
        byte bitRange,
        int explicitCandidateSortOrder = int.MaxValue,
        bool isCustomBaseline = false,
        string? canonicalKey = null,
        string sourceKind = "standard",
        string? sourceOwner = null,
        string? sourceRepository = null,
        string? sourceFileName = null,
        string? shortSourceName = null)
    {
        return new BaselineQuants(
            uniqueId,
            requiresImatrix,
            [name],
            quantizeBaseArgumentName,
            primaryTensorWeightScheme,
            learnedMatchTensorWeightSchemes,
            bannedGroupIds,
            isLearningBaseline,
            isCombinationCarrierCandidate,
            isExplicitGroupCombinationCandidate,
            isHighPrecisionExactAlias,
            bitRange,
            isCustomBaseline,
            canonicalKey ?? $"standard:{name.ToLowerInvariant()}",
            sourceKind,
            sourceOwner,
            sourceRepository,
            sourceFileName,
            shortSourceName,
            explicitCandidateSortOrder);
    }

    public static readonly BaselineQuants Q8_0 =
        Create(0, false, "Q8_0", "Q8_0", TensorWeightScheme.Q8_0, [TensorWeightScheme.Q8_0], [], true, true, true, false, 8, 16);

    public static readonly BaselineQuants Q6_K =
        Create(1, false, "Q6_K", "Q6_K", TensorWeightScheme.Q6_K, [TensorWeightScheme.Q6_K], [], true, false, true, false, 6, 15);

    public static readonly BaselineQuants Q5_K =
        Create(2, false, "Q5_K", "Q5_K", TensorWeightScheme.Q5_K, [TensorWeightScheme.Q5_K], [TReg.MoeRouter.UniqueId], true, false, true, false, 5, 14);

    public static readonly BaselineQuants Q5_K_S =
        Create(13, false, "Q5_K_S", "Q5_K_S", TensorWeightScheme.Q5_K_S, [TensorWeightScheme.Q5_K_S], [TReg.MoeRouter.UniqueId], true, false, true, false, 5, 13);

    
    public static readonly BaselineQuants Q4_K_M =
        Create(3, false, "Q4_K_M", "Q4_K_M", TensorWeightScheme.Q4_K, [TensorWeightScheme.Q4_K], [TReg.MoeRouter.UniqueId], true, false, true, false, 4, 12);

    public static readonly BaselineQuants Q4_K_S =
        Create(14, false, "Q4_K_S", "Q4_K_S", TensorWeightScheme.Q4_K_S, [TensorWeightScheme.Q4_K_S], [TReg.MoeRouter.UniqueId], true, false, true, false, 4, 11);

    
    public static readonly BaselineQuants IQ4_NL =
        Create(5, false, "IQ4_NL", "IQ4_NL", TensorWeightScheme.IQ4_NL, [TensorWeightScheme.IQ4_NL], [TReg.MoeRouter.UniqueId], true, false, true, false, 4, 10);

    public static readonly BaselineQuants IQ4_XS =
        Create(6, false, "IQ4_XS", "IQ4_XS", TensorWeightScheme.IQ4_XS, [TensorWeightScheme.IQ4_XS], [TReg.MoeRouter.UniqueId], true, false, true, false, 4, 9);

    public static readonly BaselineQuants MXFP4_MOE =
        Create(15, false, "MXFP4_MOE", "MXFP4_MOE", TensorWeightScheme.MXFP4, [TensorWeightScheme.MXFP4, TensorWeightScheme.IQ3_S, TensorWeightScheme.IQ3_XS], [TReg.MoeRouter.UniqueId], false, false, false, false, 4, 8);

    public static readonly BaselineQuants IQ3_M =
        Create(17, true, "IQ3_M", "IQ3_M", TensorWeightScheme.IQ3_S, [TensorWeightScheme.IQ3_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId], true, false, true, false, 3, 7);

    
    public static readonly BaselineQuants IQ3_S =
        Create(7, true, "IQ3_S", "IQ3_S", TensorWeightScheme.IQ3_S, [TensorWeightScheme.IQ3_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId], true, false, true, false, 3, 6);

    public static readonly BaselineQuants IQ3_XS =
        Create(8, true, "IQ3_XS", "IQ3_XS", TensorWeightScheme.IQ3_XS, [TensorWeightScheme.IQ3_XS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId], true, false, true, false, 3, 5);

    public static readonly BaselineQuants IQ3_XXS =
        Create(9, true, "IQ3_XXS", "IQ3_XXS", TensorWeightScheme.IQ3_XXS, [TensorWeightScheme.IQ3_XXS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId], true, false, true, false, 3, 4);

    public static readonly BaselineQuants IQ2_M =
        Create(16, true, "IQ2_M", "IQ2_M", TensorWeightScheme.IQ2_S, [TensorWeightScheme.IQ2_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId], true, false, true, false, 2, 3);

    
    public static readonly BaselineQuants IQ2_S =
        Create(10, true, "IQ2_S", "IQ2_S", TensorWeightScheme.IQ2_S, [TensorWeightScheme.IQ2_S], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId], true, false, true, false, 2, 2);

    public static readonly BaselineQuants IQ2_XS =
        Create(11, true, "IQ2_XS", "IQ2_XS", TensorWeightScheme.IQ2_XS, [TensorWeightScheme.IQ2_XS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId], true, false, true, false, 2, 1);

    public static readonly BaselineQuants IQ2_XXS =
        Create(12, true, "IQ2_XXS", "IQ2_XXS", TensorWeightScheme.IQ2_XXS, [TensorWeightScheme.IQ2_XXS], [TReg.Embeddings.UniqueId, TReg.LmHead.UniqueId, TReg.MoeRouter.UniqueId, TReg.MoeExperts.UniqueId, TReg.AttnKV.UniqueId], true, false, true, false, 2, 0);

    public static readonly BaselineQuants BF16_Hybrid =
        Create(201, false, "BF16", "BF16", TensorWeightScheme.BF16, [TensorWeightScheme.BF16], [], false, false, false, true, 16, int.MaxValue, false, "alias:bf16", "exact_alias", null, null, null, null);

    public static readonly BaselineQuants F16_Hybrid =
        Create(202, false, "F16", "F16", TensorWeightScheme.F16, [TensorWeightScheme.F16], [], false, false, false, true, 16, int.MaxValue, false, "alias:f16", "exact_alias", null, null, null, null);

    private static readonly ImmutableArray<BaselineQuants> StandardBaselines =
    [
        Q8_0,
        Q6_K,
        Q5_K,
        Q5_K_S,
        Q4_K_M,
        Q4_K_S,
        IQ4_NL,
        IQ4_XS,
        MXFP4_MOE,
        IQ3_M,
        IQ3_S,
        IQ3_XS,
        IQ3_XXS,
        IQ2_M,
        IQ2_S,
        IQ2_XS,
        IQ2_XXS
    ];

    private static readonly ImmutableArray<BaselineQuants> ExactAliases =
    [
        BF16_Hybrid,
        F16_Hybrid
    ];

    public static IReadOnlyList<BaselineQuants> All => GetAllRecognizedBaselines();

    public static BaselineQuants CreateDynamicCustomBaseline(
        byte uniqueId,
        string displayName,
        string quantizeBaseArgumentName,
        string sourceRepository,
        string sourceFileName,
        string shortSourceName,
        string sourceOwner,
        string sourceKind,
        string canonicalKey,
        TensorWeightScheme primaryTensorWeightScheme,
        ImmutableArray<TensorWeightScheme> learnedMatchTensorWeightSchemes,
        IReadOnlyCollection<byte> bannedGroupIds,
        bool requiresImatrix,
        bool isLearningBaseline,
        bool isCombinationCarrierCandidate,
        bool isExplicitGroupCombinationCandidate,
        byte bitRange,
        int explicitCandidateSortOrder)
    {
        return new BaselineQuants(
            uniqueId,
            requiresImatrix,
            [displayName, primaryTensorWeightScheme.Names[0]],
            quantizeBaseArgumentName,
            primaryTensorWeightScheme,
            learnedMatchTensorWeightSchemes,
            bannedGroupIds?.Distinct().OrderBy(x => x).ToImmutableArray() ?? ImmutableArray<byte>.Empty,
            isLearningBaseline,
            isCombinationCarrierCandidate,
            isExplicitGroupCombinationCandidate,
            false,
            bitRange,
            true,
            canonicalKey,
            sourceKind,
            sourceOwner,
            sourceRepository,
            sourceFileName,
            shortSourceName,
            explicitCandidateSortOrder);
    }

    public static void ResetDynamicCustomBaselines()
    {
        lock (DynamicLock)
        {
            DynamicCustomBaselines.Clear();
        }
    }

    public static byte GetFirstAvailableDynamicBaselineId()
    {
        var used = GetAllRecognizedBaselines().Select(x => x.UniqueId).ToHashSet();
        for (byte id = FirstDynamicCustomBaselineId; id < 200; id++)
        {
            if (!used.Contains(id))
                return id;
        }

        throw new InvalidOperationException("No free dynamic baseline ids remain in the configured range.");
    }

    public static void RegisterDynamicCustomBaseline(BaselineQuants baseline)
    {
        if (!baseline.IsCustomBaseline)
            throw new InvalidOperationException("Only custom baselines can be dynamically registered.");

        lock (DynamicLock)
        {
            if (GetAllRecognizedBaselines().Any(x => x.UniqueId == baseline.UniqueId))
                throw new InvalidOperationException($"Dynamic baseline id collision detected for id '{baseline.UniqueId}'.");

            if (GetAllRecognizedBaselines().Any(x => string.Equals(x.CanonicalKey, baseline.CanonicalKey, StringComparison.Ordinal)))
                throw new InvalidOperationException($"Dynamic baseline canonical key collision detected for '{baseline.CanonicalKey}'.");

            DynamicCustomBaselines.Add(baseline);
        }
    }

    public static void ConfigureStandardRoleFilters(
        IReadOnlyCollection<byte>? enabledLearningBaselineIds,
        IReadOnlyCollection<byte>? enabledCombinationCarrierIds,
        IReadOnlyCollection<byte>? enabledExplicitCandidateIds)
    {
        EnabledStandardLearningBaselineIds = enabledLearningBaselineIds == null ? null : enabledLearningBaselineIds.ToHashSet();
        EnabledStandardCombinationCarrierIds = enabledCombinationCarrierIds == null ? null : enabledCombinationCarrierIds.ToHashSet();
        EnabledStandardExplicitCandidateIds = enabledExplicitCandidateIds == null ? null : enabledExplicitCandidateIds.ToHashSet();
    }

    public static void ConfigureStandardPolicy(
        bool includeStandardLearningBaselines,
        bool includeStandardCombinationCarriers,
        bool includeStandardGroupCandidates,
        bool alwaysIncludeQ8Anchor,
        IReadOnlyCollection<string>? standardLearningBaselineAllowList,
        IReadOnlyCollection<string>? standardCarrierAllowList,
        IReadOnlyCollection<string>? standardGroupCandidateAllowList)
    {
        HashSet<byte>? learning = includeStandardLearningBaselines
            ? ResolveNamesToIdsOrNull(standardLearningBaselineAllowList)
            : new HashSet<byte>();

        HashSet<byte>? carriers = includeStandardCombinationCarriers
            ? ResolveNamesToIdsOrNull(standardCarrierAllowList)
            : new HashSet<byte>();

        HashSet<byte>? explicitCandidates = includeStandardGroupCandidates
            ? ResolveNamesToIdsOrNull(standardGroupCandidateAllowList)
            : new HashSet<byte>();

        if (alwaysIncludeQ8Anchor)
        {
            learning ??= new HashSet<byte>();
            carriers ??= new HashSet<byte>();
            explicitCandidates ??= new HashSet<byte>();
            learning.Add(Q8_0.UniqueId);
            carriers.Add(Q8_0.UniqueId);
            explicitCandidates.Add(Q8_0.UniqueId);
        }

        ConfigureStandardRoleFilters(learning, carriers, explicitCandidates);
    }

    private static HashSet<byte>? ResolveNamesToIdsOrNull(IReadOnlyCollection<string>? names)
    {
        if (names == null || names.Count == 0)
            return null;

        var set = new HashSet<byte>();
        foreach (var raw in names)
        {
            var item = ResolveBuiltInStandardBaseline(raw ?? string.Empty);
            if (item != null)
                set.Add(item.UniqueId);
        }

        return set;
    }

    public sealed class ExternalBaselineRegistration
    {
        public string CanonicalKey { get; set; } = string.Empty;
        public string DisplayName { get; set; } = string.Empty;
        public string QuantizeBaseArgumentName { get; set; } = string.Empty;
        public string Repository { get; set; } = string.Empty;
        public string RepositoryFileName { get; set; } = string.Empty;
        public string OwnerShortName { get; set; } = string.Empty;
        public string BaselineFamilyName { get; set; } = string.Empty;
        public TensorWeightScheme TensorScheme { get; set; } = default!;
        public bool RequiresImatrix { get; set; }
        public bool AddAsLearningBaseline { get; set; }
        public bool AddAsCombinationCarrier { get; set; }
        public bool AddAsGroupCandidate { get; set; }
        public byte BitRange { get; set; }
        public IReadOnlyCollection<byte> BannedGroupIds { get; set; } = Array.Empty<byte>();
    }

    public static BaselineQuants RegisterCustomExternalBaseline(ExternalBaselineRegistration registration)
    {
        var sortOrder = StandardBaselines
            .FirstOrDefault(x => string.Equals(x.Names[0], registration.BaselineFamilyName, StringComparison.OrdinalIgnoreCase))
            ?.ExplicitCandidateSortOrder ?? int.MaxValue;

        var baseline = CreateDynamicCustomBaseline(
            GetFirstAvailableDynamicBaselineId(),
            registration.DisplayName,
            registration.QuantizeBaseArgumentName,
            registration.Repository,
            registration.RepositoryFileName,
            registration.OwnerShortName,
            registration.OwnerShortName,
            "huggingface_repo",
            registration.CanonicalKey,
            registration.TensorScheme,
            [registration.TensorScheme],
            registration.BannedGroupIds,
            registration.RequiresImatrix,
            registration.AddAsLearningBaseline,
            registration.AddAsCombinationCarrier,
            registration.AddAsGroupCandidate,
            registration.BitRange,
            sortOrder);

        RegisterDynamicCustomBaseline(baseline);
        return baseline;
    }

    public static BaselineQuants GetNativeQuant()
    {
        var nativeScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        return new BaselineQuants(
            NativeSourceUniqueId,
            false,
            [nativeScheme.Names[0]],
            nativeScheme.Names[0],
            nativeScheme,
            [nativeScheme],
            [],
            false,
            false,
            false,
            true,
            16,
            false,
            $"native:{nativeScheme.Names[0].ToLowerInvariant()}",
            "native_exact_alias",
            null,
            null,
            null,
            null,
            int.MaxValue);
    }

    public static BaselineQuants GetBF16Quant() => GetNativeQuant();

    public static IReadOnlyList<BaselineQuants> GetBuiltInStandardBaselines() => StandardBaselines.OrderBy(x => x.UniqueId).ToList();

    public static BaselineQuants? ResolveBuiltInStandardBaseline(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return null;

        return StandardBaselines.FirstOrDefault(x =>
            x.Names.Any(n => string.Equals(n, name, StringComparison.OrdinalIgnoreCase)) ||
            string.Equals(x.PrimaryTensorWeightScheme.Names[0], name, StringComparison.OrdinalIgnoreCase));
    }

    public static IReadOnlyList<BaselineQuants> GetAllRecognizedBaselines() =>
        StandardBaselines
            .Concat(DynamicCustomBaselines.OrderBy(x => x.UniqueId))
            .Concat(ExactAliases)
            .OrderBy(x => x.UniqueId)
            .ToList();

    private static IEnumerable<BaselineQuants> FilterStandardByRole(
        IEnumerable<BaselineQuants> source,
        HashSet<byte>? enabledIds)
    {
        return enabledIds == null ? source : source.Where(x => enabledIds.Contains(x.UniqueId));
    }

    public static IReadOnlyList<BaselineQuants> GetLearningBaselines(bool hasUsableImatrix)
    {
        var standard = FilterStandardByRole(StandardBaselines.Where(x => x.IsLearningBaseline), EnabledStandardLearningBaselineIds);
        var custom = DynamicCustomBaselines.Where(x => x.IsLearningBaseline);

        return standard
            .Concat(custom)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static IReadOnlyList<BaselineQuants> GetPureBaselineCandidates(bool hasUsableImatrix) =>
        GetLearningBaselines(hasUsableImatrix);

    public static IReadOnlyList<BaselineQuants> GetCombinationCarrierBaselines(bool hasUsableImatrix)
    {
        var standard = FilterStandardByRole(StandardBaselines.Where(x => x.IsCombinationCarrierCandidate), EnabledStandardCombinationCarrierIds);
        var custom = DynamicCustomBaselines.Where(x => x.IsCombinationCarrierCandidate);

        var result = standard
            .Concat(custom)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.UniqueId)
            .ToList();

        return result.Count == 0 ? new[] { Q8_0 } : result;
    }

    public static IReadOnlyList<BaselineQuants> GetGroupCombinationCandidates(bool hasUsableImatrix, bool allowHighPrecisionHybrids)
    {
        var standard = FilterStandardByRole(StandardBaselines.Where(x => x.IsExplicitGroupCombinationCandidate), EnabledStandardExplicitCandidateIds);
        var custom = DynamicCustomBaselines.Where(x => x.IsExplicitGroupCombinationCandidate);

        return standard
            .Concat(custom)
            .Where(x => hasUsableImatrix || !x.RequiresImatrix)
            .OrderBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();
    }

    public static IReadOnlyList<BaselineQuants> GetGroupCombinationCandidatesSmallestFirst(bool hasUsableImatrix, bool allowHighPrecisionHybrids) =>
        GetGroupCombinationCandidates(hasUsableImatrix, allowHighPrecisionHybrids)
            .OrderBy(x => x.BitRange)
            .ThenBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();

    public static IReadOnlyList<BaselineQuants> GetExactHighPrecisionAliases(bool allowHighPrecisionHybrids)
    {
        if (!allowHighPrecisionHybrids)
            return Array.Empty<BaselineQuants>();

        return ExactAliases.OrderBy(x => x.UniqueId).ToList();
    }

    public const byte TensorConfigNullSlotValue = 0;

    public static BaselineQuants GetDefaultExplicitFallbackBaseline() => Q8_0;
    public static bool IsNullTensorConfigGroupSlot(byte storedValue) => storedValue == TensorConfigNullSlotValue;
    public static byte EncodeTensorConfigGroupSlot(BaselineQuants baseline) => EncodeTensorConfigGroupSlotBaselineId(baseline.UniqueId);
    public static byte EncodeTensorConfigGroupSlot(TensorWeightScheme exactScheme) => EncodeTensorConfigGroupSlotBaselineId(GetExactOverrideStorageId(exactScheme));

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

    public static BaselineQuants DecodeTensorConfigGroupSlotToBaseline(byte storedValue) => FromId(DecodeTensorConfigGroupSlotToBaselineId(storedValue));
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
        var all = GetAllRecognizedBaselines();

        var invalidBaselines = all
            .Where(x => !x.IsHighPrecisionExactAlias)
            .Where(x => x.LearnedMatchTensorWeightSchemes.IsDefaultOrEmpty)
            .Select(x => x.Names.IsDefaultOrEmpty ? $"id:{x.UniqueId}" : x.Names[0])
            .ToList();

        if (invalidBaselines.Count > 0)
        {
            throw new InvalidOperationException(
                "Every non-alias BaselineQuants entry must define at least one TensorWeightScheme. Missing for: " +
                string.Join(", ", invalidBaselines));
        }

        var duplicateIds = all
            .GroupBy(x => x.UniqueId)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateIds.Count > 0)
            throw new InvalidOperationException($"Duplicate baseline ids detected: {string.Join(", ", duplicateIds)}");

        var duplicateKeys = all
            .GroupBy(x => x.CanonicalKey, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateKeys.Count > 0)
            throw new InvalidOperationException($"Duplicate baseline canonical keys detected: {string.Join(", ", duplicateKeys)}");
    }

    public static BaselineQuants FromId(byte id)
    {
        if (id == NativeSourceUniqueId)
            return GetNativeQuant();

        var found = GetAllRecognizedBaselines().FirstOrDefault(x => x.UniqueId == id);
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

        if (schemeId == TensorWeightScheme.GetCurrentNativePrecisionScheme().UniqueId)
            return GetNativeQuant();

        var found = GetAllRecognizedBaselines()
            .Where(x => !x.IsHighPrecisionExactAlias)
            .FirstOrDefault(x =>
                x.PrimaryTensorWeightScheme.UniqueId == schemeId ||
                x.LearnedMatchTensorWeightSchemes.Any(s => s.UniqueId == schemeId));

        if (found == null)
            throw new InvalidOperationException($"Unknown tensor scheme id '{schemeId}' for baseline conversion.");

        return found;
    }
}