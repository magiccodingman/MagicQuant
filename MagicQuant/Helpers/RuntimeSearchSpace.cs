using MQ.DB.Models;

namespace MagicQuant.Helpers;

public sealed class RuntimeLearnedBaselineBanInfo
{
    public BaselineQuants Candidate { get; init; } = default!;
    public IReadOnlyList<BaselineQuants> MissingBaselines { get; init; } = Array.Empty<BaselineQuants>();
}

public static class RuntimeSearchSpace
{
    private static readonly Dictionary<byte, HashSet<byte>> ExplicitCandidateBansByGroup = new();
    private static readonly Dictionary<byte, Dictionary<byte, HashSet<byte>>> LearnedBaselineMissingByGroupAndCandidate = new();
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();
    private static readonly HashSet<byte> Bf16SuppressedTensorChoiceGroupIds = new();
    private static bool _imatrixAvailable;

    [Obsolete("Use candidate-based RuntimeSearchSpace APIs.")]
    [Obsolete("Use candidate-based RuntimeSearchSpace APIs.")]
    [Obsolete("Use candidate-based RuntimeSearchSpace APIs.")]
    [Obsolete("Use candidate-based RuntimeSearchSpace APIs.")]
    [Obsolete("Use candidate-based RuntimeSearchSpace APIs.")]
    public static bool AllowHighPrecisionHybrids { get; set; }

    public static void ResetForNewModel()
    {
        ExplicitCandidateBansByGroup.Clear();
        LearnedBaselineMissingByGroupAndCandidate.Clear();
        DisabledCombinationBaselineIds.Clear();
        Bf16SuppressedTensorChoiceGroupIds.Clear();
        _imatrixAvailable = false;
        AllowHighPrecisionHybrids = false;
        TensorWeightScheme.ResetAllRuntimeBans();
    }

    public static void SetImatrixAvailability(bool available) => _imatrixAvailable = available;

    public static bool HasUsableImatrix() => _imatrixAvailable;

    public static void BanCombinationCandidateForGroup(TensorGroup group, BaselineQuants candidate)
    {
        if (candidate.UniqueId == BaselineQuants.GetDefaultExplicitFallbackBaseline().UniqueId)
            return;

        if (!ExplicitCandidateBansByGroup.TryGetValue(group.UniqueId, out var set))
        {
            set = new HashSet<byte>();
            ExplicitCandidateBansByGroup[group.UniqueId] = set;
        }

        set.Add(candidate.UniqueId);
    }

    public static void BanCombinationCandidateForGroupByLearnedBaselineAbsence(
        TensorGroup group,
        BaselineQuants candidate,
        BaselineQuants sourceBaseline)
    {
        BanCombinationCandidateForGroup(group, candidate);

        if (!LearnedBaselineMissingByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate))
        {
            byCandidate = new Dictionary<byte, HashSet<byte>>();
            LearnedBaselineMissingByGroupAndCandidate[group.UniqueId] = byCandidate;
        }

        if (!byCandidate.TryGetValue(candidate.UniqueId, out var baselineIds))
        {
            baselineIds = new HashSet<byte>();
            byCandidate[candidate.UniqueId] = baselineIds;
        }

        baselineIds.Add(sourceBaseline.UniqueId);
    }

    public static void BanAllExplicitCombinationCandidatesForGroup(TensorGroup group)
    {
        foreach (var candidate in BaselineQuants.GetGroupCombinationCandidates(_imatrixAvailable, allowHighPrecisionHybrids: true))
            BanCombinationCandidateForGroup(group, candidate);
    }

    public static IReadOnlyList<BaselineQuants> GetRuntimeExplicitCandidateBansForGroup(TensorGroup group)
    {
        if (!ExplicitCandidateBansByGroup.TryGetValue(group.UniqueId, out var set))
            return Array.Empty<BaselineQuants>();

        return BaselineQuants.GetAllRecognizedBaselines()
            .Where(x => set.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool IsCombinationCandidateRuntimeBannedForGroup(TensorGroup group, BaselineQuants candidate)
        => ExplicitCandidateBansByGroup.TryGetValue(group.UniqueId, out var set) && set.Contains(candidate.UniqueId);

    public static bool HasAnyExplicitCombinationCandidateAllowed(TensorGroup group)
    {
        return BaselineQuants.GetGroupCombinationCandidates(_imatrixAvailable, allowHighPrecisionHybrids: true)
            .Any(x => !IsCombinationCandidateRuntimeBannedForGroup(group, x));
    }

    public static bool IsGroupExplicitCandidateBanned(TensorGroup group) => !HasAnyExplicitCombinationCandidateAllowed(group);

    public static IReadOnlyList<TensorGroup> GetGroupsWithExplicitQuantBanned()
        => TReg.All.Where(IsGroupExplicitCandidateBanned).OrderBy(x => x.UniqueId).ToList();

    public static bool HasLearnedBaselineMissingPrunesForGroup(TensorGroup group)
        => LearnedBaselineMissingByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate) && byCandidate.Count > 0;

    public static IReadOnlyList<TensorGroup> GetGroupsWithLearnedBaselineMissingPrunes()
        => TReg.All.Where(HasLearnedBaselineMissingPrunesForGroup).OrderBy(x => x.UniqueId).ToList();

    public static IReadOnlyList<RuntimeLearnedBaselineBanInfo> GetLearnedBaselineMissingPrunedCandidatesForGroup(TensorGroup group)
    {
        if (!LearnedBaselineMissingByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate))
            return Array.Empty<RuntimeLearnedBaselineBanInfo>();

        var result = new List<RuntimeLearnedBaselineBanInfo>();
        foreach (var kvp in byCandidate.OrderBy(x => x.Key))
        {
            var candidate = BaselineQuants.FromId(kvp.Key);
            var baselines = kvp.Value.OrderBy(x => x).Select(BaselineQuants.FromId).ToList();
            result.Add(new RuntimeLearnedBaselineBanInfo { Candidate = candidate, MissingBaselines = baselines });
        }

        return result;
    }

    public static void SuppressBf16TensorChoice(TensorGroup group) => Bf16SuppressedTensorChoiceGroupIds.Add(group.UniqueId);

    public static bool IsBf16TensorChoiceSuppressed(TensorGroup group)
        => Bf16SuppressedTensorChoiceGroupIds.Contains(group.UniqueId) && HasAnyExplicitCombinationCandidateAllowed(group);

    public static IReadOnlyList<TensorGroup> GetBf16SuppressedGroups()
        => TReg.All.Where(IsBf16TensorChoiceSuppressed).OrderBy(x => x.UniqueId).ToList();

    public static (bool ExplicitAllowed, bool Bf16Allowed) GetFinalAllowedQuantFamiliesForGroup(TensorGroup group)
    {
        bool explicitAllowed = HasAnyExplicitCombinationCandidateAllowed(group);
        bool bf16Allowed = !IsBf16TensorChoiceSuppressed(group) || !explicitAllowed;
        return (explicitAllowed, bf16Allowed);
    }

    public static IReadOnlyList<BaselineQuants> GetActiveCombinationBaselines()
    {
        return BaselineQuants.GetCombinationCarrierBaselines(_imatrixAvailable)
            .Where(x => !DisabledCombinationBaselineIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool DisableCombinationBaseline(BaselineQuants baseline, bool allowDisablingLast = false)
    {
        if (!baseline.IsCombinationCarrierCandidate || DisabledCombinationBaselineIds.Contains(baseline.UniqueId))
            return false;

        int currentlyActive = GetActiveCombinationBaselines().Count;
        if (!allowDisablingLast && currentlyActive <= 1)
            return false;

        DisabledCombinationBaselineIds.Add(baseline.UniqueId);
        return true;
    }

    public static bool IsCombinationBaselineDisabled(BaselineQuants baseline)
        => DisabledCombinationBaselineIds.Contains(baseline.UniqueId);

    // Legacy compatibility wrappers (scheme-driven callers).
    // Prefer candidate-based APIs in new code.
    [Obsolete("Use BanCombinationCandidateForGroup.")]
    public static void BanSchemeForGroup(TensorGroup group, TensorWeightScheme scheme)
        => BanCombinationCandidateForGroup(group, BaselineQuants.FromTensorSchemeId(scheme.UniqueId));

    [Obsolete("Use BanCombinationCandidateForGroupByLearnedBaselineAbsence.")]
    public static void BanSchemeForGroupByLearnedBaselineAbsence(TensorGroup group, TensorWeightScheme scheme, BaselineQuants sourceBaseline)
        => BanCombinationCandidateForGroupByLearnedBaselineAbsence(group, BaselineQuants.FromTensorSchemeId(scheme.UniqueId), sourceBaseline);

    [Obsolete("Use BanAllExplicitCombinationCandidatesForGroup.")]
    public static void BanAllExplicitTensorSchemesForGroup(TensorGroup group)
        => BanAllExplicitCombinationCandidatesForGroup(group);

    [Obsolete("Use IsCombinationCandidateRuntimeBannedForGroup.")]
    public static bool IsSchemeRuntimeBannedForGroup(TensorGroup group, TensorWeightScheme scheme)
        => IsCombinationCandidateRuntimeBannedForGroup(group, BaselineQuants.FromTensorSchemeId(scheme.UniqueId));

    [Obsolete("Use IsGroupExplicitCandidateBanned.")]
    public static bool IsGroupExplicitQuantBanned(TensorGroup group)
        => IsGroupExplicitCandidateBanned(group);
}
