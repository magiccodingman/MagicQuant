using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public sealed class RuntimeLearnedBaselineBanInfo
{
    public BaselineQuants Candidate { get; init; } = default!;
    public IReadOnlyList<byte> ExpectedTensorWeightSchemeIds { get; init; } = Array.Empty<byte>();
    public IReadOnlyList<byte> MatchedTensorWeightSchemeIds { get; init; } = Array.Empty<byte>();
    public IReadOnlyList<byte> MissingTensorWeightSchemeIds { get; init; } = Array.Empty<byte>();
    public string Note { get; init; } = string.Empty;
}

public static class RuntimeSearchSpace
{
    private static readonly Dictionary<byte, HashSet<byte>> ExplicitCandidateBansByGroup = new();
    private static readonly Dictionary<byte, Dictionary<byte, RuntimeLearnedBaselineBanInfo>> LearnedPrunesByGroupAndCandidate = new();
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();
    private static readonly HashSet<byte> Bf16SuppressedTensorChoiceGroupIds = new();
    private static bool _imatrixAvailable;

    public static bool AllowHighPrecisionHybrids { get; set; }

    public static void ResetForNewModel()
    {
        ExplicitCandidateBansByGroup.Clear();
        LearnedPrunesByGroupAndCandidate.Clear();
        DisabledCombinationBaselineIds.Clear();
        Bf16SuppressedTensorChoiceGroupIds.Clear();
        _imatrixAvailable = false;
        AllowHighPrecisionHybrids = false;
    }

    public static void SetImatrixAvailability(bool available) => _imatrixAvailable = available;

    public static void ResetForCompatibilityPass()
    {
        ExplicitCandidateBansByGroup.Clear();
        LearnedPrunesByGroupAndCandidate.Clear();
        DisabledCombinationBaselineIds.Clear();
        Bf16SuppressedTensorChoiceGroupIds.Clear();
    }

    public static bool HasUsableImatrix() => _imatrixAvailable;

    public static void BanCombinationCandidateForGroup(TensorGroup group, BaselineQuants candidate)
    {
        if (!ExplicitCandidateBansByGroup.TryGetValue(group.UniqueId, out var set))
        {
            set = new HashSet<byte>();
            ExplicitCandidateBansByGroup[group.UniqueId] = set;
        }

        set.Add(candidate.UniqueId);
    }

    public static void BanCombinationCandidateForGroupDueToLearnedSchemeMismatch(
        TensorGroup group,
        BaselineQuants candidate,
        IReadOnlyCollection<byte> expectedTensorWeightSchemeIds,
        IReadOnlyCollection<byte> matchedTensorWeightSchemeIds,
        string note)
    {
        BanCombinationCandidateForGroup(group, candidate);

        if (!LearnedPrunesByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate))
        {
            byCandidate = new Dictionary<byte, RuntimeLearnedBaselineBanInfo>();
            LearnedPrunesByGroupAndCandidate[group.UniqueId] = byCandidate;
        }

        var expected = expectedTensorWeightSchemeIds
            .Distinct()
            .OrderBy(x => x)
            .ToList();
        var matched = matchedTensorWeightSchemeIds
            .Distinct()
            .OrderBy(x => x)
            .ToList();
        var missing = expected.Except(matched).OrderBy(x => x).ToList();

        byCandidate[candidate.UniqueId] = new RuntimeLearnedBaselineBanInfo
        {
            Candidate = candidate,
            ExpectedTensorWeightSchemeIds = expected,
            MatchedTensorWeightSchemeIds = matched,
            MissingTensorWeightSchemeIds = missing,
            Note = note
        };
    }

    public static void ClearLearnedBaselinePruneBookkeeping() => LearnedPrunesByGroupAndCandidate.Clear();

    public static void ClearLearnedBaselinePruneForGroupCandidate(TensorGroup group, BaselineQuants candidate)
    {
        if (!LearnedPrunesByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate))
            return;

        byCandidate.Remove(candidate.UniqueId);
        if (byCandidate.Count == 0)
            LearnedPrunesByGroupAndCandidate.Remove(group.UniqueId);
    }

    public static void BanAllExplicitCombinationCandidatesForGroup(TensorGroup group)
    {
        foreach (var candidate in GetRealExplicitCombinationCandidatesForGroup(group))
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

    public static IReadOnlyList<BaselineQuants> GetRealExplicitCombinationCandidatesForGroup(TensorGroup group)
    {
        return BaselineQuants.GetGroupCombinationCandidates(_imatrixAvailable, allowHighPrecisionHybrids: false)
            .Where(x => !x.BannedGroupIds.Contains(group.UniqueId))
            .OrderBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();
    }

    public static IReadOnlyList<BaselineQuants> GetAllowedRealExplicitCombinationCandidatesForGroup(TensorGroup group)
    {
        return GetRealExplicitCombinationCandidatesForGroup(group)
            .Where(x => !IsCombinationCandidateRuntimeBannedForGroup(group, x))
            .ToList();
    }

    public static bool HasAnyExplicitCombinationCandidateAllowed(TensorGroup group)
        => GetAllowedRealExplicitCombinationCandidatesForGroup(group).Count > 0;

    public static bool IsGroupExplicitCandidateBanned(TensorGroup group) => !HasAnyExplicitCombinationCandidateAllowed(group);

    public static IReadOnlyList<TensorGroup> GetGroupsWithExplicitQuantBanned()
        => TReg.All.Where(IsGroupExplicitCandidateBanned).OrderBy(x => x.UniqueId).ToList();

    public static bool HasLearnedBaselineMissingPrunesForGroup(TensorGroup group)
        => LearnedPrunesByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate) && byCandidate.Count > 0;

    public static IReadOnlyList<TensorGroup> GetGroupsWithLearnedBaselineMissingPrunes()
        => TReg.All.Where(HasLearnedBaselineMissingPrunesForGroup).OrderBy(x => x.UniqueId).ToList();

    public static IReadOnlyList<RuntimeLearnedBaselineBanInfo> GetLearnedBaselineMissingPrunedCandidatesForGroup(TensorGroup group)
    {
        if (!LearnedPrunesByGroupAndCandidate.TryGetValue(group.UniqueId, out var byCandidate))
            return Array.Empty<RuntimeLearnedBaselineBanInfo>();

        return byCandidate
            .OrderBy(x => x.Key)
            .Select(x => x.Value)
            .ToList();
    }

    public static void SuppressBf16TensorChoice(TensorGroup group) => Bf16SuppressedTensorChoiceGroupIds.Add(group.UniqueId);

    public static bool IsBf16TensorChoiceSuppressed(TensorGroup group)
        => Bf16SuppressedTensorChoiceGroupIds.Contains(group.UniqueId) && HasAnyExplicitCombinationCandidateAllowed(group);

    public static IReadOnlyList<TensorGroup> GetBf16SuppressedGroups()
        => TReg.All.Where(IsBf16TensorChoiceSuppressed).OrderBy(x => x.UniqueId).ToList();

    public static string GetDisplayStateForGroup(TensorGroup group)
    {
        if (Cache.UnusedTensorGroups.Any(x => x.UniqueId == group.UniqueId))
            return "unused->NULL";

        if (IsGroupExplicitCandidateBanned(group))
            return "explicit-banned->Q8-fallback";

        if (IsBf16TensorChoiceSuppressed(group))
            return "BF16-suppressed";

        if (HasLearnedBaselineMissingPrunesForGroup(group))
            return "learned-pruned";

        return "variable";
    }

    public static (bool ExplicitAllowed, bool Bf16Allowed) GetFinalAllowedQuantFamiliesForGroup(TensorGroup group)
    {
        bool explicitAllowed = HasAnyExplicitCombinationCandidateAllowed(group);
        bool bf16Allowed = !IsBf16TensorChoiceSuppressed(group) || !explicitAllowed;
        return (explicitAllowed, bf16Allowed);
    }

    public static IReadOnlyList<BaselineQuants> GetActiveCombinationBaselines()
    {
        var active = BaselineQuants.GetCombinationCarrierBaselines(_imatrixAvailable)
            .Where(x => !DisabledCombinationBaselineIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        if (active.Count == 0)
            return new[] { BaselineQuants.Q8_0 };

        return active;
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

    [Obsolete("Use BanCombinationCandidateForGroup.")]
    public static void BanSchemeForGroup(TensorGroup group, TensorWeightScheme scheme)
        => BanCombinationCandidateForGroup(group, BaselineQuants.FromTensorSchemeId(scheme.UniqueId));

    [Obsolete("Use BanCombinationCandidateForGroupDueToLearnedSchemeMismatch.")]
    public static void BanSchemeForGroupByLearnedBaselineAbsence(TensorGroup group, TensorWeightScheme scheme, BaselineQuants sourceBaseline)
        => BanCombinationCandidateForGroupDueToLearnedSchemeMismatch(
            group,
            BaselineQuants.FromTensorSchemeId(scheme.UniqueId),
            expectedTensorWeightSchemeIds: [scheme.UniqueId],
            matchedTensorWeightSchemeIds: Array.Empty<byte>(),
            note: $"Legacy scheme-ban shim invoked for source baseline '{sourceBaseline.Names[0]}'.");

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