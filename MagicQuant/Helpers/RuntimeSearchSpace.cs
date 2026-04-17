using System;
using System.Collections.Generic;
using System.Linq;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public sealed class RuntimeLearnedBaselineBanInfo
{
    public TensorWeightScheme Scheme { get; init; } = default!;
    public IReadOnlyList<BaselineQuants> MissingBaselines { get; init; } = Array.Empty<BaselineQuants>();
}

public static class RuntimeSearchSpace
{
    private static readonly Dictionary<byte, HashSet<byte>> ExplicitSchemeBansByGroup = new();
    private static readonly Dictionary<byte, Dictionary<byte, HashSet<byte>>> LearnedBaselineMissingByGroupAndScheme = new();
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();
    private static readonly HashSet<byte> Bf16SuppressedTensorChoiceGroupIds = new();

    public static void ResetForNewModel()
    {
        ExplicitSchemeBansByGroup.Clear();
        LearnedBaselineMissingByGroupAndScheme.Clear();
        DisabledCombinationBaselineIds.Clear();
        Bf16SuppressedTensorChoiceGroupIds.Clear();
        TensorWeightScheme.ResetAllRuntimeBans();
    }

    public static void BanSchemeForGroup(TensorGroup group, TensorWeightScheme scheme)
    {
        if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId ||
            scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
            return;

        if (!ExplicitSchemeBansByGroup.TryGetValue(group.UniqueId, out var set))
        {
            set = new HashSet<byte>();
            ExplicitSchemeBansByGroup[group.UniqueId] = set;
        }

        set.Add(scheme.UniqueId);

        if (!scheme.IsBannedFor(group))
            scheme.BannedGroups.Add(group);
    }

    public static void BanSchemeForGroupByLearnedBaselineAbsence(
        TensorGroup group,
        TensorWeightScheme scheme,
        BaselineQuants sourceBaseline)
    {
        if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId ||
            scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
            return;

        BanSchemeForGroup(group, scheme);

        if (!LearnedBaselineMissingByGroupAndScheme.TryGetValue(group.UniqueId, out var byScheme))
        {
            byScheme = new Dictionary<byte, HashSet<byte>>();
            LearnedBaselineMissingByGroupAndScheme[group.UniqueId] = byScheme;
        }

        if (!byScheme.TryGetValue(scheme.UniqueId, out var baselineIds))
        {
            baselineIds = new HashSet<byte>();
            byScheme[scheme.UniqueId] = baselineIds;
        }

        baselineIds.Add(sourceBaseline.UniqueId);
    }

    public static void BanAllExplicitTensorSchemesForGroup(TensorGroup group)
    {
        foreach (var scheme in TensorWeightScheme.All_Allowed_Hybrid_Quants
                     .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId &&
                                 x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId))
        {
            BanSchemeForGroup(group, scheme);
        }
    }

    public static IReadOnlyList<TensorWeightScheme> GetRuntimeExplicitBansForGroup(TensorGroup group)
    {
        if (!ExplicitSchemeBansByGroup.TryGetValue(group.UniqueId, out var set))
            return Array.Empty<TensorWeightScheme>();

        return TensorWeightScheme.All_Allowed_Hybrid_Quants
            .Where(x => set.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool IsSchemeRuntimeBannedForGroup(TensorGroup group, TensorWeightScheme scheme)
    {
        return ExplicitSchemeBansByGroup.TryGetValue(group.UniqueId, out var set) &&
               set.Contains(scheme.UniqueId);
    }

    public static bool IsGroupExplicitQuantBanned(TensorGroup group)
    {
        return !HasAnyExplicitSchemeAllowed(group);
    }

    public static IReadOnlyList<TensorGroup> GetGroupsWithExplicitQuantBanned()
    {
        return TReg.All
            .Where(IsGroupExplicitQuantBanned)
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool HasLearnedBaselineMissingPrunesForGroup(TensorGroup group)
    {
        return LearnedBaselineMissingByGroupAndScheme.TryGetValue(group.UniqueId, out var byScheme) &&
               byScheme.Count > 0;
    }

    public static IReadOnlyList<TensorGroup> GetGroupsWithLearnedBaselineMissingPrunes()
    {
        return TReg.All
            .Where(HasLearnedBaselineMissingPrunesForGroup)
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static IReadOnlyList<RuntimeLearnedBaselineBanInfo> GetLearnedBaselineMissingPrunedSchemesForGroup(
        TensorGroup group)
    {
        if (!LearnedBaselineMissingByGroupAndScheme.TryGetValue(group.UniqueId, out var byScheme))
            return Array.Empty<RuntimeLearnedBaselineBanInfo>();

        var result = new List<RuntimeLearnedBaselineBanInfo>();

        foreach (var kvp in byScheme.OrderBy(x => x.Key))
        {
            var scheme = TensorWeightScheme.All_Allowed_Hybrid_Quants
                .FirstOrDefault(x => x.UniqueId == kvp.Key);

            if (scheme == null)
                continue;

            var baselines = kvp.Value
                .OrderBy(x => x)
                .Select(BaselineQuants.FromId)
                .ToList();

            result.Add(new RuntimeLearnedBaselineBanInfo
            {
                Scheme = scheme,
                MissingBaselines = baselines
            });
        }

        return result;
    }

    public static void SuppressBf16TensorChoice(TensorGroup group)
        => Bf16SuppressedTensorChoiceGroupIds.Add(group.UniqueId);

    public static bool IsBf16TensorChoiceSuppressed(TensorGroup group)
    {
        // BF16 suppression is only meaningful while at least one explicit tensor scheme remains.
        // If explicit schemes are all banned, BF16 becomes the only viable tensor choice.
        return Bf16SuppressedTensorChoiceGroupIds.Contains(group.UniqueId) &&
               HasAnyExplicitSchemeAllowed(group);
    }

    public static IReadOnlyList<TensorGroup> GetBf16SuppressedGroups()
    {
        return TReg.All
            .Where(IsBf16TensorChoiceSuppressed)
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static (bool ExplicitAllowed, bool Bf16Allowed) GetFinalAllowedQuantFamiliesForGroup(TensorGroup group)
    {
        bool explicitAllowed = HasAnyExplicitSchemeAllowed(group);
        bool bf16Allowed = !IsBf16TensorChoiceSuppressed(group);
        return (explicitAllowed, bf16Allowed);
    }

    private static bool HasAnyExplicitSchemeAllowed(TensorGroup group)
    {
        return TensorWeightScheme.All_Allowed_Hybrid_Quants
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .Any(x => !x.IsBannedFor(group));
    }

    public static IReadOnlyList<BaselineQuants> GetActiveCombinationBaselines()
    {
        return BaselineQuants.All
            .Where(x => x.BaseConversionBase != null)
            .Where(x => !DisabledCombinationBaselineIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool DisableCombinationBaseline(BaselineQuants baseline, bool allowDisablingLast = false)
    {
        if (baseline.BaseConversionBase == null || DisabledCombinationBaselineIds.Contains(baseline.UniqueId))
            return false;

        int currentlyActive = GetActiveCombinationBaselines().Count;
        if (!allowDisablingLast && currentlyActive <= 1)
            return false;

        DisabledCombinationBaselineIds.Add(baseline.UniqueId);
        return true;
    }

    public static bool IsCombinationBaselineDisabled(BaselineQuants baseline)
        => DisabledCombinationBaselineIds.Contains(baseline.UniqueId);
}
