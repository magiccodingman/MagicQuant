using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class RuntimeSearchSpace
{
    private static readonly Dictionary<byte, HashSet<byte>> ExplicitSchemeBansByGroup = new();
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();
    private static readonly HashSet<byte> Bf16SuppressedTensorChoiceGroupIds = new();

    public static void ResetForNewModel()
    {
        ExplicitSchemeBansByGroup.Clear();
        DisabledCombinationBaselineIds.Clear();
        Bf16SuppressedTensorChoiceGroupIds.Clear();
        TensorWeightScheme.ResetAllRuntimeBans();
    }

    public static void BanSchemeForGroup(TensorGroup group, TensorWeightScheme scheme)
    {
        if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId || scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
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

    public static void BanAllExplicitTensorSchemesForGroup(TensorGroup group)
    {
        foreach (var scheme in TensorWeightScheme.All_Allowed_Hybrid_Quants.Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId && x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId))
            BanSchemeForGroup(group, scheme);
    }

    public static IReadOnlyList<TensorWeightScheme> GetRuntimeExplicitBansForGroup(TensorGroup group)
    {
        if (!ExplicitSchemeBansByGroup.TryGetValue(group.UniqueId, out var set))
            return Array.Empty<TensorWeightScheme>();

        return TensorWeightScheme.All_Allowed_Hybrid_Quants.Where(x => set.Contains(x.UniqueId)).OrderBy(x => x.UniqueId).ToList();
    }

    public static bool IsSchemeRuntimeBannedForGroup(TensorGroup group, TensorWeightScheme scheme)
    {
        return ExplicitSchemeBansByGroup.TryGetValue(group.UniqueId, out var set) && set.Contains(scheme.UniqueId);
    }

    public static bool IsGroupExplicitQuantBanned(TensorGroup group)
    {
        var explicitSchemes = TensorWeightScheme.All_Allowed_Hybrid_Quants
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .ToList();

        return explicitSchemes.All(x => x.IsBannedFor(group));
    }

    public static IReadOnlyList<TensorGroup> GetGroupsWithExplicitQuantBanned()
    {
        return TReg.All.Where(IsGroupExplicitQuantBanned).OrderBy(x => x.UniqueId).ToList();
    }

    public static void SuppressBf16TensorChoice(TensorGroup group) => Bf16SuppressedTensorChoiceGroupIds.Add(group.UniqueId);
    public static bool IsBf16TensorChoiceSuppressed(TensorGroup group) => Bf16SuppressedTensorChoiceGroupIds.Contains(group.UniqueId);
    public static IReadOnlyList<TensorGroup> GetBf16SuppressedGroups() => TReg.All.Where(x => Bf16SuppressedTensorChoiceGroupIds.Contains(x.UniqueId)).OrderBy(x => x.UniqueId).ToList();

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

    public static bool IsCombinationBaselineDisabled(BaselineQuants baseline) => DisabledCombinationBaselineIds.Contains(baseline.UniqueId);
}