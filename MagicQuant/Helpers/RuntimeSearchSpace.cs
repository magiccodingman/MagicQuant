using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class RuntimeSearchSpace
{
    private static readonly HashSet<byte> NativeLockedGroupIds = new();
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();

    public static void ResetForNewModel()
    {
        NativeLockedGroupIds.Clear();
        DisabledCombinationBaselineIds.Clear();
        TensorWeightScheme.ResetAllRuntimeBans();
    }

    public static bool IsGroupLockedToNative(TensorGroup group)
    {
        return NativeLockedGroupIds.Contains(group.UniqueId);
    }

    public static void LockGroupToNative(TensorGroup group)
    {
        if (!NativeLockedGroupIds.Add(group.UniqueId))
            return;

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId)
                continue;

            if (scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
                continue;

            if (!scheme.BannedGroups.Any(x => x.UniqueId == group.UniqueId))
                scheme.BannedGroups.Add(group);
        }
    }

    public static IReadOnlyList<TensorGroup> GetNativeLockedGroups()
    {
        return TReg.All
            .Where(g => NativeLockedGroupIds.Contains(g.UniqueId))
            .OrderBy(g => g.UniqueId)
            .ToList();
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
        if (DisabledCombinationBaselineIds.Contains(baseline.UniqueId))
            return false;

        int currentlyActive = GetActiveCombinationBaselines().Count;
        if (!allowDisablingLast && currentlyActive <= 1)
            return false;

        DisabledCombinationBaselineIds.Add(baseline.UniqueId);
        return true;
    }

    public static bool IsCombinationBaselineDisabled(BaselineQuants baseline)
    {
        return DisabledCombinationBaselineIds.Contains(baseline.UniqueId);
    }
}