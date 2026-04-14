using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class RuntimeSearchSpace
{
    private static readonly HashSet<byte> DisabledCombinationBaselineIds = new();

    public static void ResetForNewModel()
    {
        DisabledCombinationBaselineIds.Clear();
        TensorWeightScheme.ResetAllRuntimeBans();
    }

    public static bool BanSchemeForGroup(TensorGroup group, TensorWeightScheme scheme)
    {
        if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId)
            return false;

        if (scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
            return false;

        if (scheme.BannedGroups.Any(x => x.UniqueId == group.UniqueId))
            return false;

        scheme.BannedGroups.Add(group);
        return true;
    }

    public static int BanAllExplicitTensorSchemesForGroup(TensorGroup group)
    {
        int applied = 0;

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.UniqueId == TensorWeightScheme.NULL.UniqueId)
                continue;

            if (scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
                continue;

            if (BanSchemeForGroup(group, scheme))
                applied++;
        }

        return applied;
    }

    public static IReadOnlyList<TensorWeightScheme> GetRuntimeExplicitBansForGroup(TensorGroup group)
    {
        return TensorWeightScheme.All
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .Where(x => x.IsBannedFor(group))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    public static bool IsGroupExplicitQuantBanned(TensorGroup group)
    {
        var explicitSchemes = TensorWeightScheme.All
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .ToList();

        if (explicitSchemes.Count == 0)
            return false;

        return explicitSchemes.All(x => x.IsBannedFor(group));
    }

    public static IReadOnlyList<TensorGroup> GetGroupsWithExplicitQuantBanned()
    {
        return TReg.All
            .Where(IsGroupExplicitQuantBanned)
            .OrderBy(x => x.UniqueId)
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