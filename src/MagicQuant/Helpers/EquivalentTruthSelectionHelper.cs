using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class EquivalentTruthSelectionHelper
{
    public static bool AreEquivalentTruths(
        ulong leftSizeBytes,
        double leftKld,
        double leftPpl,
        ulong rightSizeBytes,
        double rightKld,
        double rightPpl)
    {
        if (leftSizeBytes != rightSizeBytes)
            return false;

        return Math.Abs(leftKld - rightKld) <= IsolationPruningConfig.FloatingPointEpsilon &&
               Math.Abs(leftPpl - rightPpl) <= IsolationPruningConfig.FloatingPointEpsilon;
    }

    public static int GetBaselineSafetyRank(
        BaselineQuants baseline,
        bool isHybrid,
        bool isExternalPureBaseline)
    {
        // Prefer the safest / most default representative when multiple rows have identical truth.
        // 1) Higher BitRange is safer.
        // 2) Higher ExplicitCandidateSortOrder wins ties inside the same BitRange.
        // 3) Pure baseline beats hybrid when the measured truth is identical.
        // 4) Internal/non-external beats external pure reference when still tied.
        int rank = baseline.BitRange * 10_000;
        rank += baseline.ExplicitCandidateSortOrder * 10;
        rank += isHybrid ? 0 : 2;
        rank += isExternalPureBaseline ? 0 : 1;
        return rank;
    }
}
