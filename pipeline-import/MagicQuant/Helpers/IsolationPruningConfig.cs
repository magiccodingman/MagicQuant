namespace MagicQuant.Helpers;

public static class IsolationPruningConfig
{
    public static double MinimumIsolationReductionToContinueRatio => Config.Current.IsolationPruning.MinimumIsolationReductionToContinueRatio;
    public static double MinimumIsolationReductionToSuppressBf16Ratio => Config.Current.IsolationPruning.MinimumIsolationReductionToSuppressBf16Ratio;
    public static double MaximumIsolationPplDeltaPercent => Config.Current.IsolationPruning.MaximumIsolationPplDeltaPercent;
    public static double MaximumIsolationKld => Config.Current.IsolationPruning.MaximumIsolationKld;
    public static double BadTradeMaxSizeDeltaPercent => Config.Current.IsolationPruning.BadTradeMaxSizeDeltaPercent;
    public static double BadTradeKldMultiplier => Config.Current.IsolationPruning.BadTradeKldMultiplier;
    public static double BadTradePplMultiplier => Config.Current.IsolationPruning.BadTradePplMultiplier;
    public static double FloatingPointEpsilon => Config.Current.IsolationPruning.FloatingPointEpsilon;
    public static double MinimumMeaningfulBaseOnlyReductionRatio => Config.Current.IsolationPruning.MinimumMeaningfulBaseOnlyReductionRatio;
}
