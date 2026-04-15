namespace MagicQuant.Helpers;

public static class IsolationPruningConfig
{
    public const double MinimumIsolationReductionToContinueRatio = 0.04d;
    public const double MinimumIsolationReductionToSuppressBf16Ratio = 0.10d;
    public const double MaximumIsolationPplDeltaPercent = 5.0d;
    public const double MaximumIsolationKld = 0.1d;
    public const double BadTradeMaxSizeDeltaPercent = 5.0d;
    public const double BadTradeKldMultiplier = 2.0d;
    public const double BadTradePplMultiplier = 3.0d;
    public const double FloatingPointEpsilon = 1e-8d;
    public const double MinimumMeaningfulBaseOnlyReductionRatio = 0.01d;
}