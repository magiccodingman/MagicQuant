using MagicQuant.Configuration;

namespace MagicQuant;

public static class Config
{
    public static MagicQuantYamlConfig Current { get; private set; } = MagicQuantYamlConfig.CreateDefault();

    public static void Load(MagicQuantYamlConfig config)
    {
        Current = config ?? throw new ArgumentNullException(nameof(config));
    }

    public static void SetResolvedCustomBaselines(IEnumerable<ResolvedCustomBaselineSpec> baselines)
    {
        Current.Baselines.ResolvedCustomBaselines = baselines?.ToList() ?? new List<ResolvedCustomBaselineSpec>();
    }

    public static ResolvedCustomBaselineSpec? GetResolvedCustomBaseline(string canonicalKey)
    {
        return Current.Baselines.ResolvedCustomBaselines.FirstOrDefault(x =>
            string.Equals(x.CanonicalKey, canonicalKey, StringComparison.Ordinal));
    }

    public static int MaxDataCollectedPerCategory => Current.Evolution.MaxDataCollectedPerCategory;
    public static int MaxSurvivalRounds => Current.Evolution.MaxSurvivalRounds;
    public static double CollapseMultiplier => Current.Evolution.CollapseMultiplier;
    public static int BruteForceFinalCombinationThreshold => Current.Evolution.BruteForceFinalCombinationThreshold;
    public static ulong ManualMaxPredictedSizeBytes => Current.Prediction.ManualMaxPredictedSizeBytes;

    public static string? OutputDirectory => Current.Output.OutputDir;
    public static string OutputNamePrefix => string.IsNullOrWhiteSpace(Current.Output.OutputNamePrefix)
        ? "model"
        : Current.Output.OutputNamePrefix.Trim();

    public static bool ExportExternalLearnedBaselines => Current.Output.ExportExternalLearnedBaselines;

    public static int MaxSelectedChoicesPerBucket => Math.Max(1, Current.Survival.MaxSelectedChoicesPerBucket);
    public static double SurvivalMeaningfulSizeBiasPercent => Current.Survival.MeaningfulSizeBiasPercent;
    public static double SurvivalKldCloseCallAbsoluteEpsilon => Current.Survival.KldCloseCallAbsoluteEpsilon;
    public static double SurvivalKldCloseCallRelativeFraction => Current.Survival.KldCloseCallRelativeFraction;
    public static double SurvivalPplLargeDifferencePercent => Current.Survival.PplLargeDifferencePercent;
    public static double SurvivalTradeScoreSizeBiasWeight => Current.Survival.TradeScoreSizeBiasWeight;
    public static double SurvivalTradeScorePplWeight => Current.Survival.TradeScorePplWeight;

    public static List<string> SensitivityProbeGroups => Current.SensitivityProbeGroups;
    public static List<string> SensitivityProbeGroupsMoe => Current.SensitivityProbeGroupsMoe;
    public static List<string> BrainLayers => Current.BrainLayers;
    public static List<string> CollapsePenaltySchemes => Current.CollapsePenaltySchemes;
    public static List<string> MoeIndicatorTensors => Current.MoeIndicatorTensors;
}
