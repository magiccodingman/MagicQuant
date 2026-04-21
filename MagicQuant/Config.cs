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

    public static List<string> SensitivityProbeGroups => Current.SensitivityProbeGroups;
    public static List<string> SensitivityProbeGroupsMoe => Current.SensitivityProbeGroupsMoe;
    public static List<string> BrainLayers => Current.BrainLayers;
    public static List<string> CollapsePenaltySchemes => Current.CollapsePenaltySchemes;
    public static List<string> MoeIndicatorTensors => Current.MoeIndicatorTensors;
}
