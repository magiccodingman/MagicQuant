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

    // Legacy evolution knobs are retained for old helper compatibility, but the final
    // hybrid chooser now uses rank-safe isolation prediction and candidate_selection.
    public static int MaxDataCollectedPerCategory => Current.Evolution.MaxDataCollectedPerCategory;
    public static int MaxSurvivalRounds => Current.Evolution.MaxSurvivalRounds;
    public static double CollapseMultiplier => Current.Evolution.CollapseMultiplier;
    public static int BruteForceFinalCombinationThreshold => Current.Evolution.BruteForceFinalCombinationThreshold;
    public static ulong ManualMaxPredictedSizeBytes => Current.Prediction.ManualMaxPredictedSizeBytes;

    public static IReadOnlyList<double> PredictionBitStressThresholdCandidates =>
        Current.Prediction.BitStressThresholdCandidates.Count == 0
            ? new[] { Current.Prediction.DefaultBitStressThreshold }
            : Current.Prediction.BitStressThresholdCandidates;

    public static double PredictionDefaultBitStressThreshold => Current.Prediction.DefaultBitStressThreshold;
    public static int PredictionMinimumFitRows => Math.Max(2, Current.Prediction.MinimumFitRows);

    public static double SelectionNearBaselineMaxSizeGrowthPercent =>
        Math.Max(0d, Current.CandidateSelection.NearBaselineMaxSizeGrowthPercent);

    public static IReadOnlyList<double> SelectionInteriorWindowFractions =>
        Current.CandidateSelection.InteriorWindowFractions.Count == 0
            ? new[] { 0.35d, 0.35d }
            : Current.CandidateSelection.InteriorWindowFractions;

    public static int SelectionMaxCandidatesPerInteriorWindow =>
        Math.Max(1, Current.CandidateSelection.MaxCandidatesPerInteriorWindow);

    public static int SelectionMaxFallbackAttemptsPerAnchor =>
        Math.Max(1, Current.CandidateSelection.MaxFallbackAttemptsPerAnchor);

    public static double SelectionMinimumKldImprovementEpsilon =>
        Math.Max(0d, Current.CandidateSelection.MinimumKldImprovementEpsilon);

    public static double SelectionMinimumNeighborGapFractionOfGlobalSpan =>
        Math.Clamp(Current.CandidateSelection.MinimumNeighborGapFractionOfGlobalSpan, 0d, 1d);

    public static double SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan =>
        Math.Clamp(Current.CandidateSelection.NearLowerAnchorBrutalZoneFractionOfPairSpan, 0d, 1d);

    public static double SelectionNearAnchorRequiredKldGainFractionOfPairGap =>
        Math.Max(0d, Current.CandidateSelection.NearAnchorRequiredKldGainFractionOfPairGap);

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