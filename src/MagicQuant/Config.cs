using MagicQuant.Configuration;

namespace MagicQuant;

/// <summary>
/// Process-wide normalized settings for one CLI run. Load through MagicQuantYamlLoader;
/// tests changing this state must restore the previous configuration.
/// </summary>
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

    public static ulong ManualMaxPredictedSizeBytes => Current.Prediction.ManualMaxPredictedSizeBytes;

    public static IReadOnlyList<double> PredictionBitStressThresholdCandidates =>
        Current.Prediction.BitStressThresholdCandidates.Count == 0
            ? new[] { Current.Prediction.DefaultBitStressThreshold }
            : Current.Prediction.BitStressThresholdCandidates;

    public static double PredictionDefaultBitStressThreshold => Current.Prediction.DefaultBitStressThreshold;
    public static int PredictionMinimumFitRows => Math.Max(2, Current.Prediction.MinimumFitRows);
    public static long MaxInMemoryCombinationLoadRows => Math.Max(1L, Current.Prediction.MaxInMemoryCombinationLoadRows);

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

    public static bool SelectionSmartFallbackEnabled =>
        Current.CandidateSelection.SmartFallbackEnabled && SelectionSmartFallbackAttemptsPerFailure > 0;

    public static int SelectionSmartFallbackAttemptsPerFailure =>
        Math.Max(0, Current.CandidateSelection.SmartFallbackAttemptsPerFailure);

    public static int SelectionSmartFallbackMaxHigherFidelitySteps =>
        Math.Max(0, Current.CandidateSelection.SmartFallbackMaxHigherFidelitySteps);

    public static double SelectionMinimumKldImprovementEpsilon =>
        Math.Max(0d, Current.CandidateSelection.MinimumKldImprovementEpsilon);

    public static double SelectionMinimumNeighborGapFractionOfGlobalSpan =>
        Math.Clamp(Current.CandidateSelection.MinimumNeighborGapFractionOfGlobalSpan, 0d, 1d);

    public static double SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan =>
        Math.Clamp(Current.CandidateSelection.NearLowerAnchorBrutalZoneFractionOfPairSpan, 0d, 1d);

    public static double SelectionNearAnchorRequiredKldGainFractionOfPairGap =>
        Math.Max(0d, Current.CandidateSelection.NearAnchorRequiredKldGainFractionOfPairGap);

    public static bool SelectionAllowEightBitAnchorReplacements =>
        Current.CandidateSelection.AllowEightBitAnchorReplacements;

    public static bool SelectionValidateAllAnomalyStrictCandidatesAfterSuccess =>
        Current.CandidateSelection.ValidateAllAnomalyStrictCandidatesAfterSuccess;

    public static bool SelectionDiversifyValidationCandidates =>
        Current.CandidateSelection.DiversifyValidationCandidates;

    public static int SelectionDiversityScanMultiplier =>
        Math.Max(1, Current.CandidateSelection.DiversityScanMultiplier);

    public static int SelectionDiversityScanMinCandidates =>
        Math.Max(1, Current.CandidateSelection.DiversityScanMinCandidates);

    public static int SelectionDiversityScanMaxCandidates =>
        Math.Max(SelectionDiversityScanMinCandidates, Current.CandidateSelection.DiversityScanMaxCandidates);

    public static bool SelectionDiversityLowBitOnly =>
        Current.CandidateSelection.DiversityLowBitOnly;

    public static RuntimeAnomalyDetectionConfig AnomalyDetection => Current.AnomalyDetection;
    public static RuntimeSynergyDetectionConfig SynergyDetection => Current.SynergyDetection;
    public static bool AnomalyDetectionEnabled => Current.AnomalyDetection.Enabled;
    public static bool SynergyDetectionEnabled => Current.SynergyDetection.Enabled;

    public static string? OutputDirectory => Current.Output.OutputDir;
    public static string OutputNamePrefix => string.IsNullOrWhiteSpace(Current.Output.OutputNamePrefix)
        ? "Model"
        : Current.Output.OutputNamePrefix.Trim();

    public static bool ExportExternalLearnedBaselines => Current.Output.ExportExternalLearnedBaselines;
    public static bool AttemptMmprojBuild => Current.Output.AttemptMmprojBuild;
    public static bool RequireMmprojForVisionModels => Current.Output.RequireMmprojForVisionModels;
    public static bool ReuseExistingFinalArtifacts => Current.Output.ReuseExistingFinalArtifacts;

}
