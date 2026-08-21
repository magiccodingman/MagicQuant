using YamlDotNet.Serialization;

namespace MagicQuant.Configuration;

public sealed class MagicQuantYamlConfig
{
    public RuntimePathConfig Paths { get; set; } = new();
    public RuntimeFlagConfig Flags { get; set; } = new();
    public RuntimeReadmeConfig Readme { get; set; } = new();
    public RuntimeImatrixConfig Imatrix { get; set; } = new();
    public RuntimeEvolutionConfig Evolution { get; set; } = new();
    public RuntimeIsolationPruningConfig IsolationPruning { get; set; } = new();
    public RuntimePredictionConfig Prediction { get; set; } = new();
    public RuntimeIdentityConfig Identity { get; set; } = new();
    public RuntimeBaselineConfig Baselines { get; set; } = new();
    public RuntimeLearningConfig Learning { get; set; } = new();
    public RuntimeOutputConfig Output { get; set; } = new();
    public RuntimeSurvivalConfig Survival { get; set; } = new();
    public RuntimeCandidateSelectionConfig CandidateSelection { get; set; } = new();
    public RuntimeAnomalyDetectionConfig AnomalyDetection { get; set; } = new();
    public RuntimeSynergyDetectionConfig SynergyDetection { get; set; } = new();
    public RuntimeHardwareConfig Hardware { get; set; } = new();

    public List<string> SensitivityProbeGroups { get; set; } =
    [
        "embeddings",
        "lm_head",
        "attn_q",
        "attn_kv",
        "attn_output",
        "ffn_up_gate",
        "ffn_down"
    ];

    public List<string> SensitivityProbeGroupsMoe { get; set; } =
    [
        "moe_router",
        "moe_experts"
    ];

    public List<string> BrainLayers { get; set; } =
    [
        "embeddings",
        "lm_head",
        "attn_output"
    ];

    public List<string> CollapsePenaltySchemes { get; set; } =
    [
        "IQ1_S",
        "IQ1_M",
        "MXFP4",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S"
    ];

    public List<string> MoeIndicatorTensors { get; set; } =
    [
        "blk.*.ffn_up_expert_0.weight",
        "blk.*.ffn_gate_expert_0.weight",
        "blk.*.ffn_down_expert_0.weight",
        "blk.*.ffn_up_exps.weight",
        "blk.*.ffn_gate_exps.weight",
        "blk.*.ffn_down_exps.weight",
        "blk.*.ffn_gate_inp.weight",
        "router.weight",
        "gate.weight",
        "blk.*.router.*",
        "blk.*.gate_proj.*",
        "blk.*.gate_inp.*",
        "model.language_model.layers.*.mlp.experts.gate_up_proj",
        "model.language_model.layers.*.mlp.experts.down_proj",
        "model.language_model.layers.*.mlp.gate.weight",
        "model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.*.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.*.experts.gate_up_proj",
        "model.language_model.layers.*.experts.down_proj",
        "model.language_model.layers.*.router.proj.weight",
        "model.language_model.layers.*.router.per_expert_scale",
        "model.language_model.layers.*.router.scale"
    ];

    public static MagicQuantYamlConfig CreateDefault() => new();
}

public sealed class RuntimePathConfig
{
    public string? MagicQuantRoot { get; set; }
    public string? ModelDir { get; set; }
    public string? LlamaRoot { get; set; }
    public string? LlamaBin { get; set; }
    public string? ConvertScript { get; set; }
    public List<string> ScratchRoots { get; set; } = new();
    public string ExternalBaselineCacheDirName { get; set; } = "ExternalBaselines";
}

public sealed class RuntimeFlagConfig
{
    public bool UseImatrix { get; set; }
    public bool ForceImatrixRebuild { get; set; }
    public bool ForceRefreshHardwareProbe { get; set; }
    public bool AllowHighPrecisionHybrids { get; set; }
}

public sealed class RuntimeReadmeConfig
{
    public string? TitleModelNameOverride { get; set; }

    // Flexible by design: Hugging Face frontmatter can grow without requiring
    // new strongly typed C# properties for every key.
    public Dictionary<string, object?> Frontmatter { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}

public sealed class RuntimeImatrixConfig
{
    public string? ImatrixUrl { get; set; }
    public string? DatasetRepo { get; set; }
    public string? DatasetSplit { get; set; }
    public string? DatasetConfig { get; set; }
    public string? DatasetLocalFile { get; set; }
}

public sealed class RuntimeEvolutionConfig
{
    public int MaxDataCollectedPerCategory { get; set; } = 5;
    public int MaxSurvivalRounds { get; set; } = 4;
    public double CollapseMultiplier { get; set; } = 1.5d;
    public int BruteForceFinalCombinationThreshold { get; set; } = 2_000;
}

public sealed class RuntimeIsolationPruningConfig
{
    public double MinimumIsolationReductionToContinueRatio { get; set; } = 0.04d;
    public double MinimumIsolationReductionToSuppressBf16Ratio { get; set; } = 0.10d;
    public double MaximumIsolationPplDeltaPercent { get; set; } = 5.0d;
    public double MaximumIsolationKld { get; set; } = 0.1d;
    public double BadTradeMaxSizeDeltaPercent { get; set; } = 4.0d;
    public double BadTradeKldMultiplier { get; set; } = 2.5d;
    public double BadTradePplMultiplier { get; set; } = 3.5d;
    public double FloatingPointEpsilon { get; set; } = 1e-8d;
    public double MinimumMeaningfulBaseOnlyReductionRatio { get; set; } = 0.01d;
}

public sealed class RuntimePredictionConfig
{
    /// <summary>
    /// Legacy emergency ceiling. Keep at 0 for the rank-safe isolation predictor.
    /// </summary>
    public ulong ManualMaxPredictedSizeBytes { get; set; } = 0;

    /// <summary>
    /// Candidate thresholds used while fitting the low-bit interaction correction.
    /// The best threshold is selected by lowest MAE against existing general-category truth.
    /// </summary>
    public List<double> BitStressThresholdCandidates { get; set; } =
    [
        4.0d,
        5.0d,
        6.0d,
        7.0d,
        8.0d,
        9.0d,
        10.0d,
        11.0d,
        12.0d
    ];

    public double DefaultBitStressThreshold { get; set; } = 8.0d;
    public int MinimumFitRows { get; set; } = 12;
    public long MaxInMemoryCombinationLoadRows { get; set; } = 5_000_000;
}

public sealed class RuntimeIdentityConfig
{
    public string? ArchitectureFamilyName { get; set; }
    public bool AllowArchitectureFamilyAliasOverride { get; set; }
}

public sealed class RuntimeOutputConfig
{
    public string? OutputDir { get; set; }
    public string OutputNamePrefix { get; set; } = "Model";
    public bool ExportExternalLearnedBaselines { get; set; } = false;
    public bool AttemptMmprojBuild { get; set; } = true;
    public bool RequireMmprojForVisionModels { get; set; } = false;
    public bool ReuseExistingFinalArtifacts { get; set; } = false;
}

public sealed class RuntimeSurvivalConfig
{
    public int MaxSelectedChoicesPerBucket { get; set; } = 5;
    public double MeaningfulSizeBiasPercent { get; set; } = 1.0d;
    public double KldCloseCallAbsoluteEpsilon { get; set; } = 0.00075d;
    public double KldCloseCallRelativeFraction { get; set; } = 0.02d;
    public double PplLargeDifferencePercent { get; set; } = 0.75d;
    public double TradeScoreSizeBiasWeight { get; set; } = 1.25d;
    public double TradeScorePplWeight { get; set; } = 0.15d;
}

public sealed class RuntimeCandidateSelectionConfig
{
    /// <summary>
    /// Phase 2 window. 1.0 means "up to one percent larger than the smaller/higher-damage anchor".
    /// </summary>
    public double NearBaselineMaxSizeGrowthPercent { get; set; } = 1.0d;

    /// <summary>
    /// Phase 3 windows as fractions of each adjacent anchor-pair size span.
    /// Example [0.35, 0.35] tests the first 35% and next 35% of the span.
    /// </summary>
    public List<double> InteriorWindowFractions { get; set; } =
    [
        0.35d,
        0.35d
    ];

    public int MaxCandidatesPerInteriorWindow { get; set; } = 1;
    public int MaxFallbackAttemptsPerAnchor { get; set; } = 5;

    /// <summary>
    /// Enables the conservative SQLite/isolation-truth baseline tuning fallback.
    /// This does not query DuckDB and only runs after a normal phase fails to
    /// validate a candidate for its anchor/window.
    /// </summary>
    public bool SmartFallbackEnabled { get; set; } = true;

    /// <summary>
    /// Extra build/benchmark attempts permitted after the normal prediction-guided
    /// attempts fail for a strict, near-baseline, or interior window.
    /// </summary>
    public int SmartFallbackAttemptsPerFailure { get; set; } = 3;

    /// <summary>
    /// Maximum number of higher-fidelity anchor steps the smart fallback may climb
    /// for a single tensor group. Lower-fidelity swaps are still only allowed when
    /// their isolated KLD is measurably better than the baseline group state.
    /// </summary>
    public int SmartFallbackMaxHigherFidelitySteps { get; set; } = 2;

    /// <summary>
    /// Strict epsilon for "lower KLD" claims. This is intentionally tiny because
    /// the validator verifies the final relationship against real benchmark truth.
    /// </summary>
    public double MinimumKldImprovementEpsilon { get; set; } = 1e-9d;

    /// <summary>
    /// Final spacing pass: candidates closer than this fraction of the global survivor
    /// size span are collapsed to a single winner.
    /// </summary>
    public double MinimumNeighborGapFractionOfGlobalSpan { get; set; } = 0.03d;

    /// <summary>
    /// Extra-brutal near-small-anchor zone. A candidate extremely close to the smaller
    /// anchor must earn a larger KLD gain to survive.
    /// </summary>
    public double NearLowerAnchorBrutalZoneFractionOfPairSpan { get; set; } = 0.02d;

    /// <summary>
    /// Required gain fraction of the adjacent-anchor KLD gap when a candidate sits in
    /// the near-small-anchor brutal zone.
    /// </summary>
    public double NearAnchorRequiredKldGainFractionOfPairGap { get; set; } = 0.05d;

    /// <summary>
    /// When false, the prediction selector does not spend build/benchmark attempts trying
    /// to replace 8-bit anchors such as Q8_0 during strict dominance or near-anchor checks.
    /// Q8 remains the highest-fidelity practical anchor unless this is explicitly enabled.
    /// </summary>
    public bool AllowEightBitAnchorReplacements { get; set; } = false;

    /// <summary>
    /// Legacy/diagnostic mode for strict Q8/anomaly discovery. When false, once a strict
    /// candidate validates for an anchor, MagicQuant accepts it and stops spending more
    /// build/benchmark attempts on the rest of the fetched top-N list.
    /// </summary>
    public bool ValidateAllAnomalyStrictCandidatesAfterSuccess { get; set; } = false;

    /// <summary>
    /// When true, windows with more predicted candidates than validation attempts fetch a
    /// bounded scan pool and round-robin across candidate theory families before validation.
    /// </summary>
    public bool DiversifyValidationCandidates { get; set; } = true;

    /// <summary>
    /// Scan roughly attemptLimit * multiplier predicted rows before selecting final attempts.
    /// </summary>
    public int DiversityScanMultiplier { get; set; } = 25;

    /// <summary>
    /// Lower bound for the prediction-only scan pool when diversity is active.
    /// </summary>
    public int DiversityScanMinCandidates { get; set; } = 100;

    /// <summary>
    /// Upper bound for the prediction-only scan pool when diversity is active.
    /// </summary>
    public int DiversityScanMaxCandidates { get; set; } = 500;

    /// <summary>
    /// Optional escape hatch: if true, diversify only windows whose anchor band is Q4-ish or below.
    /// Defaults false because diversity is cheap and does not increase validation attempts.
    /// </summary>
    public bool DiversityLowBitOnly { get; set; } = false;
}


public sealed class RuntimeAnomalyDetectionConfig
{
    public bool Enabled { get; set; } = true;
    public int MaxAnomalyRefinementRounds { get; set; } = 1;
    public double MinActualGainVsTwinKld { get; set; } = 0.00025d;
    public double MinPredictedSizeSavingsVsTwinPercent { get; set; } = 1.0d;
    public int MaxProbeGroupCount { get; set; } = 4;
    public int MaxProbesPerSeed { get; set; } = 16;
    public int MaxTotalProbesPerRun { get; set; } = 32;
    public double MaxPredictionSpaceGapVsTwinKld { get; set; } = 0.00050d;
    public double MaxRelativePredictionPenaltyVsTwin { get; set; } = 0.35d;
    public double PredictionSpaceViolationMargin { get; set; } = 0.00005d;
    public double AnomalyAdjustmentShrinkFactor { get; set; } = 0.50d;
    public double MinRuleConfidenceToApply { get; set; } = 0.50d;
    public double MaxNegativeAdjustmentKld { get; set; } = 0.00075d;
    public double MaxPositiveAdjustmentKld { get; set; } = 0.00075d;
    public double MaxAdjustmentFractionOfBaseKld { get; set; } = 0.75d;

    /// <summary>
    /// Advisory diagnostics cap for confirmed pairwise ordering corrections. Beneficial
    /// pairwise rules are allowed to cross their own measured twin even when the required
    /// adjustment exceeds this value; the cap is reported, not used to resurrect the old
    /// broad-boost poison.
    /// </summary>
    public double MaxConfirmedPairwiseOrderingAdjustmentKld { get; set; } = 0.006d;

    public int MaxSmokeCandidatesPerReferenceZone { get; set; } = 12;
    public bool PersistSuppressionResults { get; set; } = true;
    public bool VerboseAnomalyLogging { get; set; } = true;
    public RuntimeConfirmedAnomalyExpansionConfig ConfirmedAnomalyExpansion { get; set; } = new();
}

public sealed class RuntimeConfirmedAnomalyExpansionConfig
{
    public bool Enabled { get; set; } = true;
    public int MaxNeighborsPerConfirmedRule { get; set; } = 6;
    public int MaxTotalExpansionProbes { get; set; } = 12;
    public List<string> AllowedReferenceQuants { get; set; } = ["Q8_0"];
    public List<string> AllowedCandidateQuants { get; set; } = ["Q6_K", "UD-Q6_K_XL", "Q5_K", "UD-Q5_K_XL"];
}


public sealed class RuntimeSynergyDetectionConfig
{
    public bool Enabled { get; set; } = true;
    public int MaxRefinementRounds { get; set; } = 1;
    public double ExactContextConfidenceMultiplier { get; set; } = 1.00d;
    public double SameSelectedGroupsConfidenceMultiplier { get; set; } = 0.55d;
    public double EquivalentQuantFamilyConfidenceMultiplier { get; set; } = 0.30d;
    public double GroupFamilySuspicionConfidenceMultiplier { get; set; } = 0.15d;
    public double MinConfidenceToApplyAdjustment { get; set; } = 0.35d;
    public double MinConfidenceToScheduleTransferProbe { get; set; } = 0.25d;
    public double MaxNegativeAdjustmentKld { get; set; } = 0.002d;
    public double MaxNegativeAdjustmentFractionOfBaseKld { get; set; } = 0.75d;
    public bool TransferProbeEnabled { get; set; } = true;
    public int MaxTransferProbesPerTemplate { get; set; } = 6;
    public int MaxTotalTransferProbesPerRun { get; set; } = 24;
    public RuntimeSynergyTransferProbeContextStrataConfig TransferProbeContextStrata { get; set; } = new();
    public bool ContextScopedRuleApplicationEnabled { get; set; } = true;
    public int MaxNonRuleGroupsBelowReferenceTier { get; set; } = 1;
    public bool VerboseSynergyLogging { get; set; } = true;
    public double MinSmokeScore { get; set; } = 0.55d;
    public double MaxSmokeGapKld { get; set; } = 0.004d;
    public int TopRejectedSmokePreview { get; set; } = 25;
    public bool CompositionProbeEnabled { get; set; } = true;
    public int MaxTemplateCompositionGroupCount { get; set; } = 4;
    public int MaxCompositionProbesPerRun { get; set; } = 8;
    public int MaxTemplatesToCompose { get; set; } = 4;
    public double MinTemplateConfidenceForComposition { get; set; } = 0.50d;
    public double MinCombinedExpectedSizeSavingsPercent { get; set; } = 1.0d;
    public bool ContaminatingPassengerDetectionEnabled { get; set; } = true;
    public double MinFailureMarginForContaminationKld { get; set; } = 0.00050d;
    public double ContaminationPenaltyConfidenceMultiplier { get; set; } = 0.45d;
    public bool SuppressRepeatedContaminatedAttempts { get; set; } = true;
}

public sealed class RuntimeSynergyTransferProbeContextStrataConfig
{
    public int HighFidelityMaxNonReferenceGroupsBelowQ6 { get; set; } = 1;
    public int MidFidelityMaxNonReferenceGroupsBelowQ6 { get; set; } = 3;
    public bool LowFidelityEnabled { get; set; } = false;
}

public sealed class RuntimeLearningConfig
{
    public bool ForceRelearnArchitectureFamily { get; set; }
    public List<string> ForceRelearnStandardBaselines { get; set; } = new();

    /// <summary>
    /// Safety gate for regex/profile mistakes. When true, the evolution run prints
    /// native BF16 tensor-group counts and asks before continuing.
    /// </summary>
    public bool ConfirmTensorGroupProfile { get; set; } = true;

    /// <summary>
    /// Transient repair command for regex changes. Rebuilds learned tensor/group
    /// rows for the active profile from existing DB truth where possible, avoiding
    /// needless re-download/re-quantization of pure baseline learning artifacts.
    /// </summary>
    public bool RebucketLearnedTensorGroupsFromExistingTruth { get; set; } = true;
}

public sealed class RuntimeBaselineConfig
{
    public string StandardBaselinesMode { get; set; } = "all";
    public List<string> EnabledStandardLearningBaselines { get; set; } = new();
    public List<string> EnabledStandardCombinationCarriers { get; set; } = new();
    public List<string> EnabledStandardExplicitGroupCandidates { get; set; } = new();
    public List<CustomBaselineRepositoryConfig> CustomRepositories { get; set; } = new();

    [YamlIgnore]
    public List<ResolvedCustomBaselineSpec> ResolvedCustomBaselines { get; set; } = new();
}

public sealed class CustomBaselineRepositoryConfig
{
    public string RepoId { get; set; } = string.Empty;
    public string? Revision { get; set; }
    public string? ShortSourceName { get; set; }
    public string SourceKind { get; set; } = "huggingface_gguf_repository";
    public bool Enabled { get; set; } = true;
    public bool RequireAllIncludesToResolve { get; set; } = true;
    public bool ValidateTensorNamesAgainstSourceModel { get; set; } = true;
    public bool DeletePartialOrDirtyDownloads { get; set; } = true;
    public bool ResumeOrRetryDownloads { get; set; } = true;
    public bool AllowAsCombinationCarrier { get; set; }
    public bool AllowAsExplicitGroupCandidate { get; set; } = true;
    public bool AllowAsLearningBaseline { get; set; } = true;
    public List<CustomBaselineIncludeConfig> Includes { get; set; } = new();
}

public sealed class CustomBaselineIncludeConfig
{
    public string BaselineFamily { get; set; } = string.Empty;
    public string? FileName { get; set; }
    public string? DisplayName { get; set; }
    public string? QuantizeBaseName { get; set; }
    public bool? RequiresImatrix { get; set; }
    public bool? AllowAsCombinationCarrier { get; set; }
    public bool? AllowAsExplicitGroupCandidate { get; set; }
    public bool? AllowAsLearningBaseline { get; set; }
    public bool ForceRelearn { get; set; }
    public List<byte> BannedGroupIds { get; set; } = new();
}

public sealed class ResolvedCustomBaselineSpec
{
    public byte DynamicBaselineId { get; set; }
    public string CanonicalKey { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string RepoId { get; set; } = string.Empty;
    public string? Revision { get; set; }
    public string SourceOwner { get; set; } = string.Empty;
    public string SourceFileName { get; set; } = string.Empty;
    public string ShortSourceName { get; set; } = string.Empty;
    public string BaselineFamily { get; set; } = string.Empty;
    public string QuantizeBaseName { get; set; } = string.Empty;
    public bool RequiresImatrix { get; set; }
    public bool AllowAsLearningBaseline { get; set; }
    public bool AllowAsCombinationCarrier { get; set; }
    public bool AllowAsExplicitGroupCandidate { get; set; }
    public bool ForceRelearn { get; set; }
    public int? BaselineQuantDefinitionId { get; set; }
    public bool IsActiveInCurrentConfig { get; set; } = true;
    public IReadOnlyList<byte> BannedGroupIds { get; set; } = Array.Empty<byte>();
}

public sealed class RuntimeHardwareConfig
{
    public Dictionary<int, double> GpuMemoryLimitsGb { get; set; } = new();
}
