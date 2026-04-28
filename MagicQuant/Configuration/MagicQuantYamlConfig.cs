using YamlDotNet.Serialization;

namespace MagicQuant.Configuration;

public sealed class MagicQuantYamlConfig
{
    public RuntimePathConfig Paths { get; set; } = new();
    public RuntimeFlagConfig Flags { get; set; } = new();
    public RuntimeImatrixConfig Imatrix { get; set; } = new();
    public RuntimeEvolutionConfig Evolution { get; set; } = new();
    public RuntimeIsolationPruningConfig IsolationPruning { get; set; } = new();
    public RuntimePredictionConfig Prediction { get; set; } = new();
    public RuntimeIdentityConfig Identity { get; set; } = new();
    public RuntimeBaselineConfig Baselines { get; set; } = new();
    public RuntimeOutputConfig Output { get; set; } = new();
    public RuntimeSurvivalConfig Survival { get; set; } = new();
    public RuntimeCandidateSelectionConfig CandidateSelection { get; set; } = new();
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
    public bool ForceRelearnBaselineTensorMappings { get; set; }
    public bool ForceRefreshHardwareProbe { get; set; }
    public bool AllowHighPrecisionHybrids { get; set; }
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
    public List<byte> BannedGroupIds { get; set; } = new();
}

public sealed class ResolvedCustomBaselineSpec
{
    public byte DynamicBaselineId { get; set; }
    public string CanonicalKey { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string RepoId { get; set; } = string.Empty;
    public string SourceOwner { get; set; } = string.Empty;
    public string SourceFileName { get; set; } = string.Empty;
    public string ShortSourceName { get; set; } = string.Empty;
    public string BaselineFamily { get; set; } = string.Empty;
    public string QuantizeBaseName { get; set; } = string.Empty;
    public bool RequiresImatrix { get; set; }
    public bool AllowAsLearningBaseline { get; set; }
    public bool AllowAsCombinationCarrier { get; set; }
    public bool AllowAsExplicitGroupCandidate { get; set; }
    public IReadOnlyList<byte> BannedGroupIds { get; set; } = Array.Empty<byte>();
}

public sealed class RuntimeHardwareConfig
{
    public Dictionary<int, double> GpuMemoryLimitsGb { get; set; } = new();
}
