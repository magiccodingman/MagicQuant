using MQ.DB.Models;

namespace MagicQuant.Models;

public enum QuantMovementKind
{
    Same = 0,
    Downgrade = 1,
    Upgrade = 2,
    LateralOrEquivalent = 3,
    Unknown = 4
}

public enum AnomalyMovementClassification
{
    MonotoneDowngrade = 1,
    MixedTrade = 2,
    MonotoneUpgrade = 3,
    LateralOrProviderEquivalent = 4,
    Unknown = 5,
    NoMovement = 6
}

public enum AnomalyRuleDirection
{
    Beneficial = 1,
    Harmful = 2,
    SuppressionOnly = 3
}

public enum AnomalyRuleStatus
{
    Confirmed = 1,
    Rejected = 2,
    Suppressed = 3,
    Retired = 4
}

public enum AnomalySeedClass
{
    ConfirmedHistoricalCounterfactual = 1,
    HistoricalMissingTwin = 2,
    PredictionSpaceSmoke = 3,
    ExploratorySingle = 4,
    ExploratoryPair = 5,
    ConfirmedAnomalyNeighborhoodProbe = 6,
    SynergyTransferProbe = 7,
    SynergyCompositionProbe = 8
}

public enum AnomalyProbeClassification
{
    BeneficialAnomaly = 1,
    CounterfactualMdaViolation = 2,
    HarmfulInteraction = 3,
    RejectedSmoke = 4,
    NormalGravity = 5,
    SuppressionOnly = 6,
    SingleGroupInversion = 7,
    PairSynergy = 8,
    HigherOrderSynergy = 9,
    ContextOnly = 10,
    MissingTwin = 11,
    MissingProbeBenchmark = 12,
    SuperSynergy = 13,
    AdditiveComposition = 14,
    RedundantComposition = 15,
    HarmfulInterference = 16,
    CompositionRejected = 17,
    ContaminatingPassenger = 18
}

public sealed class AnomalyChangedGroup
{
    public TensorGroup Group { get; init; } = default!;
    public byte CandidateQuantId { get; init; }
    public byte ReferenceQuantId { get; init; }
    public byte CandidateStoredSlot { get; init; }
    public byte ReferenceStoredSlot { get; init; }
    public QuantMovementKind Movement { get; init; }
}

public sealed class AnomalyMovementAnalysis
{
    public AnomalyMovementClassification Classification { get; init; }
    public IReadOnlyList<AnomalyChangedGroup> ChangedGroups { get; init; } = Array.Empty<AnomalyChangedGroup>();
    public int UpgradeCount { get; init; }
    public int DowngradeCount { get; init; }
    public int SameCount { get; init; }
    public int UnknownCount { get; init; }
    public int LateralCount { get; init; }
    public int NetBitDelta { get; init; }
}

public sealed class AnomalySmokeCandidate
{
    public string Source { get; init; } = string.Empty;
    public TensorConfig CandidateConfig { get; init; }
    public TensorConfig TwinConfig { get; init; }
    public HybridQuant CandidateQuant => (HybridQuant)CandidateConfig;
    public HybridQuant TwinQuant => (HybridQuant)TwinConfig;
    public AnomalyMovementAnalysis Movement { get; init; } = new();
    public double CandidatePredictedKld { get; init; }
    public double TwinPredictedKld { get; init; }
    public ulong? CandidatePredictedSizeBytes { get; init; }
    public ulong? TwinPredictedSizeBytes { get; init; }
    public ulong? PredictedSizeSavingsBytes { get; init; }
    public ulong? ActualSizeSavingsBytes { get; init; }
    public bool PlannedProbeWillMeasureSize { get; init; }
    public string TwinLookupMode { get; init; } = string.Empty;
    public string RejectionReason { get; init; } = string.Empty;
    public bool MatchedConfirmedAnomalyPattern { get; init; }
    public bool TwinFoundInLookupDictionary { get; init; }
    public AnomalySeedClass SeedClass { get; init; } = AnomalySeedClass.PredictionSpaceSmoke;
    public double PredictionSpaceGapVsTwin { get; init; }
    public ulong? CandidatePredictionRank { get; init; }
    public ulong? TwinPredictionRank { get; init; }
    public double SmokeScore { get; init; }
    public string SmokeStrength { get; init; } = string.Empty;
    public bool HasActualTwin { get; init; }
    public double? CandidateActualKld { get; init; }
    public double? TwinActualKld { get; init; }
    public ulong? CandidateActualSizeBytes { get; init; }
    public ulong? TwinActualSizeBytes { get; init; }
    public bool IsConfirmedFromHistory { get; init; }
    public string Message { get; init; } = string.Empty;
}

public sealed class AnomalyProbePlan
{
    public AnomalySmokeCandidate Seed { get; init; } = default!;
    public TensorConfig ReferenceConfig { get; init; }
    public TensorConfig ProbeConfig { get; init; }
    public IReadOnlyList<AnomalyChangedGroup> ProbeGroups { get; init; } = Array.Empty<AnomalyChangedGroup>();
    public string ProbeType { get; init; } = string.Empty;
    public string HypothesisLabel { get; init; } = string.Empty;
    public AnomalySeedClass SeedClass { get; init; }
    public AnomalySeedClass ProbePriorityClass { get; init; }
}

public sealed class AnomalyProbeResult
{
    public AnomalyProbePlan Plan { get; init; } = default!;
    public BenchmarkSnapshotRecord? ProbeSnapshot { get; init; }
    public BenchmarkSnapshotRecord? ReferenceSnapshot { get; init; }
    public AnomalyProbeClassification Classification { get; init; }
    public AnomalyRuleDirection RuleDirection { get; init; }
    public bool Accepted { get; init; }
    public double ActualGainVsTwin { get; init; }
    public string FailureCode { get; init; } = string.Empty;
    public string Message { get; init; } = string.Empty;
}

public sealed class AnomalyAdjustmentSummary
{
    public int AppliedRuleCount { get; init; }
    public long MatchedRowCount { get; init; }
    public string DuckDbPath { get; init; } = string.Empty;
    public IReadOnlyList<object> RuleMatches { get; init; } = Array.Empty<object>();
}

public sealed class AnomalyRunResult
{
    public IReadOnlyList<AnomalySmokeCandidate> SmokeCandidates { get; init; } = Array.Empty<AnomalySmokeCandidate>();
    public IReadOnlyList<AnomalyProbePlan> ProbePlans { get; init; } = Array.Empty<AnomalyProbePlan>();
    public IReadOnlyList<AnomalyProbeResult> ProbeResults { get; init; } = Array.Empty<AnomalyProbeResult>();
    public AnomalyAdjustmentSummary AdjustmentSummary { get; init; } = new();
    public object? BestAnomalyReconciliation { get; init; }
}

public sealed class AnomalySmokeScanDiagnostics
{
    public long PredictedRowsScanned { get; set; }
    public long SparseRowsSkipped { get; set; }
    public long SparseRowsNormalized { get; set; }
    public long Bf16ExactRowsSkipped { get; set; }
    public long PureReferenceRowsSkipped { get; set; }
    public long ContextualRowsScanned { get; set; }
    public long TwinLookupCount { get; set; }
    public long DictionaryTwinHits { get; set; }
    public long MissingTwins { get; set; }
    public long FallbackDbTwinLookups { get; set; }
    public long MovementNotMonotoneDowngrade { get; set; }
    public long MixedTradeIgnored { get; set; }
    public long SizeSavingsBelowThreshold { get; set; }
    public long PredictionSpaceGapTooLarge { get; set; }
    public long QueuedSmokeCandidates { get; set; }
    public long LoadPredictedRowsMs { get; set; }
    public long BuildLookupDictionaryMs { get; set; }
    public long ScanRowsMs { get; set; }
    public IReadOnlyList<object> RejectedPreview { get; set; } = Array.Empty<object>();
    public IReadOnlyList<object> ClosestGapFailures { get; set; } = Array.Empty<object>();
}

public sealed class ProbePlanningDiagnostics
{
    public int ExistingRuleKeysLoaded { get; set; }
    public int SkippedExistingRuleOrSuppression { get; set; }
    public int SkippedDuplicate { get; set; }
    public int SkippedInvalidMovement { get; set; }
    public int SkippedBudget { get; set; }
    public int ProbesQueued { get; set; }
    public int ExpansionProbesQueued { get; set; }
    public int CompositionProbesQueued { get; set; }
    public int TransferProbesQueued { get; set; }
    public int ExploratoryPairProbesQueued { get; set; }
    public int SkippedContaminationSuppression { get; set; }
}

public sealed class SynergyCompositionProbeRecord
{
    public string CompositionId { get; init; } = string.Empty;
    public IReadOnlyList<string> SourceTemplateIds { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> SourceTemplateLabels { get; init; } = Array.Empty<string>();
    public Dictionary<string, string> CandidateEffectiveGroups { get; init; } = new(StringComparer.OrdinalIgnoreCase);
    public Dictionary<string, string> TwinEffectiveGroups { get; init; } = new(StringComparer.OrdinalIgnoreCase);
    public int CombinedGroupCount { get; init; }
    public string Classification { get; init; } = string.Empty;
    public double? ActualCandidateKld { get; init; }
    public double? ActualTwinKld { get; init; }
    public double? ActualGainVsTwin { get; init; }
    public double? PredictedCandidateKld { get; init; }
    public double? PredictedTwinKld { get; init; }
    public double? PredictionSpaceGap { get; init; }
    public IReadOnlyList<string> Notes { get; init; } = Array.Empty<string>();
}

public sealed class SynergyWingSummary
{
    public string Zone { get; init; } = string.Empty;
    public int SmokeCount { get; set; }
    public int ConfirmedBeneficialTemplates { get; set; }
    public int HarmfulTemplates { get; set; }
    public int SuppressionOnlyTemplates { get; set; }
    public int CandidateRowsAdjustedPositively { get; set; }
    public int CandidateRowsDemoted { get; set; }
    public int ValidationSuccessCount { get; set; }
    public int ValidationFailureCount { get; set; }
    public int FinalSurvivorsFromZone { get; set; }
    public string Explanation { get; set; } = string.Empty;
}
