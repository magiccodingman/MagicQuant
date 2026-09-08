using MQ.DB.Models;

namespace MagicQuant.Models;

public sealed class RankSafePredictionRow
{
    public TensorConfig Config { get; init; }
    public HybridQuant Quant { get; init; } = default!;
    public ulong PredictedSizeBytes { get; set; }
    public bool IsSizePredictable { get; set; } = true;
    public double AdditiveKld { get; set; }
    public double InteractionKld { get; set; }
    public double PredictedKld { get; set; }
    public double PredictionConfidence { get; set; } = 1.0d;
    public double PredictedPpl { get; set; }
    public double CrossTerm { get; set; }
    public bool IsPureBaseline { get; init; }
    public bool IsHybrid => !IsPureBaseline;
    public bool IsPredictable { get; set; } = true;
    public bool HasUnknownMappings { get; set; }
    public string EffectiveStateKey { get; init; } = string.Empty;
    public List<string> Notes { get; init; } = new();

    public double ActualKld { get; set; } = double.NaN;
    public double ActualPpl { get; set; } = double.NaN;
    public ulong? ActualSizeBytes { get; set; }
    public int? ActualRank { get; set; }
    public ulong? PredictedRank { get; set; }
    public double AnomalyAdjustmentKld { get; set; }

    public double AbsoluteKldError =>
        double.IsNaN(ActualKld) ? double.NaN : Math.Abs(PredictedKld - ActualKld);

    public double SignedKldError =>
        double.IsNaN(ActualKld) ? double.NaN : PredictedKld - ActualKld;
}

public sealed class RankSafePredictionSet
{
    public IReadOnlyList<RankSafePredictionRow> Rows { get; init; } = Array.Empty<RankSafePredictionRow>();
    public RankSafePredictionFit Fit { get; init; } = new();
    public IReadOnlyList<string> Notes { get; init; } = Array.Empty<string>();

    public IReadOnlyList<RankSafePredictionRow> PredictableRows =>
        Rows.Where(x => x.IsPredictable).ToList();
}

public sealed class RankSafePredictionFit
{
    public double Alpha { get; init; } = 1.0d;
    public double Beta { get; init; } = 0.0d;
    public double BitStressThreshold { get; init; } = 8.0d;
    public int FitRowCount { get; init; }
    public double FitMae { get; init; }
    public bool UsedFallback { get; init; }
}


public sealed class PredictedAnchorRow
{
    public required TensorConfig Config { get; init; }
    public required string ConfigKey { get; init; }
    public required string DisplayName { get; init; }
    public required string BaselineCanonicalKey { get; init; }
    public byte RuntimeBaselineId { get; init; }
    public double PredictedKld { get; init; }
    public ulong PredictedSizeBytes { get; init; }
    public double PredictionConfidence { get; init; }
    public ulong PredictionRank { get; init; }
    public bool IsVirtualPredictionAnchor { get; init; } = true;
}

public sealed class HybridSelectionAnchor
{
    public BenchmarkSnapshotRecord Snapshot { get; init; } = default!;
    public string Key => TensorConfigIdentity.ToKey(Snapshot.Config);
}

public enum HybridSelectionReason
{
    StrictDominanceReplacement = 1,
    NearBaselineOnePercentReplacement = 2,
    InteriorSubspaceDiscovery = 3,

    // SQLite/isolation-truth fallback candidates. These are intentionally not
    // DuckDB prediction-space rows; they are conservative baseline-blanket
    // tuning attempts used only after the normal selector cannot validate a win.
    SmartStrictDominanceFallback = 4,
    SmartNearBaselineFallback = 5,
    SmartInteriorSubspaceFallback = 6
}

public sealed class HybridSelectionCandidate
{
    public RankSafePredictionRow Prediction { get; init; } = default!;
    public HybridSelectionReason Reason { get; init; }
    public BenchmarkSnapshotRecord LowerDamageAnchor { get; init; } = default!;
    public BenchmarkSnapshotRecord HigherDamageAnchor { get; init; } = default!;
    public ulong WindowMinSizeBytes { get; init; }
    public ulong WindowMaxSizeBytes { get; init; }
    public double LinearExpectedKld { get; init; }
    public double PredictedGainOverLine { get; init; }
    public int AttemptOrder { get; init; }
    public string WindowLabel { get; init; } = string.Empty;
    public ulong PredictionWindowMinSizeBytes { get; init; }
    public ulong PredictionWindowMaxSizeBytes { get; init; }
    public PredictedAnchorRow? HigherDamagePredictionAnchor { get; init; }
    public PredictedAnchorRow? LowerDamagePredictionAnchor { get; init; }

    // Diagnostic-only context captured at selection time. These values do not
    // change acceptance rules; they explain how the candidate was found, how
    // many neighbors existed, and how hard the retry aperture was capped.
    public long CandidatePoolSize { get; init; }
    public long WindowCandidateCount { get; init; }
    public long LineBeatingCandidateCount { get; init; }
    public int FetchedCandidateCount { get; init; }
    public int CandidatesAfterBrutalityCount { get; init; }
    public int CandidateAttemptLimit { get; init; }
    public int PhaseWindowIndex { get; init; }
    public int PhaseWindowCount { get; init; }
    public int RawSelectionRank { get; init; }
    public string CandidateTheoryFamilyKey { get; init; } = string.Empty;
    public int CandidateTheoryFamilyRank { get; init; }
    public int CandidateTheoryFamilyMemberRank { get; init; }
    public string CandidateTheoryFamilyDisplay { get; init; } = string.Empty;
    public string DiversityMode { get; init; } = string.Empty;
    public IReadOnlyList<string> CandidateSelectionNotes { get; init; } = Array.Empty<string>();
}

public sealed class CandidateValidationResult
{
    public HybridSelectionCandidate Candidate { get; init; } = default!;
    public BenchmarkSnapshotRecord? Snapshot { get; init; }
    public bool Accepted { get; init; }
    public string Message { get; init; } = string.Empty;
    public string FailureCode { get; init; } = string.Empty;
}

public sealed class BaselineEliminationRecord
{
    public BenchmarkSnapshotRecord Eliminated { get; init; } = default!;
    public BenchmarkSnapshotRecord Eliminator { get; init; } = default!;
    public string Reason { get; init; } = string.Empty;
    public bool EliminatorIsHybrid => Eliminator.IsHybrid;
    public double EliminatedKld => Eliminated.Kld;
    public double EliminatorKld => Eliminator.Kld;
    public ulong EliminatedSizeBytes => Eliminated.SizeBytes;
    public ulong EliminatorSizeBytes => Eliminator.SizeBytes;
}

public sealed class RankSafeValidationSummary
{
    public int RowCount { get; init; }
    public int PredictableCount { get; init; }
    public double Mae { get; init; }
    public double Rmse { get; init; }
    public double MaxAbsoluteError { get; init; }
    public double MeanSignedError { get; init; }
    public double PairwiseAccuracyPercent { get; init; }
    public long ConcordantPairs { get; init; }
    public long DiscordantPairs { get; init; }
    public long TiedPredictedPairs { get; init; }
    public int ExactRankMatches { get; init; }
    public int WithinOneRank { get; init; }
    public int WithinTwoRanks { get; init; }
    public int WithinFiveRanks { get; init; }
    public int WithinTenRanks { get; init; }
    public int WithinTwentyRanks { get; init; }
}

public sealed class PredictionValidationExportResult
{
    public RankSafeValidationSummary Summary { get; init; } = new();
    public string CsvPath { get; init; } = string.Empty;
    public string MarkdownPath { get; init; } = string.Empty;
    public IReadOnlyList<RankSafePredictionRow> Rows { get; init; } = Array.Empty<RankSafePredictionRow>();
}

public sealed class PhaseValidationResult
{
    public IReadOnlyList<BenchmarkSnapshotRecord> AcceptedSnapshots { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
    public IReadOnlyList<CandidateValidationResult> Attempts { get; init; } = Array.Empty<CandidateValidationResult>();
}

public sealed class PredictionGuidedSelectionResult
{
    public IReadOnlyList<BenchmarkSnapshotRecord> Survivors { get; init; } = Array.Empty<BenchmarkSnapshotRecord>();
    public IReadOnlyList<BaselineEliminationRecord> Eliminations { get; init; } = Array.Empty<BaselineEliminationRecord>();
    public IReadOnlyList<CandidateValidationResult> ValidationFailures { get; init; } = Array.Empty<CandidateValidationResult>();
}