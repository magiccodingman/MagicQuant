using MQ.DB.Models;

namespace MagicQuant.Models.Learning;

public sealed class TensorGroupingResult
{
    public TensorGroup? PrimaryGroup { get; init; }
    public IReadOnlyList<string> MatchedGroups { get; init; } = [];
    public bool IsBaseQuantException { get; init; }
    public string? MatchedExceptionPattern { get; init; }
}

public sealed class TensorGroupingAuditIssue
{
    public required string TensorName { get; init; }
    public required string IssueKind { get; init; }
    public IReadOnlyList<string> MatchedGroups { get; init; } = [];
    public string? MatchedExceptionPattern { get; init; }
    public string? FinalQuantType { get; init; }
    public string? LearningSource { get; init; }
}

public sealed class TensorGroupingAuditResult
{
    public required IReadOnlyDictionary<string, TensorGroupingResult> GroupedByTensor { get; init; }
    public required IReadOnlyList<TensorGroupingAuditIssue> Ambiguous { get; init; }
    public required IReadOnlyList<TensorGroupingAuditIssue> IllegalUnresolved { get; init; }
    public required IReadOnlyList<TensorGroupingAuditIssue> BaseQuantExceptions { get; init; }

    public bool HasFatalIssues => Ambiguous.Count > 0 || IllegalUnresolved.Count > 0;
    public int FatalIssueCount => Ambiguous.Count + IllegalUnresolved.Count;
}

public sealed record LearnedTensorTruth(string TensorName, string FinalQuantType, LearningSource Source);

public enum LearningSource
{
    LogOnly = 1,
    GgufOnly = 2,
    Both = 3,
    BothWithMismatch = 4
}

public sealed class TensorTruthMismatch
{
    public required string TensorName { get; init; }
    public required string LogQuantType { get; init; }
    public required string GgufQuantType { get; init; }
    public bool IsHighSeverity { get; init; }
}

public sealed class TensorTruthVerificationResult
{
    public required IReadOnlyDictionary<string, LearnedTensorTruth> TruthByTensor { get; init; }
    public IReadOnlyList<TensorTruthMismatch> HardMismatches { get; init; } = [];
    public IReadOnlyList<TensorTruthMismatch> SoftMismatches { get; init; } = [];
    public IReadOnlyList<string> LogOnly { get; init; } = [];

    public bool HasFatalIssues => HardMismatches.Count > 0;
}
