namespace MQ.DB.Models;

public enum RequiredSampleKind
{
    PureBaseline = 1,
    BaseOnlyIsolation = 2,
    GroupIsolationProbe = 3,
    GroupIsolationContinuation = 4
}

public sealed class RequiredSamplePlan
{
    public RequiredSampleKind Kind { get; set; }
    public string Key { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public HybridQuant Quant { get; set; } = default!;
    public byte? TargetGroupId { get; set; }
    // Legacy name kept for compatibility: this now stores the tested baseline-family candidate id.
    public byte? TestedSchemeId { get; set; }

    public byte? TestedCandidateId
    {
        get => TestedSchemeId;
        set => TestedSchemeId = value;
    }
    public byte? TestedBaselineId { get; set; }
    public bool IsSmallestProbe { get; set; }
}

public sealed class RequiredSampleGenerationResult
{
    public List<RequiredSamplePlan> Plans { get; set; } = new();
    public int PureBaselineCount { get; set; }
    public int BaseOnlyIsolationCount { get; set; }
    public int GroupIsolationCount { get; set; }
    public int TotalCount => Plans.Count;

    public RequiredSampleGenerationResult MergeWith(RequiredSampleGenerationResult other)
    {
        var merged = new RequiredSampleGenerationResult
        {
            PureBaselineCount = PureBaselineCount + other.PureBaselineCount,
            BaseOnlyIsolationCount = BaseOnlyIsolationCount + other.BaseOnlyIsolationCount,
            GroupIsolationCount = GroupIsolationCount + other.GroupIsolationCount
        };

        merged.Plans.AddRange(Plans);
        merged.Plans.AddRange(other.Plans);
        return merged;
    }
}