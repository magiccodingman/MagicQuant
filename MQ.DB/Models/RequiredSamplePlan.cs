namespace MQ.DB.Models;

public enum RequiredSampleKind
{
    PureBaseline = 1,
    BaseOnlyIsolation = 2,
    GroupIsolation = 3
}

public sealed class RequiredSamplePlan
{
    public RequiredSampleKind Kind { get; set; }
    public string Key { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public HybridQuant Quant { get; set; } = default!;

    public byte? TargetGroupId { get; set; }
    public byte? TestedSchemeId { get; set; }
    public byte? TestedBaselineId { get; set; }
}

public sealed class RequiredSampleGenerationResult
{
    public List<RequiredSamplePlan> Plans { get; set; } = new();

    public int PureBaselineCount { get; set; }
    public int BaseOnlyIsolationCount { get; set; }
    public int GroupIsolationCount { get; set; }

    public int TotalCount => Plans.Count;
}