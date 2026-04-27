namespace MagicQuant.Services.Progress;

public sealed class StageProgressOptions
{
    public string StageName { get; init; } = string.Empty;
    public int Total { get; init; }
    public int MinimumNonSkippedSamplesBeforeEta { get; init; } = 2;
    public TimeSpan MinimumPrintInterval { get; init; } = TimeSpan.FromSeconds(15);
    public int PrintEveryNFinished { get; init; } = 1;
    public bool ShowEta { get; init; } = true;
    public bool CountSkippedForEta { get; init; } = false;
    public bool PrintFinalSummary { get; init; } = true;
    public string? UnitLabel { get; init; }
}
