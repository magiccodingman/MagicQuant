namespace MagicQuant.Services.Progress;

public readonly record struct StageProgressSnapshot(
    string StageName,
    int Total,
    DateTime StartedUtc,
    int Completed,
    int Skipped,
    int Failed,
    int Finished,
    DateTime CapturedUtc,
    DateTime LastPrintedUtc,
    int LastPrintedFinished);
