using Spectre.Console;

namespace MagicQuant.Services.Progress;

public sealed class StageProgressTracker
{
    private readonly StageProgressOptions _options;
    private readonly object _printSync = new();

    private int _completed;
    private int _skipped;
    private int _failed;
    private int _lastPrintedFinished;
    private DateTime _lastPrintedUtc;

    public StageProgressTracker(StageProgressOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (_options.Total < 0)
            throw new ArgumentOutOfRangeException(nameof(options.Total), "Total cannot be negative.");

        StageName = string.IsNullOrWhiteSpace(_options.StageName) ? "Stage" : _options.StageName.Trim();
        Total = _options.Total;
        StartedUtc = DateTime.UtcNow;
        _lastPrintedUtc = StartedUtc;
    }

    public string StageName { get; }
    public int Total { get; }
    public DateTime StartedUtc { get; }

    public StageProgressSnapshot Snapshot
    {
        get
        {
            int completed = Volatile.Read(ref _completed);
            int skipped = Volatile.Read(ref _skipped);
            int failed = Volatile.Read(ref _failed);
            int finished = completed + skipped + failed;
            return new StageProgressSnapshot(
                StageName,
                Total,
                StartedUtc,
                completed,
                skipped,
                failed,
                finished,
                DateTime.UtcNow,
                _lastPrintedUtc,
                Volatile.Read(ref _lastPrintedFinished));
        }
    }

    public void ReportFinished(SampleProcessState state, string? itemName = null)
    {
        switch (state)
        {
            case SampleProcessState.Completed:
                Interlocked.Increment(ref _completed);
                break;
            case SampleProcessState.Skipped:
                Interlocked.Increment(ref _skipped);
                break;
            default:
                Interlocked.Increment(ref _failed);
                break;
        }

        MaybePrint(state, itemName);
    }

    private void MaybePrint(SampleProcessState justFinishedState, string? itemName)
    {
        var now = DateTime.UtcNow;

        lock (_printSync)
        {
            int completed = Volatile.Read(ref _completed);
            int skipped = Volatile.Read(ref _skipped);
            int failed = Volatile.Read(ref _failed);
            int finished = completed + skipped + failed;

            bool isFinal = Total > 0 && finished >= Total;
            bool intervalElapsed = now - _lastPrintedUtc >= _options.MinimumPrintInterval;
            bool countThresholdHit = finished - _lastPrintedFinished >= Math.Max(1, _options.PrintEveryNFinished);
            bool rapidSkipStorm = justFinishedState == SampleProcessState.Skipped &&
                                 finished - _lastPrintedFinished < Math.Max(1, _options.PrintEveryNFinished) &&
                                 !intervalElapsed &&
                                 !isFinal;

            if (!isFinal && !intervalElapsed && (!countThresholdHit || rapidSkipStorm))
                return;

            if (!isFinal && finished == _lastPrintedFinished)
                return;

            string escapedStage = Markup.Escape(StageName);

            if (!_options.ShowEta)
            {
                AnsiConsole.MarkupLine($"[grey][progress][/]{escapedStage}: [cyan]{finished}[/]/[cyan]{Total}[/] local GGUF outputs built");
            }
            else
            {
                var elapsed = now - StartedUtc;
                string elapsedText = FormatDuration(elapsed);

                int etaSampleCount = _options.CountSkippedForEta ? finished : completed + failed;
                string etaText = "ETA warming up...";
                string estFinishText = "est finish UTC n/a";

                if (elapsed.TotalSeconds > 0 && etaSampleCount >= Math.Max(1, _options.MinimumNonSkippedSamplesBeforeEta))
                {
                    double rate = etaSampleCount / elapsed.TotalSeconds;
                    int remaining = Math.Max(0, Total - finished);
                    if (rate > 0)
                    {
                        var estimatedRemaining = TimeSpan.FromSeconds(remaining / rate);
                        var estimatedFinishUtc = now + estimatedRemaining;
                        etaText = $"ETA {FormatEta(estimatedRemaining)}";
                        estFinishText = $"est finish UTC {estimatedFinishUtc:yyyy-MM-dd HH:mm}";
                    }
                }

                string maybeItem = string.IsNullOrWhiteSpace(itemName)
                    ? string.Empty
                    : $" | item={Markup.Escape(itemName)}";

                AnsiConsole.MarkupLine(
                    $"[grey][progress][/]{escapedStage}: [cyan]{finished}[/]/[cyan]{Total}[/] done | completed=[green]{completed}[/] skipped=[yellow]{skipped}[/] failed=[red]{failed}[/] | elapsed={elapsedText} | {etaText} | {estFinishText}{maybeItem}");
            }

            _lastPrintedUtc = now;
            _lastPrintedFinished = finished;
        }
    }

    private static string FormatDuration(TimeSpan duration)
    {
        if (duration.TotalDays >= 1)
            return $"{(int)duration.TotalDays}d {duration.Hours:00}h {duration.Minutes:00}m";

        return $"{duration.Hours:00}h {duration.Minutes:00}m {duration.Seconds:00}s";
    }

    private static string FormatEta(TimeSpan duration)
    {
        if (duration.TotalDays >= 1)
            return $"{(int)duration.TotalDays}d {duration.Hours:00}h {duration.Minutes:00}m";

        if (duration.TotalHours >= 1)
            return $"{duration.Hours:00}h {duration.Minutes:00}m";

        return $"{duration.Minutes:00}m";
    }
}
