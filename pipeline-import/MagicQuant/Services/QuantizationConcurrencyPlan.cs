namespace MagicQuant.Services;

/// <summary>CPU and storage writer limits, independent of model IO and benchmark execution.</summary>
public sealed record QuantizationConcurrencyPlan(int ReservedThreads, int UsableThreads,
    int NaturalConcurrency, int Concurrency, int ThreadsPerProcess)
{
    public static QuantizationConcurrencyPlan Create(int threadCount, int scratchWriterCapacity)
    {
        const int minimumThreadsPerProcess = 4;
        int reserved = threadCount switch { >= 8 => 2, >= 4 => 1, _ => 0 };
        int usable = Math.Max(1, threadCount - reserved);
        int natural = Math.Max(1, usable / minimumThreadsPerProcess);
        int concurrent = Math.Max(1, Math.Min(natural, scratchWriterCapacity));
        return new(reserved, usable, natural, concurrent, Math.Max(1, usable / concurrent));
    }
}
