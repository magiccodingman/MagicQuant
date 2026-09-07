namespace MQ.DB.Models;

public class BenchmarkResult
{
    public LlamaBenchMetrics? LlamaBench { get; set; }

    public Dictionary<string, PplMetrics> Perplexity { get; set; } =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Persisted into bench_metrics.json so disk-only reuse can still sync DB later
    /// even if the temporary GGUF has already been deleted.
    /// </summary>
    public ulong? ModelSizeBytes { get; set; }
}