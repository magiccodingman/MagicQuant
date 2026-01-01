namespace MQ.DB.Models;

public class BenchmarkResult
{
    public LlamaBenchMetrics? LlamaBench { get; set; }
    public Dictionary<string, PplMetrics> Perplexity { get; set; } = new();
}