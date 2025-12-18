namespace MagicQuant.Models;

public class LlamaBenchMetrics
{
    public string? LogPath { get; set; }
    public string? Backend { get; set; }
    public int? Ngl { get; set; }
    public string? Test { get; set; }
    public double? Tps { get; set; }
}