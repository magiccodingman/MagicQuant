namespace MagicQuant.Models;

public class PplMetrics
{
    public string? LogPath { get; set; }
    public double Ppl { get; set; }
    public double PplError { get; set; }
    public double? Kld { get; set; }
}