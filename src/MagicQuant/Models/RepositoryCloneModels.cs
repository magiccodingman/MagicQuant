using System.Text.Json.Serialization;

namespace MagicQuant.Models;

public sealed class MagicQuantCloneManifest
{
    public int SchemaVersion { get; set; } = 1;
    public DateTime GeneratedUtc { get; set; } = DateTime.UtcNow;
    public string Generator { get; set; } = "MagicQuant";
    public string? SourceRepository { get; set; }
    public string? SourceJson { get; set; }
    public string? SourceModelId { get; set; }
    public string? SourceArchitectureFamily { get; set; }
    public string? Notes { get; set; }

    public List<MagicQuantCloneArtifact> Artifacts { get; set; } = new();
}

public sealed class MagicQuantCloneArtifact
{
    public string FileName { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string ShortName { get; set; } = string.Empty;
    public string Provider { get; set; } = string.Empty;
    public string QuantFamily { get; set; } = string.Empty;
    public string BaseQuant { get; set; } = "Q8_0";
    public bool IsHybrid { get; set; }
    public bool UsedImatrix { get; set; }

    public double? SourceKld { get; set; }
    public double? SourcePpl { get; set; }
    public double? SourcePplDeltaPercent { get; set; }
    public ulong? SourceSizeBytes { get; set; }
    public double? SourceSizeGB { get; set; }
    public double? SourceSizeGiB { get; set; }

    /// <summary>
    /// Exact tensor-name -> final GGUF quant type map read from the exported artifact.
    /// This is the real clone payload.
    /// </summary>
    public Dictionary<string, string> TensorTypes { get; set; } = new(StringComparer.Ordinal);
}

public sealed class CloneArtifactBuildRecord
{
    public MagicQuantCloneArtifact ManifestArtifact { get; init; } = default!;
    public string OutputPath { get; init; } = string.Empty;
    public ulong ActualSizeBytes { get; set; }
    public double? Kld { get; set; }
    public double? Ppl { get; set; }
    public double? PplDeltaPercent { get; set; }
}
