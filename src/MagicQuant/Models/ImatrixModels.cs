namespace MagicQuant.Models;

public enum ImatrixSourceKind
{
    Https = 1,
    HfDataset = 2,
    LocalDatasetFile = 3
}

public sealed class ImatrixRequest
{
    public bool UseImatrix { get; init; }
    public bool ForceRebuild { get; init; }

    public string? ImatrixUrl { get; init; }

    public string? DatasetRepo { get; init; }
    public string? DatasetSplit { get; init; }
    public string? DatasetConfig { get; init; }

    public string? LocalDatasetFile { get; init; }

    public string ModelDirectory { get; init; } = default!;
    public string MagicQuantDirectory { get; init; } = default!;
}

public sealed class ImatrixEnsureResult
{
    public bool Enabled { get; init; }
    public bool Available { get; init; }
    public bool Rebuilt { get; init; }
    public string? CanonicalImatrixPath { get; init; }
    public ImatrixSourceKind? SourceKind { get; init; }
}

public sealed class ImatrixSuccessSidecar
{
    public string Status { get; init; } = "success";
    public DateTime CompletedUtc { get; init; }
    public string ArtifactType { get; init; } = "imatrix";
    public string CanonicalFileName { get; init; } = "imatrix.dat";
    public string CanonicalPath { get; init; } = string.Empty;
    public string SourceKind { get; init; } = string.Empty;
    public string SourceIdentity { get; init; } = string.Empty;
    public string? Split { get; init; }
    public string? Config { get; init; }
    public string Sha256 { get; init; } = string.Empty;
    public long FileSizeBytes { get; init; }
    public string BuilderVersion { get; init; } = "mvp-v1";
}

public sealed class ImatrixMetadataSidecar
{
    public string SourceKind { get; init; } = string.Empty;
    public string? OriginalUrl { get; init; }
    public string? OriginalDownloadName { get; init; }
    public string? DatasetRepo { get; init; }
    public string? DatasetConfig { get; init; }
    public string? DatasetSplit { get; init; }
    public string? LocalDatasetFile { get; init; }
    public string Notes { get; init; } = "Renamed to canonical imatrix.dat after acquisition/build";
}
