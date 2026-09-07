using MQ.DB.Models;

namespace MQ.DB;

public class Cache
{
    /// <summary>
    /// Full path to llama.cpp repo
    /// </summary>
    public static string? LlamaRoot;

    /// <summary>
    /// Full path to /llama.cpp/build/bin/
    /// </summary>
    public static string? LlamaBin;

    /// <summary>
    /// full path to convert_hf_to_gguf.py
    /// </summary>
    public static string? ConvertScript;

    /// <summary>
    /// System information about the PC that's detected
    /// during the initial llama cpp validation phase.
    /// </summary>
    public static SystemInfo? SysInfo;

    /// <summary>
    /// Root MagicQuant working directory. This is also where the Python
    /// environment, default config files, and shared caches live.
    /// </summary>
    public static string? MagicQuantDirectory;

    /// <summary>
    /// Full path to the desired model directory where the safetensors are.
    /// </summary>
    public static string? ModelDirectory;

    /// <summary>
    /// Per-model MagicQuant working directory.
    /// </summary>
    public static string? ModelMagicQuantDirectory;

    /// <summary>
    /// Absolute path to the active YAML config that was loaded for this run.
    /// </summary>
    public static string? ActiveConfigPath { get; set; }

    /// <summary>
    /// Root directory where external/custom baseline GGUF files are staged.
    /// </summary>
    public static string? ExternalBaselineCacheDirectory { get; set; }


    /// <summary>
    /// Normalized configured scratch roots for transient heavy GGUF writes.
    /// </summary>
    public static List<string> ScratchRoots { get; set; } = new();


    /// <summary>
    /// Aka BF16, F16, or F32
    /// </summary>
    public static MainTorchType? TorchType;

    public enum MainTorchType
    {
        BF16 = 1,
        F16 = 2,
        F32 = 3
    }

    /*
     * Groups not present in the current model graph. These are forced to NULL/ignored
     * by runtime search-space planning.
     */
    public static List<TensorGroup> UnusedTensorGroups = new();

    public static string CurrentModelId { get; set; } = string.Empty;

    public static string CurrentArchitectureFamilyName { get; set; } = string.Empty;

    public static string CurrentArchitectureFamilyNormalizedName =>
        string.IsNullOrWhiteSpace(CurrentArchitectureFamilyName)
            ? string.Empty
            : CurrentArchitectureFamilyName.Trim().ToLowerInvariant();

    public static bool AllowArchitectureFamilyAliasOverride { get; set; }

    public static int? CurrentArchitectureFamilyId { get; set; }

    public static int? CurrentTensorGroupProfileId { get; set; }

    public static string? CurrentTensorGroupProfileFingerprintHash { get; set; }

    /// <summary>
    /// When true, MagicQuant prints the BF16/native tensor grouping summary and asks
    /// for confirmation before any tensor-group-scoped learning/search work continues.
    /// </summary>
    public static bool ConfirmTensorGroupProfile { get; set; } = true;

    /// <summary>
    /// Transient repair mode for regex/profile mistakes. When true, MagicQuant tries
    /// to rebuild learned tensor mappings for the active TensorGroupProfile from
    /// existing family/profile truth instead of redownloading/requantizing pure
    /// learning baselines just to rediscover per-tensor truth.
    /// </summary>
    public static bool RebucketLearnedTensorGroupsFromExistingTruth { get; set; } = true;

    public static bool ForceRefreshHardwareProbe { get; set; }

    public static bool UseImatrix { get; set; }

    public static bool ForceImatrixRebuild { get; set; }

    public static Dictionary<int, double> GpuMemoryLimitsGb { get; set; } = new();

    public static bool IsImatrixAvailable { get; set; }

    public static string? ActiveImatrixPath { get; set; }

    public static string? ActiveImatrixIdentityHash { get; set; }


    /// <summary>
    /// When false, long-running llama.cpp child processes write their full stdout/stderr
    /// to log files only. This keeps the CLI readable during large export/clone runs.
    /// </summary>
    public static bool VerboseProcessOutput { get; set; }

    /// <summary>
    /// Clone/export-only flows may benchmark for release metadata without polluting the
    /// learning/discovery SQLite truth tables.
    /// </summary>
    public static bool SuppressBenchmarkPersistence { get; set; }


    /// <summary>
    /// Final export/output directory for selected survivor artifacts.
    /// </summary>
    public static string? OutputDirectory { get; set; }
}
