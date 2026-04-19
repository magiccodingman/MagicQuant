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
    public static  SystemInfo? SysInfo;
    
    /// <summary>
    /// The full path to the model directory being quantized,
    /// where a "MagicQuant" folder is created and used.
    /// </summary>
    public static string? MagicQuantDirectory;
    
    /// <summary>
    /// Full path to the desired model directory where the safetensors are.
    /// </summary>
    public static string? ModelDirectory;
    
    public static string? ModelMagicQuantDirectory;
    
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
     * This is properly updated, but not really used. More for generic logs because the
     * TensorWeightScheme is what's actually updated with the real ban logic both from the
     * start and during runtime
     */
    public static List<TensorGroup> UnusedTensorGroups = new List<TensorGroup>();
    
    public static string CurrentModelId { get; set; }

    public static bool ForceRelearnBaselineTensorMappings { get; set; }

    public static bool ForceRefreshHardwareProbe { get; set; }

    public static bool UseImatrix { get; set; }

    public static bool ForceImatrixRebuild { get; set; }

    public static bool IsImatrixAvailable { get; set; }

    public static string? ActiveImatrixPath { get; set; }
}
