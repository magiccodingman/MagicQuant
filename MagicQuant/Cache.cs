using MagicQuant.Models;

namespace MagicQuant;

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
}