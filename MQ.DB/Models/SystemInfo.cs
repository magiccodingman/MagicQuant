namespace MQ.DB.Models;

public enum GpuVendor
{
    Nvidia = 1, 
    Amd = 2, 
    Intel = 3, 
    Cpu = 4, 
    Unknown = 0
}

public class SystemInfo
{
    public GpuVendor GpuVendor { get; set; }
    public string GpuName { get; set; } = "Unknown";
    public double VramGb { get; set; }
    public double RamGb { get; set; }
    public int ThreadCount { get; set; }
}

public static class MagicConstants
{
    public const string MagicQuantFolder = "MagicQuant";
    public const string EnvName = "MagicQuant-Env";
    public const string LlamaRepoName = "llama.cpp";
    public const string SuccessJson = "install_success.json";
    
    // Windows Python Embed URL
    public const string WinPythonUrl = "https://www.python.org/ftp/python/3.12.3/python-3.12.3-embed-amd64.zip";
    public const string WinPythonZip = "python-3.12.3-embed.zip";
}