using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class DependencyManager
{
    // CMake Constants
    private const string CmakeVersion = "3.29.0";
    private const string CmakeWinUrl = $"https://github.com/Kitware/CMake/releases/download/v{CmakeVersion}/cmake-{CmakeVersion}-windows-x86_64.zip";

    public static async Task EnsureDependenciesAsync(SystemInfo sysInfo)
    {
        // 1. Check CMake (Download if missing on Windows)
        string cmakePath = GetCmakePath();
        if (cmakePath == null)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                await DownloadAndInstallCmakeAsync();
            }
            else
            {
                // Linux usually handles this via apt earlier, but just in case:
                throw new Exception("CMake is missing. Please run: sudo apt install cmake");
            }
        }

        // 2. Check Visual Studio (Windows Only)
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (!CheckVisualStudio())
            {
                PromptForVisualStudio();
            }
        }

        // 3. Check GPU Toolkits (CUDA / ROCm / OneAPI)
        await ValidateGpuToolkitAsync(sysInfo);
    }

    private static async Task ValidateGpuToolkitAsync(SystemInfo sysInfo)
    {
        if (sysInfo.GpuVendor == GpuVendor.Nvidia)
        {
            // Check for NVCC
            if (!CheckCommandExists("nvcc"))
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    AnsiConsole.MarkupLine("[red]CUDA Toolkit not found![/]");
                    AnsiConsole.MarkupLine("To use your NVIDIA GPU, you must install the CUDA Toolkit.");
                    AnsiConsole.MarkupLine("[link]https://developer.nvidia.com/cuda-downloads[/]");
                    
                    if (!AnsiConsole.Confirm("Have you installed the CUDA Toolkit and are ready to retry?"))
                    {
                        throw new Exception("CUDA Toolkit required for Nvidia build.");
                    }
                }
                else
                {
                    // Linux auto-install attempt or error
                    throw new Exception("CUDA Toolkit missing. Run: sudo apt install nvidia-cuda-toolkit");
                }
            }
        }
        else if (sysInfo.GpuVendor == GpuVendor.Intel)
        {
             if (!CheckCommandExists("icx")) // Intel OneAPI Compiler
             {
                 AnsiConsole.MarkupLine("[yellow]Warning: Intel OneAPI Base Toolkit not found.[/]");
                 AnsiConsole.MarkupLine("For optimal Intel performance (SYCL), install OneAPI: [blue]https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html[/]");
                 AnsiConsole.MarkupLine("Proceeding with CPU/Vulkan fallback if build fails.");
             }
        }
        // AMD on Linux usually handled by "sudo apt install hipcc" or rocm libs
    }

    // --- CMake Helpers ---

    public static string? GetCmakePath()
    {
        // 1. Check Global Path
        if (CheckCommandExists("cmake")) return "cmake";

        // 2. Check Local 'MagicQuant/cmake/bin'
        string localPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), 
                                        MagicConstants.MagicQuantFolder, "cmake", "bin", "cmake.exe");
        return File.Exists(localPath) ? localPath : null;
    }

    private static async Task DownloadAndInstallCmakeAsync()
    {
        string magicPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), MagicConstants.MagicQuantFolder);
        string zipPath = Path.Combine(magicPath, "cmake.zip");
        string extractPath = Path.Combine(magicPath, "cmake");

        AnsiConsole.Status().Start("Downloading CMake...", ctx => 
        {
            using var client = new HttpClient();
            var bytes = client.GetByteArrayAsync(CmakeWinUrl).Result;
            File.WriteAllBytes(zipPath, bytes);
        });

        AnsiConsole.MarkupLine("Extracting CMake...");
        if (Directory.Exists(extractPath)) Directory.Delete(extractPath, true);
        
        ZipFile.ExtractToDirectory(zipPath, magicPath);
        
        // Rename the extracted folder (e.g., cmake-3.29-windows...) to just "cmake"
        var extractedDir = Directory.GetDirectories(magicPath, "cmake-*").First();
        Directory.Move(extractedDir, extractPath);
        
        File.Delete(zipPath);
        AnsiConsole.MarkupLine("[green]CMake installed successfully.[/]");
    }

    // --- Visual Studio Helpers ---

    private static bool CheckVisualStudio()
    {
        // Quick check for vswhere or cl.exe
        return CheckCommandExists("cl") || File.Exists(@"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    }

    private static void PromptForVisualStudio()
    {
        AnsiConsole.Write(new Rule("[red]Missing Visual Studio[/]"));
        AnsiConsole.MarkupLine("MagicQuant requires [bold]Visual Studio Build Tools 2022[/] with C++ Desktop Development.");
        AnsiConsole.MarkupLine("[blue]https://visualstudio.microsoft.com/downloads/#build-tools[/]");
        
        if (!AnsiConsole.Confirm("Have you installed Visual Studio Build Tools?"))
        {
            throw new Exception("Visual Studio is required to compile on Windows.");
        }
    }

    private static bool CheckCommandExists(string cmd)
    {
        try 
        {
            var psi = new ProcessStartInfo
            {
                FileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "where" : "which",
                Arguments = cmd,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            using var p = Process.Start(psi);
            p?.WaitForExit();
            return p?.ExitCode == 0;
        }
        catch { return false; }
    }
}