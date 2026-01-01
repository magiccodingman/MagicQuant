using System.Runtime.InteropServices;
using MQ.DB;
using MQ.DB.Models;
using System.Diagnostics;

namespace MagicQuant.Helpers;

public static class HardwareHelper
{
    public static SystemInfo GetSystemInfo()
    {
        var info = new SystemInfo
        {
            ThreadCount = Environment.ProcessorCount,
            RamGb = GetTotalRam(),
            GpuVendor = DetectGpuVendor(out string gpuName, out double vram),
            GpuName = gpuName,
            VramGb = vram
        };
        return info;
    }
    public static double GetCudaVersion()
    {
        try
        {
            // We check nvcc because that matches the Toolkit we installed/verified
            var psi = new ProcessStartInfo
            {
                FileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcc.exe" : "nvcc",
                Arguments = "--version",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var p = Process.Start(psi);
            if (p == null) return 0;

            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit();

            // Output format: "Cuda compilation tools, release 12.4, V12.4.131"
            // Regex to find "release X.Y"
            var match = System.Text.RegularExpressions.Regex.Match(output, @"release (\d+\.\d+)");
            if (match.Success && double.TryParse(match.Groups[1].Value, out double version))
            {
                return version;
            }
        }
        catch
        {
            // Fallback: If nvcc fails, we might try parsing nvidia-smi, 
            // but for now, returning 0 triggers a safe fallback.
        }
        return 0;
    }

    private static double GetTotalRam()
    {
        // simplified generic check
        return GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1024.0 / 1024.0 / 1024.0; 
    }

    private static GpuVendor DetectGpuVendor(out string name, out double vram)
    {
        name = "Generic";
        vram = 0;

        // 1. Check NVIDIA (nvidia-smi) - Works on Linux & Windows
        try 
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo("nvidia-smi", "--query-gpu=name,memory.total --format=csv,noheader,nounits") { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true }
            };
            process.Start();
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
            {
                var parts = output.Split(',');
                name = parts[0].Trim();
                if (parts.Length > 1 && double.TryParse(parts[1], out double mem)) vram = mem / 1024.0;
                return GpuVendor.Nvidia;
            }
        }
        catch { /* Not Nvidia */ }

        // 2. Check MacOS (Metal) - Placeholder
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) return GpuVendor.Cpu; // Todo: Metal check

        // 3. Fallbacks (AMD/Intel) would go here (e.g., parsing lshw on linux)
        // For now, defaulting to CPU if Nvidia fails
        return GpuVendor.Cpu; 
    }
}