using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class HardwareHelper
{
    public static SystemInfo GetSystemInfo()
    {
        var detectedGpus = DetectGpus();
        var selectedGpus = SelectBestGpuVendorGroup(detectedGpus);

        if (selectedGpus.Count == 0)
        {
            selectedGpus = detectedGpus;
        }

        return new SystemInfo
        {
            ThreadCount = Environment.ProcessorCount,
            RamGb = GetTotalRamGb(),
            GpuInfo = selectedGpus
        };
    }
    
    private static List<GpuInfo> SelectBestGpuVendorGroup(List<GpuInfo> gpus)
    {
        if (gpus == null || gpus.Count == 0)
            return new List<GpuInfo>();

        var candidates = gpus
            .Where(x => x != null)
            .Where(x => x.GpuVendor != GpuVendor.Unknown && x.GpuVendor != GpuVendor.Cpu)
            .Where(x => x.VramGb > 0.01)
            .ToList();

        if (candidates.Count == 0)
            return new List<GpuInfo>();

        var bestVendor = candidates
            .GroupBy(x => x.GpuVendor)
            .Select(g => new
            {
                Vendor = g.Key,
                TotalVram = g.Sum(x => x.VramGb),
                Count = g.Count()
            })
            .OrderByDescending(x => x.TotalVram)
            .ThenByDescending(x => x.Count)
            .First()
            .Vendor;

        return candidates
            .Where(x => x.GpuVendor == bestVendor)
            .OrderByDescending(x => x.VramGb)
            .ThenBy(x => x.GpuName, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public static double GetCudaVersion()
    {
        try
        {
            var result = RunProcess(
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcc.exe" : "nvcc",
                "--version");

            if (!result.Success || string.IsNullOrWhiteSpace(result.StdOut))
                return 0;

            var match = Regex.Match(result.StdOut, @"release (\d+\.\d+)");
            if (match.Success &&
                double.TryParse(match.Groups[1].Value, NumberStyles.Any, CultureInfo.InvariantCulture, out var version))
            {
                return version;
            }
        }
        catch
        {
            // ignored
        }

        return 0;
    }

    private static double GetTotalRamGb()
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // More reliable than GC.GetGCMemoryInfo for actual system RAM
                var result = RunProcess(
                    "powershell",
                    "-NoProfile -ExecutionPolicy Bypass -Command \"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory\"");

                if (result.Success &&
                    TryParseFirstInteger(result.StdOut, out var bytes) &&
                    bytes > 0)
                {
                    return BytesToGb(bytes);
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                var memInfo = "/proc/meminfo";
                if (File.Exists(memInfo))
                {
                    var line = File.ReadLines(memInfo)
                        .FirstOrDefault(x => x.StartsWith("MemTotal:", StringComparison.OrdinalIgnoreCase));

                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        var match = Regex.Match(line, @"MemTotal:\s+(\d+)\s+kB", RegexOptions.IgnoreCase);
                        if (match.Success &&
                            ulong.TryParse(match.Groups[1].Value, out var kb))
                        {
                            return kb / 1024.0 / 1024.0;
                        }
                    }
                }
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                var result = RunProcess("sysctl", "-n hw.memsize");
                if (result.Success &&
                    TryParseFirstInteger(result.StdOut, out var bytes) &&
                    bytes > 0)
                {
                    return BytesToGb(bytes);
                }
            }
        }
        catch
        {
            // ignored
        }

        // Last-resort fallback. This is NOT actual total RAM, just available-to-GC-ish territory.
        var fallback = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        return fallback > 0 ? BytesToGb((ulong)fallback) : 0;
    }

    private static List<GpuInfo> DetectGpus()
    {
        var gpus = new List<GpuInfo>();

        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                gpus.AddRange(DetectGpusWindows());
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                gpus.AddRange(DetectGpusLinux());
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                gpus.AddRange(DetectGpusMac());
            }
        }
        catch
        {
            // ignored
        }

        gpus = NormalizeAndDedupe(gpus);

        if (gpus.Count == 0)
        {
            gpus.Add(new GpuInfo
            {
                GpuVendor = GpuVendor.Cpu,
                GpuName = "No discrete GPU detected",
                VramGb = 0
            });
        }

        return gpus;
    }

    // =========================
    // Windows
    // =========================

    private static IEnumerable<GpuInfo> DetectGpusWindows()
{
    var results = new List<GpuInfo>();

    var nvidia = DetectNvidiaViaSmi().ToList();
    results.AddRange(nvidia);

    var ps = RunProcess(
        "powershell",
        "-NoProfile -ExecutionPolicy Bypass -Command \"Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,PNPDeviceID | ConvertTo-Json -Depth 3\"");

    if (ps.Success && !string.IsNullOrWhiteSpace(ps.StdOut))
    {
        try
        {
            using var doc = JsonDocument.Parse(ps.StdOut);

            IEnumerable<JsonElement> items = doc.RootElement.ValueKind switch
            {
                JsonValueKind.Array => doc.RootElement.EnumerateArray().ToArray(),
                JsonValueKind.Object => new[] { doc.RootElement },
                _ => Array.Empty<JsonElement>()
            };

            foreach (var item in items)
            {
                var name = item.TryGetProperty("Name", out var nameEl)
                    ? nameEl.GetString() ?? "Unknown"
                    : "Unknown";

                var pnp = item.TryGetProperty("PNPDeviceID", out var pnpEl)
                    ? pnpEl.GetString() ?? string.Empty
                    : string.Empty;

                var vendor = DetectVendorFromNameOrId(name, pnp);

                // If we already have NVIDIA via nvidia-smi, skip weaker duplicate NVIDIA rows.
                if (vendor == GpuVendor.Nvidia && nvidia.Count > 0)
                    continue;

                double vramGb = 0;
                if (item.TryGetProperty("AdapterRAM", out var ramEl))
                {
                    if (ramEl.ValueKind == JsonValueKind.Number && ramEl.TryGetUInt64(out var bytes))
                    {
                        vramGb = BytesToGb(bytes);
                    }
                    else if (ramEl.ValueKind == JsonValueKind.String &&
                             ulong.TryParse(ramEl.GetString(), out var parsed))
                    {
                        vramGb = BytesToGb(parsed);
                    }
                }

                results.Add(new GpuInfo
                {
                    GpuName = name,
                    GpuVendor = vendor,
                    VramGb = SanitizeGb(vramGb)
                });
            }
        }
        catch
        {
            // ignored
        }
    }

    return results;
}

    // =========================
    // Linux
    // =========================

    private static IEnumerable<GpuInfo> DetectGpusLinux()
    {
        var results = new List<GpuInfo>();

        // 1. NVIDIA: highest-confidence source
        results.AddRange(DetectNvidiaViaSmi());

        // 2. lspci for names/vendors
        var lspci = RunProcess("bash", "-lc \"lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display'\"");
        var lspciEntries = new List<(string Name, GpuVendor Vendor)>();

        if (lspci.Success && !string.IsNullOrWhiteSpace(lspci.StdOut))
        {
            foreach (var line in SplitLines(lspci.StdOut))
            {
                var name = ExtractGpuNameFromLspci(line);
                var vendor = DetectVendorFromNameOrId(line, line);

                lspciEntries.Add((name, vendor));

                results.Add(new GpuInfo
                {
                    GpuName = name,
                    GpuVendor = vendor,
                    VramGb = 0
                });
            }
        }

        // 3. AMD/Intel VRAM hints from /sys/class/drm
        results = MergeLinuxSysFsData(results);

        return results;
    }

    private static List<GpuInfo> MergeLinuxSysFsData(List<GpuInfo> existing)
    {
        try
        {
            var drmPath = "/sys/class/drm";
            if (!Directory.Exists(drmPath))
                return existing;

            var cardDirs = Directory.GetDirectories(drmPath, "card*")
                .Where(x => !x.Contains("-", StringComparison.Ordinal)) // skip card0-DP-1 type connector entries
                .OrderBy(x => x)
                .ToList();

            foreach (var cardDir in cardDirs)
            {
                var deviceDir = Path.Combine(cardDir, "device");
                if (!Directory.Exists(deviceDir))
                    continue;

                string vendorId = ReadTrimmedFile(Path.Combine(deviceDir, "vendor"));
                string deviceId = ReadTrimmedFile(Path.Combine(deviceDir, "device"));

                var vendor = DetectVendorFromPciVendorId(vendorId);
                if (vendor == GpuVendor.Unknown)
                    continue;

                double vramGb = 0;

                // AMD dedicated VRAM often exposed here on amdgpu
                var amdVramPath = Path.Combine(deviceDir, "mem_info_vram_total");
                if (File.Exists(amdVramPath) &&
                    ulong.TryParse(ReadTrimmedFile(amdVramPath), out var amdBytes))
                {
                    vramGb = BytesToGb(amdBytes);
                }

                // Intel integrated usually won’t have dedicated VRAM here.
                // Leave as 0 instead of inventing nonsense.

                // Try to match an existing entry by vendor with 0 VRAM and patch it in.
                var possibleMatches = existing
                    .Where(x => x.GpuVendor == vendor && x.VramGb <= 0.01)
                    .ToList();

                GpuInfo? existingMatch = possibleMatches.Count == 1 ? possibleMatches[0] : null;

                if (existingMatch != null && vramGb > 0)
                {
                    existingMatch.VramGb = SanitizeGb(vramGb);
                }
                else
                {
                    existing.Add(new GpuInfo
                    {
                        GpuVendor = vendor,
                        GpuName = string.IsNullOrWhiteSpace(deviceId)
                            ? vendor.ToString()
                            : $"{vendor} GPU",
                        VramGb = SanitizeGb(vramGb)
                    });
                }
            }
        }
        catch
        {
            // ignored
        }

        return existing;
    }

    // =========================
    // macOS
    // =========================

    private static IEnumerable<GpuInfo> DetectGpusMac()
    {
        var results = new List<GpuInfo>();

        var sp = RunProcess("system_profiler", "SPDisplaysDataType -json");
        if (!sp.Success || string.IsNullOrWhiteSpace(sp.StdOut))
            return results;

        try
        {
            using var doc = JsonDocument.Parse(sp.StdOut);

            if (!doc.RootElement.TryGetProperty("SPDisplaysDataType", out var displays) ||
                displays.ValueKind != JsonValueKind.Array)
            {
                return results;
            }

            foreach (var gpu in displays.EnumerateArray())
            {
                string name =
                    GetJsonString(gpu, "sppci_model") ??
                    GetJsonString(gpu, "_name") ??
                    "Unknown";

                double vramGb = 0;

                // Intel/older Macs may expose strings like "1536 MB"
                var vramText =
                    GetJsonString(gpu, "spdisplays_vram") ??
                    GetJsonString(gpu, "spdisplays_vram_shared") ??
                    GetJsonString(gpu, "sppci_vram");

                if (!string.IsNullOrWhiteSpace(vramText))
                {
                    vramGb = ParseMemoryStringToGb(vramText);
                }

                // Apple Silicon often won’t expose dedicated VRAM because memory is unified.
                // So leaving 0 here is more honest than lying.

                results.Add(new GpuInfo
                {
                    GpuName = name,
                    GpuVendor = DetectVendorFromNameOrId(name, name),
                    VramGb = SanitizeGb(vramGb)
                });
            }
        }
        catch
        {
            // ignored
        }

        return results;
    }

    // =========================
    // Shared NVIDIA path
    // =========================

    private static IEnumerable<GpuInfo> DetectNvidiaViaSmi()
    {
        var results = new List<GpuInfo>();

        string exe = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvidia-smi.exe" : "nvidia-smi";
        var smi = RunProcess(exe, "--query-gpu=gpu_uuid,name,memory.total --format=csv,noheader,nounits");

        if (!smi.Success || string.IsNullOrWhiteSpace(smi.StdOut))
            return results;

        foreach (var line in SplitLines(smi.StdOut))
        {
            var parts = line.Split(',', StringSplitOptions.TrimEntries);

            if (parts.Length < 3)
                continue;

            var uuid = parts[0].Trim();
            var name = parts[1].Trim();

            double vramGb = 0;
            if (double.TryParse(parts[2].Trim(), NumberStyles.Any, CultureInfo.InvariantCulture, out var memMb))
            {
                vramGb = memMb / 1024.0;
            }

            results.Add(new GpuInfo
            {
                UniqueId = uuid,
                GpuVendor = GpuVendor.Nvidia,
                GpuName = string.IsNullOrWhiteSpace(name) ? "NVIDIA GPU" : name,
                VramGb = SanitizeGb(vramGb)
            });
        }

        return results;
    }

    // =========================
    // Helpers
    // =========================

    private static List<GpuInfo> NormalizeAndDedupe(List<GpuInfo> gpus)
    {
        var final = new List<GpuInfo>();

        foreach (var gpu in gpus)
        {
            var normalizedName = string.IsNullOrWhiteSpace(gpu.GpuName)
                ? "Unknown"
                : Regex.Replace(gpu.GpuName.Trim(), @"\s+", " ");

            var vendor = gpu.GpuVendor == GpuVendor.Unknown
                ? DetectVendorFromNameOrId(normalizedName, gpu.UniqueId ?? normalizedName)
                : gpu.GpuVendor;

            GpuInfo? existing = null;

            // Only merge by UniqueId if we actually have one
            if (!string.IsNullOrWhiteSpace(gpu.UniqueId))
            {
                existing = final.FirstOrDefault(x =>
                    !string.IsNullOrWhiteSpace(x.UniqueId) &&
                    string.Equals(x.UniqueId, gpu.UniqueId, StringComparison.OrdinalIgnoreCase));
            }

            // If no UniqueId, do NOT aggressively merge by name/vendor.
            // That breaks multi-GPU systems with identical cards.
            if (existing == null)
            {
                final.Add(new GpuInfo
                {
                    UniqueId = gpu.UniqueId,
                    GpuVendor = vendor,
                    GpuName = normalizedName,
                    VramGb = SanitizeGb(gpu.VramGb)
                });
            }
            else
            {
                if (existing.VramGb <= 0.01 && gpu.VramGb > existing.VramGb)
                    existing.VramGb = SanitizeGb(gpu.VramGb);

                if (string.Equals(existing.GpuName, "Unknown", StringComparison.OrdinalIgnoreCase) &&
                    !string.Equals(normalizedName, "Unknown", StringComparison.OrdinalIgnoreCase))
                {
                    existing.GpuName = normalizedName;
                }

                if (existing.GpuVendor == GpuVendor.Unknown && vendor != GpuVendor.Unknown)
                    existing.GpuVendor = vendor;
            }
        }

        if (final.Any(x => x.GpuVendor != GpuVendor.Cpu))
        {
            final.RemoveAll(x => x.GpuVendor == GpuVendor.Cpu);
        }

        return final;
    }

    private static GpuVendor DetectVendorFromNameOrId(string? name, string? idText)
    {
        var haystack = $"{name} {idText}".ToLowerInvariant();

        if (haystack.Contains("nvidia") || haystack.Contains("geforce") || haystack.Contains("quadro") || haystack.Contains("tesla"))
            return GpuVendor.Nvidia;

        if (haystack.Contains("amd") || haystack.Contains("advanced micro devices") || haystack.Contains("radeon") || haystack.Contains("firepro"))
            return GpuVendor.Amd;

        if (haystack.Contains("intel") || haystack.Contains("arc") || haystack.Contains("uhd") || haystack.Contains("iris"))
            return GpuVendor.Intel;

        return GpuVendor.Unknown;
    }

    private static GpuVendor DetectVendorFromPciVendorId(string? vendorId)
    {
        if (string.IsNullOrWhiteSpace(vendorId))
            return GpuVendor.Unknown;

        var v = vendorId.Trim().ToLowerInvariant();

        return v switch
        {
            "0x10de" => GpuVendor.Nvidia,
            "0x1002" => GpuVendor.Amd,
            "0x8086" => GpuVendor.Intel,
            _ => GpuVendor.Unknown
        };
    }

    private static string ExtractGpuNameFromLspci(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return "Unknown";

        var idx = line.IndexOf(':');
        if (idx >= 0 && idx < line.Length - 1)
        {
            var right = line[(idx + 1)..].Trim();

            // Strip "VGA compatible controller:" / "3D controller:" / "Display controller:"
            right = Regex.Replace(
                right,
                @"^(VGA compatible controller|3D controller|Display controller)\s*:\s*",
                "",
                RegexOptions.IgnoreCase).Trim();

            return right;
        }

        return line.Trim();
    }

    private static string? GetJsonString(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var prop))
            return null;

        return prop.ValueKind == JsonValueKind.String ? prop.GetString() : prop.ToString();
    }

    private static double ParseMemoryStringToGb(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return 0;

        var match = Regex.Match(text, @"([\d.]+)\s*(TB|GB|MB|KB)", RegexOptions.IgnoreCase);
        if (!match.Success)
            return 0;

        if (!double.TryParse(match.Groups[1].Value, NumberStyles.Any, CultureInfo.InvariantCulture, out var value))
            return 0;

        var unit = match.Groups[2].Value.ToUpperInvariant();
        return unit switch
        {
            "TB" => value * 1024.0,
            "GB" => value,
            "MB" => value / 1024.0,
            "KB" => value / 1024.0 / 1024.0,
            _ => 0
        };
    }

    private static bool TryParseFirstInteger(string? text, out ulong value)
    {
        value = 0;
        if (string.IsNullOrWhiteSpace(text))
            return false;

        var match = Regex.Match(text, @"\d+");
        return match.Success && ulong.TryParse(match.Value, out value);
    }

    private static string ReadTrimmedFile(string path)
    {
        try
        {
            return File.Exists(path) ? File.ReadAllText(path).Trim() : string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }

    private static IEnumerable<string> SplitLines(string text)
    {
        return text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    }

    private static double BytesToGb(ulong bytes)
    {
        return bytes / 1024.0 / 1024.0 / 1024.0;
    }

    private static double SanitizeGb(double gb)
    {
        if (double.IsNaN(gb) || double.IsInfinity(gb) || gb < 0)
            return 0;

        // Keep this nice and user-facing
        return Math.Round(gb, 2);
    }

    private static ProcessResult RunProcess(string fileName, string arguments, int timeoutMs = 5000)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = psi };
            process.Start();

            var stdOutTask = process.StandardOutput.ReadToEndAsync();
            var stdErrTask = process.StandardError.ReadToEndAsync();

            if (!process.WaitForExit(timeoutMs))
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return ProcessResult.Failure("Process timed out.");
            }

            Task.WaitAll(stdOutTask, stdErrTask);

            return new ProcessResult(
                process.ExitCode == 0,
                stdOutTask.Result ?? string.Empty,
                stdErrTask.Result ?? string.Empty,
                process.ExitCode);
        }
        catch (Exception ex)
        {
            return ProcessResult.Failure(ex.Message);
        }
    }

    private readonly record struct ProcessResult(bool Success, string StdOut, string StdErr, int ExitCode)
    {
        public static ProcessResult Failure(string error) => new(false, string.Empty, error, -1);
    }
}