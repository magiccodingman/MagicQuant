using MagicQuant.Models;
using Spectre.Console;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using MagicQuant.Helpers;

namespace MagicQuant.Services;

public class QuantizationService
{
    private readonly BenchmarkService _benchmarker;
    private readonly string _ggufDir;
    private readonly string _benchDir;

    private readonly PythonManager _python;

    // Threading Control
    // We limit CPU-heavy quantization jobs to (TotalThreads / 8) to avoid choking the system
    // while leaving room for the GPU-heavy Perplexity tasks.
    private readonly SemaphoreSlim _cpuQuantLock;

    // The Queue
    private readonly ConcurrentQueue<Func<Task>> _jobQueue = new();
    private bool _isQueueRunning = false;

    public QuantizationService(BenchmarkService benchmarker)
    {
        _benchmarker = benchmarker;
        _python = _benchmarker._pyManager;
        // Setup Directories based on Cache (assumed populated by Evolution command)
        if (Cache.MagicQuantDirectory == null)
            throw new Exception("MagicQuant Directory not set. Run initialization first.");

        _ggufDir = Path.Combine(Cache.ModelMagicQuantDirectory, "GGUF");
        _benchDir = Path.Combine(Cache.ModelMagicQuantDirectory, "Benchmarks");

        Directory.CreateDirectory(_ggufDir);
        Directory.CreateDirectory(_benchDir);

        // Limit concurrent quantizations.
        // Example: 32 threads -> 4 concurrent quants (leaving threads for PPL)
        int maxConcurrent = Math.Max(1, (Cache.SysInfo?.ThreadCount ?? 4) / 8);
        _cpuQuantLock = new SemaphoreSlim(maxConcurrent, maxConcurrent);
    }

    // ----------------------------------------------------------------
    // 1. High-Level Entry Point: Build & Benchmark
    // ----------------------------------------------------------------

    public void QueueJob(HybridQuant quant)
    {
        _jobQueue.Enqueue(async () => await ProcessHybridQuantAsync(quant));
        StartQueueProcessor();
    }

    private void StartQueueProcessor()
    {
        if (_isQueueRunning) return;
        _isQueueRunning = true;

        // Fire and forget the processor loop
        Task.Run(async () =>
        {
            while (_jobQueue.TryDequeue(out var job))
            {
                await job();
            }

            _isQueueRunning = false;
        });
    }

    private async Task ProcessHybridQuantAsync(HybridQuant quant)
    {
        try
        {
            // 1. Ensure Base Model Exists (Dynamic BF16/F16/F32)
            string basePath = await EnsureBaseModelAsync();

            // 2. Determine Output Name & Path
            string modelName = GenerateHybridName(quant);
            string quantPath = Path.Combine(_ggufDir, $"{modelName}.gguf");

            // 3. Quantize (CPU Bound - Parallel)
            await _cpuQuantLock.WaitAsync();
            try
            {
                if (!File.Exists(quantPath))
                {
                    AnsiConsole.MarkupLine($"[cyan]Building Hybrid Model:[/] {modelName}");
                    await RunLlamaQuantizeAsync(basePath, quantPath, quant);
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            // 4. Benchmark (Mixed CPU/GPU/Exclusive)
            string modelBenchDir = Path.Combine(_benchDir, modelName);
            string metricsPath = Path.Combine(modelBenchDir, "bench_metrics.json");

            if (!File.Exists(metricsPath))
            {
                AnsiConsole.MarkupLine($"[yellow]Benchmarking:[/] {modelName}");

                await _benchmarker.RunAllBenchmarksAsync(
                    quantPath,
                    modelBenchDir,
                    saveLogits: false // Only base models save logits
                );

                // Cleanup: Delete GGUF after benchmark (unless protected base)
                if (File.Exists(quantPath) && !IsProtectedModel(modelName))
                {
                    AnsiConsole.MarkupLine($"[grey]Deleting temp model: {modelName}[/]");
                    File.Delete(quantPath);
                }
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.WriteException(ex);
        }
    }

    private bool IsProtectedModel(string name)
    {
        // Don't delete the BF16/F16/F32 base files
        return name.EndsWith("BF16") || name.EndsWith("F16") || name.EndsWith("F32");
    }

    // ----------------------------------------------------------------
    // 2. Base Model Generation (Dynamic BF16 / F16 / F32)
    // ----------------------------------------------------------------
    public async Task<string> EnsureBaseModelAsync()
    {
        // Resolve model name
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;

        // Determine torch type (default BF16)
        var torchType = Cache.TorchType ?? Cache.MainTorchType.BF16;
        string typeStr = torchType.ToString(); // BF16, F16, F32

        // Output paths
        string fileName = $"{modelName}-{typeStr}.gguf";
        string outputPath = Path.Combine(_ggufDir, fileName);
        string successFile = Path.Combine(_ggufDir, $"{fileName}.success.json");

        // Already converted?
        if (!File.Exists(outputPath) || !File.Exists(successFile))
        {
            // ---- Conversion ----
            AnsiConsole.MarkupLine($"[bold cyan]Converting to {typeStr}...[/]");

            if (File.Exists(outputPath))
                File.Delete(outputPath);

            string convertScript = Cache.ConvertScript
                                   ?? throw new Exception("ConvertScript path missing in Cache");

            string outTypeArg = typeStr.ToLowerInvariant(); // bf16 / f16 / f32

            string arguments =
                $"\"{convertScript}\" \"{Cache.ModelDirectory}\" " +
                $"--outtype {outTypeArg} " +
                $"--outfile \"{outputPath}\"";

            string python = _python.GetPythonExecutable();

            var psi = new ProcessStartInfo
            {
                FileName = python,
                Arguments = arguments,
                WorkingDirectory = Cache.LlamaRoot,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi)
                                ?? throw new InvalidOperationException("Failed to start conversion process");

            // UNTRUSTED OUTPUT → WriteLine ONLY
            process.OutputDataReceived += (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    AnsiConsole.WriteLine(e.Data);
            };

            process.ErrorDataReceived += (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    AnsiConsole.WriteLine(e.Data);
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
                throw new Exception($"{typeStr} conversion failed");

            // Write success marker
            await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
        }

        // ---- Benchmark Base Model ----
        string benchPath = Path.Combine(_benchDir, typeStr);
        string logitsDir = Path.Combine(benchPath, "logits");

        AnsiConsole.MarkupLine(
            $"[bold yellow]Benchmarking Base {typeStr} (Saving Logits)...[/]"
        );

        await _benchmarker.RunAllBenchmarksAsync(
            modelPath: outputPath,
            benchDir: benchPath,
            klLogitsDir: logitsDir,
            saveLogits: true
        );

        return outputPath;
    }


    // ----------------------------------------------------------------
    // 3. Hybrid Quantization Execution
    // ----------------------------------------------------------------

    public async Task RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridQuant quant)
    {
        var args = new List<string>(capacity: 64);

        if (quant.Tensors is { Count: > 0 })
        {
            foreach (var hybrid in quant.Tensors)
            {
                if (hybrid?.TGroup == null) continue;

                // Resolve scheme dynamically (handles BF16/F16 shared ID)
                string schemeName = ResolveSchemeName(hybrid.TensorType);

                foreach (var tensorPattern in hybrid.TGroup.Tensors)
                {
                    args.Add($"--tensor-type \"{tensorPattern}={schemeName}\"");
                }
            }
        }

        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(ResolveBaseName(quant.BaseQuant));
        args.Add("8");

        string arguments = string.Join(" ", args);

        string bin = Cache.LlamaBin + (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "/llama-quantize.exe"
            : "/llama-quantize");

        var psi = new ProcessStartInfo
        {
            FileName = bin,
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var p = Process.Start(psi);
        if (p == null) throw new InvalidOperationException($"Failed to start process: {bin}");

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        await p.WaitForExitAsync();

        if (p.ExitCode != 0)
            throw new Exception($"Quantization failed for {outputFile}");
    }

    private static string ResolveBaseName(BaselineQuants b)
    {
        if (b.Names.IsDefaultOrEmpty)
            throw new InvalidOperationException($"BaselineQuants '{b.UniqueId}' has no Names.");
        return b.Names[0];
    }

    private static string ResolveSchemeName(TensorWeightScheme s)
    {
        if (s.Names.IsDefaultOrEmpty)
            throw new InvalidOperationException($"TensorWeightScheme '{s.UniqueId}' has no Names.");

        // Special case: BF16_F16 shares UniqueId and has two names ["BF16","F16"].
        if (s.UniqueId == TensorWeightScheme.BF16_F16.UniqueId && s.Names.Length >= 2)
        {
            // Dynamic check against Cache
            if (Cache.TorchType == Cache.MainTorchType.F16)
            {
                return "F16";
            }

            // Default to BF16 for BF16 or F32 types (safer modern default)
            return "BF16";
        }

        return s.Names[0];
    }

    // ----------------------------------------------------------------
    // 4. Naming Scheme Logic (E-H-Q-K-O...)
    // ----------------------------------------------------------------

    public string GenerateHybridName(HybridQuant quant)
    {
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        string baseName = ResolveBaseName(quant.BaseQuant);

        if (quant.Tensors == null || quant.Tensors.Count == 0)
        {
            return $"{modelName}-{baseName}";
        }

        var grouped = quant.Tensors
            .GroupBy(t => ResolveSchemeName(t.TensorType))
            .Select(g => new
            {
                Type = g.Key,
                Codes = g.Select(x => x.TGroup.ShortCode)
                    .OrderBy(c => GetOrder(c))
                    .ToArray()
            })
            .OrderBy(x => GetOrder(x.Codes.FirstOrDefault()))
            .ToList();

        var nameParts = new List<string>(capacity: grouped.Count);

        foreach (var group in grouped)
        {
            string codeStr = new string(group.Codes);
            string quantStr = SimplifyQuant(group.Type);
            nameParts.Add($"{codeStr}-{quantStr}");
        }

        string suffix = string.Join("-", nameParts);
        return $"{modelName}-{baseName}-{suffix}";
    }

    private int GetOrder(char c)
    {
        // E, H, Q, K, O, U, D, X, R
        return "EHQKOUDXR".IndexOf(c);
    }

    private string SimplifyQuant(string quant)
    {
        // Optional: Simplify quantization names for the filename
        // BF16 -> B16, Q4_K_M -> Q4KM
        return quant.Replace("_", "").Replace("BF16", "B16").Replace("F16", "F16");
    }
}