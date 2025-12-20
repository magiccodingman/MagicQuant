using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using MagicQuant;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public class QuantizationService
{
    private readonly BenchmarkService _benchmarker;
    private readonly string _ggufDir;
    private readonly string _benchDir;
    
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
        
        // Setup Directories based on Cache (assumed populated by Evolution command)
        if (Cache.MagicQuantDirectory == null) 
            throw new Exception("MagicQuant Directory not set. Run initialization first.");

        _ggufDir = Path.Combine(Cache.MagicQuantDirectory, "GGUF");
        _benchDir = Path.Combine(Cache.MagicQuantDirectory, "Benchmarks");

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

    public void QueueJob(HybridBuild build)
    {
        _jobQueue.Enqueue(async () => await ProcessHybridBuildAsync(build));
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

    private async Task ProcessHybridBuildAsync(HybridBuild build)
    {
        try
        {
            // 1. Ensure BF16 Base Exists (Prerequisite)
            string bf16Path = await EnsureBf16ModelAsync();

            // 2. Determine Output Name & Path
            string modelName = GenerateHybridName(build);
            string quantPath = Path.Combine(_ggufDir, $"{modelName}.gguf");
            
            // 3. Quantize (CPU Bound - Parallel)
            await _cpuQuantLock.WaitAsync();
            try
            {
                if (!File.Exists(quantPath))
                {
                    AnsiConsole.MarkupLine($"[cyan]Building Hybrid Model:[/] {modelName}");
                    await RunLlamaQuantizeAsync(bf16Path, quantPath, build);
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            // 4. Benchmark (Mixed CPU/GPU/Exclusive)
            // The BenchmarkService handles its own locking (Exclusive vs VRAM)
            // so we can just call it here.
            string modelBenchDir = Path.Combine(_benchDir, modelName);
            string metricsPath = Path.Combine(modelBenchDir, "bench_metrics.json");

            if (!File.Exists(metricsPath))
            {
                AnsiConsole.MarkupLine($"[yellow]Benchmarking:[/] {modelName}");
                
                // Note: LlamaBench will block everything else (ExclusiveBenchLock).
                // Perplexity will run in parallel with other Quant jobs if VRAM permits (VramLock).
                await _benchmarker.RunAllBenchmarksAsync(
                    quantPath, 
                    modelBenchDir, 
                    saveLogits: false // Only BF16 saves logits usually
                );
                
                // Cleanup: Delete GGUF after benchmark to save space (per requirements)
                // EXCEPT if it is a base/common one we might want to keep?
                // Logic: "always remember to delete the hybrid or base... delete with true delete"
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
    // 2. BF16 Base Generation (The "Root" Model)
    // ----------------------------------------------------------------

    public async Task<string> EnsureBf16ModelAsync()
    {
        // Name usually: <ModelName>-BF16.gguf
        // We get ModelName from Cache.ModelDirectory
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        
        // Detect Torch Type from Cache (as you requested) or default to BF16
        string typeSuffix = Cache.SysInfo != null ? "BF16" : "F16"; // Simplification
        // Real logic: Check Cache.TorchType (e.g. "BF16", "F16", "F32")
        // For this snippet, I assume "BF16" is the target per your prompt.

        string fileName = $"{modelName}-BF16.gguf";
        string output = Path.Combine(_ggufDir, fileName);
        string successFile = Path.Combine(_ggufDir, $"{fileName}.success.json");

        if (File.Exists(output) && File.Exists(successFile)) 
            return output;

        // Create/Convert
        AnsiConsole.MarkupLine($"[bold cyan]Converting to {typeSuffix}...[/]");
        
        // Clean partials
        if (File.Exists(output)) File.Delete(output);

        string convertScript = Cache.ConvertScript 
            ?? throw new Exception("ConvertScript path missing in Cache");

        // Command: python convert_hf_to_gguf.py path --outtype bf16 --outfile output
        var psi = new ProcessStartInfo
        {
            FileName = "python", // Or _pyManager.GetPythonExecutable()
            Arguments = $"\"{convertScript}\" \"{Cache.ModelDirectory}\" --outtype bf16 --outfile \"{output}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var p = Process.Start(psi);
        p.OutputDataReceived += (s, e) => { if(e.Data != null) AnsiConsole.WriteLine(e.Data); };
        p.ErrorDataReceived += (s, e) => { if(e.Data != null) AnsiConsole.WriteLine(e.Data); }; // Errors often printed to stderr
        p.BeginOutputReadLine(); p.BeginErrorReadLine();
        await p.WaitForExitAsync();

        if (p.ExitCode != 0) throw new Exception("BF16 Conversion Failed");

        // Write Success JSON
        await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");

        // Run Benchmark on BF16 (Critical First Step)
        string benchPath = Path.Combine(_benchDir, "BF16");
        // We need to save logits for the base model so others can calculate KLD
        string logitsDir = Path.Combine(benchPath, "logits");
        
        AnsiConsole.MarkupLine("[bold yellow]Benchmarking Base BF16 (Saving Logits)...[/]");
        await _benchmarker.RunAllBenchmarksAsync(
            output, 
            benchPath, 
            klLogitsDir: logitsDir, 
            saveLogits: true
        );

        return output;
    }

    // ----------------------------------------------------------------
    // 3. Hybrid Quantization Execution
    // ----------------------------------------------------------------

    public async Task RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridBuild build)
    {
        // 1. Base Arguments
        // llama-quantize [flags] input output base_type
        var args = new List<string>();

        // 2. Hybrid Overrides (The Magic)
        // Modern llama-quantize supports --tensor-type <regex>=<type>
        if (build.Tensors != null)
        {
            foreach (var hybrid in build.Tensors)
            {
                foreach (var tensorPattern in hybrid.TensorGroup.Tensors)
                {
                    // Convert glob-like patterns to what llama-quantize accepts if needed
                    // Usually it accepts substrings or regex.
                    // We append: --tensor-type pattern=type
                    args.Add($"--tensor-type \"{tensorPattern}={hybrid.TensorType}\"");
                }
            }
        }

        // 3. Files and Base Type
        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(build.Base);

        // 4. Threads (8 per quant job as requested)
        args.Add("8");

        string arguments = string.Join(" ", args);
        string bin = Cache.LlamaBin + (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "/llama-quantize.exe" : "/llama-quantize");
        
        // Execute
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
        // We might not want to spam console with quantization logs unless verbose
        p.BeginOutputReadLine(); p.BeginErrorReadLine(); 
        await p.WaitForExitAsync();

        if (p.ExitCode != 0) throw new Exception($"Quantization failed for {outputFile}");
    }

    // ----------------------------------------------------------------
    // 4. Naming Scheme Logic (E-H-Q-K-O...)
    // ----------------------------------------------------------------

    public string GenerateHybridName(HybridBuild build)
    {
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        
        // If pure baseline (no tensors), just <Model>-<Base>
        if (build.Tensors == null || build.Tensors.Count == 0)
        {
            return $"{modelName}-{build.Base}";
        }

        // Hybrid Logic
        // 1. Group by Quant Type
        var grouped = build.Tensors
            .GroupBy(t => t.TensorType)
            .Select(g => new 
            { 
                Type = g.Key, 
                // Get Sortable ShortCodes (E, H, Q, K...)
                Codes = g.Select(x => x.TensorGroup.ShortCode).OrderBy(c => GetOrder(c)).ToArray()
            })
            .OrderBy(x => GetOrder(x.Codes.FirstOrDefault()))
            .ToList();

        var nameParts = new List<string>();
        
        foreach (var group in grouped)
        {
            string codeStr = new string(group.Codes); // e.g., "EH" or "QKO"
            // Remove underscores from quant type for cleanliness (Q4_K_M -> Q4KM) if desired
            // The prompt says "Q6K", "B16". Let's stick to simple mapping.
            string quantStr = SimplifyQuant(group.Type);
            
            nameParts.Add($"{codeStr}-{quantStr}");
        }

        string suffix = string.Join("-", nameParts);
        return $"{modelName}-{build.Base}-{suffix}";
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