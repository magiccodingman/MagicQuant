using MagicQuant.Models;
using Spectre.Console;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;

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
            // 1. Ensure BF16 Base Exists (Prerequisite)
            string bf16Path = await EnsureBf16ModelAsync();

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
                    await RunLlamaQuantizeAsync(bf16Path, quantPath, quant);
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
        p.OutputDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.WriteLine(e.Data); };
        p.ErrorDataReceived += (s, e) => { if (e.Data != null) AnsiConsole.WriteLine(e.Data); };
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
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

   public async Task RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridQuant quant)
    {
        // llama-quantize [flags] input output base_type threads
        var args = new List<string>(capacity: 64);

        // 1) Hybrid overrides
        // llama-quantize supports: --tensor-type "<pattern>=<type>"
        // We emit one flag per tensor pattern per group.
        if (quant.Tensors is { Count: > 0 })
        {
            foreach (var hybrid in quant.Tensors)
            {
                // Skip null guard (shouldn't happen, but keep robust)
                if (hybrid?.TGroup == null)
                    continue;

                // Resolve actual llama quant name for this scheme
                // (handles BF16/F16 shared ID)
                string schemeName = ResolveSchemeName(hybrid.TensorType);

                foreach (var tensorPattern in hybrid.TGroup.Tensors)
                {
                    args.Add($"--tensor-type \"{tensorPattern}={schemeName}\"");
                }
            }
        }

        // 2) Files + base type + threads
        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");

        // Base type comes from the BaselineQuants record
        args.Add(ResolveBaseName(quant.BaseQuant));

        // Threads (8 per quant job as requested)
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
        if (p == null)
            throw new InvalidOperationException($"Failed to start process: {bin}");

        // If you want logging, attach handlers like you do elsewhere.
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

        // Most bases only have a single name.
        // If you ever add aliases, this keeps it deterministic.
        return b.Names[0];
    }

    private static string ResolveSchemeName(TensorWeightScheme s)
    {
        if (s.Names.IsDefaultOrEmpty)
            throw new InvalidOperationException($"TensorWeightScheme '{s.UniqueId}' has no Names.");

        // Special case: BF16_F16 shares UniqueId and has two names ["BF16","F16"].
        // Pick based on runtime float type when available.
        if (s.UniqueId == TensorWeightScheme.BF16_F16.UniqueId && s.Names.Length >= 2)
        {
            // Plug in your real logic here (Cache.TorchType etc.)
            // For now, default BF16 if unknown.
            // Example expected values: "BF16", "F16", "F32"
            var torch = Cache.TorchType; // if you have it; otherwise this can be null
            if (string.Equals(torch, "F16", StringComparison.OrdinalIgnoreCase))
                return "F16";

            return "BF16";
        }

        // Normal case: first name is canonical
        return s.Names[0];
    }

    // ----------------------------------------------------------------
    // 4. Naming Scheme Logic (E-H-Q-K-O...)
    // ----------------------------------------------------------------

    public string GenerateHybridName(HybridQuant quant)
    {
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;

        string baseName = ResolveBaseName(quant.BaseQuant);

        // If pure baseline (no tensors), just <Model>-<Base>
        if (quant.Tensors == null || quant.Tensors.Count == 0)
        {
            return $"{modelName}-{baseName}";
        }

        // Group by quant scheme (resolved to a stable string)
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
            string codeStr = new string(group.Codes);     // e.g., "EH" or "QKO"
            string quantStr = SimplifyQuant(group.Type);  // e.g., "Q6K", "B16", "IQ4XS"

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
