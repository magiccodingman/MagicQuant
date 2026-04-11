using System.Diagnostics;
using System.Runtime.InteropServices;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public enum SampleProcessState
{
    Completed = 1,
    Skipped = 2,
    Failed = 3
}

public sealed class SampleProcessingSummary
{
    public int Requested { get; set; }
    public int Completed { get; set; }
    public int Skipped { get; set; }
    public int Failed { get; set; }
}

public class QuantizationService
{
    private readonly BenchmarkService _benchmarker;
    private readonly string _ggufDir;
    private readonly string _benchDir;
    private readonly PythonManager _python;
    private readonly SemaphoreSlim _cpuQuantLock;
    private readonly int _maxConcurrentQuantizations;

    private static readonly SemaphoreSlim BaseModelLock = new(1, 1);

    public QuantizationService(BenchmarkService benchmarker)
    {
        _benchmarker = benchmarker;
        _python = _benchmarker._pyManager;

        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new Exception("Cache.ModelMagicQuantDirectory not set. Evolution must set this before quantization starts.");

        _ggufDir = Path.Combine(Cache.ModelMagicQuantDirectory, "GGUF");
        _benchDir = Path.Combine(Cache.ModelMagicQuantDirectory, "Benchmarks");

        Directory.CreateDirectory(_ggufDir);
        Directory.CreateDirectory(_benchDir);

        int threadCount = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        _maxConcurrentQuantizations = Math.Max(1, threadCount / 8);
        _cpuQuantLock = new SemaphoreSlim(_maxConcurrentQuantizations, _maxConcurrentQuantizations);
    }

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<HybridQuant> quants,
        CancellationToken ct = default)
    {
        if (quants == null)
            throw new ArgumentNullException(nameof(quants));

        int completed = 0;
        int skipped = 0;
        int failed = 0;

        // Warm the base model once so workers don't all race into conversion.
        await EnsureBaseModelAsync(false);

        await Parallel.ForEachAsync(
            quants,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = _maxConcurrentQuantizations,
                CancellationToken = ct
            },
            async (quant, token) =>
            {
                try
                {
                    var state = await ProcessHybridQuantAsync(quant, token);

                    switch (state)
                    {
                        case SampleProcessState.Completed:
                            Interlocked.Increment(ref completed);
                            break;
                        case SampleProcessState.Skipped:
                            Interlocked.Increment(ref skipped);
                            break;
                        default:
                            Interlocked.Increment(ref failed);
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref failed);
                    AnsiConsole.MarkupLine($"[red]Sample failed:[/] {Markup.Escape(GenerateHybridName(quant))}");
                    AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");
                }
            });

        return new SampleProcessingSummary
        {
            Requested = quants.Count,
            Completed = completed,
            Skipped = skipped,
            Failed = failed
        };
    }

    public async Task<SampleProcessState> ProcessHybridQuantAsync(
    HybridQuant quant,
    CancellationToken ct = default)
{
    string modelName = GenerateHybridName(quant);
    string quantPath = Path.Combine(_ggufDir, $"{modelName}.gguf");
    string modelBenchDir = Path.Combine(_benchDir, modelName);
    string baseLogitsDir = GetBaseLogitsDirectory();

    // 1. Fast path: if the benchmark artifacts on disk are already valid, reuse them
    // and sync SQLite without rebuilding the sample GGUF.
    if (await _benchmarker.TryReuseExistingBenchmarksAsync(
            quantConfig: quant,
            modelPath: quantPath,
            benchDir: modelBenchDir,
            klLogitsDir: baseLogitsDir,
            domainsOverride: new[] { "general" }))
    {
        AnsiConsole.MarkupLine($"[grey]Reused existing benchmark artifacts:[/] {Markup.Escape(modelName)}");

        if (!IsProtectedModel(modelName))
            await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

        return SampleProcessState.Skipped;
    }

    // 2. DB truth still matters too
    if (await BenchmarkExistsAsync(quant, ct))
    {
        AnsiConsole.MarkupLine($"[grey]Skipping already completed sample:[/] {Markup.Escape(modelName)}");

        if (!IsProtectedModel(modelName))
            await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

        return SampleProcessState.Skipped;
    }

    try
    {
        string basePath = await EnsureBaseModelAsync();

        await _cpuQuantLock.WaitAsync(ct);
        try
        {
            if (!File.Exists(quantPath))
            {
                AnsiConsole.MarkupLine($"[cyan]Building sample:[/] {Markup.Escape(modelName)}");
                await RunLlamaQuantizeAsync(basePath, quantPath, quant);
            }
        }
        finally
        {
            _cpuQuantLock.Release();
        }

        // Re-check after build in case another worker finished the DB sync while we were quantizing
        if (await BenchmarkExistsAsync(quant, ct))
        {
            if (!IsProtectedModel(modelName))
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

            return SampleProcessState.Skipped;
        }

        AnsiConsole.MarkupLine($"[yellow]Benchmarking:[/] {Markup.Escape(modelName)}");

        await _benchmarker.RunAllBenchmarksAsync(
            quantConfig: quant,
            modelPath: quantPath,
            benchDir: modelBenchDir,
            klLogitsDir: baseLogitsDir,
            saveLogits: false,
            domainsOverride: new[] { "general" });

        return SampleProcessState.Completed;
    }
    finally
    {
        if (!IsProtectedModel(modelName))
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);
        }
    }
}

    private string GetBaseLogitsDirectory()
    {
        string typeStr = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        return Path.Combine(_benchDir, typeStr, "logits");
    }

    private async Task<bool> BenchmarkExistsAsync(HybridQuant quant, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var lookup = BuildTensorLookup(quant);

        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return false;

        var comboId = await db.TensorCombos
            .AsNoTracking()
            .Where(x =>
                x.BaseQuant == lookup.BaseQuant &&
                x.Embeddings == lookup.Embeddings &&
                x.LmHead == lookup.LmHead &&
                x.AttnQ == lookup.AttnQ &&
                x.AttnKV == lookup.AttnKV &&
                x.AttnOutput == lookup.AttnOutput &&
                x.FfnUpGate == lookup.FfnUpGate &&
                x.FfnDown == lookup.FfnDown &&
                x.MoeExperts == lookup.MoeExperts &&
                x.MoeRouter == lookup.MoeRouter)
            .Select(x => (uint?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (!comboId.HasValue)
            return false;

        return await db.AiBenchmarks
            .AsNoTracking()
            .AnyAsync(x => x.AiModelHashId == model.Id && x.TensorComboId == comboId.Value, ct);
    }

    private static (
        byte BaseQuant,
        byte Embeddings,
        byte LmHead,
        byte AttnQ,
        byte AttnKV,
        byte AttnOutput,
        byte FfnUpGate,
        byte FfnDown,
        byte MoeExperts,
        byte MoeRouter) BuildTensorLookup(HybridQuant quant)
    {
        byte embeddings = 0;
        byte lmHead = 0;
        byte attnQ = 0;
        byte attnKV = 0;
        byte attnOutput = 0;
        byte ffnUpGate = 0;
        byte ffnDown = 0;
        byte moeExperts = 0;
        byte moeRouter = 0;

        if (quant.Tensors != null)
        {
            foreach (var tensor in quant.Tensors)
            {
                if (tensor?.TGroup == null)
                    continue;

                if (tensor.TGroup.UniqueId == TReg.Embeddings.UniqueId) embeddings = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.LmHead.UniqueId) lmHead = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnQ.UniqueId) attnQ = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnKV.UniqueId) attnKV = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.AttnOutput.UniqueId) attnOutput = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.FfnUpGate.UniqueId) ffnUpGate = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.FfnDown.UniqueId) ffnDown = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.MoeExperts.UniqueId) moeExperts = tensor.TensorType.UniqueId;
                else if (tensor.TGroup.UniqueId == TReg.MoeRouter.UniqueId) moeRouter = tensor.TensorType.UniqueId;
            }
        }

        return (
            quant.BaseQuant.UniqueId,
            embeddings,
            lmHead,
            attnQ,
            attnKV,
            attnOutput,
            ffnUpGate,
            ffnDown,
            moeExperts,
            moeRouter
        );
    }

    private bool IsProtectedModel(string name)
    {
        return name.EndsWith("BF16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F32", StringComparison.OrdinalIgnoreCase);
    }

    public async Task<string> EnsureBaseModelAsync(bool deleteProcess = false)
    {
        await BaseModelLock.WaitAsync();
        try
        {
            string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
            var torchType = Cache.TorchType ?? Cache.MainTorchType.BF16;
            string typeStr = torchType.ToString();

            string fileName = $"{modelName}-{typeStr}.gguf";
            string outputPath = Path.Combine(_ggufDir, fileName);
            string successFile = Path.Combine(_ggufDir, $"{fileName}.success.json");

            if (deleteProcess)
            {
                if (!Directory.Exists(_ggufDir))
                    Directory.CreateDirectory(_ggufDir);

                var normalizedFileName = Path.GetFileName(fileName);
                var successFileName = normalizedFileName + ".success.json";
                var successFilePath = Path.Combine(_ggufDir, successFileName);
                bool isImmune = File.Exists(successFilePath);

                foreach (var filePath in Directory.EnumerateFiles(_ggufDir, "*.gguf", SearchOption.TopDirectoryOnly))
                {
                    var currentFileName = Path.GetFileName(filePath);

                    if (isImmune &&
                        string.Equals(currentFileName, normalizedFileName, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    await HardDeleteHelper.DeleteFileIfExistsAsync(filePath);
                }
            }

            if (!File.Exists(outputPath) || !File.Exists(successFile))
            {
                AnsiConsole.MarkupLine($"[bold cyan]Converting to {typeStr}...[/]");

                await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                string convertScript = Cache.ConvertScript
                                       ?? throw new Exception("ConvertScript path missing in Cache");

                string outTypeArg = typeStr.ToLowerInvariant();

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

                await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
            }

            string benchPath = Path.Combine(_benchDir, typeStr);
            string logitsDir = Path.Combine(benchPath, "logits");

            AnsiConsole.MarkupLine($"[bold yellow]Benchmarking Base {typeStr} (Saving Logits)...[/]");

            var baseModelQuant = new HybridQuant
            {
                BaseQuant = BaselineQuants.GetBF16Quant(),
                Tensors = new List<HybridTensor>()
            };

            await _benchmarker.RunAllBenchmarksAsync(
                quantConfig: baseModelQuant,
                modelPath: outputPath,
                benchDir: benchPath,
                klLogitsDir: logitsDir,
                saveLogits: true,
                domainsOverride: new[] { "general", "code", "math" }
            );

            return outputPath;
        }
        finally
        {
            BaseModelLock.Release();
        }
    }

    public async Task RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridQuant quant)
    {
        var args = new List<string>(capacity: 64);

        if (quant.Tensors is { Count: > 0 })
        {
            foreach (var hybrid in quant.Tensors)
            {
                if (hybrid?.TGroup == null)
                    continue;

                // NULL is an internal sentinel only.
                // It means "do not emit an override for this tensor group".
                if (hybrid.TensorType.UniqueId == TensorWeightScheme.NULL.UniqueId)
                    continue;

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

        string bin = Path.Combine(
            Cache.LlamaBin!,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama-quantize.exe" : "llama-quantize");

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

        p.OutputDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                AnsiConsole.WriteLine(e.Data);
        };

        p.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                AnsiConsole.WriteLine(e.Data);
        };

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

        if (s.UniqueId == TensorWeightScheme.BF16_F16.UniqueId && s.Names.Length >= 2)
        {
            if (Cache.TorchType == Cache.MainTorchType.F16)
                return "F16";

            if (Cache.TorchType == Cache.MainTorchType.F32)
                return "F32";

            return "BF16";
        }

        return s.Names[0];
    }

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
        return "EHQKOUDXR".IndexOf(c);
    }

    private string SimplifyQuant(string quant)
    {
        return quant.Replace("_", "")
            .Replace("BF16", "B16")
            .Replace("F16", "F16")
            .Replace("F32", "F32");
    }
}