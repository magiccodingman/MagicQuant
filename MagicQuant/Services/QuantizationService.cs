using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public enum SampleProcessState
{
    Completed = 1,
    Skipped = 2,
    Failed = 3
}

public sealed class SampleProcessingRecord
{
    public RequiredSamplePlan Plan { get; set; } = default!;
    public SampleProcessState State { get; set; }
    public string ModelName { get; set; } = string.Empty;
    public uint? TensorComboId { get; set; }
    public uint? BenchmarkId { get; set; }
    public string? Error { get; set; }
}

public sealed class SampleProcessingSummary
{
    public int Requested { get; set; }
    public int Completed { get; set; }
    public int Skipped { get; set; }
    public int Failed { get; set; }
    public List<SampleProcessingRecord> Records { get; set; } = new();
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
        _benchmarker = benchmarker ?? throw new ArgumentNullException(nameof(benchmarker));
        _python = _benchmarker._pyManager;

        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new Exception(
                "Cache.ModelMagicQuantDirectory not set. Evolution must set this before quantization starts.");

        if (string.IsNullOrWhiteSpace(Cache.ModelDirectory))
            throw new Exception("Cache.ModelDirectory not set. Evolution must set this before quantization starts.");

        if (string.IsNullOrWhiteSpace(Cache.LlamaBin))
            throw new Exception("Cache.LlamaBin not set. Initialization must complete before quantization starts.");

        _ggufDir = Path.Combine(Cache.ModelMagicQuantDirectory, "GGUF");
        _benchDir = Path.Combine(Cache.ModelMagicQuantDirectory, "Benchmarks");

        Directory.CreateDirectory(_ggufDir);
        Directory.CreateDirectory(_benchDir);

        int threadCount = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        _maxConcurrentQuantizations = Math.Max(1, threadCount / 8);
        _cpuQuantLock = new SemaphoreSlim(_maxConcurrentQuantizations, _maxConcurrentQuantizations);
    }

    // ----------------------------------------------------------------
    // Batch processing
    // ----------------------------------------------------------------

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<HybridQuant> quants,
        CancellationToken ct = default)
    {
        if (quants == null)
            throw new ArgumentNullException(nameof(quants));

        var shimmedPlans = quants
            .Select((quant, index) => new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.GroupIsolationContinuation,
                Key = $"legacy:{index}",
                Description = "Legacy batch item",
                Quant = quant
            })
            .ToList();

        return await ProcessHybridBatchAsync(shimmedPlans, ct);
    }

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<RequiredSamplePlan> plans,
        CancellationToken ct = default)
    {
        if (plans == null)
            throw new ArgumentNullException(nameof(plans));

        int completed = 0;
        int skipped = 0;
        int failed = 0;
        var records = new ConcurrentBag<SampleProcessingRecord>();

        await EnsureBaseModelFileAsync(false);

        var learnableBaselinePlans = plans
            .Where(p => IsLearnableBaselineRun(p.Quant))
            .OrderBy(p => p.Quant.BaseQuant.UniqueId)
            .ToList();

        foreach (var baselinePlan in learnableBaselinePlans)
        {
            var baselineRecord = new SampleProcessingRecord
            {
                Plan = baselinePlan,
                ModelName = GenerateHybridName(baselinePlan.Quant)
            };

            try
            {
                var state = await ProcessHybridQuantAsync(baselinePlan.Quant, ct);
                baselineRecord.State = state;

                var identity = await ResolveBenchmarkIdentityAsync(baselinePlan.Quant, ct);
                baselineRecord.TensorComboId = identity.TensorComboId;
                baselineRecord.BenchmarkId = identity.BenchmarkId;

                switch (state)
                {
                    case SampleProcessState.Completed:
                        completed++;
                        break;
                    case SampleProcessState.Skipped:
                        skipped++;
                        break;
                    default:
                        failed++;
                        break;
                }
            }
            catch (Exception ex)
            {
                baselineRecord.State = SampleProcessState.Failed;
                baselineRecord.Error = ex.Message;
                failed++;

                AnsiConsole.MarkupLine($"[red]Baseline sample failed:[/] {Markup.Escape(baselineRecord.ModelName)}");
                AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");
            }
            finally
            {
                records.Add(baselineRecord);
            }
        }

        var remainingPlans = plans.Except(learnableBaselinePlans).ToList();

        await Parallel.ForEachAsync(
            remainingPlans,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = _maxConcurrentQuantizations,
                CancellationToken = ct
            },
            async (plan, token) =>
            {
                var record = new SampleProcessingRecord
                {
                    Plan = plan,
                    ModelName = GenerateHybridName(plan.Quant)
                };

                try
                {
                    var state = await ProcessHybridQuantAsync(plan.Quant, token);
                    record.State = state;

                    var identity = await ResolveBenchmarkIdentityAsync(plan.Quant, token);
                    record.TensorComboId = identity.TensorComboId;
                    record.BenchmarkId = identity.BenchmarkId;

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
                    record.State = SampleProcessState.Failed;
                    record.Error = ex.Message;
                    Interlocked.Increment(ref failed);

                    AnsiConsole.MarkupLine($"[red]Sample failed:[/] {Markup.Escape(record.ModelName)}");
                    AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");
                }
                finally
                {
                    records.Add(record);
                }
            });

        return new SampleProcessingSummary
        {
            Requested = plans.Count,
            Completed = completed,
            Skipped = skipped,
            Failed = failed,
            Records = records.OrderBy(x => x.Plan.Key).ToList()
        };
    }

    private async Task<(uint? TensorComboId, uint? BenchmarkId)> ResolveBenchmarkIdentityAsync(
        HybridQuant quant,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var lookup = BuildTensorLookup(quant);

        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return (null, null);

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
            .Select(x => x.Id)
            .FirstOrDefaultAsync(ct);

        if (comboId == 0)
            return (null, null);

        var benchmarkId = await db.AiBenchmarks
            .AsNoTracking()
            .Where(x => x.AiModelHashId == model.Id && x.TensorComboId == comboId)
            .Select(x => x.Id)
            .FirstOrDefaultAsync(ct);

        return (comboId, benchmarkId == 0 ? null : benchmarkId);
    }

    public async Task<SampleProcessState> ProcessHybridQuantAsync(
        HybridQuant quant,
        CancellationToken ct = default)
    {
        string modelName = GenerateHybridName(quant);
        string quantPath = Path.Combine(_ggufDir, $"{modelName}.gguf");
        string modelBenchDir = Path.Combine(_benchDir, modelName);
        string baseLogitsDir = GetBaseLogitsDirectory();

        DateTime startedUtc = DateTime.UtcNow;
        var stopwatch = Stopwatch.StartNew();
        var forceBaselineRelearn = Cache.ForceRelearnBaselineTensorMappings && IsLearnableBaselineRun(quant);

        // 1. Fast path: valid artifacts already exist on disk and can be synced/reused
        if (!forceBaselineRelearn && await _benchmarker.TryReuseExistingBenchmarksAsync(
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
        if (!forceBaselineRelearn && await BenchmarkExistsAsync(quant, ct))
        {
            AnsiConsole.MarkupLine($"[grey]Skipping already completed sample:[/] {Markup.Escape(modelName)}");

            if (!IsProtectedModel(modelName))
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

            return SampleProcessState.Skipped;
        }

        try
        {
            string basePath = await EnsureBaseModelFileAsync();
            QuantizationExecutionReport? quantizationReport = null;

            await _cpuQuantLock.WaitAsync(ct);
            try
            {
                if (!File.Exists(quantPath) || forceBaselineRelearn)
                {
                    AnsiConsole.MarkupLine($"[cyan]Building sample:[/] {Markup.Escape(modelName)}");
                    quantizationReport = await RunLlamaQuantizeAsync(basePath, quantPath, quant);
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            // Re-check after build in case another worker finished the DB sync while we were quantizing
            if (!forceBaselineRelearn && await BenchmarkExistsAsync(quant, ct))
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

            stopwatch.Stop();

            await PersistQuantizationRunAsync(
                quant: quant,
                startedUtc: startedUtc,
                completedUtc: DateTime.UtcNow,
                succeeded: true,
                outputModelPath: quantPath,
                error: null,
                ct: ct);

            if (IsLearnableBaselineRun(quant))
            {
                await LearnAndPersistBaselineTensorMapAsync(quant, quantPath, quantizationReport, ct);
            }

            return SampleProcessState.Completed;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();

            try
            {
                await PersistQuantizationRunAsync(
                    quant: quant,
                    startedUtc: startedUtc,
                    completedUtc: DateTime.UtcNow,
                    succeeded: false,
                    outputModelPath: quantPath,
                    error: ex.ToString(),
                    ct: ct);
            }
            catch
            {
                // Never hide the original exception because timing persistence failed.
            }

            throw;
        }
        finally
        {
            if (!IsProtectedModel(modelName))
            {
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);
            }
        }
    }

    // ----------------------------------------------------------------
    // Benchmark/logit helpers
    // ----------------------------------------------------------------

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

        var bench = await db.AiBenchmarks
            .AsNoTracking()
            .Where(x => x.AiModelHashId == model.Id)
            .Join(
                db.TensorCombos.AsNoTracking(),
                benchmark => benchmark.TensorComboId,
                combo => combo.Id,
                (benchmark, combo) => new { benchmark, combo })
            .Where(x =>
                x.combo.BaseQuant == lookup.BaseQuant &&
                x.combo.Embeddings == lookup.Embeddings &&
                x.combo.LmHead == lookup.LmHead &&
                x.combo.AttnQ == lookup.AttnQ &&
                x.combo.AttnKV == lookup.AttnKV &&
                x.combo.AttnOutput == lookup.AttnOutput &&
                x.combo.FfnUpGate == lookup.FfnUpGate &&
                x.combo.FfnDown == lookup.FfnDown &&
                x.combo.MoeExperts == lookup.MoeExperts &&
                x.combo.MoeRouter == lookup.MoeRouter)
            .Select(x => x.benchmark.Id)
            .FirstOrDefaultAsync(ct);

        if (bench == 0)
            return false;

        // Require at least one category row too, so a half-baked parent row doesn't count as complete.
        bool hasCategory = await db.Set<CategoryBenchmark>()
            .AsNoTracking()
            .AnyAsync(x => x.AiBenchmarkId == bench, ct);

        return hasCategory;
    }

    private static TensorConfig BuildTensorLookup(HybridQuant quant)
    {
        return (TensorConfig)quant;
    }

    private async Task PersistQuantizationRunAsync(
        HybridQuant quant,
        DateTime startedUtc,
        DateTime completedUtc,
        bool succeeded,
        string? outputModelPath,
        string? error,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var lookup = BuildTensorLookup(quant);

        await using var db = new MagicQuantContext();

        var aiModelHash = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash
            {
                UniqueHash = Cache.CurrentModelId
            };

            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync(ct);
        }

        var tensorCombo = await db.TensorCombos.FirstOrDefaultAsync(x =>
            x.BaseQuant == lookup.BaseQuant &&
            x.Embeddings == lookup.Embeddings &&
            x.LmHead == lookup.LmHead &&
            x.AttnQ == lookup.AttnQ &&
            x.AttnKV == lookup.AttnKV &&
            x.AttnOutput == lookup.AttnOutput &&
            x.FfnUpGate == lookup.FfnUpGate &&
            x.FfnDown == lookup.FfnDown &&
            x.MoeExperts == lookup.MoeExperts &&
            x.MoeRouter == lookup.MoeRouter, ct);

        if (tensorCombo == null)
        {
            tensorCombo = new TensorCombo(lookup);
            db.TensorCombos.Add(tensorCombo);
            await db.SaveChangesAsync(ct);
        }

        uint? aiBenchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == aiModelHash.Id && x.TensorComboId == tensorCombo.Id)
            .Select(x => (uint?)x.Id)
            .FirstOrDefaultAsync(ct);

        var row = new QuantizationRun
        {
            Id = Guid.NewGuid(),
            AiModelHashId = aiModelHash.Id,
            TensorComboId = tensorCombo.Id,
            AiBenchmarkId = aiBenchmarkId,
            StartedUtc = startedUtc,
            CompletedUtc = completedUtc,
            DurationMs = Math.Max(0L, (long)(completedUtc - startedUtc).TotalMilliseconds),
            Succeeded = succeeded,
            Error = error,
            OutputModelPath = outputModelPath
        };

        db.QuantizationRuns.Add(row);
        await db.SaveChangesAsync(ct);
    }

    private bool IsProtectedModel(string name)
    {
        return name.EndsWith("BF16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F16", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("F32", StringComparison.OrdinalIgnoreCase) ||
               name.EndsWith("Q8_0", StringComparison.OrdinalIgnoreCase);
    }

    // ----------------------------------------------------------------
    // Base/native model helpers
    // ----------------------------------------------------------------

    public async Task<string> EnsureBaseModelAsync(bool deleteProcess = false)
    {
        string outputPath = await EnsureBaseModelFileAsync(deleteProcess);

        string typeStr = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
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

    public async Task<string> EnsureBaseModelFileAsync(bool deleteProcess = false)
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
            string convertLogPath = outputPath + ".convert.log";

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
                    WorkingDirectory = Cache.LlamaRoot
                };

                var result = await RunLoggedProcessAsync(psi, convertLogPath);

                if (result.ExitCode != 0)
                {
                    await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                    throw new Exception(
                        $"{typeStr} conversion failed. ExitCode={result.ExitCode}. See '{convertLogPath}'.");
                }

                if (!File.Exists(outputPath) || new FileInfo(outputPath).Length == 0)
                {
                    await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

                    throw new InvalidOperationException(
                        $"Conversion exited successfully but produced no valid GGUF output: {outputPath}");
                }

                await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
            }

            return outputPath;
        }
        finally
        {
            BaseModelLock.Release();
        }
    }

    public async Task<string> EnsurePureQ8ModelAsync()
    {
        string basePath = await EnsureBaseModelFileAsync();

        var pureQ8 = new HybridQuant
        {
            BaseQuant = BaselineQuants.Q8_0,
            Tensors = new List<HybridTensor>()
        };

        string modelName = GenerateHybridName(pureQ8);
        string q8Path = Path.Combine(_ggufDir, $"{modelName}.gguf");
        string successFile = Path.Combine(_ggufDir, $"{Path.GetFileName(q8Path)}.success.json");

        if (!File.Exists(q8Path) || !File.Exists(successFile))
        {
            await _cpuQuantLock.WaitAsync();
            try
            {
                if (!File.Exists(q8Path))
                {
                    AnsiConsole.MarkupLine($"[cyan]Building pure Q8 baseline:[/] {Markup.Escape(modelName)}");
                    await RunLlamaQuantizeAsync(basePath, q8Path, pureQ8);
                    AnsiConsole.MarkupLine(
                        $"[green]Pure Q8 baseline quantization finished:[/] {Markup.Escape(q8Path)}");
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            await File.WriteAllTextAsync(successFile, "{\"status\":\"success\"}");
        }
        else
        {
            AnsiConsole.MarkupLine($"[grey]Pure Q8 baseline already exists:[/] {Markup.Escape(q8Path)}");
        }

        return q8Path;
    }

    // ----------------------------------------------------------------
    // Quantization
    // ----------------------------------------------------------------

    private async Task<QuantizationExecutionReport> RunLlamaQuantizeAsync(string inputFile, string outputFile, HybridQuant quant)
    {
        if (string.IsNullOrWhiteSpace(inputFile) || !File.Exists(inputFile))
            throw new FileNotFoundException($"Input GGUF not found: {inputFile}");

        Directory.CreateDirectory(Path.GetDirectoryName(outputFile)!);

        var requestedOverrides = BuildRequestedTensorOverrides(quant);

        // Keep this resolution step:
        // it is not output validation; it is how logical group rules become real tensor names.
        var concreteOverrides = await ResolveConcreteTensorOverridesAsync(
            inputGgufPath: inputFile,
            outputFilePath: outputFile,
            requestedOverrides: requestedOverrides);

        if (requestedOverrides.Count > 0 && concreteOverrides.Count == 0)
        {
            throw new InvalidOperationException(
                $"No concrete tensors were resolved for requested overrides when quantizing '{outputFile}'. " +
                "This means the requested tensor selectors did not match the input GGUF.");
        }

        var args = new List<string>(capacity: 256);

        foreach (var overrideItem in concreteOverrides)
        {
            args.Add($"--tensor-type \"{overrideItem.TensorName}={overrideItem.SchemeName}\"");
        }

        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(ResolveQuantizeBaseArgument(quant, concreteOverrides));
        args.Add("8");

        string arguments = string.Join(" ", args);

        string bin = Path.Combine(
            Cache.LlamaBin!,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama-quantize.exe" : "llama-quantize");

        string quantizeLogPath = outputFile + ".quantize.log";

        var psi = new ProcessStartInfo
        {
            FileName = bin,
            Arguments = arguments
        };

        var result = await RunLoggedProcessAsync(psi, quantizeLogPath);

        if (result.ExitCode != 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);

            throw new InvalidOperationException(
                $"Quantization failed for '{outputFile}'. ExitCode={result.ExitCode}. See '{quantizeLogPath}'.");
        }

        if (!File.Exists(outputFile) || new FileInfo(outputFile).Length == 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);

            throw new InvalidOperationException(
                $"Quantization process exited successfully but produced no valid GGUF output: {outputFile}");
        }

        AnsiConsole.MarkupLine($"[green]Quantized model ready:[/] {Markup.Escape(outputFile)}");

        return new QuantizationExecutionReport
        {
            LogPath = quantizeLogPath,
            ResolvedOverrides = concreteOverrides
        };
    }

    private static string ResolveQuantizeBaseArgument(
        HybridQuant quant,
        List<ConcreteTensorOverride> concreteOverrides)
    {
        if (quant.BaseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId &&
            concreteOverrides.Count > 0)
        {
            throw new InvalidOperationException(
                "Native BF16/F16/F32 + tensor overrides is disabled. " +
                "In this build of llama-quantize it produced no-op outputs for isolation tests. " +
                "Use a real carrier baseline (Q8_0 recommended), force all known groups to BF16/F16, " +
                "and quantize only the target group.");
        }

        return ResolveBaseName(quant.BaseQuant);
    }

    public async Task ClearLearnedBaselineTensorMappingsAsync(CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        int removed = await db.LearnedBaselineTensorQuants.ExecuteDeleteAsync(ct);
        AnsiConsole.MarkupLine($"[yellow]Relearn requested:[/] removed [red]{removed:N0}[/] learned baseline tensor mapping rows.");
    }

    private static bool IsLearnableBaselineRun(HybridQuant quant)
    {
        return quant.Tensors.Count == 0 &&
               quant.BaseQuant.UniqueId != BaselineQuants.NativeSourceUniqueId &&
               quant.BaseQuant.DefaultTensorScheme != null;
    }

    private async Task LearnAndPersistBaselineTensorMapAsync(
        HybridQuant quant,
        string quantizedModelPath,
        QuantizationExecutionReport? report,
        CancellationToken ct)
    {
        if (!IsLearnableBaselineRun(quant))
            return;

        var tensorScheme = quant.BaseQuant.DefaultTensorScheme!;
        var parsed = ParseQuantizeLogForTensorTypes(report?.LogPath ?? (quantizedModelPath + ".quantize.log"));
        if (parsed.Count == 0)
        {
            AnsiConsole.MarkupLine(
                $"[red]WARNING:[/] learned mapping parse returned no tensors for baseline [yellow]{quant.BaseQuant.Names[0]}[/].");
            return;
        }

        var grouped = AssignGroups(parsed.Keys);
        var ambiguous = grouped.Where(x => x.Value.MatchedGroups.Count > 1).ToList();
        if (ambiguous.Count > 0)
        {
            AnsiConsole.MarkupLine(
                $"[red]WARNING:[/] {ambiguous.Count} tensor(s) matched multiple groups while learning baseline {quant.BaseQuant.Names[0]}.");
            AnsiConsole.MarkupLine($"[grey]Example: {Markup.Escape(ambiguous[0].Key)} => {string.Join(", ", ambiguous[0].Value.MatchedGroups)}[/]");
        }

        var unresolved = grouped.Where(x => x.Value.PrimaryGroup == null).Select(x => x.Key).ToList();
        if (unresolved.Count > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]WARNING:[/] {unresolved.Count} tensor(s) had no tensor-group match while learning baseline {quant.BaseQuant.Names[0]}.");
        }

        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (model == null)
            throw new InvalidOperationException("Unable to persist learned mappings because AiModelHash row was not found.");

        var combo = await db.TensorCombos
            .AsNoTracking()
            .FirstAsync(x => x.BaseQuant == quant.BaseQuant.UniqueId &&
                             x.Embeddings == 0 && x.LmHead == 0 && x.AttnQ == 0 && x.AttnKV == 0 &&
                             x.AttnOutput == 0 && x.FfnUpGate == 0 && x.FfnDown == 0 && x.MoeExperts == 0 && x.MoeRouter == 0, ct);

        var benchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == model.Id && x.TensorComboId == combo.Id)
            .OrderByDescending(x => x.Id)
            .Select(x => (uint?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (!benchmarkId.HasValue)
            throw new InvalidOperationException($"Unable to persist learned mappings because no AiBenchmark exists for baseline '{quant.BaseQuant.Names[0]}'.");

        await db.LearnedBaselineTensorQuants
            .Where(x => x.AiModelHashId == model.Id &&
                        x.BaselineQuantId == quant.BaseQuant.UniqueId &&
                        x.TensorWeightSchemeId == tensorScheme.UniqueId)
            .ExecuteDeleteAsync(ct);

        var rows = new List<LearnedBaselineTensorQuant>(parsed.Count);
        foreach (var kv in parsed.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            var match = grouped[kv.Key];
            if (match.PrimaryGroup == null)
                continue;

            rows.Add(new LearnedBaselineTensorQuant
            {
                AiBenchmarkId = benchmarkId.Value,
                AiModelHashId = model.Id,
                BaselineQuantId = quant.BaseQuant.UniqueId,
                TensorWeightSchemeId = tensorScheme.UniqueId,
                TensorGroupId = match.PrimaryGroup.UniqueId,
                TensorName = kv.Key,
                FinalQuantType = kv.Value
            });
        }

        if (rows.Count == 0)
            throw new InvalidOperationException($"Learning baseline '{quant.BaseQuant.Names[0]}' produced no persistable rows.");

        db.LearnedBaselineTensorQuants.AddRange(rows);
        await db.SaveChangesAsync(ct);

        AnsiConsole.MarkupLine(
            $"[green]Learned baseline tensor mapping persisted:[/] [cyan]{rows.Count:N0}[/] row(s) for [yellow]{quant.BaseQuant.Names[0]}[/].");
    }

    private Dictionary<string, string> ParseQuantizeLogForTensorTypes(string logPath)
    {
        if (!File.Exists(logPath))
        {
            AnsiConsole.MarkupLine($"[red]WARNING:[/] quantization log does not exist, cannot learn tensor mapping: {Markup.Escape(logPath)}");
            return new Dictionary<string, string>(StringComparer.Ordinal);
        }

        var byTensor = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var raw in File.ReadLines(logPath))
        {
            var match = TensorLogLineRegex.Match(raw);
            if (!match.Success)
                continue;

            string tensorName = match.Groups["tensor"].Value.Trim();
            string declaredType = NormalizeQuantName(match.Groups["type"].Value);

            string final = declaredType;
            var convert = match.Groups["convert"];
            if (convert.Success && !string.IsNullOrWhiteSpace(convert.Value))
                final = NormalizeQuantName(convert.Value);

            byTensor[tensorName] = final;
        }

        return byTensor;
    }

    private Dictionary<string, TensorGroupingResult> AssignGroups(IEnumerable<string> tensorNames)
    {
        var dict = new Dictionary<string, TensorGroupingResult>(StringComparer.Ordinal);

        foreach (var tensorName in tensorNames)
        {
            var matched = new List<TensorGroup>();

            foreach (var group in TReg.All)
            {
                if (group.Tensors.Any(pattern => Regex.IsMatch(tensorName, $"^{pattern}$")))
                    matched.Add(group);
            }

            dict[tensorName] = new TensorGroupingResult
            {
                MatchedGroups = matched.Select(x => x.Name).ToList(),
                PrimaryGroup = matched.FirstOrDefault()
            };
        }

        return dict;
    }

    private static TensorWeightScheme? TryResolveBaseTensorScheme(BaselineQuants baseQuant)
    {
        if (baseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId)
            return TensorWeightScheme.BF16_F16;

        if (baseQuant.Names.IsDefaultOrEmpty)
            return null;

        return TensorWeightScheme.All.FirstOrDefault(s =>
            !s.Names.IsDefaultOrEmpty &&
            s.Names.Any(sn => baseQuant.Names.Contains(sn, StringComparer.OrdinalIgnoreCase)));
    }

    private List<RequestedTensorOverride> BuildRequestedTensorOverrides(HybridQuant quant)
    {
        var result = new List<RequestedTensorOverride>();

        if (quant.Tensors == null || quant.Tensors.Count == 0)
            return result;

        var baseScheme = TryResolveBaseTensorScheme(quant.BaseQuant);

        foreach (var hybrid in quant.Tensors)
        {
            if (hybrid?.TGroup == null)
                continue;

            if (hybrid.TensorType.UniqueId == TensorWeightScheme.NULL.UniqueId)
                continue;

            // Do not emit a redundant override if this tensor type is already the same
            // as the blanket base quant.
            if (baseScheme != null && hybrid.TensorType.UniqueId == baseScheme.UniqueId)
                continue;

            var learned = TryLoadLearnedTensorMapping(hybrid.TensorType, hybrid.TGroup);
            if (learned.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Missing required learned baseline mapping for group '{hybrid.TGroup.Name}' + scheme '{hybrid.TensorType.Names[0]}'. " +
                    "Run with --relearn-baseline-mappings to regenerate.");
            }

            foreach (var kv in learned)
            {
                result.Add(new RequestedTensorOverride
                {
                    GroupName = hybrid.TGroup.Name,
                    TensorName = kv.Key,
                    SchemeName = kv.Value
                });
            }
        }

        return result;
    }

    private Dictionary<string, string> TryLoadLearnedTensorMapping(TensorWeightScheme sourceScheme, TensorGroup targetGroup)
    {
        using var db = new MagicQuantContext();

        var model = db.AiModelHashes
            .AsNoTracking()
            .FirstOrDefault(x => x.UniqueHash == Cache.CurrentModelId);

        if (model == null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var baseline = BaselineQuants.All.FirstOrDefault(x => x.DefaultTensorScheme?.UniqueId == sourceScheme.UniqueId);
        if (baseline == null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var rows = db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == model.Id)
            .Where(x => x.BaselineQuantId == baseline.UniqueId)
            .Where(x => x.TensorWeightSchemeId == sourceScheme.UniqueId)
            .Where(x => x.TensorGroupId == targetGroup.UniqueId)
            .OrderBy(x => x.TensorName)
            .ToList();

        if (rows.Count == 0)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        return rows.ToDictionary(x => x.TensorName, x => x.FinalQuantType, StringComparer.Ordinal);
    }

    private async Task<List<ConcreteTensorOverride>> ResolveConcreteTensorOverridesAsync(
        string inputGgufPath,
        string outputFilePath,
        List<RequestedTensorOverride> requestedOverrides)
    {
        if (requestedOverrides.Count == 0)
            return new List<ConcreteTensorOverride>();
        var allTensorNames = await ReadTensorNamesFromGgufAsync(inputGgufPath, outputFilePath);
        var nameSet = allTensorNames.ToHashSet(StringComparer.Ordinal);

        var missing = requestedOverrides
            .Where(x => !nameSet.Contains(x.TensorName))
            .ToList();

        if (missing.Count > 0)
        {
            throw new InvalidOperationException(
                $"Required learned tensor mappings were missing in source GGUF ({missing.Count} tensors). " +
                $"Examples: {string.Join(", ", missing.Take(10).Select(x => x.TensorName))}");
        }

        var duplicates = requestedOverrides
            .GroupBy(x => x.TensorName, StringComparer.Ordinal)
            .Where(g => g.Select(x => x.SchemeName).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicates.Count > 0)
        {
            throw new InvalidOperationException(
                $"Conflicting learned mappings tried to assign multiple quant types to the same tensor: " +
                $"{string.Join(", ", duplicates.Take(20))}");
        }

        return requestedOverrides
            .GroupBy(x => x.TensorName, StringComparer.Ordinal)
            .Select(g => g.First())
            .Select(x => new ConcreteTensorOverride
            {
                GroupName = x.GroupName,
                SchemeName = x.SchemeName,
                TensorName = x.TensorName
            })
            .ToList();
    }

    private async Task<List<string>> ReadTensorNamesFromGgufAsync(string ggufPath, string outputFilePath)
    {
        string workingDir = Path.GetDirectoryName(outputFilePath)!;
        string unique = Guid.NewGuid().ToString("N");
        string payloadPath = Path.Combine(workingDir, $"read_gguf_tensors_{unique}.json");
        string resultPath = Path.Combine(workingDir, $"read_gguf_tensors_result_{unique}.json");
        string scriptPath = Path.Combine(workingDir, $"read_gguf_tensors_{unique}.py");

        try
        {
            await File.WriteAllTextAsync(payloadPath, JsonSerializer.Serialize(new { gguf_path = ggufPath, output_path = resultPath }));

            const string py = """
                              import json
                              import sys

                              payload_path = sys.argv[1]
                              with open(payload_path, "r", encoding="utf-8") as f:
                                  payload = json.load(f)

                              output_path = payload["output_path"]

                              try:
                                  import gguf
                                  reader = gguf.GGUFReader(payload["gguf_path"])
                                  tensor_names = [t.name for t in reader.tensors]
                                  result = {"TensorNames": tensor_names}
                              except Exception as e:
                                  result = {"Error": str(e), "TensorNames": []}

                              with open(output_path, "w", encoding="utf-8") as f:
                                  json.dump(result, f, indent=2)
                              """;

            await File.WriteAllTextAsync(scriptPath, py);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            var result = JsonSerializer.Deserialize<TensorNameReadResult>(await File.ReadAllTextAsync(resultPath));
            if (result == null)
                throw new InvalidOperationException("Failed to parse GGUF tensor list result.");
            if (!string.IsNullOrWhiteSpace(result.Error))
                throw new InvalidOperationException($"Failed to read GGUF tensor names: {result.Error}");

            return result.TensorNames;
        }
        finally
        {
            if (File.Exists(payloadPath)) File.Delete(payloadPath);
            if (File.Exists(resultPath)) File.Delete(resultPath);
            if (File.Exists(scriptPath)) File.Delete(scriptPath);
        }
    }

    // ----------------------------------------------------------------
    // Internal DTOs
    // ----------------------------------------------------------------

    private static readonly Regex TensorLogLineRegex = new(
        @"\]\s+(?<tensor>[^\s]+)\s+-\s+\[[^\]]+\],\s+type\s*=\s*(?<type>[A-Za-z0-9_]+)(?:.*?converting to\s+(?<convert>[A-Za-z0-9_]+))?",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static string NormalizeQuantName(string value)
    {
        var normalized = value.Trim().ToUpperInvariant();
        return normalized.Replace("Q5_K", "Q5_K")
            .Replace("Q6_K", "Q6_K")
            .Replace("Q8_0", "Q8_0")
            .Replace("IQ4_XS", "IQ4_XS")
            .Replace("IQ4_NL", "IQ4_NL")
            .Replace("BF16", "BF16")
            .Replace("F16", "F16")
            .Replace("F32", "F32");
    }

    private sealed class QuantizationExecutionReport
    {
        public string LogPath { get; set; } = string.Empty;
        public List<ConcreteTensorOverride> ResolvedOverrides { get; set; } = new();
    }

    private sealed class RequestedTensorOverride
    {
        public string GroupName { get; set; } = string.Empty;
        public string TensorName { get; set; } = string.Empty;
        public string SchemeName { get; set; } = string.Empty;
    }

    private sealed class ConcreteTensorOverride
    {
        public string TensorName { get; set; } = string.Empty;
        public string SchemeName { get; set; } = string.Empty;
        public string GroupName { get; set; } = string.Empty;
    }

    private sealed class TensorGroupingResult
    {
        public TensorGroup? PrimaryGroup { get; set; }
        public List<string> MatchedGroups { get; set; } = new();
    }

    private sealed class TensorNameReadResult
    {
        public string? Error { get; set; }
        public List<string> TensorNames { get; set; } = new();
    }

    // ----------------------------------------------------------------
    // Naming helpers
    // ----------------------------------------------------------------

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

        var effectiveTensors = quant.Tensors?
            .Where(t => t?.TGroup != null && t.TensorType.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .ToList();

        if (effectiveTensors == null || effectiveTensors.Count == 0)
            return $"{modelName}-{baseName}";

        var grouped = effectiveTensors
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

        var nameParts = new List<string>(grouped.Count);

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

    private sealed class LoggedProcessResult
    {
        public int ExitCode { get; init; }
        public string StdOut { get; init; } = string.Empty;
        public string StdErr { get; init; } = string.Empty;
    }

    private async Task<LoggedProcessResult> RunLoggedProcessAsync(
        ProcessStartInfo psi,
        string? logPath,
        CancellationToken ct = default)
    {
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;

        using var process = new Process
        {
            StartInfo = psi,
            EnableRaisingEvents = true
        };

        var stdoutBuilder = new StringBuilder();
        var stderrBuilder = new StringBuilder();
        object sync = new();

        var stdoutClosed = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var stderrClosed = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        StreamWriter? logWriter = null;
        FileStream? logStream = null;

        if (!string.IsNullOrWhiteSpace(logPath))
        {
            logStream = new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            logWriter = new StreamWriter(logStream) { AutoFlush = true };
        }

        void HandleLine(string? line, bool isError)
        {
            if (line == null)
            {
                if (isError)
                    stderrClosed.TrySetResult(true);
                else
                    stdoutClosed.TrySetResult(true);

                return;
            }

            lock (sync)
            {
                if (isError)
                    stderrBuilder.AppendLine(line);
                else
                    stdoutBuilder.AppendLine(line);

                logWriter?.WriteLine(line);
            }

            AnsiConsole.WriteLine(line);
        }

        process.OutputDataReceived += (_, e) => HandleLine(e.Data, isError: false);
        process.ErrorDataReceived += (_, e) => HandleLine(e.Data, isError: true);

        if (!process.Start())
            throw new InvalidOperationException($"Failed to start process: {psi.FileName}");

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using var ctr = ct.Register(() =>
        {
            try
            {
                if (!process.HasExited)
                    process.Kill(entireProcessTree: true);
            }
            catch
            {
                // ignored
            }
        });

        await process.WaitForExitAsync(ct);
        await Task.WhenAll(stdoutClosed.Task, stderrClosed.Task);

        logWriter?.Dispose();
        logStream?.Dispose();

        return new LoggedProcessResult
        {
            ExitCode = process.ExitCode,
            StdOut = stdoutBuilder.ToString(),
            StdErr = stderrBuilder.ToString()
        };
    }
}
