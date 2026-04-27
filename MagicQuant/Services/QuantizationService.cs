using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using MagicQuant.Helpers;
using MagicQuant.Models.Learning;
using MagicQuant.Services.Learning;
using MagicQuant.Services.Progress;
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
    public Guid? TensorComboId { get; set; }
    public Guid? BenchmarkId { get; set; }
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
    private readonly int _quantThreadsPerProcess;
    private readonly int _maxConcurrentQuantizations;
    private readonly ImatrixService _imatrixService;
    private readonly HuggingFaceBaselineService _huggingFaceBaselineService;
    private readonly TensorGroupingAuditService _tensorGroupingAuditService;
    private readonly TensorLearningDiagnosticWriter _tensorLearningDiagnosticWriter;

    private static readonly SemaphoreSlim BaseModelLock = new(1, 1);
    private const byte UnknownTensorGroupId = 255;

    private static readonly Lazy<Dictionary<string, string>> QuantAliasLookup =
        new(BuildQuantAliasLookup, LazyThreadSafetyMode.ExecutionAndPublication);

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
        _imatrixService = new ImatrixService();
        _huggingFaceBaselineService = new HuggingFaceBaselineService(_python);
        _tensorGroupingAuditService = new TensorGroupingAuditService();
        _tensorLearningDiagnosticWriter = new TensorLearningDiagnosticWriter();

        Directory.CreateDirectory(_ggufDir);
        Directory.CreateDirectory(_benchDir);

        int threadCount = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;

// Minimum desired threads per llama-quantize process.
// This is used to decide the natural concurrency first.
        const int minimumQuantThreadsPerProcess = 8;

// Hard safety cap for large GGUF quantization.
// More than 2 concurrent 35B quantizers can overwhelm the output NVMe queue.
        const int maxConcurrentQuantizationCap = 2;

// Keep a little workstation breathing room.
        int reservedThreads = threadCount switch
        {
            >= 16 => 2,
            >= 8  => 2,
            >= 4  => 1,
            _     => 0
        };

        int usableThreads = Math.Max(1, threadCount - reservedThreads);

// First decide how many quantization processes the CPU budget would naturally allow.
        int naturalConcurrentQuantizations = Math.Max(
            1,
            usableThreads / minimumQuantThreadsPerProcess);

// Then cap it to avoid hammering the output drive with too many giant writers.
        _maxConcurrentQuantizations = Math.Max(
            1,
            Math.Min(maxConcurrentQuantizationCap, naturalConcurrentQuantizations));

// Divide the usable thread budget evenly across the allowed quantization processes.
// Example on 7950X3D:
// 32 total - 2 reserved = 30 usable
// natural = 30 / 8 = 3
// capped = min(2, 3) = 2
// threads/process = 30 / 2 = 15
        _quantThreadsPerProcess = Math.Max(
            1,
            usableThreads / _maxConcurrentQuantizations);

        _cpuQuantLock = new SemaphoreSlim(
            _maxConcurrentQuantizations,
            _maxConcurrentQuantizations);

        AnsiConsole.MarkupLine(
            $"[grey]Quantization CPU plan:[/] " +
            $"threads={threadCount}, reserved={reservedThreads}, usable={usableThreads}, " +
            $"naturalConcurrent={naturalConcurrentQuantizations}, " +
            $"concurrent={_maxConcurrentQuantizations}, " +
            $"threads/process={_quantThreadsPerProcess}");
    }

    public static void ValidateQuantNameNormalizationOrThrow()
    {
        var collisions = TensorWeightScheme.All
            .Where(x => !x.Names.IsDefaultOrEmpty)
            .SelectMany(s => s.Names.Select(name => new
            {
                SchemeId = s.UniqueId,
                Canonical = s.Names[0],
                Alias = CanonicalizeQuantToken(name)
            }))
            .GroupBy(x => x.Alias, StringComparer.Ordinal)
            .Where(g => g.Select(x => x.SchemeId).Distinct().Count() > 1)
            .Select(g => $"{g.Key} => {string.Join(", ", g.Select(x => x.Canonical).Distinct(StringComparer.Ordinal))}")
            .ToList();

        if (collisions.Count > 0)
        {
            throw new InvalidOperationException(
                "Quant alias registry has conflicting aliases across TensorWeightScheme definitions: " +
                string.Join(" | ", collisions));
        }
    }

    // ----------------------------------------------------------------
    // Batch processing
    // ----------------------------------------------------------------

    public Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<HybridQuant> quants,
        CancellationToken ct = default)
        => ProcessHybridBatchAsync(quants, progressOptions: null, ct);

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<HybridQuant> quants,
        StageProgressOptions? progressOptions,
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

        return await ProcessHybridBatchAsync(shimmedPlans, progressOptions, ct);
    }

    public Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<RequiredSamplePlan> plans,
        CancellationToken ct = default)
        => ProcessHybridBatchAsync(plans, progressOptions: null, ct);

    public async Task<SampleProcessingSummary> ProcessHybridBatchAsync(
        IReadOnlyCollection<RequiredSamplePlan> plans,
        StageProgressOptions? progressOptions,
        CancellationToken ct = default)
    {
        if (plans == null)
            throw new ArgumentNullException(nameof(plans));

        if (plans.Count == 0)
        {
            return new SampleProcessingSummary
            {
                Requested = 0,
                Records = new List<SampleProcessingRecord>()
            };
        }

        var records = new ConcurrentBag<SampleProcessingRecord>();
        var stageProgress = progressOptions != null && progressOptions.Total > 0
            ? new StageProgressTracker(progressOptions)
            : null;

        await EnsureBaseModelFileAsync(false);

        var learnableBaselinePlans = plans
            .Where(p => IsLearnableBaselineRun(p.Quant))
            .OrderBy(p => p.Quant.BaseQuant.UniqueId)
            .ToList();

        var duplicateLearnableNames = learnableBaselinePlans
            .Select(p => GenerateHybridName(p.Quant))
            .GroupBy(x => x, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        if (duplicateLearnableNames.Count > 0)
        {
            throw new InvalidOperationException(
                "Duplicate learnable baseline output names were queued in the same batch: " +
                string.Join(", ", duplicateLearnableNames));
        }

        int baselineWorkerCount = CalculateBatchWorkerCount(learnableBaselinePlans.Count);

        await Parallel.ForEachAsync(
            learnableBaselinePlans,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = baselineWorkerCount,
                CancellationToken = ct
            },
            async (baselinePlan, token) =>
            {
                records.Add(await ExecutePlanAsync(baselinePlan, stageProgress, token));
            });

        var remainingPlans = plans.Except(learnableBaselinePlans).ToList();
        var equivalenceMap = await BuildIsolationDeduplicationPlanAsync(remainingPlans, ct);

        var planByKey = remainingPlans.ToDictionary(p => p.Key, StringComparer.Ordinal);
        var primaryGroups = new List<(RequiredSamplePlan Source, List<RequiredSamplePlan> Duplicates)>();

        foreach (var plan in remainingPlans)
        {
            ct.ThrowIfCancellationRequested();

            if (!equivalenceMap.TryGetValue(plan.Key, out var sourceKey) ||
                string.IsNullOrWhiteSpace(sourceKey) ||
                string.Equals(sourceKey, plan.Key, StringComparison.Ordinal))
            {
                primaryGroups.Add((plan, new List<RequiredSamplePlan>()));
                continue;
            }

            if (!planByKey.TryGetValue(sourceKey, out _))
            {
                primaryGroups.Add((plan, new List<RequiredSamplePlan>()));
            }
        }

        var groupBySource = primaryGroups.ToDictionary(g => g.Source.Key, g => g, StringComparer.Ordinal);
        foreach (var plan in remainingPlans)
        {
            if (!equivalenceMap.TryGetValue(plan.Key, out var sourceKey) ||
                string.IsNullOrWhiteSpace(sourceKey) ||
                string.Equals(sourceKey, plan.Key, StringComparison.Ordinal))
            {
                continue;
            }

            if (groupBySource.TryGetValue(sourceKey, out var group))
                group.Duplicates.Add(plan);
        }

        int workerCount = CalculateBatchWorkerCount(primaryGroups.Count);

        await Parallel.ForEachAsync(
            primaryGroups,
            new ParallelOptions { MaxDegreeOfParallelism = workerCount, CancellationToken = ct },
            async (group, token) =>
            {
                records.Add(await ExecutePlanAsync(group.Source, stageProgress, token));

                foreach (var duplicatePlan in group.Duplicates)
                {
                    token.ThrowIfCancellationRequested();
                    records.Add(await ExecuteDuplicatePlanAsync(group.Source, duplicatePlan, stageProgress, token));
                }
            });

        var finalRecords = records.OrderBy(x => x.Plan.Key, StringComparer.Ordinal).ToList();

        return new SampleProcessingSummary
        {
            Requested = plans.Count,
            Completed = finalRecords.Count(x => x.State == SampleProcessState.Completed),
            Skipped = finalRecords.Count(x => x.State == SampleProcessState.Skipped),
            Failed = finalRecords.Count(x => x.State == SampleProcessState.Failed),
            Records = finalRecords
        };
    }


    private int CalculateBatchWorkerCount(int itemCount)
    {
        if (itemCount <= 0)
            return 1;

        return Math.Max(1, Math.Min(
            itemCount,
            _maxConcurrentQuantizations + _benchmarker.CurrentParallelSlotCount));
    }


    private async Task<SampleProcessingRecord> ExecutePlanAsync(
        RequiredSamplePlan plan,
        StageProgressTracker? progress,
        CancellationToken ct)
    {
        var record = new SampleProcessingRecord
        {
            Plan = plan,
            ModelName = GenerateHybridName(plan.Quant)
        };

        try
        {
            var state = await ProcessHybridQuantAsync(plan.Quant, ct);
            record.State = state;

            var identity = await ResolveBenchmarkIdentityAsync(plan.Quant, ct);
            record.TensorComboId = identity.TensorComboId;
            record.BenchmarkId = identity.BenchmarkId;

            progress?.ReportFinished(state, record.ModelName);
            return record;
        }
        catch (Exception ex)
        {
            record.State = SampleProcessState.Failed;
            record.Error = ex.Message;

            AnsiConsole.MarkupLine($"[red]Sample failed:[/] {Markup.Escape(record.ModelName)}");
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");

            progress?.ReportFinished(SampleProcessState.Failed, record.ModelName);
            return record;
        }
    }

    private async Task<SampleProcessingRecord> ExecuteDuplicatePlanAsync(
        RequiredSamplePlan sourcePlan,
        RequiredSamplePlan duplicatePlan,
        StageProgressTracker? progress,
        CancellationToken ct)
    {
        var record = new SampleProcessingRecord
        {
            Plan = duplicatePlan,
            ModelName = GenerateHybridName(duplicatePlan.Quant)
        };

        try
        {
            bool cloned = await CloneEquivalentIsolationBenchmarkAsync(sourcePlan, duplicatePlan, ct);

            if (cloned)
            {
                var identity = await ResolveBenchmarkIdentityAsync(duplicatePlan.Quant, ct);
                record.State = SampleProcessState.Completed;
                record.TensorComboId = identity.TensorComboId;
                record.BenchmarkId = identity.BenchmarkId;
                progress?.ReportFinished(SampleProcessState.Completed, record.ModelName);
                return record;
            }

            return await ExecutePlanAsync(duplicatePlan, progress, ct);
        }
        catch (Exception ex)
        {
            record.State = SampleProcessState.Failed;
            record.Error = ex.Message;

            AnsiConsole.MarkupLine($"[red]Sample failed:[/] {Markup.Escape(record.ModelName)}");
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(ex.Message)}[/]");

            progress?.ReportFinished(SampleProcessState.Failed, record.ModelName);
            return record;
        }
    }

    private async Task<(Guid? TensorComboId, Guid? BenchmarkId)> ResolveBenchmarkIdentityAsync(
        HybridQuant quant,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var lookup = BuildTensorLookup(quant);

        await using var db = new MagicQuantContext();

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdOrNullAsync(db, ct);

        if (exactAiModelHashId == null)
            return (null, null);

        var imatrixDefinitionId =
            await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, exactAiModelHashId.Value,
                createIfMissing: false, ct);

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

        if (comboId == Guid.Empty)
            return (null, null);

        var benchmarkId = await db.AiBenchmarks
            .AsNoTracking()
            .Where(x => x.AiModelHashId == exactAiModelHashId.Value && x.ImatrixDefinitionId == imatrixDefinitionId &&
                        x.TensorComboId == comboId)
            .Select(x => x.Id)
            .FirstOrDefaultAsync(ct);

        return (comboId, benchmarkId == Guid.Empty ? null : benchmarkId);
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
        bool pureExternalBaseline = ShouldDownloadExternalBaselineInsteadOfQuantizing(quant);
        bool baselineLearnedTruthExists =
            !forceBaselineRelearn && await HasLearnedTruthForBaselineAsync(quant.BaseQuant, ct);
        string benchmarkModelPath = quantPath;
        PreparedExternalBaselineBuild? preparedExternalBaseline = null;
        string? transientExternalDownloadPath = null;

        if (!forceBaselineRelearn && baselineLearnedTruthExists && await _benchmarker.TryReuseExistingBenchmarksAsync(
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

        if (!forceBaselineRelearn && baselineLearnedTruthExists && await BenchmarkExistsAsync(quant, ct))
        {
            AnsiConsole.MarkupLine($"[grey]Skipping already completed sample:[/] {Markup.Escape(modelName)}");

            if (!IsProtectedModel(modelName))
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

            return SampleProcessState.Skipped;
        }

        try
        {
            string inputPath = await GetEffectiveInputModelPathAsync(quant, forceBaselineRelearn, ct);
            if (pureExternalBaseline)
                transientExternalDownloadPath = inputPath;

            QuantizationExecutionReport? quantizationReport = null;

            await _cpuQuantLock.WaitAsync(ct);
            try
            {
                if (pureExternalBaseline)
                {
                    preparedExternalBaseline = await PrepareExternalBaselineRebuildAsync(
                        quant,
                        downloadedExternalBaselinePath: inputPath,
                        rebuiltOutputPath: quantPath,
                        forceBaselineRelearn: forceBaselineRelearn,
                        ct: ct);

                    benchmarkModelPath = preparedExternalBaseline.BenchmarkModelPath;
                }
                else
                {
                    benchmarkModelPath = quantPath;
                    if (!File.Exists(quantPath) || forceBaselineRelearn)
                    {
                        var quantToExecute = quant.BaseQuant.IsExternalRepositoryBaseline
                            ? CreateEquivalentStandardCarrierQuantForExternalRebuild(quant)
                            : quant;

                        var effectiveInputPath = quant.BaseQuant.IsExternalRepositoryBaseline
                            ? await EnsureBaseModelFileAsync()
                            : inputPath;

                        if (quant.BaseQuant.IsExternalRepositoryBaseline)
                        {
                            AnsiConsole.MarkupLine(
                                $"[cyan]Building sample:[/] {Markup.Escape(modelName)} [grey](native input, surrogate carrier={Markup.Escape(quantToExecute.BaseQuant.Names[0])})[/]");
                        }
                        else
                        {
                            AnsiConsole.MarkupLine($"[cyan]Building sample:[/] {Markup.Escape(modelName)}");
                        }

                        quantizationReport = await RunLlamaQuantizeAsync(effectiveInputPath, quantPath, quantToExecute);
                    }
                }
            }
            finally
            {
                _cpuQuantLock.Release();
            }

            if (!forceBaselineRelearn && baselineLearnedTruthExists && await BenchmarkExistsAsync(quant, ct))
            {
                if (!IsProtectedModel(modelName) && benchmarkModelPath == quantPath)
                    await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);

                return SampleProcessState.Skipped;
            }

            AnsiConsole.MarkupLine($"[yellow]Benchmarking:[/] {Markup.Escape(modelName)}");

            await _benchmarker.RunAllBenchmarksAsync(
                quantConfig: quant,
                modelPath: benchmarkModelPath,
                benchDir: modelBenchDir,
                klLogitsDir: baseLogitsDir,
                saveLogits: false,
                domainsOverride: new[] { "general" });

            stopwatch.Stop();

            await PersistQuantizationRunAsync(
                quant: quant,
                imatrixDefinitionId: null,
                startedUtc: startedUtc,
                completedUtc: DateTime.UtcNow,
                succeeded: true,
                outputModelPath: benchmarkModelPath,
                error: null,
                ct: ct);

            if (IsLearnableBaselineRun(quant))
            {
                if (preparedExternalBaseline?.HasPreparedLearningTruth == true)
                    await PersistLearnedBaselineTensorMapFromPreparedAsync(quant, preparedExternalBaseline, ct);
                else if (!baselineLearnedTruthExists || forceBaselineRelearn)
                    await LearnAndPersistBaselineTensorMapAsync(quant, benchmarkModelPath, quantizationReport, ct);
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
                    imatrixDefinitionId: null,
                    startedUtc: startedUtc,
                    completedUtc: DateTime.UtcNow,
                    succeeded: false,
                    outputModelPath: benchmarkModelPath,
                    error: ex.ToString(),
                    ct: ct);
            }
            catch
            {
            }

            throw;
        }
        finally
        {
            if (pureExternalBaseline && !string.IsNullOrWhiteSpace(transientExternalDownloadPath))
                await CleanupExternalBaselineDownloadArtifactsAsync(transientExternalDownloadPath);

            if (!IsProtectedModel(modelName) && benchmarkModelPath == quantPath)
                await HardDeleteHelper.DeleteFileIfExistsAsync(quantPath);
        }
    }

    private bool ShouldDownloadExternalBaselineInsteadOfQuantizing(HybridQuant quant)
        => quant.BaseQuant.IsExternalRepositoryBaseline && quant.Tensors.Count == 0;

    private async Task<string> GetEffectiveInputModelPathAsync(HybridQuant quant, bool forceRefresh,
        CancellationToken ct)
    {
        string basePath = await EnsureBaseModelFileAsync();
        if (!quant.BaseQuant.IsExternalRepositoryBaseline)
            return basePath;

        // Pure external baselines are downloaded so MagicQuant can learn their tensor truth.
        // Any continuation / isolation / hybrid that uses that external baseline must rebuild
        // from the native base GGUF instead of requantizing the staged external GGUF.
        if (quant.Tensors.Count > 0)
            return basePath;

        string externalPath = GetExternalBaselineCachePath(quant.BaseQuant);
        await _huggingFaceBaselineService.DownloadBaselineAsync(quant.BaseQuant, externalPath, forceRefresh, ct);
        await ValidateExternalBaselineTensorParityOrThrow(basePath, externalPath);
        return externalPath;
    }

    private string GetExternalBaselineCachePath(BaselineQuants baseline)
    {
        string root = Cache.ExternalBaselineCacheDirectory ??
                      Path.Combine(Cache.ModelMagicQuantDirectory!, "ExternalBaselines");
        Directory.CreateDirectory(root);
        string safe =
            string.Concat(baseline.CanonicalKey.Select(ch => Path.GetInvalidFileNameChars().Contains(ch) ? '_' : ch));
        string extension = Path.GetExtension(baseline.SourceFileName ?? string.Empty);
        if (string.IsNullOrWhiteSpace(extension))
            extension = ".gguf";
        return Path.Combine(root, safe + extension);
    }

    private async Task ValidateExternalBaselineTensorParityOrThrow(string baseModelPath, string externalBaselinePath)
    {
        var baseMeta = await ReadTensorMetadataFromGgufAsync(baseModelPath, externalBaselinePath + ".nativecheck");
        var externalMeta =
            await ReadTensorMetadataFromGgufAsync(externalBaselinePath, externalBaselinePath + ".externalcheck");

        var baseNames = baseMeta.TensorNames.OrderBy(x => x, StringComparer.Ordinal).ToList();
        var externalNames = externalMeta.TensorNames.OrderBy(x => x, StringComparer.Ordinal).ToList();

        var missing = baseNames.Except(externalNames, StringComparer.Ordinal).Take(20).ToList();
        var unexpected = externalNames.Except(baseNames, StringComparer.Ordinal).Take(20).ToList();

        if (missing.Count > 0 || unexpected.Count > 0 || baseNames.Count != externalNames.Count)
        {
            throw new InvalidOperationException(
                $"External/custom baseline tensor mismatch detected. Missing=[{string.Join(", ", missing)}] Unexpected=[{string.Join(", ", unexpected)}]. " +
                "MagicQuant will not persist or use a custom baseline whose tensor names do not exactly match the source model.");
        }
    }

    private async Task<bool> HasLearnedTruthForBaselineAsync(BaselineQuants baseline, CancellationToken ct = default)
    {
        if (baseline.UniqueId == BaselineQuants.NativeSourceUniqueId)
            return await HasNativeSourceLearnedTruthAsync(ct);

        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            return false;

        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);

        if (scopedAiModelHashId == null)
            return false;

        var query = db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.BaselineCanonicalKey == baseline.CanonicalKey);

        if (baseline.DefaultTensorScheme != null)
            query = query.Where(x => x.TensorWeightSchemeId == baseline.DefaultTensorScheme.UniqueId);

        return await query.AnyAsync(ct);
    }

    private async Task<PreparedExternalBaselineBuild> PrepareExternalBaselineRebuildAsync(
        HybridQuant quant,
        string downloadedExternalBaselinePath,
        string rebuiltOutputPath,
        bool forceBaselineRelearn,
        CancellationToken ct)
    {
        if (!quant.BaseQuant.IsExternalRepositoryBaseline)
            throw new InvalidOperationException(
                "PrepareExternalBaselineRebuildAsync was called for a non-external baseline.");

        string nativeBasePath = await EnsureBaseModelFileAsync();
        bool canReuseLearnedTruth = !forceBaselineRelearn && await HasLearnedTruthForBaselineAsync(quant.BaseQuant, ct);

        if (canReuseLearnedTruth)
        {
            var blanket = TryLoadAllLearnedTensorMappings(
                canonicalBaselineKey: quant.BaseQuant.CanonicalKey,
                preferredSourceScheme: quant.BaseQuant.DefaultTensorScheme,
                allowDominantFallback: false);

            if (blanket.Count == 0)
                throw new InvalidOperationException(
                    $"Custom baseline '{quant.BaseQuant.Names[0]}' was marked as already learned, but no blanket learned tensor mapping could be loaded.");

            if (!File.Exists(rebuiltOutputPath) || forceBaselineRelearn)
            {
                AnsiConsole.MarkupLine(
                    $"[cyan]Rebuilding normalized custom baseline from learned truth:[/] {Markup.Escape(quant.BaseQuant.Names[0])}");
                await RunLlamaQuantizeAsync(nativeBasePath, rebuiltOutputPath, quant, blanket);
            }

            return new PreparedExternalBaselineBuild
            {
                BenchmarkModelPath = rebuiltOutputPath,
                DownloadedExternalModelPath = downloadedExternalBaselinePath,
                HasPreparedLearningTruth = false
            };
        }

        AnsiConsole.MarkupLine(
            $"[cyan]Learning external baseline truth from downloaded artifact:[/] {Markup.Escape(quant.BaseQuant.Names[0])}");
        await ValidateExternalBaselineTensorParityOrThrow(nativeBasePath, downloadedExternalBaselinePath);

        var ggufMetadata =
            await ReadTensorMetadataFromGgufAsync(downloadedExternalBaselinePath, rebuiltOutputPath + ".learn");
        var ggufTruth = ggufMetadata.TensorTypes
            .ToDictionary(x => x.Key, x => NormalizeQuantName(x.Value), StringComparer.Ordinal);

        if (ggufTruth.Count == 0)
            throw new InvalidOperationException(
                $"Downloaded external baseline '{quant.BaseQuant.Names[0]}' produced no readable GGUF tensor truth.");

        var truth = ggufTruth
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(
                x => x.Key,
                x => new LearnedTensorTruth(x.Key, x.Value, LearningSource.GgufOnly),
                StringComparer.Ordinal);

        var verification = new TensorTruthVerificationResult
        {
            TruthByTensor = truth
        };
        var audit = _tensorGroupingAuditService.Audit(truth.Keys.ToList(), truth);

        if (audit.HasFatalIssues)
        {
            var diagnosticPath = await _tensorLearningDiagnosticWriter.WriteFailureAsync(
                baselineName: quant.BaseQuant.Names[0],
                schemeName: quant.BaseQuant.DefaultTensorScheme?.Names[0] ?? "external",
                sourceKind: quant.BaseQuant.SourceKind.ToString(),
                sourceRepository: quant.BaseQuant.SourceRepository,
                sourceFileName: quant.BaseQuant.SourceFileName,
                truthByTensor: truth,
                audit: audit,
                verification: verification,
                ct: ct);

            AnsiConsole.MarkupLine(
                $"[red]Tensor group learning failed.[/] See diagnostic log: [yellow]{Markup.Escape(diagnosticPath)}[/]");
            throw new InvalidOperationException(
                $"Strict tensor-group learning validation failed for external baseline '{quant.BaseQuant.Names[0]}' " +
                $"from '{quant.BaseQuant.SourceRepository}/{quant.BaseQuant.SourceFileName}'. " +
                $"No normalized rebuilt baseline was produced and no learned tensor mappings were persisted. " +
                $"Diagnostic log: {diagnosticPath}");
        }

        var normalizedOverrides = truth.ToDictionary(
            x => x.Key,
            x => NativePrecisionNormalization.NormalizeLearnedFinalQuantTypeForApplication(x.Value.FinalQuantType),
            StringComparer.Ordinal);

        if (normalizedOverrides.Values.Any(string.IsNullOrWhiteSpace))
            throw new InvalidOperationException(
                $"External baseline '{quant.BaseQuant.Names[0]}' produced one or more empty normalized tensor scheme names.");

        AnsiConsole.MarkupLine(
            $"[cyan]Rebuilding normalized benchmark artifact for custom baseline:[/] {Markup.Escape(quant.BaseQuant.Names[0])}");
        await RunLlamaQuantizeAsync(nativeBasePath, rebuiltOutputPath, quant, normalizedOverrides);

        return new PreparedExternalBaselineBuild
        {
            BenchmarkModelPath = rebuiltOutputPath,
            DownloadedExternalModelPath = downloadedExternalBaselinePath,
            TruthByTensor = truth,
            GroupedByTensor = audit.GroupedByTensor,
            AllTensorNamesInDownloadedArtifact = ggufMetadata.TensorNames,
            AmbiguousGroupingRows = audit.Ambiguous,
            UnresolvedTensorNames = audit.IllegalUnresolved.Select(x => x.TensorName).ToList(),
            BaseQuantExceptionRows = audit.BaseQuantExceptions,
            Verification = verification,
            HasPreparedLearningTruth = true
        };
    }

    private async Task PersistLearnedBaselineTensorMapFromPreparedAsync(
        HybridQuant quant,
        PreparedExternalBaselineBuild prepared,
        CancellationToken ct)
    {
        if (!IsLearnableBaselineRun(quant) || !prepared.HasPreparedLearningTruth || prepared.TruthByTensor == null ||
            prepared.GroupedByTensor == null)
            return;

        var tensorScheme = quant.BaseQuant.DefaultTensorScheme!;
        var verification = prepared.Verification ?? new TensorTruthVerificationResult
            { TruthByTensor = prepared.TruthByTensor };
        var audit = new TensorGroupingAuditResult
        {
            GroupedByTensor = prepared.GroupedByTensor,
            Ambiguous = prepared.AmbiguousGroupingRows ?? [],
            IllegalUnresolved = (prepared.UnresolvedTensorNames ?? []).Select(x => new TensorGroupingAuditIssue
            {
                TensorName = x,
                IssueKind = "IllegalUnresolvedTensor"
            }).ToList(),
            BaseQuantExceptions = prepared.BaseQuantExceptionRows ?? []
        };

        if (audit.HasFatalIssues || verification.HasFatalIssues)
        {
            var diagnosticPath = await _tensorLearningDiagnosticWriter.WriteFailureAsync(
                baselineName: quant.BaseQuant.Names[0],
                schemeName: tensorScheme.Names[0],
                sourceKind: quant.BaseQuant.SourceKind.ToString(),
                sourceRepository: quant.BaseQuant.SourceRepository,
                sourceFileName: quant.BaseQuant.SourceFileName,
                truthByTensor: prepared.TruthByTensor,
                audit: audit,
                verification: verification,
                ct: ct);

            AnsiConsole.MarkupLine(
                $"[red]Tensor group learning failed.[/] See diagnostic log: [yellow]{Markup.Escape(diagnosticPath)}[/]");
            throw new InvalidOperationException(
                $"Strict tensor-group learning validation failed for external baseline '{quant.BaseQuant.Names[0]}' " +
                $"from '{quant.BaseQuant.SourceRepository}/{quant.BaseQuant.SourceFileName}'. " +
                $"Prepared external baseline learning truth was invalid. No learned tensor mappings were persisted. " +
                $"Diagnostic log: {diagnosticPath}");
        }

        await using var db = new MagicQuantContext();

        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            throw new InvalidOperationException(
                "Unable to persist learned mappings because scoped AiModelHash row was not found.");

        var combo = await db.TensorCombos
            .AsNoTracking()
            .FirstAsync(x => x.BaseQuant == quant.BaseQuant.UniqueId &&
                             x.Embeddings == 0 && x.LmHead == 0 && x.AttnQ == 0 && x.AttnKV == 0 &&
                             x.AttnOutput == 0 && x.FfnUpGate == 0 && x.FfnDown == 0 && x.MoeExperts == 0 &&
                             x.MoeRouter == 0, ct);

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId =
            await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, exactAiModelHashId,
                createIfMissing: false, ct);

        var benchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == exactAiModelHashId && x.ImatrixDefinitionId == imatrixDefinitionId &&
                        x.TensorComboId == combo.Id)
            .OrderByDescending(x => x.Id)
            .Select(x => (Guid?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (!benchmarkId.HasValue)
            throw new InvalidOperationException(
                $"Unable to persist learned mappings because no AiBenchmark exists for rebuilt baseline '{quant.BaseQuant.Names[0]}'.");

        var rows = prepared.TruthByTensor
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(kv =>
            {
                var match = prepared.GroupedByTensor[kv.Key];

                return new LearnedBaselineTensorQuant
                {
                    Id = Guid.NewGuid(),
                    AiBenchmarkId = benchmarkId.Value,
                    AiModelHashId = scopedAiModelHashId.Value,
                    BaselineQuantId = quant.BaseQuant.UniqueId,
                    TensorWeightSchemeId = tensorScheme.UniqueId,
                    TensorGroupId = match.PrimaryGroup?.UniqueId ?? UnknownTensorGroupId,
                    BaselineCanonicalKey = quant.BaseQuant.CanonicalKey,
                    BaselineSourceKind = quant.BaseQuant.SourceKind,
                    BaselineSourceRepository = quant.BaseQuant.SourceRepository,
                    BaselineSourceFileName = quant.BaseQuant.SourceFileName,
                    TensorName = kv.Key,
                    FinalQuantType = kv.Value.FinalQuantType
                };
            })
            .ToList();

        if (rows.Count == 0)
            throw new InvalidOperationException(
                $"Prepared learning truth for baseline '{quant.BaseQuant.Names[0]}' produced no persistable rows.");

        await db.LearnedBaselineTensorQuants
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value &&
                        x.BaselineCanonicalKey == quant.BaseQuant.CanonicalKey &&
                        x.TensorWeightSchemeId == tensorScheme.UniqueId)
            .ExecuteDeleteAsync(ct);

        db.LearnedBaselineTensorQuants.AddRange(rows);
        await db.SaveChangesAsync(ct);

        await WriteLearningDiagnosticArtifactAsync(
            baselineName: quant.BaseQuant.Names[0],
            schemeName: tensorScheme.Names[0],
            truthByTensor: prepared.TruthByTensor,
            grouped: prepared.GroupedByTensor,
            allTensorNamesInModel: prepared.AllTensorNamesInDownloadedArtifact ?? prepared.TruthByTensor.Keys.ToList(),
            ambiguous: audit.Ambiguous,
            unresolved: prepared.UnresolvedTensorNames ?? new List<string>());

        AnsiConsole.MarkupLine(
            $"[green]Persisted rebuilt custom-baseline learning truth:[/] [cyan]{rows.Count:N0}[/] row(s) for [yellow]{Markup.Escape(quant.BaseQuant.Names[0])}[/].");
    }

    private async Task CleanupExternalBaselineDownloadArtifactsAsync(string downloadedExternalBaselinePath)
    {
        if (string.IsNullOrWhiteSpace(downloadedExternalBaselinePath))
            return;

        await HardDeleteHelper.DeleteFileIfExistsAsync(downloadedExternalBaselinePath);
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

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdOrNullAsync(db, ct);

        if (exactAiModelHashId == null)
            return false;

        var imatrixDefinitionId =
            await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, exactAiModelHashId.Value,
                createIfMissing: false, ct);

        var bench = await db.AiBenchmarks
            .AsNoTracking()
            .Where(x => x.AiModelHashId == exactAiModelHashId.Value && x.ImatrixDefinitionId == imatrixDefinitionId)
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

        if (bench == Guid.Empty)
            return false;

        bool hasCategory = await db.Set<CategoryBenchmark>()
            .AsNoTracking()
            .AnyAsync(x => x.AiBenchmarkId == bench, ct);

        return hasCategory;
    }

    private static TensorConfig BuildTensorLookup(HybridQuant quant)
    {
        return (TensorConfig)quant;
    }

    private static async Task<uint?> ResolveCurrentScopedAiModelHashIdOrNullAsync(MagicQuantContext db,
        CancellationToken ct)
    {
        return await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
    }

    private static async Task<uint> ResolveCurrentScopedAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct)
    {
        return await ArchitectureFamilyService.ResolveScopedAiModelHashIdAsync(db, ct);
    }

    private static async Task<uint?> ResolveCurrentExactAiModelHashIdOrNullAsync(MagicQuantContext db,
        CancellationToken ct)
    {
        return await ArchitectureFamilyService.ResolveExactCurrentAiModelHashIdOrNullAsync(db, ct);
    }

    private static async Task<uint> ResolveCurrentExactAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct)
    {
        return await ArchitectureFamilyService.ResolveExactCurrentAiModelHashIdAsync(db, ct);
    }


    private async Task PersistQuantizationRunAsync(
        HybridQuant quant,
        int? imatrixDefinitionId,
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

        uint persistenceAiModelHashId = aiModelHash.Id;

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

        imatrixDefinitionId ??=
            await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, persistenceAiModelHashId,
                createIfMissing: true, ct);
        await ImatrixIdentityService.ValidateOwnershipAsync(db, persistenceAiModelHashId, imatrixDefinitionId, ct);

        Guid? aiBenchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == persistenceAiModelHashId && x.ImatrixDefinitionId == imatrixDefinitionId &&
                        x.TensorComboId == tensorCombo.Id)
            .Select(x => (Guid?)x.Id)
            .FirstOrDefaultAsync(ct);

        var row = new QuantizationRun
        {
            Id = Guid.NewGuid(),
            AiModelHashId = persistenceAiModelHashId,
            ImatrixDefinitionId = imatrixDefinitionId,
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

        AnsiConsole.MarkupLine($"[bold yellow]Benchmarking Base {Markup.Escape(typeStr)} (Saving Logits)...[/]");

        var baseModelQuant = new HybridQuant
        {
            BaseQuant = BaselineQuants.GetNativeQuant(),
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
                    var currentModelName = Path.GetFileNameWithoutExtension(currentFileName);

                    if (isImmune &&
                        string.Equals(currentFileName, normalizedFileName, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (!string.IsNullOrWhiteSpace(currentModelName) && IsProtectedModel(currentModelName))
                    {
                        continue;
                    }

                    await HardDeleteHelper.DeleteFileIfExistsAsync(filePath);
                }
            }

            if (!File.Exists(outputPath) || !File.Exists(successFile))
            {
                AnsiConsole.MarkupLine($"[bold cyan]Converting to {Markup.Escape(typeStr)}...[/]");

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


    public async Task<string> BuildExportArtifactFromExactTensorMapAsync(
        IReadOnlyDictionary<string, string> tensorTypes,
        string outputPath,
        string baseQuantName,
        bool forceRebuild = false,
        CancellationToken ct = default)
    {
        if (tensorTypes == null || tensorTypes.Count == 0)
            throw new ArgumentException("A clone tensor map must contain at least one tensor entry.",
                nameof(tensorTypes));

        if (string.IsNullOrWhiteSpace(outputPath))
            throw new InvalidOperationException("Export output path is required.");

        var baseQuant = BaselineQuants.ResolveBuiltInStandardBaseline(baseQuantName)
                        ?? BaselineQuants.Q8_0;

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        if (!forceRebuild && File.Exists(outputPath) && new FileInfo(outputPath).Length > 0)
            return outputPath;

        if (forceRebuild && File.Exists(outputPath))
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

        await _cpuQuantLock.WaitAsync(ct);
        try
        {
            string nativeBasePath = await EnsureBaseModelFileAsync();
            await RunLlamaQuantizeWithExactTensorMapAsync(
                inputFile: nativeBasePath,
                outputFile: outputPath,
                tensorTypes: tensorTypes,
                baseQuant: baseQuant,
                ct: ct);

            await File.WriteAllTextAsync(outputPath + ".success.json", "{\"status\":\"success\"}", ct);
            return outputPath;
        }
        finally
        {
            _cpuQuantLock.Release();
        }
    }

    public async Task<string> BuildExportArtifactAsync(
        HybridQuant quant,
        string outputPath,
        bool forceRebuild = false,
        CancellationToken ct = default)
    {
        if (quant == null)
            throw new ArgumentNullException(nameof(quant));

        if (string.IsNullOrWhiteSpace(outputPath))
            throw new InvalidOperationException("Export output path is required.");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        if (!forceRebuild && File.Exists(outputPath) && new FileInfo(outputPath).Length > 0)
            return outputPath;

        if (forceRebuild && File.Exists(outputPath))
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputPath);

        await _cpuQuantLock.WaitAsync(ct);
        try
        {
            string nativeBasePath = await EnsureBaseModelFileAsync();
            HybridQuant quantToExecute = quant.BaseQuant.IsExternalRepositoryBaseline
                ? CreateEquivalentStandardCarrierQuantForExternalRebuild(quant)
                : quant;

            IReadOnlyDictionary<string, string>? temporaryCarrierOverrides = null;

            if (quant.BaseQuant.IsExternalRepositoryBaseline)
            {
                string downloadedExternalBaselinePath = GetExternalBaselineCachePath(quant.BaseQuant);
                await _huggingFaceBaselineService.DownloadBaselineAsync(
                    quant.BaseQuant,
                    downloadedExternalBaselinePath,
                    forceRedownload: false,
                    ct: ct);

                temporaryCarrierOverrides = TryLoadAllLearnedTensorMappings(
                    canonicalBaselineKey: quant.BaseQuant.CanonicalKey,
                    preferredSourceScheme: quant.BaseQuant.DefaultTensorScheme,
                    allowDominantFallback: true);

                if (temporaryCarrierOverrides.Count == 0)
                {
                    throw new InvalidOperationException(
                        $"Missing blanket learned mapping for external/custom baseline '{quant.BaseQuant.Names[0]}'. " +
                        "MagicQuant cannot export a hybrid from an external baseline until that baseline has been learned.");
                }
            }

            await RunLlamaQuantizeAsync(
                inputFile: nativeBasePath,
                outputFile: outputPath,
                quant: quantToExecute,
                temporaryCarrierOverrides: temporaryCarrierOverrides);

            await File.WriteAllTextAsync(outputPath + ".success.json", "{\"status\":\"success\"}", ct);
            return outputPath;
        }
        finally
        {
            _cpuQuantLock.Release();
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

    public async Task CleanupPureQ8ModelAsync()
    {
        var pureQ8 = new HybridQuant
        {
            BaseQuant = BaselineQuants.Q8_0,
            Tensors = new List<HybridTensor>()
        };

        string modelName = GenerateHybridName(pureQ8);
        string q8Path = Path.Combine(_ggufDir, $"{modelName}.gguf");
        string successFile = Path.Combine(_ggufDir, $"{Path.GetFileName(q8Path)}.success.json");
        string quantLog = q8Path + ".quantize.log";

        bool hadQ8 = File.Exists(q8Path) || File.Exists(successFile) || File.Exists(quantLog);

        await HardDeleteHelper.DeleteFileIfExistsAsync(q8Path);
        await HardDeleteHelper.DeleteFileIfExistsAsync(successFile);
        await HardDeleteHelper.DeleteFileIfExistsAsync(quantLog);

        if (hadQ8)
            AnsiConsole.MarkupLine($"[grey]Removed probe-only Q8 artifacts:[/] {Markup.Escape(modelName)}");
        else
            AnsiConsole.MarkupLine("[grey]No probe-only Q8 artifacts to clean up.[/]");
    }

    // ----------------------------------------------------------------
    // Quantization
    // ----------------------------------------------------------------


    private async Task<QuantizationExecutionReport> RunLlamaQuantizeWithExactTensorMapAsync(
        string inputFile,
        string outputFile,
        IReadOnlyDictionary<string, string> tensorTypes,
        BaselineQuants baseQuant,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(inputFile) || !File.Exists(inputFile))
            throw new FileNotFoundException($"Input GGUF not found: {inputFile}");

        Directory.CreateDirectory(Path.GetDirectoryName(outputFile)!);

        var inputTensorMetadata = await ReadTensorMetadataFromGgufAsync(inputFile, outputFile);
        var requestedOverrides = tensorTypes
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(x => new RequestedTensorOverride
            {
                GroupName = "clone_exact_tensor_map",
                TensorName = x.Key,
                SchemeName = NormalizeQuantName(x.Value)
            })
            .ToList();

        var concreteOverrides = ResolveConcreteTensorOverrides(
            allTensorNames: inputTensorMetadata.TensorNames,
            requestedOverrides: requestedOverrides);

        var missingInManifest = inputTensorMetadata.TensorNames
            .Except(tensorTypes.Keys, StringComparer.Ordinal)
            .Take(20)
            .ToList();

        var unexpectedInManifest = tensorTypes.Keys
            .Except(inputTensorMetadata.TensorNames, StringComparer.Ordinal)
            .Take(20)
            .ToList();

        if (missingInManifest.Count > 0 || unexpectedInManifest.Count > 0 ||
            inputTensorMetadata.TensorNames.Count != tensorTypes.Count)
        {
            throw new InvalidOperationException(
                $"Clone tensor manifest does not exactly match this model architecture. " +
                $"MissingInManifest=[{string.Join(", ", missingInManifest)}] UnexpectedInManifest=[{string.Join(", ", unexpectedInManifest)}] " +
                $"ModelTensorCount={inputTensorMetadata.TensorNames.Count} ManifestTensorCount={tensorTypes.Count}.");
        }

        var args = new List<string>(capacity: concreteOverrides.Count + 8);

        foreach (var overrideItem in concreteOverrides)
            args.Add($"--tensor-type \"{overrideItem.TensorName}={overrideItem.SchemeName}\"");

        if (_imatrixService.ShouldUseImatrixForQuant(HybridQuant.CreatePureBaseline(baseQuant)))
        {
            string imatrixPath = _imatrixService.GetCanonicalImatrixPath();
            if (!File.Exists(imatrixPath))
                throw new InvalidOperationException(
                    $"Imatrix was marked active but canonical artifact is missing: {imatrixPath}");

            args.Add($"--imatrix \"{imatrixPath}\"");
        }

        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(baseQuant.QuantizeBaseArgumentName);
        args.Add(_quantThreadsPerProcess.ToString());

        string bin = Path.Combine(
            Cache.LlamaBin!,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "llama-quantize.exe" : "llama-quantize");

        string quantizeLogPath = outputFile + ".quantize.log";
        AnsiConsole.MarkupLine(
            $"[cyan]Quantizing clone artifact:[/] {Markup.Escape(Path.GetFileName(outputFile))} [grey](log: {Markup.Escape(quantizeLogPath)})[/]");

        var result = await RunLoggedProcessAsync(new ProcessStartInfo
        {
            FileName = bin,
            Arguments = string.Join(" ", args)
        }, quantizeLogPath, ct);

        if (result.ExitCode != 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);
            throw new InvalidOperationException(
                $"Clone quantization failed for '{outputFile}'. ExitCode={result.ExitCode}. See '{quantizeLogPath}'.");
        }

        if (!File.Exists(outputFile) || new FileInfo(outputFile).Length == 0)
        {
            await HardDeleteHelper.DeleteFileIfExistsAsync(outputFile);
            throw new InvalidOperationException(
                $"Clone quantization exited successfully but produced no valid GGUF output: {outputFile}");
        }

        AnsiConsole.MarkupLine($"[green]Clone quantized model ready:[/] {Markup.Escape(outputFile)}");

        return new QuantizationExecutionReport
        {
            LogPath = quantizeLogPath,
            ResolvedOverrides = concreteOverrides
        };
    }

    private async Task<QuantizationExecutionReport> RunLlamaQuantizeAsync(string inputFile, string outputFile,
        HybridQuant quant, IReadOnlyDictionary<string, string>? temporaryCarrierOverrides = null)
    {
        if (string.IsNullOrWhiteSpace(inputFile) || !File.Exists(inputFile))
            throw new FileNotFoundException($"Input GGUF not found: {inputFile}");

        if (!string.IsNullOrWhiteSpace(Cache.ExternalBaselineCacheDirectory))
        {
            string fullInput = Path.GetFullPath(inputFile);
            string fullExternalRoot = Path.GetFullPath(Cache.ExternalBaselineCacheDirectory);

            if (fullInput.StartsWith(fullExternalRoot, StringComparison.OrdinalIgnoreCase) &&
                (temporaryCarrierOverrides != null || quant.BaseQuant.IsExternalRepositoryBaseline))
            {
                throw new InvalidOperationException(
                    $"Quantization attempted to use staged external GGUF '{inputFile}' as the carrier input. " +
                    "External/custom baselines must rebuild from the native base GGUF instead.");
            }
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputFile)!);

        var inputTensorMetadata = await ReadTensorMetadataFromGgufAsync(inputFile, outputFile);
        var requestedOverrides =
            BuildRequestedTensorOverrides(quant, inputTensorMetadata.TensorNames, temporaryCarrierOverrides);
        var concreteOverrides = ResolveConcreteTensorOverrides(
            allTensorNames: inputTensorMetadata.TensorNames,
            requestedOverrides: requestedOverrides);
        bool shouldRequireFullLearnedCoverage = ShouldApplyLearnedBaseCarrierBlanket(quant, temporaryCarrierOverrides);

        if (requestedOverrides.Count > 0 && concreteOverrides.Count == 0)
        {
            throw new InvalidOperationException(
                $"No concrete tensors were resolved for requested overrides when quantizing '{outputFile}'. " +
                "This means the requested tensor selectors did not match the input GGUF.");
        }

        if (shouldRequireFullLearnedCoverage)
        {
            var concreteNames = concreteOverrides
                .Select(x => x.TensorName)
                .ToHashSet(StringComparer.Ordinal);
            var missing = inputTensorMetadata.TensorNames
                .Except(concreteNames, StringComparer.Ordinal)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();

            if (missing.Count > 0)
            {
                throw new InvalidOperationException(
                    $"Full learned base-carrier coverage is incomplete for baseline '{quant.BaseQuant.Names[0]}'. " +
                    $"Missing={missing.Count}. Examples=[{string.Join(", ", missing.Take(15))}]. " +
                    "Run with --relearn-baseline-mappings.");
            }
        }

        var args = new List<string>(capacity: 256);

        foreach (var overrideItem in concreteOverrides)
        {
            args.Add($"--tensor-type \"{overrideItem.TensorName}={overrideItem.SchemeName}\"");
        }

        if (ShouldApplyImatrix(quant))
        {
            string imatrixPath = _imatrixService.GetCanonicalImatrixPath();
            if (!File.Exists(imatrixPath))
                throw new InvalidOperationException(
                    $"Imatrix was marked active but canonical artifact is missing: {imatrixPath}");

            args.Add($"--imatrix \"{imatrixPath}\"");
        }

        args.Add($"\"{inputFile}\"");
        args.Add($"\"{outputFile}\"");
        args.Add(ResolveQuantizeBaseArgument(quant, concreteOverrides));
        args.Add(_quantThreadsPerProcess.ToString());

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

        AnsiConsole.MarkupLine(
            $"[cyan]Quantizing:[/] {Markup.Escape(Path.GetFileName(outputFile))} [grey](log: {Markup.Escape(quantizeLogPath)})[/]");
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
                "Use a real carrier baseline (Q8_0 recommended) and apply only learned exact tensor overrides for the target configuration.");
        }

        return quant.BaseQuant.QuantizeBaseArgumentName;
    }

    private static HybridQuant CreateEquivalentStandardCarrierQuantForExternalRebuild(HybridQuant quant)
    {
        if (!quant.BaseQuant.IsExternalRepositoryBaseline)
            return quant;

        var standardFamily = BaselineQuants.ResolveBuiltInStandardBaseline(quant.BaseQuant.QuantizeBaseArgumentName)
                             ?? BaselineQuants.ResolveBuiltInStandardBaseline(quant.BaseQuant.Names[0])
                             ?? throw new InvalidOperationException(
                                 $"Could not resolve a built-in carrier baseline for external baseline '{quant.BaseQuant.Names[0]}' using quantize base name '{quant.BaseQuant.QuantizeBaseArgumentName}'.");

        var clone = quant.Clone();
        clone.BaseQuant = standardFamily;
        return clone;
    }

    private bool ShouldApplyImatrix(HybridQuant quant)
    {
        return _imatrixService.ShouldUseImatrixForQuant(quant);
    }

    public async Task<IReadOnlyDictionary<string, string>> ReadExactTensorTypesAsync(
        string ggufPath,
        CancellationToken ct = default)
    {
        var meta = await ReadTensorMetadataFromGgufAsync(ggufPath, ggufPath);
        return meta.TensorTypes
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .ToDictionary(x => x.Key, x => NormalizeQuantName(x.Value), StringComparer.Ordinal);
    }

    public async Task ClearLearnedBaselineTensorMappingsAsync(CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        int removed = await db.LearnedBaselineTensorQuants.ExecuteDeleteAsync(ct);
        AnsiConsole.MarkupLine(
            $"[yellow]Relearn requested:[/] removed [red]{removed:N0}[/] learned baseline tensor mapping rows.");
    }

    public async Task InvalidateBaselineArtifactsAsync(CancellationToken ct = default)
    {
        await ClearLearnedBaselineTensorMappingsAsync(ct);

        foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines())
        {
            var pure = HybridQuant.CreatePureBaseline(baseline);
            var name = GenerateHybridName(pure);
            var ggufPath = Path.Combine(_ggufDir, $"{name}.gguf");
            var success = Path.Combine(_ggufDir, $"{name}.gguf.success.json");
            var log = ggufPath + ".quantize.log";

            await HardDeleteHelper.DeleteFileIfExistsAsync(ggufPath);
            await HardDeleteHelper.DeleteFileIfExistsAsync(success);
            await HardDeleteHelper.DeleteFileIfExistsAsync(log);

            string benchDir = Path.Combine(_benchDir, name);
            if (Directory.Exists(benchDir))
                Directory.Delete(benchDir, recursive: true);
        }

        string debugDir = Path.Combine(_benchDir, "_learning_debug");
        if (Directory.Exists(debugDir))
            Directory.Delete(debugDir, recursive: true);

        if (!string.IsNullOrWhiteSpace(Cache.ExternalBaselineCacheDirectory) &&
            Directory.Exists(Cache.ExternalBaselineCacheDirectory))
            Directory.Delete(Cache.ExternalBaselineCacheDirectory, recursive: true);

        string nativeType = (Cache.TorchType ?? Cache.MainTorchType.BF16).ToString();
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        string nativeBaseFile = Path.Combine(_ggufDir, $"{modelName}-{nativeType}.gguf");
        await HardDeleteHelper.DeleteFileIfExistsAsync(nativeBaseFile);
        await HardDeleteHelper.DeleteFileIfExistsAsync(nativeBaseFile + ".success.json");
        await HardDeleteHelper.DeleteFileIfExistsAsync(nativeBaseFile + ".convert.log");

        string nativeBenchDir = Path.Combine(_benchDir, nativeType);
        if (Directory.Exists(nativeBenchDir))
            Directory.Delete(nativeBenchDir, recursive: true);

        AnsiConsole.MarkupLine(
            "[yellow]Relearn requested:[/] baseline artifacts, benchmark caches, and learning diagnostics were invalidated.");
    }

    public async Task<bool> HasNativeSourceLearnedTruthAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            return false;

        var nativeScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);

        if (scopedAiModelHashId == null)
            return false;

        return await db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value &&
                        x.BaselineQuantId == BaselineQuants.NativeSourceUniqueId &&
                        x.TensorWeightSchemeId == nativeScheme.UniqueId)
            .AnyAsync(ct);
    }

    public async Task LearnNativeSourceTruthAsync(
        string nativeGgufPath,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(nativeGgufPath) || !File.Exists(nativeGgufPath))
            throw new FileNotFoundException($"Native GGUF path not found for learning: {nativeGgufPath}");

        var nativeScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        if (!Cache.ForceRelearnBaselineTensorMappings)
        {
            await using var precheckDb = new MagicQuantContext();
            var precheckScopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(precheckDb, ct);

            if (precheckScopedAiModelHashId != null)
            {
                int existingRows = await precheckDb.LearnedBaselineTensorQuants
                    .AsNoTracking()
                    .Where(x => x.AiModelHashId == precheckScopedAiModelHashId.Value &&
                                x.BaselineQuantId == BaselineQuants.NativeSourceUniqueId &&
                                x.TensorWeightSchemeId == nativeScheme.UniqueId)
                    .CountAsync(ct);

                if (existingRows > 0)
                {
                    AnsiConsole.MarkupLine(
                        $"[grey]Native-source learned truth already exists:[/] [cyan]{existingRows:N0}[/] row(s) for [yellow]{Markup.Escape(nativeScheme.Names[0])}[/]. Skipping relearn. Use [green]--relearn-baseline-mappings[/] to regenerate.");
                    return;
                }
            }
        }

        var metadata = await ReadTensorMetadataFromGgufAsync(nativeGgufPath, nativeGgufPath);
        var ggufTruth = metadata.TensorTypes
            .ToDictionary(x => x.Key, x => NormalizeQuantName(x.Value), StringComparer.Ordinal);

        var verification = BuildTruthMapWithVerification(
            logTruth: new Dictionary<string, string>(StringComparer.Ordinal),
            ggufTruth: ggufTruth,
            baselineName: "NATIVE");
        var truth = verification.TruthByTensor;

        var audit = _tensorGroupingAuditService.Audit(truth.Keys.ToList(), truth);

        if (audit.HasFatalIssues || verification.HasFatalIssues)
        {
            var diagnosticPath = await _tensorLearningDiagnosticWriter.WriteFailureAsync(
                baselineName: $"NATIVE_{nativeScheme.Names[0]}",
                schemeName: nativeScheme.Names[0],
                sourceKind: "NativeSource",
                sourceRepository: null,
                sourceFileName: Path.GetFileName(nativeGgufPath),
                truthByTensor: truth,
                audit: audit,
                verification: verification,
                ct: ct);

            AnsiConsole.MarkupLine(
                $"[red]Tensor group learning failed.[/] See diagnostic log: [yellow]{Markup.Escape(diagnosticPath)}[/]");
            throw new InvalidOperationException(
                $"Strict tensor-group learning validation failed for baseline 'NATIVE_{nativeScheme.Names[0]}'. " +
                $"Ambiguous={audit.Ambiguous.Count}, IllegalUnresolved={audit.IllegalUnresolved.Count}, " +
                $"AllowedBaseQuantFallback={audit.BaseQuantExceptions.Count}. " +
                $"No learned tensor mappings were persisted. Diagnostic log: {diagnosticPath}");
        }

        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct)
                                  ?? throw new InvalidOperationException(
                                      "Could not persist native-source learning because scoped AiModelHash row was missing.");

        var combo = await db.TensorCombos
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.BaseQuant == BaselineQuants.NativeSourceUniqueId &&
                                      x.Embeddings == 0 && x.LmHead == 0 && x.AttnQ == 0 && x.AttnKV == 0 &&
                                      x.AttnOutput == 0 && x.FfnUpGate == 0 && x.FfnDown == 0 &&
                                      x.MoeExperts == 0 && x.MoeRouter == 0, ct);

        if (combo == null)
            throw new InvalidOperationException(
                "Native-source benchmark TensorCombo is missing; benchmark base model first.");

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(
            db,
            exactAiModelHashId,
            createIfMissing: false,
            ct);

        var benchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == exactAiModelHashId &&
                        x.ImatrixDefinitionId == imatrixDefinitionId &&
                        x.TensorComboId == combo.Id)
            .OrderByDescending(x => x.Id)
            .Select(x => (Guid?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (!benchmarkId.HasValue)
            throw new InvalidOperationException(
                "Native-source benchmark row is missing; benchmark base model before native-source learning.");

        var rows = truth
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(x =>
            {
                var primaryGroup = audit.GroupedByTensor[x.Key].PrimaryGroup;

                return new LearnedBaselineTensorQuant
                {
                    Id = Guid.NewGuid(),
                    AiBenchmarkId = benchmarkId.Value,
                    AiModelHashId = scopedAiModelHashId,
                    BaselineQuantId = BaselineQuants.NativeSourceUniqueId,
                    TensorWeightSchemeId = nativeScheme.UniqueId,
                    TensorGroupId = primaryGroup?.UniqueId ?? UnknownTensorGroupId,
                    TensorName = x.Key,
                    FinalQuantType = x.Value.FinalQuantType
                };
            })
            .ToList();

        if (rows.Count == 0)
            throw new InvalidOperationException("Native-source learning produced no persistable rows.");

        await db.LearnedBaselineTensorQuants
            .Where(x => x.AiModelHashId == scopedAiModelHashId &&
                        x.BaselineQuantId == BaselineQuants.NativeSourceUniqueId &&
                        x.TensorWeightSchemeId == nativeScheme.UniqueId)
            .ExecuteDeleteAsync(ct);

        db.LearnedBaselineTensorQuants.AddRange(rows);
        await db.SaveChangesAsync(ct);

        await WriteLearningDiagnosticArtifactAsync(
            baselineName: $"NATIVE_{nativeScheme.Names[0]}",
            schemeName: nativeScheme.Names[0],
            truthByTensor: truth,
            grouped: audit.GroupedByTensor,
            allTensorNamesInModel: metadata.TensorNames,
            ambiguous: audit.Ambiguous,
            unresolved: audit.IllegalUnresolved.Select(x => x.TensorName).ToList());

        var sourcePrecision = nativeScheme.Names[0];
        var distribution = rows.GroupBy(x => x.FinalQuantType)
            .OrderByDescending(g => g.Count())
            .Select(g => $"{g.Key}:{g.Count()}")
            .ToList();

        AnsiConsole.MarkupLine(
            $"[green]Native-source learned truth:[/] precision={Markup.Escape(sourcePrecision)}, tensors={rows.Count}, unresolved={audit.IllegalUnresolved.Count}, ambiguous={audit.Ambiguous.Count}, baseFallback={audit.BaseQuantExceptions.Count}, dist={Markup.Escape($"[{string.Join(", ", distribution)}]")}");
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

        if (quant.BaseQuant.IsExternalRepositoryBaseline)
        {
            string nativeBasePath = await EnsureBaseModelFileAsync();
            await ValidateExternalBaselineTensorParityOrThrow(nativeBasePath, quantizedModelPath);
        }

        var tensorScheme = quant.BaseQuant.DefaultTensorScheme!;
        string logPath = report?.LogPath ?? (quantizedModelPath + ".quantize.log");
        var parsed = ParseQuantizeLogForTensorTypes(logPath);
        var ggufMetadata = await ReadTensorMetadataFromGgufAsync(quantizedModelPath, quantizedModelPath);
        var ggufTruth = ggufMetadata.TensorTypes
            .ToDictionary(x => x.Key, x => NormalizeQuantName(x.Value), StringComparer.Ordinal);

        if (parsed.Count == 0 && ggufTruth.Count == 0)
        {
            throw new InvalidOperationException(
                $"Strict tensor learning failed for baseline '{quant.BaseQuant.Names[0]}': no tensor truth could be read from either the quantize log or GGUF metadata. " +
                $"QuantizedModelPath={quantizedModelPath}; LogPath={logPath}");
        }

        var verification = BuildTruthMapWithVerification(parsed, ggufTruth, quant.BaseQuant.Names[0]);
        var truth = verification.TruthByTensor;
        if (truth.Count == 0)
            throw new InvalidOperationException(
                $"No verified tensor truth entries were available for baseline '{quant.BaseQuant.Names[0]}'.");

        var audit = _tensorGroupingAuditService.Audit(truth.Keys.ToList(), truth);
        if (audit.HasFatalIssues || verification.HasFatalIssues)
        {
            var diagnosticPath = await _tensorLearningDiagnosticWriter.WriteFailureAsync(
                baselineName: quant.BaseQuant.Names[0],
                schemeName: tensorScheme.Names[0],
                sourceKind: quant.BaseQuant.SourceKind.ToString(),
                sourceRepository: quant.BaseQuant.SourceRepository,
                sourceFileName: quant.BaseQuant.SourceFileName,
                truthByTensor: truth,
                audit: audit,
                verification: verification,
                ct: ct);

            AnsiConsole.MarkupLine(
                $"[red]Tensor group learning failed.[/] See diagnostic log: [yellow]{Markup.Escape(diagnosticPath)}[/]");
            throw new InvalidOperationException(
                $"Strict tensor-group learning validation failed for baseline '{quant.BaseQuant.Names[0]}'. " +
                $"Ambiguous={audit.Ambiguous.Count}, " +
                $"IllegalUnresolved={audit.IllegalUnresolved.Count}, " +
                $"AllowedBaseQuantFallback={audit.BaseQuantExceptions.Count}. " +
                $"No learned tensor mappings were persisted. " +
                $"Diagnostic log: {diagnosticPath}");
        }

        await using var db = new MagicQuantContext();

        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            throw new InvalidOperationException(
                "Unable to persist learned mappings because scoped AiModelHash row was not found.");

        var combo = await db.TensorCombos
            .AsNoTracking()
            .FirstAsync(x => x.BaseQuant == quant.BaseQuant.UniqueId &&
                             x.Embeddings == 0 && x.LmHead == 0 && x.AttnQ == 0 && x.AttnKV == 0 &&
                             x.AttnOutput == 0 && x.FfnUpGate == 0 && x.FfnDown == 0 && x.MoeExperts == 0 &&
                             x.MoeRouter == 0, ct);

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId =
            await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, exactAiModelHashId,
                createIfMissing: false, ct);

        var benchmarkId = await db.AiBenchmarks
            .Where(x => x.AiModelHashId == exactAiModelHashId && x.ImatrixDefinitionId == imatrixDefinitionId &&
                        x.TensorComboId == combo.Id)
            .OrderByDescending(x => x.Id)
            .Select(x => (Guid?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (!benchmarkId.HasValue)
            throw new InvalidOperationException(
                $"Unable to persist learned mappings because no AiBenchmark exists for baseline '{quant.BaseQuant.Names[0]}'.");

        var rows = truth
            .OrderBy(x => x.Key, StringComparer.Ordinal)
            .Select(kv =>
            {
                var match = audit.GroupedByTensor[kv.Key];

                return new LearnedBaselineTensorQuant
                {
                    Id = Guid.NewGuid(),
                    AiBenchmarkId = benchmarkId.Value,
                    AiModelHashId = scopedAiModelHashId.Value,
                    BaselineQuantId = quant.BaseQuant.UniqueId,
                    TensorWeightSchemeId = tensorScheme.UniqueId,
                    TensorGroupId = match.PrimaryGroup?.UniqueId ?? UnknownTensorGroupId,
                    BaselineCanonicalKey = quant.BaseQuant.CanonicalKey,
                    BaselineSourceKind = quant.BaseQuant.SourceKind,
                    BaselineSourceRepository = quant.BaseQuant.SourceRepository,
                    BaselineSourceFileName = quant.BaseQuant.SourceFileName,
                    TensorName = kv.Key,
                    FinalQuantType = kv.Value.FinalQuantType
                };
            })
            .ToList();

        if (rows.Count == 0)
            throw new InvalidOperationException(
                $"Learning baseline '{quant.BaseQuant.Names[0]}' produced no persistable rows.");

        await db.LearnedBaselineTensorQuants
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value &&
                        x.BaselineCanonicalKey == quant.BaseQuant.CanonicalKey &&
                        x.TensorWeightSchemeId == tensorScheme.UniqueId)
            .ExecuteDeleteAsync(ct);

        db.LearnedBaselineTensorQuants.AddRange(rows);
        await db.SaveChangesAsync(ct);

        await WriteLearningDiagnosticArtifactAsync(
            baselineName: quant.BaseQuant.Names[0],
            schemeName: tensorScheme.Names[0],
            truthByTensor: truth,
            grouped: audit.GroupedByTensor,
            allTensorNamesInModel: ggufMetadata.TensorNames,
            ambiguous: audit.Ambiguous,
            unresolved: audit.IllegalUnresolved.Select(x => x.TensorName).ToList());

        AnsiConsole.MarkupLine(
            $"[green]Learned baseline tensor mapping persisted:[/] [cyan]{rows.Count:N0}[/] row(s) for [yellow]{Markup.Escape(quant.BaseQuant.Names[0])}[/].");
    }

    private Dictionary<string, string> ParseQuantizeLogForTensorTypes(string logPath)
    {
        if (!File.Exists(logPath))
        {
            AnsiConsole.MarkupLine(
                $"[red]WARNING:[/] quantization log does not exist, cannot learn tensor mapping: {Markup.Escape(logPath)}");
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

    private TensorTruthVerificationResult BuildTruthMapWithVerification(
        IReadOnlyDictionary<string, string> logTruth,
        IReadOnlyDictionary<string, string> ggufTruth,
        string baselineName)
    {
        var allNames = logTruth.Keys
            .Concat(ggufTruth.Keys)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        var result = new Dictionary<string, LearnedTensorTruth>(StringComparer.Ordinal);
        var hardMismatches = new List<TensorTruthMismatch>();
        var softMismatches = new List<TensorTruthMismatch>();
        var logOnly = new List<string>();

        foreach (var name in allNames)
        {
            var inLog = logTruth.TryGetValue(name, out var logType);
            var inGguf = ggufTruth.TryGetValue(name, out var ggufType);

            if (inLog && inGguf)
            {
                if (string.Equals(logType, ggufType, StringComparison.OrdinalIgnoreCase))
                {
                    result[name] = new LearnedTensorTruth(name, ggufType!, LearningSource.Both);
                }
                else
                {
                    // GGUF is the final artifact truth. The log is secondary evidence only.
                    // A log/GGUF disagreement is useful diagnostic information, but it is
                    // not fatal as long as GGUF truth exists.
                    result[name] = new LearnedTensorTruth(name, ggufType!, LearningSource.BothWithMismatch);

                    softMismatches.Add(new TensorTruthMismatch
                    {
                        TensorName = name,
                        LogQuantType = logType!,
                        GgufQuantType = ggufType!,
                        IsHighSeverity = IsHighSeverityMismatch(logType!, ggufType!)
                    });
                }
            }
            else if (inGguf)
            {
                result[name] = new LearnedTensorTruth(name, ggufType!, LearningSource.GgufOnly);
            }
            else if (inLog)
            {
                // Log-only entries are not reliable enough to learn from because there is no
                // final GGUF artifact truth confirming them.
                logOnly.Add($"{name}:{logType}");
            }
        }

        if (softMismatches.Count > 0)
        {
            var highSeverityCount = softMismatches.Count(x => x.IsHighSeverity);
            var lowSeverityCount = softMismatches.Count - highSeverityCount;

            var severitySummary = highSeverityCount > 0 && lowSeverityCount > 0
                ? $"{highSeverityCount} high-severity, {lowSeverityCount} low-severity"
                : highSeverityCount > 0
                    ? $"{highSeverityCount} high-severity"
                    : $"{lowSeverityCount} low-severity";

            AnsiConsole.MarkupLine(
                $"[yellow]GGUF/log mismatch:[/] Baseline [yellow]{Markup.Escape(baselineName)}[/] had {softMismatches.Count} tensor type disagreement(s) ({severitySummary}). GGUF artifact truth was used.");

            AnsiConsole.MarkupLine(
                $"[grey]Examples: {Markup.Escape(string.Join(" | ", softMismatches.Take(6).Select(x => $"{x.TensorName}: log={x.LogQuantType} gguf={x.GgufQuantType}")))}[/]");
        }

        if (logOnly.Count > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]GGUF/log mismatch:[/] Baseline [yellow]{Markup.Escape(baselineName)}[/] produced {logOnly.Count} log-only tensor mapping(s) with no GGUF artifact truth. They were ignored.");

            AnsiConsole.MarkupLine(
                $"[grey]Examples: {Markup.Escape(string.Join(" | ", logOnly.Take(6)))}[/]");
        }

        return new TensorTruthVerificationResult
        {
            TruthByTensor = result,

            // Deliberately empty for log-vs-GGUF disagreements where GGUF truth exists.
            // GGUF wins, so these are diagnostics, not fatal validation failures.
            HardMismatches = hardMismatches,

            SoftMismatches = softMismatches,
            LogOnly = logOnly
        };
    }

    private static bool IsHighSeverityMismatch(string logType, string ggufType)
    {
        bool logHighPrecision = IsHighPrecisionType(logType);
        bool ggufHighPrecision = IsHighPrecisionType(ggufType);
        return logHighPrecision != ggufHighPrecision;
    }

    private async Task WriteLearningDiagnosticArtifactAsync(
        string baselineName,
        string schemeName,
        IReadOnlyDictionary<string, LearnedTensorTruth> truthByTensor,
        IReadOnlyDictionary<string, TensorGroupingResult> grouped,
        IReadOnlyCollection<string> allTensorNamesInModel,
        IReadOnlyCollection<TensorGroupingAuditIssue> ambiguous,
        IReadOnlyCollection<string> unresolved)
    {
        var summaries = new List<object>();
        var severeCoverageIssues = new List<string>();

        foreach (var group in TReg.All.OrderBy(x => x.UniqueId))
        {
            var expected = allTensorNamesInModel
                .Where(x => group.Tensors.Any(p => Regex.IsMatch(x, $"^{p}$")))
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();

            var learned = truthByTensor
                .Where(x => grouped.TryGetValue(x.Key, out var g) && g.PrimaryGroup?.UniqueId == group.UniqueId)
                .Select(x => x.Key)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();

            var unmatched = expected.Except(learned, StringComparer.Ordinal).Take(20).ToList();
            var unexpected = learned.Except(expected, StringComparer.Ordinal).Take(20).ToList();

            var distribution = truthByTensor
                .Where(x => learned.Contains(x.Key, StringComparer.Ordinal))
                .GroupBy(x => x.Value.FinalQuantType)
                .OrderByDescending(g => g.Count())
                .ToDictionary(g => g.Key, g => g.Count());

            var sourceCounts = truthByTensor
                .Where(x => learned.Contains(x.Key, StringComparer.Ordinal))
                .GroupBy(x => x.Value.Source.ToString())
                .ToDictionary(g => g.Key, g => g.Count());

            summaries.Add(new
            {
                Group = group.Name,
                ExpectedTensorCount = expected.Count,
                LearnedTensorCount = learned.Count,
                UnmatchedExpected = unmatched,
                UnexpectedLearned = unexpected,
                Ambiguous = ambiguous.Where(x => x.MatchedGroups.Contains(group.Name)).Select(x => x.TensorName)
                    .Take(20).ToList(),
                QuantDistribution = distribution,
                SourceDistribution = sourceCounts
            });

            var distShort = distribution.Count == 0
                ? "none"
                : string.Join(", ", distribution.Select(kv => $"{kv.Key}:{kv.Value}"));

            var srcShort = sourceCounts.Count == 0
                ? "none"
                : string.Join(", ", sourceCounts.Select(kv => $"{kv.Key}:{kv.Value}"));

            var label = $"[learn:{baselineName}:{group.Name}]";

            AnsiConsole.MarkupLine(
                $"[grey]{Markup.Escape(label)} " +
                $"expected={expected.Count} " +
                $"learned={learned.Count} " +
                $"unmatched={unmatched.Count} " +
                $"ambiguous={ambiguous.Count(x => x.MatchedGroups.Contains(group.Name))} " +
                $"dist={Markup.Escape($"[{distShort}]")} " +
                $"src={Markup.Escape($"[{srcShort}]")}[/]");

            if (expected.Count > 0 && unmatched.Count > 0)
            {
                severeCoverageIssues.Add(
                    $"{group.Name}: expected={expected.Count} learned={learned.Count} unmatched={unmatched.Count}");
            }
        }

        var artifact = new
        {
            Baseline = baselineName,
            Scheme = schemeName,
            TotalTruthTensors = truthByTensor.Count,
            UnresolvedTensorCount = unresolved.Count,
            AmbiguousTensorCount = ambiguous.Count,
            GeneratedUtc = DateTime.UtcNow,
            Groups = summaries
        };

        string debugDir = Path.Combine(_benchDir, "_learning_debug");
        Directory.CreateDirectory(debugDir);
        string path = Path.Combine(debugDir, $"{baselineName}_{schemeName}_learned_map.json");
        await File.WriteAllTextAsync(path,
            JsonSerializer.Serialize(artifact, new JsonSerializerOptions { WriteIndented = true }));

        AnsiConsole.MarkupLine(
            $"[grey]Learned mapping diagnostic written:[/] {Markup.Escape(path)}");

        if (severeCoverageIssues.Count > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]WARNING:[/] Baseline learning coverage was incomplete for {severeCoverageIssues.Count} group(s): " +
                $"{Markup.Escape(string.Join(" | ", severeCoverageIssues.Take(8)))}");
        }
    }

    private static TensorWeightScheme? TryResolveBaseTensorScheme(BaselineQuants baseQuant)
    {
        if (baseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId)
            return TensorWeightScheme.GetCurrentNativePrecisionScheme();

        return baseQuant.DefaultTensorScheme;
    }

    private static HashSet<string> GetExpectedTensorNamesForGroup(
        TensorGroup group,
        IReadOnlyCollection<string> sourceTensorNames)
    {
        return sourceTensorNames
            .Where(x => group.Tensors.Any(p => Regex.IsMatch(x, $"^{p}$")))
            .ToHashSet(StringComparer.Ordinal);
    }


    private List<RequestedTensorOverride> BuildRequestedTensorOverrides(
        HybridQuant quant,
        IReadOnlyCollection<string> sourceTensorNames,
        IReadOnlyDictionary<string, string>? temporaryCarrierOverrides = null)
    {
        var result = new List<RequestedTensorOverride>();

        bool hasTemporaryCarrierOverrides = temporaryCarrierOverrides != null && temporaryCarrierOverrides.Count > 0;
        bool hasExplicitGroupOverrides = quant.Tensors != null && quant.Tensors.Count > 0;
        bool shouldApplyBaseCarrierBlanket = ShouldApplyLearnedBaseCarrierBlanket(quant, temporaryCarrierOverrides);

        if (!shouldApplyBaseCarrierBlanket)
            return result;

        var baseScheme = TryResolveBaseTensorScheme(quant.BaseQuant);
        var blanket = LoadBaseCarrierTensorMappingsOrThrow(
            quant: quant,
            temporaryCarrierOverrides: temporaryCarrierOverrides,
            requireFullCoverage: shouldApplyBaseCarrierBlanket);

        foreach (var kv in blanket.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            result.Add(new RequestedTensorOverride
            {
                GroupName = "base_carrier",
                TensorName = kv.Key,
                SchemeName = kv.Value
            });
        }

        if (!hasExplicitGroupOverrides)
            return result;

        foreach (var hybrid in quant.Tensors)
        {
            if (hybrid?.TGroup == null)
                continue;

            hybrid.ValidateOrThrow();

            if (hybrid.MaterializedTensorScheme.UniqueId == TensorWeightScheme.NULL.UniqueId)
                continue;

            var expectedForGroup = GetExpectedTensorNamesForGroup(hybrid.TGroup, sourceTensorNames);
            if (expectedForGroup.Count == 0)
                continue;

            switch (hybrid.OverrideMode)
            {
                case HybridTensorOverrideMode.ExactTensorScheme:
                {
                    var exactScheme = hybrid.ExactTensorScheme!;
                    if (!quant.BaseQuant.IsExternalRepositoryBaseline && baseScheme != null &&
                        exactScheme.UniqueId == baseScheme.UniqueId)
                        continue;

                    string schemeName = ResolveSchemeName(exactScheme);
                    foreach (var tensorName in expectedForGroup.OrderBy(x => x, StringComparer.Ordinal))
                    {
                        result.Add(new RequestedTensorOverride
                        {
                            GroupName = hybrid.TGroup.Name,
                            TensorName = tensorName,
                            SchemeName = schemeName
                        });
                    }

                    break;
                }

                case HybridTensorOverrideMode.LearnedBaselineCandidate:
                {
                    var sourceBaseline = hybrid.CandidateBaseline!;
                    var learned = TryLoadLearnedTensorMapping(
                        sourceBaseline: sourceBaseline,
                        targetGroup: hybrid.TGroup,
                        preferredSourceScheme: sourceBaseline.DefaultTensorScheme,
                        allowDominantFallback: false);

                    if (learned.Count == 0)
                        throw new InvalidOperationException(
                            $"Missing required learned baseline mapping for group '{hybrid.TGroup.Name}' + baseline '{sourceBaseline.Names[0]}'. Run with --relearn-baseline-mappings to regenerate.");

                    var learnedNames = learned.Keys.ToHashSet(StringComparer.Ordinal);
                    var missingExpected = expectedForGroup.Except(learnedNames).OrderBy(x => x).ToList();
                    var unexpectedLearned = learnedNames.Except(expectedForGroup).OrderBy(x => x).ToList();

                    if (missingExpected.Count > 0 || unexpectedLearned.Count > 0)
                    {
                        var missingText = missingExpected.Count == 0
                            ? "none"
                            : string.Join(", ", missingExpected.Take(15));
                        var unexpectedText = unexpectedLearned.Count == 0
                            ? "none"
                            : string.Join(", ", unexpectedLearned.Take(15));
                        throw new InvalidOperationException(
                            $"Learned mapping coverage mismatch for group '{hybrid.TGroup.Name}' + baseline '{sourceBaseline.Names[0]}'. Expected={expectedForGroup.Count}, Learned={learnedNames.Count}, Missing=[{missingText}], Unexpected=[{unexpectedText}].");
                    }

                    foreach (var kv in learned.OrderBy(x => x.Key, StringComparer.Ordinal))
                    {
                        result.Add(new RequestedTensorOverride
                        {
                            GroupName = hybrid.TGroup.Name,
                            TensorName = kv.Key,
                            SchemeName = kv.Value
                        });
                    }

                    break;
                }

                default:
                    throw new InvalidOperationException(
                        $"Hybrid tensor for group '{hybrid.TGroup.Name}' has unsupported override mode '{hybrid.OverrideMode}'.");
            }
        }

        return result;
    }

    private bool ShouldApplyLearnedBaseCarrierBlanket(
        HybridQuant quant,
        IReadOnlyDictionary<string, string>? temporaryCarrierOverrides = null)
    {
        bool hasTemporaryCarrierOverrides = temporaryCarrierOverrides != null && temporaryCarrierOverrides.Count > 0;
        bool hasExplicitGroupOverrides = quant.Tensors != null && quant.Tensors.Count > 0;

        return hasTemporaryCarrierOverrides ||
               quant.BaseQuant.IsExternalRepositoryBaseline ||
               hasExplicitGroupOverrides;
    }

    private Dictionary<string, string> LoadBaseCarrierTensorMappingsOrThrow(
        HybridQuant quant,
        IReadOnlyDictionary<string, string>? temporaryCarrierOverrides,
        bool requireFullCoverage)
    {
        var blanket = temporaryCarrierOverrides != null && temporaryCarrierOverrides.Count > 0
            ? new Dictionary<string, string>(temporaryCarrierOverrides, StringComparer.Ordinal)
            : TryLoadAllLearnedTensorMappings(
                canonicalBaselineKey: quant.BaseQuant.CanonicalKey,
                preferredSourceScheme: quant.BaseQuant.DefaultTensorScheme,
                allowDominantFallback: false);

        if (requireFullCoverage && blanket.Count == 0)
        {
            throw new InvalidOperationException(
                $"Missing full learned base-carrier mapping for baseline '{quant.BaseQuant.Names[0]}'. " +
                "Run with --relearn-baseline-mappings before applying learned tensor configurations.");
        }

        return blanket;
    }

    private Dictionary<string, string> TryLoadAllLearnedTensorMappings(
        string canonicalBaselineKey,
        TensorWeightScheme? preferredSourceScheme = null,
        bool allowDominantFallback = false)
    {
        using var db = new MagicQuantContext();
        var scopedAiModelHashId =
            ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db).GetAwaiter().GetResult();
        if (scopedAiModelHashId == null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var allRows = db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.BaselineCanonicalKey == canonicalBaselineKey)
            .OrderBy(x => x.TensorName)
            .ToList();

        if (allRows.Count == 0)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var rows = allRows;
        if (preferredSourceScheme != null)
        {
            var preferred = allRows.Where(x => x.TensorWeightSchemeId == preferredSourceScheme.UniqueId).ToList();
            if (preferred.Count > 0)
                rows = preferred;
            else if (!allowDominantFallback)
                return new Dictionary<string, string>(StringComparer.Ordinal);
        }

        if (rows.Select(x => x.TensorWeightSchemeId).Distinct().Count() > 1)
        {
            if (!allowDominantFallback)
                return new Dictionary<string, string>(StringComparer.Ordinal);

            var dominantSchemeId = rows.GroupBy(x => x.TensorWeightSchemeId)
                .OrderByDescending(g => g.Count())
                .ThenBy(g => g.Key)
                .Select(g => g.Key)
                .First();
            rows = rows.Where(x => x.TensorWeightSchemeId == dominantSchemeId).ToList();
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var row in rows)
        {
            var appliedSchemeName =
                NativePrecisionNormalization.NormalizeLearnedFinalQuantTypeForApplication(row.FinalQuantType);

            if (string.IsNullOrWhiteSpace(appliedSchemeName))
            {
                throw new InvalidOperationException(
                    $"Learned tensor mapping for tensor '{row.TensorName}' on baseline key '{canonicalBaselineKey}' " +
                    $"returned an empty normalized scheme name. Observed FinalQuantType='{row.FinalQuantType}'.");
            }

            result[row.TensorName] = appliedSchemeName;
        }

        return result;
    }

    private Dictionary<string, string> TryLoadLearnedTensorMapping(
        BaselineQuants sourceBaseline,
        TensorGroup targetGroup,
        TensorWeightScheme? preferredSourceScheme = null,
        bool allowDominantFallback = false)
    {
        using var db = new MagicQuantContext();

        var scopedAiModelHashId =
            ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db).GetAwaiter().GetResult();

        if (scopedAiModelHashId == null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var allRows = db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.BaselineCanonicalKey == sourceBaseline.CanonicalKey)
            .Where(x => x.TensorGroupId == targetGroup.UniqueId)
            .OrderBy(x => x.TensorName)
            .ToList();

        if (allRows.Count == 0)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        List<MQ.DB.Models.DbModels.LearnedBaselineTensorQuant> rows = allRows;

        if (preferredSourceScheme != null)
        {
            var preferred = allRows
                .Where(x => x.TensorWeightSchemeId == preferredSourceScheme.UniqueId)
                .ToList();

            if (preferred.Count > 0)
            {
                rows = preferred;
            }
            else if (!allowDominantFallback)
            {
                return new Dictionary<string, string>(StringComparer.Ordinal);
            }
        }

        if (rows.Count == 0)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        if (rows.Select(x => x.TensorWeightSchemeId).Distinct().Count() > 1)
        {
            if (!allowDominantFallback)
                return new Dictionary<string, string>(StringComparer.Ordinal);

            var dominantSchemeId = rows
                .GroupBy(x => x.TensorWeightSchemeId)
                .OrderByDescending(g => g.Count())
                .ThenBy(g => g.Key)
                .Select(g => g.Key)
                .First();

            rows = rows.Where(x => x.TensorWeightSchemeId == dominantSchemeId).ToList();
        }

        var result = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var row in rows)
        {
            var appliedSchemeName =
                NativePrecisionNormalization.NormalizeLearnedFinalQuantTypeForApplication(row.FinalQuantType);

            if (string.IsNullOrWhiteSpace(appliedSchemeName))
            {
                throw new InvalidOperationException(
                    $"Learned tensor mapping for tensor '{row.TensorName}' in group '{targetGroup.Name}' " +
                    $"returned an empty normalized scheme name. Observed FinalQuantType='{row.FinalQuantType}'.");
            }

            result[row.TensorName] = appliedSchemeName;
        }

        return result;
    }


    private List<ConcreteTensorOverride> ResolveConcreteTensorOverrides(
        IReadOnlyCollection<string> allTensorNames,
        List<RequestedTensorOverride> requestedOverrides)
    {
        if (requestedOverrides.Count == 0)
            return new List<ConcreteTensorOverride>();

        var nameSet = allTensorNames.ToHashSet(StringComparer.Ordinal);

        var missing = requestedOverrides
            .Where(x => !nameSet.Contains(x.TensorName))
            .ToList();

        if (missing.Count > 0)
        {
            throw new InvalidOperationException(
                $"Required learned tensor mappings were missing in source GGUF ({missing.Count} tensors). Examples: {string.Join(", ", missing.Take(10).Select(x => x.TensorName))}");
        }

        var lastWins = new Dictionary<string, RequestedTensorOverride>(StringComparer.Ordinal);
        foreach (var item in requestedOverrides)
            lastWins[item.TensorName] = item;

        return lastWins.Values
            .Select(x => new ConcreteTensorOverride
            {
                GroupName = x.GroupName,
                SchemeName = x.SchemeName,
                TensorName = x.TensorName
            })
            .OrderBy(x => x.TensorName, StringComparer.Ordinal)
            .ToList();
    }

    private async Task<GgufTensorReadResult> ReadTensorMetadataFromGgufAsync(string ggufPath, string outputFilePath)
    {
        string workingDir = Path.GetDirectoryName(outputFilePath)!;
        string unique = Guid.NewGuid().ToString("N");
        string payloadPath = Path.Combine(workingDir, $"read_gguf_tensors_{unique}.json");
        string resultPath = Path.Combine(workingDir, $"read_gguf_tensors_result_{unique}.json");
        string scriptPath = Path.Combine(workingDir, $"read_gguf_tensors_{unique}.py");

        try
        {
            await File.WriteAllTextAsync(payloadPath,
                JsonSerializer.Serialize(new { gguf_path = ggufPath, output_path = resultPath }));

            const string py = """
                              import json
                              import sys

                              payload_path = sys.argv[1]
                              with open(payload_path, "r", encoding="utf-8") as f:
                                  payload = json.load(f)

                              output_path = payload["output_path"]

                              def resolve_type_name(t):
                                  for attr in ["type_name", "tensor_type", "type"]:
                                      v = getattr(t, attr, None)
                                      if v is None:
                                          continue
                                      if hasattr(v, "name"):
                                          return str(v.name)
                                      return str(v)
                                  return "UNKNOWN"

                              try:
                                  import gguf
                                  reader = gguf.GGUFReader(payload["gguf_path"])
                                  tensor_names = [t.name for t in reader.tensors]
                                  tensor_types = {t.name: resolve_type_name(t) for t in reader.tensors}
                                  result = {"TensorNames": tensor_names, "TensorTypes": tensor_types}
                              except Exception as e:
                                  result = {"Error": str(e), "TensorNames": [], "TensorTypes": {}}

                              with open(output_path, "w", encoding="utf-8") as f:
                                  json.dump(result, f, indent=2)
                              """;

            await File.WriteAllTextAsync(scriptPath, py);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");

            var result = JsonSerializer.Deserialize<GgufTensorReadResult>(await File.ReadAllTextAsync(resultPath));
            if (result == null)
                throw new InvalidOperationException("Failed to parse GGUF tensor list result.");
            if (!string.IsNullOrWhiteSpace(result.Error))
                throw new InvalidOperationException($"Failed to read GGUF tensor names: {result.Error}");

            return result;
        }
        finally
        {
            if (File.Exists(payloadPath)) File.Delete(payloadPath);
            if (File.Exists(resultPath)) File.Delete(resultPath);
            if (File.Exists(scriptPath)) File.Delete(scriptPath);
        }
    }


    private async Task<Dictionary<string, string>> BuildIsolationDeduplicationPlanAsync(
        IReadOnlyCollection<RequiredSamplePlan> plans,
        CancellationToken ct)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        var firstBySignature = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var plan in plans)
        {
            string? signature = await TryBuildIsolationEquivalenceKeyAsync(plan, ct);
            if (string.IsNullOrWhiteSpace(signature))
            {
                result[plan.Key] = plan.Key;
                continue;
            }

            if (!firstBySignature.TryGetValue(signature, out var firstKey))
            {
                firstBySignature[signature] = plan.Key;
                result[plan.Key] = plan.Key;
                continue;
            }

            result[plan.Key] = firstKey;
            AnsiConsole.MarkupLine(
                $"[grey]Isolation dedupe planned:[/] {Markup.Escape(plan.Key)} -> {Markup.Escape(firstKey)}");
        }

        return result;
    }

    private async Task<string?> TryBuildIsolationEquivalenceKeyAsync(
        RequiredSamplePlan plan,
        CancellationToken ct)
    {
        if (plan.Kind != RequiredSampleKind.GroupIsolationProbe &&
            plan.Kind != RequiredSampleKind.GroupIsolationContinuation)
            return null;

        if (plan.TargetGroupId == null || string.IsNullOrWhiteSpace(plan.TestedCandidateCanonicalKey))
            return null;

        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            return null;

        var rows = await db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.BaselineCanonicalKey == plan.TestedCandidateCanonicalKey)
            .Where(x => x.TensorGroupId == plan.TargetGroupId.Value)
            .OrderBy(x => x.TensorName)
            .Select(x => new { x.TensorName, x.FinalQuantType })
            .ToListAsync(ct);

        if (rows.Count == 0)
            return null;

        var sb = new StringBuilder();
        sb.Append("group=").Append(plan.TargetGroupId.Value).Append('|');
        foreach (var row in rows)
        {
            sb.Append(row.TensorName).Append('=')
                .Append(NormalizeLearnedIsolationQuantToken(row.FinalQuantType))
                .Append(';');
        }

        return sb.ToString();
    }

    private static string NormalizeLearnedIsolationQuantToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return string.Empty;

        return value.Trim().Replace(" ", string.Empty).Replace("-", "_").ToUpperInvariant();
    }

    private async Task<bool> CloneEquivalentIsolationBenchmarkAsync(
        RequiredSamplePlan sourcePlan,
        RequiredSamplePlan duplicatePlan,
        CancellationToken ct)
    {
        var sourceIdentity = await ResolveBenchmarkIdentityAsync(sourcePlan.Quant, ct);
        if (sourceIdentity.BenchmarkId == null)
            return false;

        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ResolveCurrentScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            return false;

        var sourceBench = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .FirstOrDefaultAsync(x => x.Id == sourceIdentity.BenchmarkId.Value, ct);

        if (sourceBench == null)
            return false;

        var duplicateLookup = BuildTensorLookup(duplicatePlan.Quant);
        var duplicateCombo = await db.TensorCombos.FirstOrDefaultAsync(x =>
            x.BaseQuant == duplicateLookup.BaseQuant &&
            x.Embeddings == duplicateLookup.Embeddings &&
            x.LmHead == duplicateLookup.LmHead &&
            x.AttnQ == duplicateLookup.AttnQ &&
            x.AttnKV == duplicateLookup.AttnKV &&
            x.AttnOutput == duplicateLookup.AttnOutput &&
            x.FfnUpGate == duplicateLookup.FfnUpGate &&
            x.FfnDown == duplicateLookup.FfnDown &&
            x.MoeExperts == duplicateLookup.MoeExperts &&
            x.MoeRouter == duplicateLookup.MoeRouter, ct);

        if (duplicateCombo == null)
        {
            duplicateCombo = new TensorCombo(duplicateLookup);
            db.TensorCombos.Add(duplicateCombo);
            await db.SaveChangesAsync(ct);
        }

        var exactAiModelHashId = await ResolveCurrentExactAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(
            db,
            exactAiModelHashId,
            createIfMissing: true,
            ct);

        var existing = await db.AiBenchmarks
            .FirstOrDefaultAsync(x => x.AiModelHashId == exactAiModelHashId &&
                                      x.ImatrixDefinitionId == imatrixDefinitionId &&
                                      x.TensorComboId == duplicateCombo.Id, ct);

        if (existing != null)
            return true;

        var clonedBenchmark = new AiBenchmark
        {
            Id = Guid.NewGuid(),
            Ngl = sourceBench.Ngl,
            SizeBytes = sourceBench.SizeBytes,
            TokensPerSecond = sourceBench.TokensPerSecond,
            TensorComboId = duplicateCombo.Id,
            AiModelHashId = exactAiModelHashId,
            ImatrixDefinitionId = imatrixDefinitionId
        };
        db.AiBenchmarks.Add(clonedBenchmark);

        var clonedCategories = sourceBench.CategorBenchmarks
            .Select(x => new CategoryBenchmark
            {
                Id = Guid.NewGuid(),
                AiBenchmarkId = clonedBenchmark.Id,
                Category = x.Category,
                Kld = x.Kld,
                Ppl = x.Ppl,
                PplError = x.PplError
            })
            .ToList();
        db.AddRange(clonedCategories);

        db.QuantizationRuns.Add(new QuantizationRun
        {
            Id = Guid.NewGuid(),
            AiModelHashId = exactAiModelHashId,
            ImatrixDefinitionId = imatrixDefinitionId,
            TensorComboId = duplicateCombo.Id,
            AiBenchmarkId = clonedBenchmark.Id,
            StartedUtc = DateTime.UtcNow,
            CompletedUtc = DateTime.UtcNow,
            DurationMs = 0,
            Succeeded = true,
            Error = $"Cloned from equivalent isolation benchmark '{sourcePlan.Key}'.",
            OutputModelPath = null
        });

        foreach (var cat in clonedCategories)
        {
            db.BenchmarkRuns.Add(new BenchmarkRun
            {
                Id = Guid.NewGuid(),
                AiModelHashId = exactAiModelHashId,
                ImatrixDefinitionId = imatrixDefinitionId,
                TensorComboId = duplicateCombo.Id,
                AiBenchmarkId = clonedBenchmark.Id,
                CategoryBenchmarkId = cat.Id,
                Category = cat.Category,
                StartedUtc = DateTime.UtcNow,
                CompletedUtc = DateTime.UtcNow,
                DurationMs = 0,
                Succeeded = true,
                Error = $"Cloned from equivalent isolation benchmark '{sourcePlan.Key}'."
            });
        }

        await db.SaveChangesAsync(ct);

        AnsiConsole.MarkupLine(
            $"[green]Isolation dedupe clone:[/] {Markup.Escape(duplicatePlan.Key)} reused benchmark data from {Markup.Escape(sourcePlan.Key)}");
        return true;
    }

    // ----------------------------------------------------------------
    // Internal DTOs
    // ----------------------------------------------------------------

    private static readonly Regex TensorLogLineRegex = new(
        @"\]\s+(?<tensor>[^\s]+)\s+-\s+\[[^\]]+\],\s+type\s*=\s*(?<type>[^\s,]+)(?:.*?converting to\s+(?<convert>[^\s,]+))?",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static string NormalizeQuantName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "UNKNOWN";

        string token = CanonicalizeQuantToken(value);

        if (QuantAliasLookup.Value.TryGetValue(token, out var canonical))
            return canonical;

        return token;
    }

    private static Dictionary<string, string> BuildQuantAliasLookup()
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.Names.IsDefaultOrEmpty)
                continue;

            string canonical = scheme.Names[0];

            foreach (var alias in scheme.Names)
            {
                string token = CanonicalizeQuantToken(alias);

                if (!map.TryAdd(token, canonical))
                {
                    if (!string.Equals(map[token], canonical, StringComparison.Ordinal))
                    {
                        throw new InvalidOperationException(
                            $"Quant alias collision detected for token '{token}'. Existing='{map[token]}', New='{canonical}'.");
                    }
                }
            }
        }

        return map;
    }

    private static string CanonicalizeQuantToken(string value)
    {
        return value
            .Trim()
            .Replace("-", "_")
            .Replace(" ", string.Empty)
            .ToUpperInvariant();
    }

    private static bool IsHighPrecisionType(string value)
    {
        string normalized = NormalizeQuantName(value);
        return normalized is "BF16" or "F16" or "F32";
    }

    private sealed class PreparedExternalBaselineBuild
    {
        public string BenchmarkModelPath { get; set; } = string.Empty;
        public string? DownloadedExternalModelPath { get; set; }
        public Dictionary<string, LearnedTensorTruth>? TruthByTensor { get; set; }
        public IReadOnlyDictionary<string, TensorGroupingResult>? GroupedByTensor { get; set; }
        public IReadOnlyCollection<string>? AllTensorNamesInDownloadedArtifact { get; set; }
        public IReadOnlyList<TensorGroupingAuditIssue>? AmbiguousGroupingRows { get; set; }
        public IReadOnlyList<string>? UnresolvedTensorNames { get; set; }
        public IReadOnlyList<TensorGroupingAuditIssue>? BaseQuantExceptionRows { get; set; }
        public TensorTruthVerificationResult? Verification { get; set; }
        public bool HasPreparedLearningTruth { get; set; }
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

    private sealed class GgufTensorReadResult
    {
        public string? Error { get; set; }
        public List<string> TensorNames { get; set; } = new();
        public Dictionary<string, string> TensorTypes { get; set; } = new(StringComparer.Ordinal);
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

        return s.Names[0];
    }

    public string GenerateHybridName(HybridQuant quant)
    {
        string modelName = new DirectoryInfo(Cache.ModelDirectory!).Name;
        string baseName = ResolveBaseName(quant.BaseQuant);

        var effectiveTensors = quant.Tensors?
            .Where(t => t?.TGroup != null)
            .ToList();

        if (effectiveTensors == null || effectiveTensors.Count == 0)
            return $"{modelName}-{baseName}";

        var grouped = effectiveTensors
            .Select(t =>
            {
                t.ValidateOrThrow();

                string typeName = t.OverrideMode switch
                {
                    HybridTensorOverrideMode.LearnedBaselineCandidate => t.CandidateBaseline!.Names[0],
                    HybridTensorOverrideMode.ExactTensorScheme => ResolveSchemeName(t.ExactTensorScheme!),
                    _ => throw new InvalidOperationException($"Unknown override mode '{t.OverrideMode}'.")
                };

                return new
                {
                    Type = typeName,
                    Code = t.TGroup.ShortCode
                };
            })
            .GroupBy(x => x.Type)
            .Select(g => new
            {
                Type = g.Key,
                Codes = g.Select(x => x.Code)
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

            if (Cache.VerboseProcessOutput)
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