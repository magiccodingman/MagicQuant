using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
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

public class BenchmarkService
{
    private readonly LlamaBinaries _bins;
    public readonly PythonManager _pyManager;

    private static readonly string[] BaseDomains = { "general", "code", "math" };
    private static readonly string[] SampleDomains = { "general" };

    private static readonly string[] OomMarkers =
    {
        "out of memory",
        "cudamalloc failed",
        "unable to allocate cuda",
        "try reducing --n-gpu-layers",
        "cannot fulfill margin",
        "failed to fit params",
        "cuda error"
    };

    private static readonly int[] NglCandidates = { 35, 30, 24, 20, 16, 12, 8, 4 };

    // ----------------------------------------------------------------
    // Static execution-plan state
    // ----------------------------------------------------------------

    private static readonly SemaphoreSlim PlanInitLock = new(1, 1);
    private static readonly object SlotSync = new();

    private static BenchmarkExecutionPlan? _currentPlan;
    private static string _currentPlanQuantizationKey = "Q8_0";
    private static Queue<BenchmarkSlot> _availableSlots = new();
    private static SemaphoreSlim? _slotSemaphore;

    // ----------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------

    public BenchmarkService(PythonManager pyManager)
    {
        _bins = new LlamaBinaries(Cache.LlamaRoot);
        _bins.Validate();
        _pyManager = pyManager;
    }

    // ----------------------------------------------------------------
    // Execution-plan discovery
    // ----------------------------------------------------------------

    public async Task EnsureExecutionPlanAsync(
        string q8ModelPath,
        int discoveryTokenTarget = 8192,
        string quantizationKey = "Q8_0",
        bool forceRediscovery = false,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(q8ModelPath))
            throw new ArgumentException("Q8 model path was null or empty.", nameof(q8ModelPath));
        if (string.IsNullOrWhiteSpace(quantizationKey))
            throw new ArgumentException("Quantization key was null or empty.", nameof(quantizationKey));

        string normalizedPath = Path.GetFullPath(q8ModelPath);
        string normalizedQuantizationKey = quantizationKey.Trim().ToUpperInvariant();

        if (!forceRediscovery &&
            _currentPlan != null &&
            string.Equals(_currentPlanQuantizationKey, normalizedQuantizationKey, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(_currentPlan.PlanModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        await PlanInitLock.WaitAsync(ct);
        try
        {
            if (!forceRediscovery &&
                _currentPlan != null &&
                string.Equals(_currentPlanQuantizationKey, normalizedQuantizationKey, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(_currentPlan.PlanModelPath, normalizedPath, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            var cacheKey = BuildExecutionPlanCacheKey(normalizedPath, discoveryTokenTarget, normalizedQuantizationKey);

            BenchmarkExecutionPlan? plan = null;
            if (!forceRediscovery)
            {
                plan = await TryLoadCachedExecutionPlanAsync(cacheKey, ct);
                if (plan != null)
                    AnsiConsole.MarkupLine("[green]Loaded benchmark execution plan from SQLite cache.[/]");
            }

            if (plan == null)
            {
                plan = await BuildExecutionPlanAsync(normalizedPath, discoveryTokenTarget, ct);
                await UpsertCachedExecutionPlanAsync(cacheKey, plan, ct);
            }

            lock (SlotSync)
            {
                _currentPlan = plan;
                _currentPlanQuantizationKey = normalizedQuantizationKey;
                _availableSlots = new Queue<BenchmarkSlot>(plan.Slots);
                _slotSemaphore = new SemaphoreSlim(plan.Slots.Count, plan.Slots.Count);
            }

            AnsiConsole.Write(new Rule("[yellow]Benchmark Execution Plan[/]") { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[green]Static ngl:[/] [cyan]{plan.StaticNgl}[/]");
            AnsiConsole.MarkupLine($"[green]Uses GPU:[/] [cyan]{plan.UsesGpu}[/]");
            AnsiConsole.MarkupLine($"[green]GPU group size:[/] [cyan]{plan.GroupSize}[/]");
            AnsiConsole.MarkupLine($"[green]Parallel benchmark slots:[/] [cyan]{plan.Slots.Count}[/]");
            AnsiConsole.MarkupLine($"[green]Quantization key:[/] [cyan]{Markup.Escape(normalizedQuantizationKey)}[/]");

            foreach (var slot in plan.Slots)
            {
                AnsiConsole.MarkupLine($"  [grey]Slot {slot.SlotId}:[/] {Markup.Escape(slot.DisplayName)}");
            }
        }
        finally
        {
            PlanInitLock.Release();
        }
    }

    public async Task ClampStaticNglWithBaseModelAsync(
        string baseModelPath,
        int discoveryTokenTarget = 8192,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(baseModelPath))
            throw new ArgumentException("Base model path was null or empty.", nameof(baseModelPath));

        if (_currentPlan == null)
            throw new InvalidOperationException(
                "Benchmark execution plan has not been initialized. Call EnsureExecutionPlanAsync() first.");

        if (!_currentPlan.UsesGpu)
            return;

        await PlanInitLock.WaitAsync(ct);
        try
        {
            if (_currentPlan == null || !_currentPlan.UsesGpu)
                return;

            var slot = _currentPlan.Slots[0];

            string probeRoot = Path.Combine(Cache.ModelMagicQuantDirectory!, "_benchmark_plan_probe_base");
            Directory.CreateDirectory(probeRoot);

            string probeCorpusDir = Path.Combine(probeRoot, "_ppl_corpora");
            Directory.CreateDirectory(probeCorpusDir);

            string corpusPath = Path.Combine(probeCorpusDir, "ppl_corpus_general.txt");
            await PreparePplCorpusAsync("general", corpusPath, discoveryTokenTarget);

            int startingNgl = _currentPlan.StaticNgl;
            int? chosen = null;

            AnsiConsole.Write(new Rule("[yellow]Clamping Static ngl With Base Model[/]")
                { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[grey]Base model:[/] {Markup.Escape(baseModelPath)}");
            AnsiConsole.MarkupLine($"[grey]Starting from Q8-discovered ngl:[/] [cyan]{startingNgl}[/]");

            foreach (int ngl in NglCandidates.Where(n => n <= startingNgl).OrderByDescending(n => n))
            {
                ct.ThrowIfCancellationRequested();

                AnsiConsole.MarkupLine($"[grey]Base clamp probe:[/] [cyan]ngl={ngl}[/]");

                bool benchOk = await ProbeLlamaBenchAtFixedNglAsync(baseModelPath, slot, ngl, probeRoot);
                if (!benchOk)
                {
                    AnsiConsole.MarkupLine($"[grey]  llama-bench failed at ngl={ngl}[/]");
                    continue;
                }

                bool pplOk = await ProbePerplexityAtFixedNglAsync(baseModelPath, slot, ngl, corpusPath, probeRoot);
                if (!pplOk)
                {
                    AnsiConsole.MarkupLine($"[grey]  perplexity failed at ngl={ngl}[/]");
                    continue;
                }

                chosen = ngl;
                break;
            }

            if (!chosen.HasValue)
            {
                AnsiConsole.MarkupLine(
                    "[yellow]Base model could not sustain the discovered GPU ngl. Falling back to a CPU benchmark plan.[/]");

                var cpuPlan = BenchmarkExecutionPlan.CreateCpuPlan(_currentPlan.PlanModelPath);

                lock (SlotSync)
                {
                    _currentPlan = cpuPlan;
                    _availableSlots = new Queue<BenchmarkSlot>(cpuPlan.Slots);
                    _slotSemaphore = new SemaphoreSlim(cpuPlan.Slots.Count, cpuPlan.Slots.Count);
                }

                var cacheKeyCpu = BuildExecutionPlanCacheKey(
                    _currentPlan.PlanModelPath,
                    discoveryTokenTarget,
                    _currentPlanQuantizationKey);
                await UpsertCachedExecutionPlanAsync(cacheKeyCpu, _currentPlan, ct);

                return;
            }

            if (chosen.Value != _currentPlan.StaticNgl)
            {
                var updated = new BenchmarkExecutionPlan(
                    planModelPath: _currentPlan.PlanModelPath,
                    staticNgl: chosen.Value,
                    usesGpu: _currentPlan.UsesGpu,
                    groupSize: _currentPlan.GroupSize,
                    slots: _currentPlan.Slots);

                lock (SlotSync)
                {
                    _currentPlan = updated;
                    _availableSlots = new Queue<BenchmarkSlot>(updated.Slots);
                    _slotSemaphore = new SemaphoreSlim(updated.Slots.Count, updated.Slots.Count);
                }
            }

            var cacheKey = BuildExecutionPlanCacheKey(
                _currentPlan.PlanModelPath,
                discoveryTokenTarget,
                _currentPlanQuantizationKey);
            await UpsertCachedExecutionPlanAsync(cacheKey, _currentPlan, ct);

            AnsiConsole.MarkupLine($"[green]Base-model clamped static ngl:[/] [cyan]{chosen.Value}[/]");
        }
        finally
        {
            PlanInitLock.Release();
        }
    }

    private async Task<BenchmarkExecutionPlan> BuildExecutionPlanAsync(
        string q8ModelPath,
        int discoveryTokenTarget,
        CancellationToken ct)
    {
        int gpuCount = Cache.SysInfo?.GpuInfo?
            .Count(x => x.GpuVendor != GpuVendor.Cpu && x.GpuVendor != GpuVendor.Unknown) ?? 0;

        if (gpuCount <= 0)
        {
            return BenchmarkExecutionPlan.CreateCpuPlan(q8ModelPath);
        }

        var allGpuIndices = Enumerable.Range(0, gpuCount).ToArray();
        var allGpuSlot = new BenchmarkSlot(0, allGpuIndices);

        string probeRoot = Path.Combine(Cache.ModelMagicQuantDirectory!, "_benchmark_plan_probe");
        Directory.CreateDirectory(probeRoot);

        int? targetNgl = await ProbeHighestStableNglAsync(
            q8ModelPath,
            allGpuSlot,
            probeRoot,
            discoveryTokenTarget,
            ct);

        if (!targetNgl.HasValue || targetNgl.Value <= 0)
        {
            AnsiConsole.MarkupLine(
                "[yellow]Q8 discovery could not establish a stable GPU ngl. Falling back to a single CPU slot.[/]");
            return BenchmarkExecutionPlan.CreateCpuPlan(q8ModelPath);
        }

        foreach (var groupSize in GetCandidateGroupSizes(gpuCount))
        {
            var groups = BuildContiguousGroups(allGpuIndices, groupSize);
            var slots = new List<BenchmarkSlot>();

            bool allGroupsPass = true;
            for (int i = 0; i < groups.Count; i++)
            {
                var slot = new BenchmarkSlot(i, groups[i]);

                bool ok = await ValidateSlotForFixedPlanAsync(
                    q8ModelPath,
                    slot,
                    targetNgl.Value,
                    probeRoot,
                    discoveryTokenTarget,
                    ct);

                if (!ok)
                {
                    allGroupsPass = false;
                    break;
                }

                slots.Add(slot);
            }

            if (allGroupsPass && slots.Count > 0)
            {
                return new BenchmarkExecutionPlan(
                    planModelPath: q8ModelPath,
                    staticNgl: targetNgl.Value,
                    usesGpu: true,
                    groupSize: groupSize,
                    slots: slots);
            }
        }

        return new BenchmarkExecutionPlan(
            planModelPath: q8ModelPath,
            staticNgl: targetNgl.Value,
            usesGpu: true,
            groupSize: gpuCount,
            slots: new List<BenchmarkSlot> { allGpuSlot });
    }

    private async Task<BenchmarkExecutionPlan?> TryLoadCachedExecutionPlanAsync(
        ExecutionPlanCacheKey key,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var aiModelHashId = await GetOrCreateAiModelHashIdAsync(db, ct);

        var row = await db.ExecutionPlanProbeCaches
            .AsNoTracking()
            .FirstOrDefaultAsync(x =>
                x.AiModelHashId == aiModelHashId &&
                x.HardwareFingerprint == key.HardwareFingerprint &&
                x.QuantizedModelFingerprint == key.QuantizedModelFingerprint &&
                x.QuantizationKey == key.QuantizationKey &&
                x.DiscoveryTokenTarget == key.DiscoveryTokenTarget, ct);

        if (row == null)
            return null;

        List<int[]> slotDevices;
        try
        {
            slotDevices = JsonSerializer.Deserialize<List<int[]>>(row.SlotsJson) ?? new List<int[]>();
        }
        catch
        {
            return null;
        }

        if (slotDevices.Count == 0)
            return null;

        var slots = slotDevices
            .Select((devices, idx) => new BenchmarkSlot(idx, devices ?? Array.Empty<int>()))
            .ToList();

        return new BenchmarkExecutionPlan(
            planModelPath: key.PlanModelPath,
            staticNgl: row.StaticNgl,
            usesGpu: row.UsesGpu,
            groupSize: row.GroupSize,
            slots: slots);
    }

    private async Task UpsertCachedExecutionPlanAsync(
        ExecutionPlanCacheKey key,
        BenchmarkExecutionPlan plan,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var aiModelHashId = await GetOrCreateAiModelHashIdAsync(db, ct);

        var existing = await db.ExecutionPlanProbeCaches
            .FirstOrDefaultAsync(x =>
                x.AiModelHashId == aiModelHashId &&
                x.HardwareFingerprint == key.HardwareFingerprint &&
                x.QuantizedModelFingerprint == key.QuantizedModelFingerprint &&
                x.QuantizationKey == key.QuantizationKey &&
                x.DiscoveryTokenTarget == key.DiscoveryTokenTarget, ct);

        string slotsJson = JsonSerializer.Serialize(plan.Slots.Select(x => x.DeviceIndices).ToList());
        var now = DateTime.UtcNow;

        if (existing == null)
        {
            existing = new ExecutionPlanProbeCache
            {
                AiModelHashId = aiModelHashId,
                HardwareFingerprint = key.HardwareFingerprint,
                QuantizedModelFingerprint = key.QuantizedModelFingerprint,
                QuantizationKey = key.QuantizationKey,
                DiscoveryTokenTarget = key.DiscoveryTokenTarget,
                CreatedUtc = now
            };

            db.ExecutionPlanProbeCaches.Add(existing);
        }

        existing.StaticNgl = plan.StaticNgl;
        existing.UsesGpu = plan.UsesGpu;
        existing.GroupSize = plan.GroupSize;
        existing.SlotsJson = slotsJson;
        existing.UpdatedUtc = now;

        await db.SaveChangesAsync(ct);
    }

    private static ExecutionPlanCacheKey BuildExecutionPlanCacheKey(
        string normalizedModelPath,
        int discoveryTokenTarget,
        string quantizationKey)
    {
        string quantizedModelFingerprint;
        if (File.Exists(normalizedModelPath))
        {
            var info = new FileInfo(normalizedModelPath);
            quantizedModelFingerprint =
                $"{normalizedModelPath}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
        }
        else
        {
            quantizedModelFingerprint = normalizedModelPath;
        }

        var sys = Cache.SysInfo;
        string hardwareFingerprint = sys == null
            ? "unknown-hardware"
            : string.Join("|", new[]
            {
                $"threads:{sys.ThreadCount}",
                $"ram:{sys.RamGb:F2}",
                $"gpu:{string.Join(";", sys.GpuInfo.Select(g => $"{g.GpuVendor}:{g.GpuName}:{g.VramGb:F2}:{g.UniqueId ?? "none"}"))}"
            });

        return new ExecutionPlanCacheKey(
            hardwareFingerprint,
            quantizedModelFingerprint,
            quantizationKey,
            discoveryTokenTarget,
            normalizedModelPath);
    }

    private static async Task<uint> GetOrCreateAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (model != null)
            return model.Id;

        model = new AiModelHash { UniqueHash = Cache.CurrentModelId };
        db.AiModelHashes.Add(model);
        await db.SaveChangesAsync(ct);
        return model.Id;
    }

    private async Task<int?> ProbeHighestStableNglAsync(
        string modelPath,
        BenchmarkSlot slot,
        string probeRoot,
        int tokenTarget,
        CancellationToken ct)
    {
        string probeCorpusDir = Path.Combine(probeRoot, "_ppl_corpora");
        Directory.CreateDirectory(probeCorpusDir);

        string corpusPath = Path.Combine(probeCorpusDir, "ppl_corpus_general.txt");
        await PreparePplCorpusAsync("general", corpusPath, tokenTarget);

        foreach (int ngl in NglCandidates)
        {
            ct.ThrowIfCancellationRequested();

            AnsiConsole.MarkupLine(
                $"[grey]Plan probe:[/] testing [cyan]{Markup.Escape(slot.DisplayName)}[/] at [cyan]ngl={ngl}[/]");

            bool benchOk = await ProbeLlamaBenchAtFixedNglAsync(modelPath, slot, ngl, probeRoot);
            if (!benchOk)
            {
                AnsiConsole.MarkupLine($"[grey]  llama-bench failed at ngl={ngl}[/]");
                continue;
            }

            bool pplOk = await ProbePerplexityAtFixedNglAsync(modelPath, slot, ngl, corpusPath, probeRoot);
            if (!pplOk)
            {
                AnsiConsole.MarkupLine($"[grey]  perplexity failed at ngl={ngl}[/]");
                continue;
            }

            AnsiConsole.MarkupLine($"[green]  stable ngl discovered:[/] [cyan]{ngl}[/]");
            return ngl;
        }

        return null;
    }

    private async Task<bool> ValidateSlotForFixedPlanAsync(
        string modelPath,
        BenchmarkSlot slot,
        int fixedNgl,
        string probeRoot,
        int tokenTarget,
        CancellationToken ct)
    {
        string probeCorpusDir = Path.Combine(probeRoot, "_ppl_corpora");
        Directory.CreateDirectory(probeCorpusDir);

        string corpusPath = Path.Combine(probeCorpusDir, "ppl_corpus_general.txt");
        await PreparePplCorpusAsync("general", corpusPath, tokenTarget);

        bool benchOk = await ProbeLlamaBenchAtFixedNglAsync(modelPath, slot, fixedNgl, probeRoot);
        if (!benchOk)
            return false;

        bool pplOk = await ProbePerplexityAtFixedNglAsync(modelPath, slot, fixedNgl, corpusPath, probeRoot);
        return pplOk;
    }

    private async Task<bool> ProbeLlamaBenchAtFixedNglAsync(
        string modelPath,
        BenchmarkSlot slot,
        int fixedNgl,
        string probeRoot)
    {
        string logFile = Path.Combine(
            probeRoot,
            $"probe_llamabench_slot{slot.SlotId}_g{slot.DeviceCount}_ngl{fixedNgl}.md");

        string cmd = slot.UsesGpu
            ? $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {fixedNgl} -o md"
            : $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -backend cpu -o md";

        var result = await RunShellCommandAsync(cmd, logFile, slot.BuildProcessEnv());

        if (!result.Success)
            return false;

        try
        {
            var parsed = ParseLlamaBench(logFile);
            return parsed.Tps.HasValue && parsed.Tps.Value > 0;
        }
        catch
        {
            return false;
        }
    }

    private async Task<bool> ProbePerplexityAtFixedNglAsync(
        string modelPath,
        BenchmarkSlot slot,
        int fixedNgl,
        string corpusPath,
        string probeRoot)
    {
        string logFile = Path.Combine(
            probeRoot,
            $"probe_ppl_general_slot{slot.SlotId}_g{slot.DeviceCount}_ngl{fixedNgl}.log");

        string cmd = slot.UsesGpu
            ? $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {fixedNgl} -t 4 -c 2048 --file \"{corpusPath}\""
            : $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl 0 -t 4 -c 2048 --file \"{corpusPath}\"";

        var result = await RunShellCommandAsync(cmd, logFile, slot.BuildProcessEnv());

        if (!result.Success)
            return false;

        try
        {
            var parsed = ParsePerplexity(logFile, allowMissingKld: true);
            return parsed.Ppl > 0;
        }
        catch
        {
            return false;
        }
    }

    private static List<int> GetCandidateGroupSizes(int gpuCount)
    {
        var divisors = new List<int>();

        for (int i = 1; i <= gpuCount; i++)
        {
            if (gpuCount % i == 0)
                divisors.Add(i);
        }

        return divisors;
    }

    private static List<int[]> BuildContiguousGroups(int[] gpuIndices, int groupSize)
    {
        if (gpuIndices.Length % groupSize != 0)
        {
            throw new InvalidOperationException(
                $"GPU count {gpuIndices.Length} was not divisible by group size {groupSize}.");
        }

        var groups = new List<int[]>();
        for (int i = 0; i < gpuIndices.Length; i += groupSize)
        {
            groups.Add(gpuIndices.Skip(i).Take(groupSize).ToArray());
        }

        return groups;
    }

    private static async Task<BenchmarkSlotLease> AcquireBenchmarkSlotAsync(CancellationToken ct = default)
    {
        if (_currentPlan == null)
            throw new InvalidOperationException(
                "Benchmark execution plan has not been initialized. Call EnsureExecutionPlanAsync() first.");

        if (_slotSemaphore == null)
            throw new InvalidOperationException("Benchmark slot semaphore is not initialized.");

        await _slotSemaphore.WaitAsync(ct);

        lock (SlotSync)
        {
            if (_availableSlots.Count == 0)
            {
                _slotSemaphore.Release();
                throw new InvalidOperationException("No benchmark slots were available after semaphore acquisition.");
            }

            var slot = _availableSlots.Dequeue();
            return new BenchmarkSlotLease(slot);
        }
    }

    private static void ReturnBenchmarkSlot(BenchmarkSlot slot)
    {
        lock (SlotSync)
        {
            _availableSlots.Enqueue(slot);
            _slotSemaphore!.Release();
        }
    }

    // ----------------------------------------------------------------
    // Public entry points
    // ----------------------------------------------------------------

    private static bool IsNativeBaseModel(HybridQuant quantConfig)
    {
        return quantConfig.BaseQuant.UniqueId == BaselineQuants.NativeSourceUniqueId;
    }

    private static IReadOnlyCollection<string> ResolveRequestedDomains(
        HybridQuant quantConfig,
        IReadOnlyCollection<string>? domainsOverride)
    {
        if (domainsOverride != null && domainsOverride.Count > 0)
        {
            return domainsOverride
                .Select(x => x.Trim().ToLowerInvariant())
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        return IsNativeBaseModel(quantConfig) ? BaseDomains : SampleDomains;
    }

    private static bool RequiresKld(HybridQuant quantConfig)
    {
        return !IsNativeBaseModel(quantConfig);
    }

    private static ulong TryGetModelSize(string modelPath)
    {
        return File.Exists(modelPath) ? (ulong)new FileInfo(modelPath).Length : 0UL;
    }

    private const double KldEpsilon = 1e-8;

    private static bool HasMeaningfulKld(double? kld)
    {
        return kld.HasValue &&
               !double.IsNaN(kld.Value) &&
               !double.IsInfinity(kld.Value) &&
               Math.Abs(kld.Value) > KldEpsilon;
    }

    public async Task<bool> TryReuseExistingBenchmarksAsync(
        HybridQuant quantConfig,
        string modelPath,
        string benchDir,
        string? klLogitsDir,
        IReadOnlyCollection<string>? domainsOverride = null)
    {
        var requestedDomains = ResolveRequestedDomains(quantConfig, domainsOverride);
        bool requireKld = RequiresKld(quantConfig);

        if (!TryReadExistingBenchmarkArtifacts(
                benchDir: benchDir,
                requestedDomains: requestedDomains,
                requireKld: requireKld,
                result: out var reused))
        {
            return false;
        }

        if (!reused.ModelSizeBytes.HasValue || reused.ModelSizeBytes.Value == 0)
        {
            var actualSize = TryGetModelSize(modelPath);
            if (actualSize > 0)
                reused.ModelSizeBytes = actualSize;
        }

        using var db = new MagicQuantContext();

        var identity = await GetOrCreateBenchmarkIdentityAsync(db, quantConfig);
        await SaveBenchmarkToDbAsync(
            db: db,
            model: identity.AiModelHash,
            combo: identity.TensorCombo,
            res: reused,
            modelPath: modelPath,
            executedRunTimings: new List<PendingBenchmarkRunTiming>());

        await WriteMetricsJsonAsync(benchDir, reused);

        return true;
    }

    public async Task<BenchmarkResult> RunAllBenchmarksAsync(
        HybridQuant quantConfig,
        string modelPath,
        string benchDir,
        int tokenTarget = 32768,
        int? startNgl = null,
        string? klLogitsDir = null,
        bool saveLogits = false,
        IReadOnlyCollection<string>? domainsOverride = null)
    {
        Directory.CreateDirectory(benchDir);

        var requestedDomains = ResolveRequestedDomains(quantConfig, domainsOverride);
        bool requireKld = RequiresKld(quantConfig);

        using var db = new MagicQuantContext();

        var identity = await GetOrCreateBenchmarkIdentityAsync(db, quantConfig);
        var aiModelHash = identity.AiModelHash;
        var tensorCombo = identity.TensorCombo;

        var existingBench = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .AsNoTracking()
            .FirstOrDefaultAsync(b => b.AiModelHashId == aiModelHash.Id && b.TensorComboId == tensorCombo.Id);

        // 1. DB truth first
        if (existingBench != null && HasRequiredCategories(existingBench, requestedDomains, requireKld))
        {
            if (existingBench.SizeBytes == 0)
            {
                var repairedSize = TryGetModelSize(modelPath);
                if (repairedSize > 0)
                {
                    var trackedRepair = await db.AiBenchmarks
                        .FirstOrDefaultAsync(x => x.Id == existingBench.Id);

                    if (trackedRepair != null)
                    {
                        trackedRepair.SizeBytes = repairedSize;
                        await db.SaveChangesAsync();
                        existingBench.SizeBytes = repairedSize;
                    }
                }
            }

            if (TryReadExistingBenchmarkArtifacts(benchDir, requestedDomains, requireKld, out var diskResult))
            {
                if (!diskResult.ModelSizeBytes.HasValue || diskResult.ModelSizeBytes.Value == 0)
                    diskResult.ModelSizeBytes = existingBench.SizeBytes > 0
                        ? existingBench.SizeBytes
                        : TryGetModelSize(modelPath);

                return diskResult;
            }

            return BuildResultFromDb(existingBench, requestedDomains);
        }

        // 2. Disk truth second
        if (TryReadExistingBenchmarkArtifacts(benchDir, requestedDomains, requireKld, out var reused))
        {
            reused.ModelSizeBytes ??= TryGetModelSize(modelPath);
            await SaveBenchmarkToDbAsync(
                db: db,
                model: aiModelHash,
                combo: tensorCombo,
                res: reused,
                modelPath: modelPath,
                executedRunTimings: new List<PendingBenchmarkRunTiming>());

            await WriteMetricsJsonAsync(benchDir, reused);
            return reused;
        }

        // 3. Real execution: fixed slot + fixed ngl
        if (_currentPlan == null)
        {
            throw new InvalidOperationException(
                "No benchmark execution plan has been discovered yet. " +
                "You must call EnsureExecutionPlanAsync() with the pure Q8 model first.");
        }

        var trackedBench = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .FirstOrDefaultAsync(x => x.AiModelHashId == aiModelHash.Id && x.TensorComboId == tensorCombo.Id);

        if (trackedBench == null)
        {
            trackedBench = new AiBenchmark
            {
                AiModelHashId = aiModelHash.Id,
                TensorComboId = tensorCombo.Id,
                Ngl = 0,
                SizeBytes = 0,
                TokensPerSecond = 0
            };

            db.AiBenchmarks.Add(trackedBench);
            await db.SaveChangesAsync();
        }

        await using var slotLease = await AcquireBenchmarkSlotAsync();
        var slot = slotLease.Slot;

        int effectiveNgl = slot.UsesGpu
            ? _currentPlan.StaticNgl
            : 0;

        var result = new BenchmarkResult
        {
            ModelSizeBytes = TryGetModelSize(modelPath)
        };

        var executedRunTimings = new List<PendingBenchmarkRunTiming>();

        // Disabled for now. Too many variables that're annoying to track
        result.LlamaBench = new LlamaBenchMetrics
        {
            LogPath = null,
            Backend = slot.UsesGpu ? "disabled" : "cpu-disabled",
            Ngl = effectiveNgl,
            Test = "disabled",
            Tps = 0
        };

        var corporaRoot = Path.Combine(Path.GetDirectoryName(benchDir)!, "_ppl_corpora");
        Directory.CreateDirectory(corporaRoot);

        if (saveLogits && !string.IsNullOrEmpty(klLogitsDir))
            Directory.CreateDirectory(klLogitsDir);

        foreach (var domain in requestedDomains)
        {
            if (TryReadExistingPplLog(
                    benchDir: benchDir,
                    domain: domain,
                    allowMissingKld: !requireKld,
                    requirePositiveKld: requireKld,
                    metrics: out var existingPpl))
            {
                result.Perplexity[domain] = existingPpl;
                continue;
            }

            string corpusPath = Path.Combine(corporaRoot, $"ppl_corpus_{domain}.txt");
            await PreparePplCorpusAsync(domain, corpusPath, tokenTarget);

            DateTime startedUtc = DateTime.UtcNow;
            var sw = Stopwatch.StartNew();

            try
            {
                AnsiConsole.MarkupLine(
                    $"[yellow]Running Perplexity ({Markup.Escape(domain)})[/] [grey]({Markup.Escape(slot.DisplayName)}, ngl={effectiveNgl})[/]");

                var metrics = await RunPplBenchmarkAsync(
                    modelPath: modelPath,
                    benchDir: benchDir,
                    domain: domain,
                    corpusPath: corpusPath,
                    fixedNgl: effectiveNgl,
                    slot: slot,
                    klLogitsDir: klLogitsDir,
                    saveLogits: saveLogits);

                if (requireKld && !HasMeaningfulKld(metrics.Kld))
                {
                    throw new InvalidOperationException(
                        $"Non-base benchmark produced invalid KLD for domain '{domain}'. " +
                        $"KLD must exist and be > 0. Parsed value: {(metrics.Kld.HasValue ? metrics.Kld.Value.ToString(CultureInfo.InvariantCulture) : "null")}");
                }

                sw.Stop();

                result.Perplexity[domain] = metrics;

                executedRunTimings.Add(new PendingBenchmarkRunTiming
                {
                    Domain = domain,
                    Category = DomainToCategory(domain),
                    StartedUtc = startedUtc,
                    CompletedUtc = DateTime.UtcNow,
                    Succeeded = true,
                    Error = null
                });
            }
            catch (Exception ex)
            {
                sw.Stop();

                await PersistFailedBenchmarkRunAsync(
                    db: db,
                    aiModelHashId: aiModelHash.Id,
                    tensorComboId: tensorCombo.Id,
                    aiBenchmarkId: trackedBench.Id,
                    category: DomainToCategory(domain),
                    startedUtc: startedUtc,
                    completedUtc: DateTime.UtcNow,
                    error: ex.ToString());

                throw;
            }
        }

        await WriteMetricsJsonAsync(benchDir, result);

        await SaveBenchmarkToDbAsync(
            db: db,
            model: aiModelHash,
            combo: tensorCombo,
            res: result,
            modelPath: modelPath,
            executedRunTimings: executedRunTimings);

        return result;
    }

    // ----------------------------------------------------------------
    // Database helpers
    // ----------------------------------------------------------------

    private async Task<(AiModelHash AiModelHash, TensorCombo TensorCombo)> GetOrCreateBenchmarkIdentityAsync(
        MagicQuantContext db,
        HybridQuant quantConfig,
        CancellationToken ct = default)
    {
        var currentHashStr = Cache.CurrentModelId;
        if (string.IsNullOrWhiteSpace(currentHashStr))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var aiModelHash = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == currentHashStr, ct);

        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash { UniqueHash = currentHashStr };
            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync(ct);
        }

        var tensorCombo = await GetOrCreateTensorComboAsync(db, quantConfig, ct);

        return (aiModelHash, tensorCombo);
    }

    private async Task<TensorCombo> GetOrCreateTensorComboAsync(
        MagicQuantContext db,
        HybridQuant quant,
        CancellationToken ct = default)
    {
        var c = (TensorConfig)quant;

        var existing = await db.TensorCombos.FirstOrDefaultAsync(x =>
            x.BaseQuant == c.BaseQuant &&
            x.Embeddings == c.Embeddings &&
            x.LmHead == c.LmHead &&
            x.AttnQ == c.AttnQ &&
            x.AttnKV == c.AttnKV &&
            x.AttnOutput == c.AttnOutput &&
            x.FfnUpGate == c.FfnUpGate &&
            x.FfnDown == c.FfnDown &&
            x.MoeExperts == c.MoeExperts &&
            x.MoeRouter == c.MoeRouter, ct);

        if (existing != null)
            return existing;

        var newCombo = new TensorCombo(c);
        db.TensorCombos.Add(newCombo);
        await db.SaveChangesAsync(ct);
        return newCombo;
    }

    private async Task SaveBenchmarkToDbAsync(
        MagicQuantContext db,
        AiModelHash model,
        TensorCombo combo,
        BenchmarkResult res,
        string modelPath,
        IReadOnlyCollection<PendingBenchmarkRunTiming> executedRunTimings)
    {
        using var transaction = await db.Database.BeginTransactionAsync();

        try
        {
            bool isBaseModel =
                combo.BaseQuant == BaselineQuants.NativeSourceUniqueId &&
                combo.Embeddings == 0 &&
                combo.LmHead == 0 &&
                combo.AttnQ == 0 &&
                combo.AttnKV == 0 &&
                combo.AttnOutput == 0 &&
                combo.FfnUpGate == 0 &&
                combo.FfnDown == 0 &&
                combo.MoeExperts == 0 &&
                combo.MoeRouter == 0;

            ulong sizeBytes =
                res.ModelSizeBytes.GetValueOrDefault() > 0
                    ? res.ModelSizeBytes!.Value
                    : (File.Exists(modelPath) ? (ulong)new FileInfo(modelPath).Length : 0UL);

            var bench = await db.AiBenchmarks
                .Include(x => x.CategorBenchmarks)
                .FirstOrDefaultAsync(x =>
                    x.AiModelHashId == model.Id &&
                    x.TensorComboId == combo.Id);

            if (bench == null)
            {
                bench = new AiBenchmark
                {
                    AiModelHashId = model.Id,
                    TensorComboId = combo.Id
                };

                db.AiBenchmarks.Add(bench);
                await db.SaveChangesAsync();
            }

            bench.TokensPerSecond = res.LlamaBench?.Tps ?? 0;
            bench.Ngl = (byte)(res.LlamaBench?.Ngl ?? 0);

            if (sizeBytes > 0)
            {
                bench.SizeBytes = sizeBytes;
            }
            else if (bench.SizeBytes == 0)
            {
                bench.SizeBytes = 0;
            }

            await db.SaveChangesAsync();

            if (bench.CategorBenchmarks != null && bench.CategorBenchmarks.Count > 0)
            {
                db.Set<CategoryBenchmark>().RemoveRange(bench.CategorBenchmarks);
                await db.SaveChangesAsync();
            }

            var categories = new List<CategoryBenchmark>();

            foreach (var kvp in res.Perplexity)
            {
                string domain = kvp.Key.ToLowerInvariant();
                var m = kvp.Value;

                byte category = DomainToCategory(domain);

                double kld;
                if (isBaseModel)
                {
                    kld = 0d;
                }
                else
                {
                    if (!HasMeaningfulKld(m.Kld))
                    {
                        throw new InvalidOperationException(
                            $"Refusing to save non-base benchmark with invalid KLD. Domain='{domain}', KLD='{m.Kld?.ToString(CultureInfo.InvariantCulture) ?? "null"}'");
                    }

                    kld = m.Kld!.Value;
                }

                categories.Add(new CategoryBenchmark
                {
                    AiBenchmarkId = bench.Id,
                    Category = category,
                    Ppl = m.Ppl,
                    PplError = m.PplError,
                    Kld = kld
                });
            }

            if (categories.Count > 0)
            {
                db.Set<CategoryBenchmark>().AddRange(categories);
                await db.SaveChangesAsync();
            }

            if (executedRunTimings.Count > 0)
            {
                var categoryIdLookup = await db.Set<CategoryBenchmark>()
                    .Where(x => x.AiBenchmarkId == bench.Id)
                    .ToDictionaryAsync(x => x.Category, x => x.Id);

                foreach (var timing in executedRunTimings)
                {
                    Guid? categoryBenchmarkId = null;
                    if (categoryIdLookup.TryGetValue(timing.Category, out var foundCategoryId))
                        categoryBenchmarkId = foundCategoryId;

                    db.BenchmarkRuns.Add(new BenchmarkRun
                    {
                        Id = Guid.NewGuid(),
                        AiModelHashId = model.Id,
                        TensorComboId = combo.Id,
                        AiBenchmarkId = bench.Id,
                        CategoryBenchmarkId = categoryBenchmarkId,
                        Category = timing.Category,
                        StartedUtc = timing.StartedUtc,
                        CompletedUtc = timing.CompletedUtc,
                        DurationMs = Math.Max(0L, (long)(timing.CompletedUtc - timing.StartedUtc).TotalMilliseconds),
                        Succeeded = timing.Succeeded,
                        Error = timing.Error
                    });
                }

                await db.SaveChangesAsync();
            }

            await transaction.CommitAsync();
        }
        catch (Exception ex)
        {
            await transaction.RollbackAsync();

            var inner = ex.InnerException?.Message;
            if (!string.IsNullOrWhiteSpace(inner))
            {
                AnsiConsole.MarkupLine($"[red]Failed to save benchmarks to DB:[/] {Markup.Escape(ex.Message)}");
                AnsiConsole.MarkupLine($"[red]Inner Exception:[/] {Markup.Escape(inner)}");
            }
            else
            {
                AnsiConsole.MarkupLine($"[red]Failed to save benchmarks to DB:[/] {Markup.Escape(ex.Message)}");
            }

            throw;
        }
    }

    private static byte DomainToCategory(string domain)
    {
        return domain.Trim().ToLowerInvariant() switch
        {
            "general" => (byte)BenchmarkCategory.General,
            "math" => (byte)BenchmarkCategory.Math,
            "code" => (byte)BenchmarkCategory.Code,
            _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
        };
    }

    private async Task PersistFailedBenchmarkRunAsync(
        MagicQuantContext db,
        uint aiModelHashId,
        Guid tensorComboId,
        Guid aiBenchmarkId,
        byte category,
        DateTime startedUtc,
        DateTime completedUtc,
        string error)
    {
        db.BenchmarkRuns.Add(new BenchmarkRun
        {
            Id = Guid.NewGuid(),
            AiModelHashId = aiModelHashId,
            TensorComboId = tensorComboId,
            AiBenchmarkId = aiBenchmarkId,
            CategoryBenchmarkId = null,
            Category = category,
            StartedUtc = startedUtc,
            CompletedUtc = completedUtc,
            DurationMs = Math.Max(0L, (long)(completedUtc - startedUtc).TotalMilliseconds),
            Succeeded = false,
            Error = error
        });

        await db.SaveChangesAsync();
    }

    private sealed class PendingBenchmarkRunTiming
    {
        public string Domain { get; set; } = string.Empty;
        public byte Category { get; set; }
        public DateTime StartedUtc { get; set; }
        public DateTime CompletedUtc { get; set; }
        public bool Succeeded { get; set; }
        public string? Error { get; set; }
    }

    // ----------------------------------------------------------------
    // Artifact reuse helpers
    // ----------------------------------------------------------------

    private async Task WriteMetricsJsonAsync(string benchDir, BenchmarkResult result)
    {
        Directory.CreateDirectory(benchDir);

        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");
        await File.WriteAllTextAsync(
            jsonPath,
            JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
    }

    private bool TryReadExistingBenchmarkArtifacts(
        string benchDir,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld,
        out BenchmarkResult result)
    {
        result = new BenchmarkResult();

        string jsonPath = Path.Combine(benchDir, "bench_metrics.json");
        if (File.Exists(jsonPath))
        {
            try
            {
                var parsed = JsonSerializer.Deserialize<BenchmarkResult>(File.ReadAllText(jsonPath));
                if (parsed != null && IsReusableBenchmarkResult(parsed, requestedDomains, requireKld))
                {
                    result = parsed;
                    return true;
                }
            }
            catch
            {
                // fall through
            }
        }

        var rebuilt = new BenchmarkResult
        {
            LlamaBench = new LlamaBenchMetrics
            {
                LogPath = null,
                Backend = "disabled",
                Ngl = 0,
                Test = "disabled",
                Tps = 0
            }
        };

        foreach (var domain in requestedDomains)
        {
            if (!TryReadExistingPplLog(
                    benchDir: benchDir,
                    domain: domain,
                    allowMissingKld: !requireKld,
                    requirePositiveKld: requireKld,
                    metrics: out var ppl))
            {
                return false;
            }

            rebuilt.Perplexity[domain] = ppl;
        }

        result = rebuilt;
        return true;
    }

    private bool IsReusableBenchmarkResult(
        BenchmarkResult result,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld)
    {
        foreach (var domain in requestedDomains)
        {
            if (!result.Perplexity.TryGetValue(domain, out var ppl))
                return false;

            if (ppl.Ppl <= 0 || ppl.PplError < 0)
                return false;

            if (requireKld && !HasMeaningfulKld(ppl.Kld))
                return false;
        }

        return true;
    }

    private bool TryReadExistingLlamaBenchLog(string logPath, out LlamaBenchMetrics metrics)
    {
        metrics = null!;

        if (!File.Exists(logPath) || new FileInfo(logPath).Length == 0)
            return false;

        try
        {
            var parsed = ParseLlamaBench(logPath);
            if (parsed.Tps.HasValue && parsed.Tps.Value > 0)
            {
                metrics = parsed;
                return true;
            }
        }
        catch
        {
            // ignore
        }

        return false;
    }

    private bool TryReadExistingPplLog(
        string benchDir,
        string domain,
        bool allowMissingKld,
        bool requirePositiveKld,
        out PplMetrics metrics)
    {
        metrics = null!;

        string logPath = Path.Combine(benchDir, $"perplexity_{domain}.log");
        if (!File.Exists(logPath) || new FileInfo(logPath).Length == 0)
            return false;

        try
        {
            var parsed = ParsePerplexity(logPath, allowMissingKld);

            if (parsed.Ppl <= 0)
                return false;

            if (requirePositiveKld && !HasMeaningfulKld(parsed.Kld))
                return false;

            metrics = parsed;
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool HasRequiredCategories(
        AiBenchmark bench,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld)
    {
        if (bench.CategorBenchmarks == null || bench.CategorBenchmarks.Count == 0)
            return false;

        foreach (var domain in requestedDomains)
        {
            byte category = domain switch
            {
                "general" => (byte)BenchmarkCategory.General,
                "math" => (byte)BenchmarkCategory.Math,
                "code" => (byte)BenchmarkCategory.Code,
                _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
            };

            var existing = bench.CategorBenchmarks.FirstOrDefault(x => x.Category == category);
            if (existing == null)
                return false;

            if (existing.Ppl <= 0)
                return false;

            if (requireKld && existing.Kld <= 0)
                return false;
        }

        return bench.SizeBytes > 0;
    }

    private BenchmarkResult BuildResultFromDb(
        AiBenchmark bench,
        IReadOnlyCollection<string> requestedDomains)
    {
        var result = new BenchmarkResult
        {
            ModelSizeBytes = bench.SizeBytes,
            LlamaBench = new LlamaBenchMetrics
            {
                Ngl = bench.Ngl,
                Tps = bench.TokensPerSecond
            }
        };

        foreach (var domain in requestedDomains)
        {
            byte category = domain switch
            {
                "general" => (byte)BenchmarkCategory.General,
                "math" => (byte)BenchmarkCategory.Math,
                "code" => (byte)BenchmarkCategory.Code,
                _ => throw new InvalidOperationException($"Unknown benchmark domain '{domain}'.")
            };

            var existing = bench.CategorBenchmarks.First(x => x.Category == category);

            result.Perplexity[domain] = new PplMetrics
            {
                Ppl = existing.Ppl,
                PplError = existing.PplError,
                Kld = existing.Kld
            };
        }

        return result;
    }

    // ----------------------------------------------------------------
    // Real benchmark execution (fixed slot + fixed ngl)
    // ----------------------------------------------------------------

    private async Task<LlamaBenchMetrics> RunLlamaBenchAsync(
        string modelPath,
        string benchDir,
        int fixedNgl,
        BenchmarkSlot slot)
    {
        string logFile = Path.Combine(benchDir, "llamabench.md");

        string cmd = slot.UsesGpu
            ? $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {fixedNgl} -o md"
            : $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -backend cpu -o md";

        await RunFixedCommandWithRetryAsync(
            label: "llama-bench",
            cmd: cmd,
            logFile: logFile,
            slot: slot,
            attempts: 2,
            requirePplMarker: false);

        var parsed = ParseLlamaBench(logFile);
        if (!parsed.Tps.HasValue || parsed.Tps.Value <= 0)
        {
            throw new InvalidOperationException(
                $"llama-bench completed but no valid TPS could be parsed from {logFile}");
        }

        return parsed;
    }

    private async Task<PplMetrics> RunPplBenchmarkAsync(
        string modelPath,
        string benchDir,
        string domain,
        string corpusPath,
        int fixedNgl,
        BenchmarkSlot slot,
        string? klLogitsDir,
        bool saveLogits)
    {
        string logFile = Path.Combine(benchDir, $"perplexity_{domain}.log");

        string kldArgs = "";
        bool expectKld = false;

        if (!string.IsNullOrEmpty(klLogitsDir))
        {
            string logitsFile = Path.Combine(klLogitsDir, $"kld_logits_{domain}.bin");

            if (saveLogits)
            {
                kldArgs = $"--kl-divergence-base \"{logitsFile}\"";
                expectKld = false;
            }
            else if (File.Exists(logitsFile))
            {
                kldArgs = $"--kl-divergence-base \"{logitsFile}\" --kl-divergence";
                expectKld = true;
            }
        }

        string cmd = slot.UsesGpu
            ? $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {fixedNgl} -t 4 -c 2048 --file \"{corpusPath}\" {kldArgs}"
            : $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl 0 -t 4 -c 2048 --file \"{corpusPath}\" {kldArgs}";

        await RunFixedCommandWithRetryAsync(
            label: $"perplexity-{domain}",
            cmd: cmd,
            logFile: logFile,
            slot: slot,
            attempts: 2,
            requirePplMarker: true);

        bool allowMissingKld = !expectKld;
        var parsed = ParsePerplexity(logFile, allowMissingKld);

        if (expectKld && !HasMeaningfulKld(parsed.Kld))
        {
            throw new InvalidOperationException(
                $"Expected a real KLD for domain '{domain}', but parsed '{parsed.Kld?.ToString(CultureInfo.InvariantCulture) ?? "null"}' from {logFile}");
        }

        return parsed;
    }

    private async Task RunFixedCommandWithRetryAsync(
        string label,
        string cmd,
        string logFile,
        BenchmarkSlot slot,
        int attempts,
        bool requirePplMarker)
    {
        CommandRunResult? last = null;

        for (int attempt = 1; attempt <= attempts; attempt++)
        {
            last = await RunShellCommandAsync(cmd, logFile, slot.BuildProcessEnv());

            string logContent = !string.IsNullOrWhiteSpace(last.LogOutput)
                ? last.LogOutput
                : (File.Exists(logFile) ? File.ReadAllText(logFile) : string.Empty);

            bool success = last.Success && logContent.Length >= 50;

            if (success && requirePplMarker)
            {
                success = LooksLikeSuccessfulPerplexityRun(logFile, logContent);
            }

            if (success)
                return;

            bool retryable = LooksLikeRetryableGpuFailure(logContent);

            if (attempt < attempts && retryable)
            {
                AnsiConsole.MarkupLine(
                    $"[yellow]Transient benchmark failure detected on slot {slot.SlotId} ({Markup.Escape(slot.DisplayName)}). Retrying same fixed plan...[/]");
                await Task.Delay(1500);
                continue;
            }

            throw new InvalidOperationException(
                $"{label} failed on fixed benchmark slot {slot.SlotId} ({Markup.Escape(slot.DisplayName)}).\n" +
                $"Command: {cmd}\n\nLog Output:\n{logContent}");
        }

        throw new InvalidOperationException(
            $"{label} failed after {attempts} attempts on slot {slot.SlotId} ({Markup.Escape(slot.DisplayName)}).\n" +
            $"{last?.LogOutput}");
    }

    private bool LooksLikeSuccessfulPerplexityRun(string logFile, string logContent)
    {
        if (string.IsNullOrWhiteSpace(logContent) || logContent.Length < 50)
            return false;

        try
        {
            var parsed = ParsePerplexity(logFile, allowMissingKld: true);
            return parsed.Ppl > 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool LooksLikeRetryableGpuFailure(string logContent)
    {
        if (string.IsNullOrWhiteSpace(logContent))
            return false;

        if (OomMarkers.Any(m => logContent.Contains(m, StringComparison.OrdinalIgnoreCase)))
            return true;

        if (logContent.Contains("failed to load model", StringComparison.OrdinalIgnoreCase))
            return true;

        if (logContent.Contains("error:", StringComparison.OrdinalIgnoreCase))
            return true;

        return false;
    }

    // ----------------------------------------------------------------
    // Parsers
    // ----------------------------------------------------------------

    private LlamaBenchMetrics ParseLlamaBench(string logPath)
    {
        var metrics = new LlamaBenchMetrics { LogPath = GetRelativePath(logPath) };
        if (!File.Exists(logPath))
            return metrics;

        var lines = File.ReadAllLines(logPath);
        int headerIdx = -1;
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].Contains("|") && lines[i].Contains("backend"))
            {
                headerIdx = i;
                break;
            }
        }

        if (headerIdx == -1 || lines.Length <= headerIdx + 2)
            return metrics;

        var headers = lines[headerIdx]
            .Split('|', StringSplitOptions.RemoveEmptyEntries)
            .Select(h => h.Trim())
            .ToList();

        var dataRow = lines[headerIdx + 2]
            .Split('|', StringSplitOptions.RemoveEmptyEntries)
            .Select(d => d.Trim())
            .ToList();

        if (headers.Count != dataRow.Count)
            return metrics;

        var row = headers
            .Zip(dataRow, (h, d) => new { Header = h, Data = d })
            .ToDictionary(x => x.Header, x => x.Data, StringComparer.OrdinalIgnoreCase);

        string tpsStr = row.ContainsKey("t/s")
            ? row["t/s"]
            : (row.ContainsKey("tps") ? row["tps"] : "0");

        var match = Regex.Match(tpsStr, @"([0-9.]+)");
        if (match.Success &&
            double.TryParse(match.Groups[1].Value, NumberStyles.Any, CultureInfo.InvariantCulture, out double tps))
        {
            metrics.Tps = tps;
            metrics.Backend = row.ContainsKey("backend") ? row["backend"] : "unknown";
            metrics.Test = row.ContainsKey("test") ? row["test"] : "unknown";

            if (row.ContainsKey("ngl") &&
                int.TryParse(row["ngl"], NumberStyles.Any, CultureInfo.InvariantCulture, out int ngl))
            {
                metrics.Ngl = ngl;
            }
        }

        return metrics;
    }

    private PplMetrics ParsePerplexity(string logPath, bool allowMissingKld)
    {
        var metrics = new PplMetrics { LogPath = GetRelativePath(logPath) };

        if (!File.Exists(logPath))
            throw new FileNotFoundException($"Perplexity log file was not created: {logPath}");

        string text = File.ReadAllText(logPath);
        string cleanText = StripAnsi(text);

        var pplMatch = Regex.Match(
            cleanText,
            @"(?:Mean PPL\(Q\)|PPL)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:±|\+/-)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            RegexOptions.IgnoreCase);

        if (!pplMatch.Success)
        {
            throw new InvalidOperationException(
                $"Failed to parse PPL from log: {logPath}\n\nLast log content:\n{cleanText}");
        }

        metrics.Ppl = double.Parse(pplMatch.Groups[1].Value, CultureInfo.InvariantCulture);
        metrics.PplError = double.Parse(pplMatch.Groups[2].Value, CultureInfo.InvariantCulture);

        var kldMatch = Regex.Match(
            cleanText,
            @"(?:Mean\s+KLD|Mean\s+KL|KL[-_\s]*divergence|KLD|kl[-_\s]*div)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            RegexOptions.IgnoreCase);

        if (kldMatch.Success)
        {
            metrics.Kld = double.Parse(kldMatch.Groups[1].Value, CultureInfo.InvariantCulture);
        }
        else if (!allowMissingKld)
        {
            throw new InvalidOperationException(
                $"KLD was expected but could not be parsed from log: {logPath}\n\nLast log content:\n{cleanText}");
        }

        return metrics;
    }

    // ----------------------------------------------------------------
    // Corpus preparation
    // ----------------------------------------------------------------

    private async Task PreparePplCorpusAsync(string domain, string outPath, int tokenTarget)
    {
        if (File.Exists(outPath) && new FileInfo(outPath).Length > 0)
            return;

        AnsiConsole.MarkupLine($"[grey]Generating corpus for domain: {domain}[/]");

        string pyScript = $@"
import sys
from datasets import load_dataset

domain = '{domain}'
out_path = r'{outPath}'
max_chars = {tokenTarget} * 4

def get_sources(d):
    if d == 'general': return [('wikitext', 'wikitext-103-raw-v1', 'test', 'text'), ('wikitext', 'wikitext-2-raw-v1', 'test', 'text')]
    if d == 'code': return [('codeparrot/codeparrot-clean', None, 'train', 'content')]
    if d == 'math': return [('gsm8k', 'main', 'test', 'question')]
    return []

parts = []
total = 0
for ds, conf, split, field in get_sources(domain):
    try:
        d = load_dataset(ds, conf) if conf else load_dataset(ds)
        for text in d[split][field]:
            if not text or not isinstance(text, str):
                continue
            chunk = text.strip() + '\n'
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
    except Exception as e:
        print(f'Error loading {{ds}}: {{e}}')
    if total >= max_chars:
        break

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(''.join(parts))
";

        string scriptPath = Path.Combine(Path.GetDirectoryName(outPath)!, $"gen_{domain}.py");
        await File.WriteAllTextAsync(scriptPath, pyScript);

        string pythonExe = _pyManager.GetPythonExecutable();
        string args = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? $"/c \"{pythonExe}\" \"{scriptPath}\""
            : $"\"{scriptPath}\"";

        string runner = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cmd.exe" : pythonExe;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            args = scriptPath;

        await _pyManager.RunPipInstallAsync("datasets");
        await RunShellCommandAsync(runner + " " + args, null);

        if (File.Exists(scriptPath))
            File.Delete(scriptPath);
    }

    // ----------------------------------------------------------------
    // Process / shell utilities
    // ----------------------------------------------------------------

    private sealed class CommandRunResult
    {
        public bool Success { get; init; }
        public int ExitCode { get; init; }
        public string LogOutput { get; init; } = string.Empty;
    }

    private async Task<CommandRunResult> RunShellCommandAsync(
        string cmd,
        string? logPath,
        IReadOnlyDictionary<string, string>? extraEnv = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "cmd.exe" : "/bin/bash",
            Arguments = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? $"/c {cmd}" : $"-c \"{cmd}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        if (extraEnv != null)
        {
            foreach (var kvp in extraEnv)
            {
                startInfo.Environment[kvp.Key] = kvp.Value;
            }
        }

        using var process = new Process { StartInfo = startInfo };
        FileStream? fs = null;
        StreamWriter? sw = null;

        if (logPath != null)
        {
            fs = new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            sw = new StreamWriter(fs) { AutoFlush = true };
        }

        process.Start();

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (!string.IsNullOrWhiteSpace(stdout))
            sw?.WriteLine(stdout);

        if (!string.IsNullOrWhiteSpace(stderr))
            sw?.WriteLine(stderr);

        sw?.Dispose();
        fs?.Dispose();

        string combinedLog;
        if (logPath != null && File.Exists(logPath))
            combinedLog = File.ReadAllText(logPath);
        else
            combinedLog = $"{stdout}\n{stderr}";

        return new CommandRunResult
        {
            Success = process.ExitCode == 0,
            ExitCode = process.ExitCode,
            LogOutput = combinedLog
        };
    }

    private string StripAnsi(string text)
    {
        return Regex.Replace(text, @"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "");
    }

    private string GetRelativePath(string fullPath)
    {
        return Path.GetFileName(fullPath);
    }

    // ----------------------------------------------------------------
    // Internal plan / slot types
    // ----------------------------------------------------------------

    private sealed class BenchmarkExecutionPlan
    {
        public string PlanModelPath { get; }
        public int StaticNgl { get; }
        public bool UsesGpu { get; }
        public int GroupSize { get; }
        public IReadOnlyList<BenchmarkSlot> Slots { get; }

        public BenchmarkExecutionPlan(
            string planModelPath,
            int staticNgl,
            bool usesGpu,
            int groupSize,
            IReadOnlyList<BenchmarkSlot> slots)
        {
            PlanModelPath = planModelPath;
            StaticNgl = staticNgl;
            UsesGpu = usesGpu;
            GroupSize = groupSize;
            Slots = slots;
        }

        public static BenchmarkExecutionPlan CreateCpuPlan(string q8ModelPath)
        {
            return new BenchmarkExecutionPlan(
                planModelPath: q8ModelPath,
                staticNgl: 0,
                usesGpu: false,
                groupSize: 0,
                slots: new List<BenchmarkSlot> { new(0, Array.Empty<int>()) });
        }
    }

    private sealed class BenchmarkSlot
    {
        public int SlotId { get; }
        public int[] DeviceIndices { get; }
        public bool UsesGpu => DeviceIndices.Length > 0;
        public int DeviceCount => DeviceIndices.Length;

        public BenchmarkSlot(int slotId, int[] deviceIndices)
        {
            SlotId = slotId;
            DeviceIndices = deviceIndices;
        }

        public string DisplayName =>
            UsesGpu
                ? $"GPU[{string.Join(",", DeviceIndices)}]"
                : "CPU";

        public IReadOnlyDictionary<string, string>? BuildProcessEnv()
        {
            if (!UsesGpu)
                return null;

            string visible = string.Join(",", DeviceIndices);

            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["CUDA_VISIBLE_DEVICES"] = visible,
                ["HIP_VISIBLE_DEVICES"] = visible,
                ["ROCR_VISIBLE_DEVICES"] = visible
            };
        }
    }

    private sealed class BenchmarkSlotLease : IAsyncDisposable
    {
        public BenchmarkSlot Slot { get; }

        public BenchmarkSlotLease(BenchmarkSlot slot)
        {
            Slot = slot;
        }

        public ValueTask DisposeAsync()
        {
            ReturnBenchmarkSlot(Slot);
            return ValueTask.CompletedTask;
        }
    }

    private sealed record ExecutionPlanCacheKey(
        string HardwareFingerprint,
        string QuantizedModelFingerprint,
        string QuantizationKey,
        int DiscoveryTokenTarget,
        string PlanModelPath);
}
