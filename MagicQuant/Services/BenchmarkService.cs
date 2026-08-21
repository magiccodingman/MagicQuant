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
    private readonly GgufMetadataReader _ggufMetadataReader;
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

    private static readonly int[] LegacyNglFallbacks = { 35, 30, 24, 20, 16, 12, 8, 4 };

    // Version 4 adds measured shared/independent GPU topology profiles, per-device Q8
    // anchors, exact model-layer ceilings, and the llama.cpp binary fingerprint.
    private const int DynamicProbeSchemaVersion = 4;
    private const int PplCharsPerTokenEstimate = 4;
    internal const string GeneralPplDatasetId = "Salesforce/wikitext";
    internal const string MathPplDatasetId = "openai/gsm8k";

    // ----------------------------------------------------------------
    // Static execution-plan state
    // ----------------------------------------------------------------

    private static readonly SemaphoreSlim PlanInitLock = new(1, 1);
    private static readonly object SlotSync = new();

    private static BenchmarkExecutionPlan? _currentPlan;
    private static string _currentPlanQuantizationKey = "Q8_0";
    private static GpuResourceScheduler _resourceScheduler = new();

    public int CurrentParallelSlotCount
    {
        get
        {
            lock (SlotSync)
            {
                if (_currentPlan == null)
                    return 1;

                return Math.Max(
                    1,
                    Math.Max(_currentPlan.SharedProfile.Slots.Count, _currentPlan.IndependentProfile.Slots.Count));
            }
        }
    }

    // ----------------------------------------------------------------
    // Construction
    // ----------------------------------------------------------------

    public BenchmarkService(PythonManager pyManager)
    {
        _bins = new LlamaBinaries(Cache.LlamaRoot);
        _bins.Validate();
        _pyManager = pyManager;
        _ggufMetadataReader = new GgufMetadataReader(pyManager);
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
        await EnsureDynamicExecutionPlanAsync(
            q8ModelPath: q8ModelPath,
            nativeModelPath: q8ModelPath,
            q8QuantizationKey: quantizationKey,
            nativeQuantizationKey: quantizationKey,
            discoveryTokenTarget: discoveryTokenTarget,
            forceRediscovery: forceRediscovery,
            ct: ct);
    }

    public async Task EnsureDynamicExecutionPlanAsync(
        string q8ModelPath,
        string nativeModelPath,
        string q8QuantizationKey = "Q8_0",
        string nativeQuantizationKey = "BF16",
        int discoveryTokenTarget = 8192,
        bool forceRediscovery = false,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(q8ModelPath))
            throw new ArgumentException("Q8 model path was null or empty.", nameof(q8ModelPath));
        if (string.IsNullOrWhiteSpace(nativeModelPath))
            throw new ArgumentException("Native model path was null or empty.", nameof(nativeModelPath));
        if (string.IsNullOrWhiteSpace(q8QuantizationKey))
            throw new ArgumentException("Quantization key was null or empty.", nameof(q8QuantizationKey));

        string normalizedPath = Path.GetFullPath(q8ModelPath);
        string normalizedNativePath = Path.GetFullPath(nativeModelPath);
        string normalizedQuantizationKey = q8QuantizationKey.Trim().ToUpperInvariant();
        string normalizedNativeQuantizationKey = nativeQuantizationKey.Trim().ToUpperInvariant();

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
                plan = await TryLoadCachedExecutionPlanAsync(
                    key: cacheKey,
                    nativeModelPath: normalizedNativePath,
                    nativeQuantizationKey: normalizedNativeQuantizationKey,
                    ct: ct);
                if (plan != null)
                    AnsiConsole.MarkupLine("[green]Loaded benchmark execution plan from SQLite cache.[/]");
            }

            if (plan == null)
            {
                plan = await BuildDynamicExecutionPlanAsync(
                    q8ModelPath: normalizedPath,
                    nativeModelPath: normalizedNativePath,
                    q8QuantizationKey: normalizedQuantizationKey,
                    nativeQuantizationKey: normalizedNativeQuantizationKey,
                    discoveryTokenTarget: discoveryTokenTarget,
                    ct: ct);
                await UpsertCachedExecutionPlanAsync(cacheKey, plan, ct);
            }

            lock (SlotSync)
            {
                _currentPlan = plan;
                _currentPlanQuantizationKey = normalizedQuantizationKey;
                _resourceScheduler = new GpuResourceScheduler();
            }

            AnsiConsole.Write(new Rule("[yellow]Benchmark Execution Plan[/]") { Justification = Justify.Left });
            AnsiConsole.MarkupLine($"[green]Static ngl:[/] [cyan]{plan.StaticNgl}[/]");
            AnsiConsole.MarkupLine($"[green]Q8 anchor:[/] [cyan]{(plan.Q8ModelSizeBytes / 1024d / 1024d / 1024d):F2} GB @ ngl={plan.Q8StableNgl}[/]");
            AnsiConsole.MarkupLine($"[green]Native anchor:[/] [cyan]{(plan.NativeModelSizeBytes / 1024d / 1024d / 1024d):F2} GB @ ngl={plan.NativeStableNgl}[/]");
            AnsiConsole.MarkupLine($"[green]Uses GPU:[/] [cyan]{plan.UsesGpu}[/]");
            AnsiConsole.MarkupLine($"[green]GPU group size:[/] [cyan]{plan.GroupSize}[/]");
            AnsiConsole.MarkupLine($"[green]Max parallel benchmark slots:[/] [cyan]{CurrentParallelSlotCount}[/]");
            if (plan.IndependentMaxModelSizeBytes > 0)
            {
                AnsiConsole.MarkupLine(
                    $"[green]Independent-worker crossover:[/] [cyan]{plan.IndependentMaxModelSizeBytes / 1024d / 1024d / 1024d:F2} GB[/]");
            }
            AnsiConsole.MarkupLine($"[green]Quantization key:[/] [cyan]{Markup.Escape(normalizedQuantizationKey)}[/]");
            if (Cache.GpuMemoryLimitsGb.Count == 0)
            {
                AnsiConsole.MarkupLine("[green]GPU memory limits:[/] [grey]none[/]");
            }
            else
            {
                string limits = string.Join(", ", Cache.GpuMemoryLimitsGb.OrderBy(x => x.Key).Select(x => $"GPU {x.Key}={x.Value:0.###} GB"));
                AnsiConsole.MarkupLine($"[green]GPU memory limits:[/] [cyan]{Markup.Escape(limits)}[/]");
            }

            foreach (var slot in plan.SharedProfile.Slots.Concat(plan.IndependentProfile.Slots))
            {
                AnsiConsole.MarkupLine(
                    $"  [grey]Slot {slot.SlotId}:[/] {Markup.Escape(slot.DisplayName)} @ Q8 ngl={slot.Q8StableNgl}");
                string tensorSplit = BuildTensorSplitArgs(slot, LlamaGpuTool.CommonCli);
                if (!string.IsNullOrWhiteSpace(tensorSplit))
                {
                    AnsiConsole.MarkupLine($"    [grey]tensor split:[/] {Markup.Escape(tensorSplit.Trim())}");
                }
            }
        }
        finally
        {
            PlanInitLock.Release();
        }
    }

    public async Task<bool> TryInitializeExecutionPlanFromCacheAsync(
        int discoveryTokenTarget = 8192,
        string quantizationKey = "Q8_0",
        string? nativeModelPath = null,
        string nativeQuantizationKey = "BF16",
        string? preferredPlanModelPath = null,
        CancellationToken ct = default)
        => await TryInitializeDynamicExecutionPlanFromCacheAsync(
            discoveryTokenTarget: discoveryTokenTarget,
            q8QuantizationKey: quantizationKey,
            nativeModelPath: nativeModelPath,
            nativeQuantizationKey: nativeQuantizationKey,
            preferredPlanModelPath: preferredPlanModelPath,
            ct: ct);

    public async Task<bool> TryInitializeDynamicExecutionPlanFromCacheAsync(
        int discoveryTokenTarget = 8192,
        string q8QuantizationKey = "Q8_0",
        string? nativeModelPath = null,
        string nativeQuantizationKey = "BF16",
        string? preferredPlanModelPath = null,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(q8QuantizationKey))
            throw new ArgumentException("Quantization key was null or empty.", nameof(q8QuantizationKey));

        string normalizedQuantizationKey = q8QuantizationKey.Trim().ToUpperInvariant();
        string planModelPath = string.IsNullOrWhiteSpace(preferredPlanModelPath)
            ? $"cached://{normalizedQuantizationKey}"
            : Path.GetFullPath(preferredPlanModelPath);

        var cacheKey = BuildExecutionPlanCacheKey(planModelPath, discoveryTokenTarget, normalizedQuantizationKey);
        AnsiConsole.MarkupLine(
            $"[grey]Checking execution-plan cache:[/] quant={Markup.Escape(normalizedQuantizationKey)}, tokens={discoveryTokenTarget}");

        var plan = await TryLoadCachedExecutionPlanAsync(
            key: cacheKey,
            nativeModelPath: nativeModelPath,
            nativeQuantizationKey: nativeQuantizationKey,
            ct: ct);
        if (plan == null)
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache miss:[/] full Q8 probe will run.");
            return false;
        }

        lock (SlotSync)
        {
            _currentPlan = plan;
            _currentPlanQuantizationKey = normalizedQuantizationKey;
            _resourceScheduler = new GpuResourceScheduler();
        }

        AnsiConsole.MarkupLine("[green]Loaded benchmark execution plan from SQLite cache (no Q8 rebuild needed).[/]");
        return true;
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

            foreach (int ngl in BuildNglFallbackList(startingNgl).Where(n => n > 0))
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

                var cpuPlan = BenchmarkExecutionPlan.CreateCpuPlan(_currentPlan.PlanModelPath) with
                {
                    ProbeSchemaVersion = _currentPlan.ProbeSchemaVersion,
                    Q8ModelSizeBytes = _currentPlan.Q8ModelSizeBytes,
                    Q8StableNgl = 0,
                    NativeModelSizeBytes = _currentPlan.NativeModelSizeBytes,
                    NativeStableNgl = 0,
                    NativeQuantizationKey = _currentPlan.NativeQuantizationKey,
                    MaxCandidateNgl = _currentPlan.MaxCandidateNgl,
                    GpuMemoryLimitsJson = _currentPlan.GpuMemoryLimitsJson,
                    TensorSplitJson = "{}"
                };

                lock (SlotSync)
                {
                    _currentPlan = cpuPlan;
                    _resourceScheduler = new GpuResourceScheduler();
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
                var updated = _currentPlan with { StaticNgl = chosen.Value };

                lock (SlotSync)
                {
                    _currentPlan = updated;
                    _resourceScheduler = new GpuResourceScheduler();
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

    private async Task<BenchmarkExecutionPlan> BuildDynamicExecutionPlanAsync(
        string q8ModelPath,
        string nativeModelPath,
        string q8QuantizationKey,
        string nativeQuantizationKey,
        int discoveryTokenTarget,
        CancellationToken ct)
    {
        var plan = await BuildExecutionPlanAsync(q8ModelPath, discoveryTokenTarget, ct);
        if (!plan.UsesGpu)
        {
            return plan with
            {
                ProbeSchemaVersion = DynamicProbeSchemaVersion,
                Q8ModelSizeBytes = TryGetModelSize(q8ModelPath),
                Q8StableNgl = 0,
                NativeModelSizeBytes = TryGetModelSize(nativeModelPath),
                NativeStableNgl = 0,
                NativeQuantizationKey = nativeQuantizationKey,
                MaxCandidateNgl = plan.MaxCandidateNgl,
                GpuMemoryLimitsJson = SerializeGpuMemoryLimits(),
                TensorSplitJson = SerializeTensorSplitMap(plan.Slots)
            };
        }

        string nativeProbeRoot = Path.Combine(Cache.ModelMagicQuantDirectory!, "_benchmark_plan_probe_native");
        Directory.CreateDirectory(nativeProbeRoot);
        string nativeCorpusDir = Path.Combine(nativeProbeRoot, "_ppl_corpora");
        Directory.CreateDirectory(nativeCorpusDir);
        string nativeCorpusPath = Path.Combine(nativeCorpusDir, "ppl_corpus_general.txt");
        await PreparePplCorpusAsync("general", nativeCorpusPath, discoveryTokenTarget);

        BenchmarkSlot? nativeProbeSlot = await ProbeSlotCapacityAsync(
            nativeModelPath,
            plan.Slots[0] with { ProbeSamples = [] },
            plan.MaxCandidateNgl,
            nativeCorpusPath,
            nativeProbeRoot,
            ct);
        int? nativeStableNgl = nativeProbeSlot?.Q8StableNgl;

        if (!nativeStableNgl.HasValue || nativeStableNgl.Value <= 0)
        {
            AnsiConsole.MarkupLine(
                "[yellow]Native anchor unavailable; keeping Q8 GPU plan and using conservative Q8-only dynamic NGL fallback.[/]");
            return plan with
            {
                ProbeSchemaVersion = DynamicProbeSchemaVersion,
                Q8ModelSizeBytes = TryGetModelSize(q8ModelPath),
                Q8StableNgl = plan.StaticNgl,
                NativeModelSizeBytes = TryGetModelSize(nativeModelPath),
                NativeStableNgl = 0,
                NativeQuantizationKey = nativeQuantizationKey,
                MaxCandidateNgl = plan.MaxCandidateNgl,
                GpuMemoryLimitsJson = SerializeGpuMemoryLimits(),
                TensorSplitJson = SerializeTensorSplitMap(plan.Slots)
            };
        }

        return plan with
        {
            ProbeSchemaVersion = DynamicProbeSchemaVersion,
            Q8ModelSizeBytes = TryGetModelSize(q8ModelPath),
            Q8StableNgl = plan.StaticNgl,
            NativeModelSizeBytes = TryGetModelSize(nativeModelPath),
            NativeStableNgl = nativeStableNgl.Value,
            NativeQuantizationKey = nativeQuantizationKey,
            MaxCandidateNgl = plan.MaxCandidateNgl,
            GpuMemoryLimitsJson = SerializeGpuMemoryLimits(),
            TensorSplitJson = SerializeTensorSplitMap(plan.Slots)
        };
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

        string probeRoot = Path.Combine(Cache.ModelMagicQuantDirectory!, "_benchmark_plan_probe");
        Directory.CreateDirectory(probeRoot);

        var metadata = await _ggufMetadataReader.ReadAsync(q8ModelPath, probeRoot, ct);
        int maxOffloadNgl = metadata.BlockCount is > 0
            ? checked(metadata.BlockCount.Value + 1)
            : throw new InvalidOperationException(
                "Q8 GGUF metadata did not expose a positive architecture block_count; " +
                "an exact full-offload ceiling cannot be planned safely.");

        string probeCorpusDir = Path.Combine(probeRoot, "_ppl_corpora");
        Directory.CreateDirectory(probeCorpusDir);
        string corpusPath = Path.Combine(probeCorpusDir, "ppl_corpus_general.txt");
        await PreparePplCorpusAsync("general", corpusPath, discoveryTokenTarget);

        var allGpuIndices = Enumerable.Range(0, gpuCount).ToArray();
        var sharedSeed = new BenchmarkSlot(0, "shared", allGpuIndices, 0, []);
        _ = BuildTensorSplitArgs(sharedSeed, LlamaGpuTool.CommonCli);

        BenchmarkSlot? sharedSlot = await ProbeSlotCapacityAsync(
            q8ModelPath,
            sharedSeed,
            maxOffloadNgl,
            corpusPath,
            probeRoot,
            ct);

        if (sharedSlot == null || sharedSlot.Q8StableNgl <= 0)
        {
            AnsiConsole.MarkupLine(
                "[yellow]Q8 discovery could not establish a stable GPU ngl. Falling back to a single CPU slot.[/]");
            return BenchmarkExecutionPlan.CreateCpuPlan(q8ModelPath);
        }

        var independentSlots = new List<BenchmarkSlot>();
        if (gpuCount > 1)
        {
            for (int gpuIndex = 0; gpuIndex < gpuCount; gpuIndex++)
            {
                var seed = new BenchmarkSlot(gpuIndex, "independent", [gpuIndex], 0, []);
                BenchmarkSlot? discovered = await ProbeSlotCapacityAsync(
                    q8ModelPath,
                    seed,
                    maxOffloadNgl,
                    corpusPath,
                    probeRoot,
                    ct);

                if (discovered == null || discovered.Q8StableNgl <= 0)
                {
                    AnsiConsole.MarkupLine(
                        $"[yellow]Independent slot GPU[{gpuIndex}] was not stable; independent topology disabled.[/]");
                    independentSlots.Clear();
                    break;
                }

                independentSlots.Add(discovered);
            }
        }

        BenchmarkTopologyProfile sharedProfile = await MeasureTopologyProfileAsync(
            "shared", q8ModelPath, [sharedSlot], corpusPath, probeRoot, ct);

        if (sharedProfile.Slots.Count == 0)
        {
            AnsiConsole.MarkupLine(
                "[yellow]Shared GPU topology failed its concurrent throughput validation. Falling back to CPU.[/]");
            return BenchmarkExecutionPlan.CreateCpuPlan(q8ModelPath);
        }

        BenchmarkTopologyProfile independentProfile = independentSlots.Count > 1
            ? await MeasureTopologyProfileAsync(
                "independent", q8ModelPath, independentSlots, corpusPath, probeRoot, ct)
            : new BenchmarkTopologyProfile("independent", [], 0, 0);

        ulong q8Size = TryGetModelSize(q8ModelPath);
        ulong independentMaxModelSizeBytes = independentProfile.Slots.Count > 1
            ? BenchmarkGpuPlanner.EstimateIndependentCrossoverBytes(
                q8Size,
                maxOffloadNgl,
                sharedProfile.MeasuredSecondsPerPass,
                independentProfile.Slots)
            : 0;

        if (independentMaxModelSizeBytes > 0)
        {
            AnsiConsole.MarkupLine(
                $"[green]Measured topology crossover:[/] models up to " +
                $"[cyan]{independentMaxModelSizeBytes / 1024d / 1024d / 1024d:F2} GB[/] use independent GPU workers; larger models use shared GPUs.");
        }

        return new BenchmarkExecutionPlan(
            PlanModelPath: q8ModelPath,
            StaticNgl: sharedSlot.Q8StableNgl,
            UsesGpu: true,
            GroupSize: gpuCount,
            Slots: sharedProfile.Slots,
            MaxCandidateNgl: maxOffloadNgl,
            IndependentSlots: independentProfile.Slots,
            IndependentMaxModelSizeBytes: independentMaxModelSizeBytes,
            SharedMeasuredJobsPerSecond: sharedProfile.MeasuredJobsPerSecond,
            SharedMeasuredSecondsPerPass: sharedProfile.MeasuredSecondsPerPass,
            IndependentMeasuredJobsPerSecond: independentProfile.MeasuredJobsPerSecond,
            IndependentMeasuredSecondsPerPass: independentProfile.MeasuredSecondsPerPass);
    }

    private async Task<BenchmarkSlot?> ProbeSlotCapacityAsync(
        string modelPath,
        BenchmarkSlot seed,
        int maxOffloadNgl,
        string corpusPath,
        string probeRoot,
        CancellationToken ct)
    {
        var samples = new List<GpuProbeSample>();

        async Task<GpuProbeSample> Probe(int ngl)
        {
            var sample = await ProbePerplexitySampleAsync(
                modelPath, seed, ngl, corpusPath, probeRoot, "capacity", ct);
            samples.Add(sample);
            return sample;
        }

        AnsiConsole.MarkupLine(
            $"[grey]Capacity probe:[/] {Markup.Escape(seed.DisplayName)} full-offload ngl={maxOffloadNgl}");

        var full = await Probe(maxOffloadNgl);
        int stableNgl;

        if (full.Success)
        {
            stableNgl = maxOffloadNgl;
            int lowerNgl = Math.Max(1, (int)Math.Floor(maxOffloadNgl * 0.72d));
            if (lowerNgl < stableNgl)
                await Probe(lowerNgl);
        }
        else
        {
            int low = 0;
            int high = maxOffloadNgl - 1;

            while (low < high)
            {
                ct.ThrowIfCancellationRequested();
                int candidate = low + ((high - low + 1) / 2);
                var sample = await Probe(candidate);
                if (sample.Success)
                    low = candidate;
                else
                    high = candidate - 1;
            }

            stableNgl = low;
        }

        if (stableNgl <= 0)
            return null;

        int successfulDistinct = samples.Where(x => x.Success).Select(x => x.Ngl).Distinct().Count();
        if (successfulDistinct < 2)
        {
            int lowerNgl = Math.Max(1, (int)Math.Floor(stableNgl * 0.72d));
            if (lowerNgl < stableNgl)
                await Probe(lowerNgl);
        }

        AnsiConsole.MarkupLine(
            $"[green]Stable capacity:[/] {Markup.Escape(seed.DisplayName)} ngl={stableNgl}/{maxOffloadNgl}");

        return seed with
        {
            Q8StableNgl = stableNgl,
            ProbeSamples = samples
        };
    }

    private async Task<BenchmarkTopologyProfile> MeasureTopologyProfileAsync(
        string profileName,
        string modelPath,
        IReadOnlyList<BenchmarkSlot> slots,
        string corpusPath,
        string probeRoot,
        CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();
        var tasks = slots.Select(slot => ProbePerplexitySampleAsync(
            modelPath,
            slot,
            slot.Q8StableNgl,
            corpusPath,
            probeRoot,
            $"throughput_{profileName}",
            ct));

        GpuProbeSample[] samples = await Task.WhenAll(tasks);
        stopwatch.Stop();

        if (samples.Any(x => !x.Success))
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Topology throughput validation failed for {Markup.Escape(profileName)}.[/]");
            return new BenchmarkTopologyProfile(profileName, [], 0, 0);
        }

        double jobsPerSecond = slots.Count / Math.Max(0.001d, stopwatch.Elapsed.TotalSeconds);
        double secondsPerPass = samples.Average(x => x.SecondsPerPass);
        AnsiConsole.MarkupLine(
            $"[green]Topology throughput:[/] {Markup.Escape(profileName)} = " +
            $"[cyan]{jobsPerSecond:F4} jobs/s[/], {secondsPerPass:F2} s/pass");

        return new BenchmarkTopologyProfile(
            profileName,
            slots,
            jobsPerSecond,
            secondsPerPass);
    }

    private async Task<GpuProbeSample> ProbePerplexitySampleAsync(
        string modelPath,
        BenchmarkSlot slot,
        int fixedNgl,
        string corpusPath,
        string probeRoot,
        string phase,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        string devices = slot.DeviceIndices.Length == 0
            ? "cpu"
            : string.Join("-", slot.DeviceIndices);
        string logFile = Path.Combine(
            probeRoot,
            $"probe_ppl_{phase}_{slot.ProfileName}_gpu{devices}_ngl{fixedNgl}.log");

        string cmd = slot.UsesGpu
            ? $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {fixedNgl}{BuildTensorSplitArgs(slot, LlamaGpuTool.CommonCli)} -t 4 -c 2048 --file \"{corpusPath}\""
            : $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl 0 -t 4 -c 2048 --file \"{corpusPath}\"";

        var stopwatch = Stopwatch.StartNew();
        var result = await RunShellCommandAsync(cmd, logFile, slot.BuildProcessEnv());
        stopwatch.Stop();

        if (!result.Success)
            return new GpuProbeSample(fixedNgl, false, 0, stopwatch.Elapsed.TotalSeconds);

        try
        {
            var parsed = ParsePerplexity(logFile, allowMissingKld: true);
            string clean = StripAnsi(result.LogOutput);
            var passMatch = Regex.Match(
                clean,
                @"([0-9]+(?:\.[0-9]+)?)\s+seconds per pass",
                RegexOptions.IgnoreCase);
            double secondsPerPass = passMatch.Success
                ? double.Parse(passMatch.Groups[1].Value, CultureInfo.InvariantCulture)
                : stopwatch.Elapsed.TotalSeconds;

            return new GpuProbeSample(
                fixedNgl,
                parsed.Ppl > 0,
                secondsPerPass,
                stopwatch.Elapsed.TotalSeconds);
        }
        catch
        {
            return new GpuProbeSample(fixedNgl, false, 0, stopwatch.Elapsed.TotalSeconds);
        }
    }

    private async Task<BenchmarkExecutionPlan?> TryLoadCachedExecutionPlanAsync(
        ExecutionPlanCacheKey key,
        string? nativeModelPath,
        string nativeQuantizationKey,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var aiModelHashId = await GetOrCreateAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, aiModelHashId, createIfMissing: false, ct);
        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();

        // Hardware execution-plan probes are intentionally NOT invalidated by tensor grouping
        // profile changes. Regex/profile edits change benchmark/learned-truth semantics, but the
        // Q8/native hardware capability plan is still valid for the same architecture family,
        // exact model hash, imatrix identity, quantized artifact fingerprint, hardware, and token
        // target. Prefer a current-profile row when present, then fall back to the newest
        // compatible row from any prior TensorGroupProfile.
        var compatibleRows = await db.ExecutionPlanProbeCaches
            .AsNoTracking()
            .Where(x =>
                x.ArchitectureFamilyId == architectureFamilyId &&
                x.AiModelHashId == aiModelHashId &&
                x.ImatrixDefinitionId == imatrixDefinitionId &&
                x.HardwareFingerprint == key.HardwareFingerprint &&
                x.QuantizedModelFingerprint == key.QuantizedModelFingerprint &&
                x.QuantizationKey == key.QuantizationKey &&
                x.DiscoveryTokenTarget == key.DiscoveryTokenTarget)
            .OrderByDescending(x => x.TensorGroupProfileId == tensorGroupProfileId)
            .ThenByDescending(x => x.UpdatedUtc)
            .ThenByDescending(x => x.CreatedUtc)
            .ToListAsync(ct);

        var row = compatibleRows.FirstOrDefault();

        if (row == null)
            return null;

        if (row.TensorGroupProfileId != tensorGroupProfileId)
        {
            AnsiConsole.MarkupLine(
                $"[grey]Execution-plan cache reused from prior tensor profile {row.TensorGroupProfileId}; hardware probe cache is profile-compatible.[/]");
        }

        if (row.ProbeSchemaVersion < DynamicProbeSchemaVersion)
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache row uses old probe schema; re-probing.[/]");
            return null;
        }

        if (row.GpuMemoryLimitsJson != SerializeGpuMemoryLimits())
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache row GPU memory limits differ from current config; re-probing.[/]");
            return null;
        }

        string normalizedNativeQuantizationKey = (nativeQuantizationKey ?? string.Empty).Trim().ToUpperInvariant();
        if (!string.Equals((row.NativeQuantizationKey ?? string.Empty).Trim().ToUpperInvariant(), normalizedNativeQuantizationKey, StringComparison.Ordinal))
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache native quantization key changed; re-probing.[/]");
            return null;
        }

        if (!string.IsNullOrWhiteSpace(nativeModelPath))
        {
            ulong nativeSize = TryGetModelSize(Path.GetFullPath(nativeModelPath));
            if (nativeSize > 0 && row.NativeModelSizeBytes != nativeSize)
            {
                AnsiConsole.MarkupLine("[yellow]Execution-plan cache native model size changed; re-probing.[/]");
                return null;
            }
        }

        if (!row.UsesGpu)
        {
            return BenchmarkExecutionPlan.CreateCpuPlan(key.PlanModelPath) with
            {
                ProbeSchemaVersion = row.ProbeSchemaVersion,
                Q8ModelSizeBytes = row.Q8ModelSizeBytes,
                NativeModelSizeBytes = row.NativeModelSizeBytes,
                NativeQuantizationKey = row.NativeQuantizationKey ?? string.Empty,
                GpuMemoryLimitsJson = row.GpuMemoryLimitsJson ?? "{}"
            };
        }

        if (!BenchmarkTopologyCacheCodec.TryDeserialize(
                row.SlotsJson,
                out int maxOffloadNgl,
                out ulong independentMaxModelSizeBytes,
                out var sharedProfile,
                out var independentProfile) ||
            sharedProfile == null ||
            independentProfile == null)
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache topology JSON was unreadable. Re-probing.[/]");
            return null;
        }

        foreach (var slot in sharedProfile.Slots.Concat(independentProfile.Slots))
        {
            _ = BuildTensorSplitArgs(slot, LlamaGpuTool.CommonCli);
        }

        // NativeStableNgl == 0 is valid and represents "native anchor unavailable"
        // while still running a Q8-based GPU plan.
        if (row.UsesGpu &&
            (row.Q8ModelSizeBytes == 0 || row.NativeModelSizeBytes == 0 || row.Q8StableNgl <= 0))
        {
            AnsiConsole.MarkupLine("[yellow]Execution-plan cache row is missing dynamic anchor metadata; re-probing.[/]");
            return null;
        }

        return new BenchmarkExecutionPlan(
            PlanModelPath: key.PlanModelPath,
            StaticNgl: row.StaticNgl,
            UsesGpu: row.UsesGpu,
            GroupSize: row.GroupSize,
            Slots: sharedProfile.Slots,
            ProbeSchemaVersion: row.ProbeSchemaVersion,
            Q8ModelSizeBytes: row.Q8ModelSizeBytes,
            Q8StableNgl: row.Q8StableNgl,
            NativeModelSizeBytes: row.NativeModelSizeBytes,
            NativeStableNgl: row.NativeStableNgl,
            NativeQuantizationKey: row.NativeQuantizationKey ?? string.Empty,
            MaxCandidateNgl: maxOffloadNgl,
            GpuMemoryLimitsJson: row.GpuMemoryLimitsJson ?? "{}",
            TensorSplitJson: row.TensorSplitJson ?? "{}",
            IndependentSlots: independentProfile.Slots,
            IndependentMaxModelSizeBytes: independentMaxModelSizeBytes,
            SharedMeasuredJobsPerSecond: sharedProfile.MeasuredJobsPerSecond,
            SharedMeasuredSecondsPerPass: sharedProfile.MeasuredSecondsPerPass,
            IndependentMeasuredJobsPerSecond: independentProfile.MeasuredJobsPerSecond,
            IndependentMeasuredSecondsPerPass: independentProfile.MeasuredSecondsPerPass);
    }

    private async Task UpsertCachedExecutionPlanAsync(
        ExecutionPlanCacheKey key,
        BenchmarkExecutionPlan plan,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var aiModelHashId = await GetOrCreateAiModelHashIdAsync(db, ct);
        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, aiModelHashId, createIfMissing: true, ct);
        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();

        var existing = await db.ExecutionPlanProbeCaches
            .FirstOrDefaultAsync(x =>
                x.ArchitectureFamilyId == architectureFamilyId &&
                x.TensorGroupProfileId == tensorGroupProfileId &&
                x.AiModelHashId == aiModelHashId &&
                x.ImatrixDefinitionId == imatrixDefinitionId &&
                x.HardwareFingerprint == key.HardwareFingerprint &&
                x.QuantizedModelFingerprint == key.QuantizedModelFingerprint &&
                x.QuantizationKey == key.QuantizationKey &&
                x.DiscoveryTokenTarget == key.DiscoveryTokenTarget, ct);

        string slotsJson = plan.UsesGpu
            ? BenchmarkTopologyCacheCodec.Serialize(
                plan.MaxCandidateNgl,
                plan.IndependentMaxModelSizeBytes,
                plan.SharedProfile,
                plan.IndependentProfile)
            : "[]";
        var now = DateTime.UtcNow;

        if (existing == null)
        {
            existing = new ExecutionPlanProbeCache
            {
                ArchitectureFamilyId = architectureFamilyId,
                TensorGroupProfileId = tensorGroupProfileId,
                AiModelHashId = aiModelHashId,
                ImatrixDefinitionId = imatrixDefinitionId,
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
        existing.ProbeSchemaVersion = plan.ProbeSchemaVersion;
        existing.Q8ModelSizeBytes = plan.Q8ModelSizeBytes;
        existing.Q8StableNgl = plan.Q8StableNgl;
        existing.NativeModelSizeBytes = plan.NativeModelSizeBytes;
        existing.NativeStableNgl = plan.NativeStableNgl;
        existing.NativeQuantizationKey = plan.NativeQuantizationKey;
        existing.MaxCandidateNgl = plan.MaxCandidateNgl;
        existing.GpuMemoryLimitsJson = plan.GpuMemoryLimitsJson;
        existing.TensorSplitJson = plan.TensorSplitJson;
        existing.UpdatedUtc = now;

        await db.SaveChangesAsync(ct);
    }

    private static ExecutionPlanCacheKey BuildExecutionPlanCacheKey(
        string planModelPath,
        int discoveryTokenTarget,
        string quantizationKey)
    {
        string quantizedModelFingerprint = BuildQuantizedModelFingerprint(quantizationKey);

        var sys = Cache.SysInfo;
        string hardwareFingerprint = sys == null
            ? $"unknown-hardware|llama:{BuildLlamaRuntimeFingerprint()}"
            : string.Join("|", new[]
            {
                $"threads:{sys.ThreadCount}",
                $"ram:{sys.RamGb:F2}",
                $"gpu:{string.Join(";", sys.GpuInfo.Select(g => $"{g.GpuVendor}:{g.GpuName}:{g.VramGb:F2}:{g.UniqueId ?? "none"}"))}",
                $"llama:{BuildLlamaRuntimeFingerprint()}"
            });

        return new ExecutionPlanCacheKey(
            hardwareFingerprint,
            quantizedModelFingerprint,
            quantizationKey,
            discoveryTokenTarget,
            planModelPath);
    }

    private static string BuildLlamaRuntimeFingerprint()
    {
        try
        {
            var bins = new LlamaBinaries(Cache.LlamaRoot);
            var ppl = new FileInfo(bins.Ppl);
            if (!ppl.Exists)
                return "missing";

            return $"ppl:{ppl.Length}:{ppl.LastWriteTimeUtc.Ticks}";
        }
        catch
        {
            return "unknown";
        }
    }

    private static string BuildQuantizedModelFingerprint(string quantizationKey)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        string imatrix = Cache.IsImatrixAvailable ? (Cache.ActiveImatrixIdentityHash ?? "imatrix-unknown") : "no-imatrix";
        string family = string.IsNullOrWhiteSpace(Cache.CurrentArchitectureFamilyName) ? Cache.CurrentModelId : Cache.CurrentArchitectureFamilyNormalizedName;
        return $"family:{family}|model:{Cache.CurrentModelId}|imatrix:{imatrix}|quant:{quantizationKey}";
    }

    private static string SerializeGpuMemoryLimits()
    {
        var ordered = Cache.GpuMemoryLimitsGb
            .OrderBy(x => x.Key)
            .ToDictionary(x => x.Key, x => x.Value);
        return JsonSerializer.Serialize(ordered);
    }

    private static string SerializeTensorSplitMap(IReadOnlyList<BenchmarkSlot> slots)
    {
        var map = slots
            .Where(s => s.UsesGpu && s.DeviceIndices.Length > 1)
            .ToDictionary(
                s => s.DisplayName,
                s => BuildTensorSplitArgs(s, LlamaGpuTool.CommonCli).Trim(),
                StringComparer.Ordinal);
        return JsonSerializer.Serialize(map);
    }

    private static string BuildTensorSplitArgs(BenchmarkSlot slot, LlamaGpuTool tool)
    {
        try
        {
            return LlamaGpuArgumentBuilder.BuildTensorSplitArgs(
                slot.DeviceIndices,
                Cache.GpuMemoryLimitsGb,
                tool);
        }
        catch (InvalidOperationException ex)
        {
            throw new InvalidOperationException(
                $"Invalid tensor split for slot {slot.DisplayName}: {ex.Message}", ex);
        }
    }

    private static async Task<uint> GetOrCreateAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (model == null)
        {
            model = new AiModelHash { UniqueHash = Cache.CurrentModelId };
            db.AiModelHashes.Add(model);
            await db.SaveChangesAsync(ct);
        }

        return model.Id;
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
            ? $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {fixedNgl}{BuildTensorSplitArgs(slot, LlamaGpuTool.LlamaBench)} -o md"
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
            ? $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {fixedNgl}{BuildTensorSplitArgs(slot, LlamaGpuTool.CommonCli)} -t 4 -c 2048 --file \"{corpusPath}\""
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

    private static async ValueTask<GpuResourceScheduler.GpuResourceLease> AcquireBenchmarkSlotAsync(
        ulong modelSizeBytes,
        bool allowIndependentTopology,
        CancellationToken ct = default)
    {
        if (_currentPlan == null)
            throw new InvalidOperationException(
                "Benchmark execution plan has not been initialized. Call EnsureExecutionPlanAsync() first.");

        BenchmarkTopologyProfile profile = BenchmarkGpuPlanner.ShouldUseIndependentTopology(
                modelSizeBytes,
                _currentPlan.IndependentMaxModelSizeBytes,
                _currentPlan.IndependentProfile.Slots.Count,
                allowIndependentTopology)
                ? _currentPlan.IndependentProfile
                : _currentPlan.SharedProfile;

        return await _resourceScheduler.AcquireAsync(profile.Slots, ct);
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

    private int ResolveDynamicNglForModel(ulong modelSizeBytes, BenchmarkSlot slot)
    {
        if (_currentPlan == null || !_currentPlan.UsesGpu || !slot.UsesGpu)
            return 0;

        if (modelSizeBytes == 0)
            return Math.Max(0, _currentPlan.StaticNgl);

        if (_currentPlan.Q8ModelSizeBytes == 0 || _currentPlan.Q8StableNgl <= 0)
            return Math.Max(0, _currentPlan.StaticNgl);

        int maxNgl = _currentPlan.MaxCandidateNgl > 0 ? _currentPlan.MaxCandidateNgl : _currentPlan.StaticNgl;
        int slotQ8StableNgl = slot.Q8StableNgl > 0
            ? slot.Q8StableNgl
            : _currentPlan.Q8StableNgl;

        if (modelSizeBytes <= _currentPlan.Q8ModelSizeBytes)
        {
            return BenchmarkGpuPlanner.ResolveNglForModel(
                _currentPlan.Q8ModelSizeBytes,
                slotQ8StableNgl,
                maxNgl,
                modelSizeBytes);
        }

        double estimateRaw;
        if (_currentPlan.NativeModelSizeBytes > _currentPlan.Q8ModelSizeBytes && _currentPlan.NativeStableNgl > 0 && modelSizeBytes < _currentPlan.NativeModelSizeBytes)
        {
            double t = (modelSizeBytes - _currentPlan.Q8ModelSizeBytes) / (double)(_currentPlan.NativeModelSizeBytes - _currentPlan.Q8ModelSizeBytes);
            estimateRaw = Math.Floor(slotQ8StableNgl + ((_currentPlan.NativeStableNgl - slotQ8StableNgl) * t));
        }
        else if (_currentPlan.NativeModelSizeBytes > 0 && _currentPlan.NativeStableNgl > 0)
        {
            estimateRaw = Math.Floor(_currentPlan.NativeStableNgl * (_currentPlan.NativeModelSizeBytes / (double)modelSizeBytes));
            estimateRaw = Math.Min(estimateRaw, _currentPlan.NativeStableNgl);
        }
        else
        {
            estimateRaw = Math.Floor(slotQ8StableNgl * (_currentPlan.Q8ModelSizeBytes / (double)modelSizeBytes));
        }

        int estimate = (int)Math.Clamp(estimateRaw, 0, maxNgl);
        return Math.Max(0, Math.Min(estimate, maxNgl));
    }

    private static List<int> BuildNglFallbackList(int startNgl)
    {
        var result = new List<int> { Math.Max(0, startNgl) };
        result.AddRange(Enumerable.Range(Math.Max(1, startNgl - 4), Math.Min(4, Math.Max(0, startNgl - 1)))
            .Reverse());
        result.AddRange(LegacyNglFallbacks.Where(x => x < startNgl).OrderByDescending(x => x));
        if (!result.Contains(0))
            result.Add(0);

        return result.Distinct().ToList();
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

        if (!reused.ModelSizeBytes.HasValue || reused.ModelSizeBytes.Value == 0)
        {
            // Reuse cannot safely persist DB truth with unknown size.
            // This is expected for transient scratch samples where modelPath may be intentionally empty.
            return false;
        }

        using var db = new MagicQuantContext();

        var identity = await GetOrCreateBenchmarkIdentityAsync(db, quantConfig);
        await SaveBenchmarkToDbAsync(
            db: db,
            model: identity.AiModelHash,
            combo: identity.TensorCombo,
            imatrixDefinitionId: identity.ImatrixDefinitionId,
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
        IReadOnlyCollection<string>? domainsOverride = null,
        bool allowIndependentGpuTopology = true)
    {
        Directory.CreateDirectory(benchDir);

        var requestedDomains = ResolveRequestedDomains(quantConfig, domainsOverride);
        bool requireKld = RequiresKld(quantConfig);

        if (Cache.SuppressBenchmarkPersistence)
        {
            return await RunAllBenchmarksTransientAsync(
                quantConfig: quantConfig,
                modelPath: modelPath,
                benchDir: benchDir,
                tokenTarget: tokenTarget,
                klLogitsDir: klLogitsDir,
                saveLogits: saveLogits,
                requestedDomains: requestedDomains,
                requireKld: requireKld,
                allowIndependentGpuTopology: allowIndependentGpuTopology);
        }

        using var db = new MagicQuantContext();

        var identity = await GetOrCreateBenchmarkIdentityAsync(db, quantConfig);
        var aiModelHash = identity.AiModelHash;
        var tensorCombo = identity.TensorCombo;

        var existingBench = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .AsNoTracking()
            .FirstOrDefaultAsync(b => b.ArchitectureFamilyId == TensorGroupProfileService.RequireCurrentArchitectureFamilyId() && b.TensorGroupProfileId == TensorGroupProfileService.RequireCurrentProfileId() && b.AiModelHashId == aiModelHash.Id && b.ImatrixDefinitionId == identity.ImatrixDefinitionId && b.TensorComboId == tensorCombo.Id);

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
                imatrixDefinitionId: identity.ImatrixDefinitionId,
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
            .FirstOrDefaultAsync(x => x.ArchitectureFamilyId == TensorGroupProfileService.RequireCurrentArchitectureFamilyId() && x.TensorGroupProfileId == TensorGroupProfileService.RequireCurrentProfileId() && x.AiModelHashId == aiModelHash.Id && x.ImatrixDefinitionId == identity.ImatrixDefinitionId && x.TensorComboId == tensorCombo.Id);

        if (trackedBench == null)
        {
            trackedBench = new AiBenchmark
            {
                ArchitectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId(),
                TensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId(),
                AiModelHashId = aiModelHash.Id,
                ImatrixDefinitionId = identity.ImatrixDefinitionId,
                TensorComboId = tensorCombo.Id,
                Ngl = 0,
                SizeBytes = 0,
                TokensPerSecond = 0
            };

            db.AiBenchmarks.Add(trackedBench);
            await db.SaveChangesAsync();
        }

        ulong modelSizeBytes = TryGetModelSize(modelPath);
        await using var slotLease = await AcquireBenchmarkSlotAsync(
            modelSizeBytes,
            allowIndependentGpuTopology);
        var slot = slotLease.Slot;

        int initialNgl = ResolveDynamicNglForModel(modelSizeBytes, slot);
        var runtimeNgl = new RuntimeNglState
        {
            CurrentNgl = initialNgl,
            LastSuccessfulNgl = initialNgl
        };
        AnsiConsole.MarkupLine(
            $"[grey]Dynamic NGL:[/] model={Markup.Escape(Path.GetFileName(modelPath))}, size={(modelSizeBytes / 1024d / 1024d / 1024d):F2} GB, q8={( _currentPlan.Q8ModelSizeBytes / 1024d / 1024d / 1024d):F2} GB/{_currentPlan.Q8StableNgl}, native={(_currentPlan.NativeModelSizeBytes / 1024d / 1024d / 1024d):F2} GB/{_currentPlan.NativeStableNgl}, slot={Markup.Escape(slot.DisplayName)}, chosen={initialNgl}");

        var result = new BenchmarkResult
        {
            ModelSizeBytes = modelSizeBytes
        };

        var executedRunTimings = new List<PendingBenchmarkRunTiming>();

        // Disabled for now. Too many variables that're annoying to track
        result.LlamaBench = new LlamaBenchMetrics
        {
            LogPath = null,
            Backend = slot.UsesGpu ? "disabled" : "cpu-disabled",
            Ngl = runtimeNgl.CurrentNgl,
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
                    $"[yellow]Running Perplexity ({Markup.Escape(domain)})[/] [grey]({Markup.Escape(slot.DisplayName)}, ngl={runtimeNgl.CurrentNgl})[/]");

                var metrics = await RunPplBenchmarkWithNglFallbackAsync(
                    modelPath: modelPath,
                    benchDir: benchDir,
                    domain: domain,
                    corpusPath: corpusPath,
                    slot: slot,
                    runtimeNgl: runtimeNgl,
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
                result.LlamaBench.Ngl = runtimeNgl.LastSuccessfulNgl;

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

                if (!Cache.SuppressBenchmarkPersistence)
                {
                    await PersistFailedBenchmarkRunAsync(
                        db: db,
                        aiModelHashId: aiModelHash.Id,
                        tensorComboId: tensorCombo.Id,
                        aiBenchmarkId: trackedBench.Id,
                        imatrixDefinitionId: identity.ImatrixDefinitionId,
                        category: DomainToCategory(domain),
                        startedUtc: startedUtc,
                        completedUtc: DateTime.UtcNow,
                        error: ex.ToString());
                }

                throw;
            }
        }

        await WriteMetricsJsonAsync(benchDir, result);

        await SaveBenchmarkToDbAsync(
            db: db,
            model: aiModelHash,
            combo: tensorCombo,
            imatrixDefinitionId: identity.ImatrixDefinitionId,
            res: result,
            modelPath: modelPath,
            executedRunTimings: executedRunTimings);

        return result;
    }


    private async Task<BenchmarkResult> RunAllBenchmarksTransientAsync(
        HybridQuant quantConfig,
        string modelPath,
        string benchDir,
        int tokenTarget,
        string? klLogitsDir,
        bool saveLogits,
        IReadOnlyCollection<string> requestedDomains,
        bool requireKld,
        bool allowIndependentGpuTopology)
    {
        if (TryReadExistingBenchmarkArtifacts(benchDir, requestedDomains, requireKld, out var reused))
        {
            reused.ModelSizeBytes ??= TryGetModelSize(modelPath);
            await WriteMetricsJsonAsync(benchDir, reused);
            return reused;
        }

        if (_currentPlan == null)
        {
            throw new InvalidOperationException(
                "No benchmark execution plan has been discovered yet. " +
                "You must call EnsureExecutionPlanAsync() with the pure Q8 model first.");
        }

        ulong modelSizeBytes = TryGetModelSize(modelPath);
        await using var slotLease = await AcquireBenchmarkSlotAsync(
            modelSizeBytes,
            allowIndependentGpuTopology);
        var slot = slotLease.Slot;

        int initialNgl = ResolveDynamicNglForModel(modelSizeBytes, slot);
        var runtimeNgl = new RuntimeNglState
        {
            CurrentNgl = initialNgl,
            LastSuccessfulNgl = initialNgl
        };
        AnsiConsole.MarkupLine(
            $"[grey]Dynamic NGL:[/] model={Markup.Escape(Path.GetFileName(modelPath))}, size={(modelSizeBytes / 1024d / 1024d / 1024d):F2} GB, q8={( _currentPlan.Q8ModelSizeBytes / 1024d / 1024d / 1024d):F2} GB/{_currentPlan.Q8StableNgl}, native={(_currentPlan.NativeModelSizeBytes / 1024d / 1024d / 1024d):F2} GB/{_currentPlan.NativeStableNgl}, slot={Markup.Escape(slot.DisplayName)}, chosen={initialNgl}");

        var result = new BenchmarkResult
        {
            ModelSizeBytes = modelSizeBytes,
            LlamaBench = new LlamaBenchMetrics
            {
                LogPath = null,
                Backend = slot.UsesGpu ? "disabled" : "cpu-disabled",
                Ngl = runtimeNgl.CurrentNgl,
                Test = "disabled",
                Tps = 0
            }
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

            AnsiConsole.MarkupLine(
                $"[yellow]Running transient Perplexity ({Markup.Escape(domain)})[/] [grey]({Markup.Escape(slot.DisplayName)}, ngl={runtimeNgl.CurrentNgl})[/]");

            var metrics = await RunPplBenchmarkWithNglFallbackAsync(
                modelPath: modelPath,
                benchDir: benchDir,
                domain: domain,
                corpusPath: corpusPath,
                slot: slot,
                runtimeNgl: runtimeNgl,
                klLogitsDir: klLogitsDir,
                saveLogits: saveLogits);

            if (requireKld && !HasMeaningfulKld(metrics.Kld))
            {
                throw new InvalidOperationException(
                    $"Non-base transient benchmark produced invalid KLD for domain '{domain}'. " +
                    $"KLD must exist and be > 0. Parsed value: {(metrics.Kld.HasValue ? metrics.Kld.Value.ToString(CultureInfo.InvariantCulture) : "null")}");
            }

            result.Perplexity[domain] = metrics;
            result.LlamaBench.Ngl = runtimeNgl.LastSuccessfulNgl;
        }

        await WriteMetricsJsonAsync(benchDir, result);
        return result;
    }

    // ----------------------------------------------------------------
    // Database helpers
    // ----------------------------------------------------------------

    private async Task<(AiModelHash AiModelHash, TensorCombo TensorCombo, int? ImatrixDefinitionId)> GetOrCreateBenchmarkIdentityAsync(
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

        if (Cache.CurrentArchitectureFamilyId != null)
        {
            uint scopedId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdAsync(db, ct);
            aiModelHash = await db.AiModelHashes.FirstAsync(x => x.Id == scopedId, ct);
        }

        var tensorCombo = await GetOrCreateTensorComboAsync(db, quantConfig, ct);

        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, aiModelHash.Id, createIfMissing: true, ct);
        return (aiModelHash, tensorCombo, imatrixDefinitionId);
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
        int? imatrixDefinitionId,
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

            int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
            int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();

            var bench = await db.AiBenchmarks
                .Include(x => x.CategorBenchmarks)
                .FirstOrDefaultAsync(x =>
                    x.ArchitectureFamilyId == architectureFamilyId &&
                    x.TensorGroupProfileId == tensorGroupProfileId &&
                    x.AiModelHashId == model.Id &&
                    x.ImatrixDefinitionId == imatrixDefinitionId &&
                    x.TensorComboId == combo.Id);

            if (bench == null)
            {
                bench = new AiBenchmark
                {
                    ArchitectureFamilyId = architectureFamilyId,
                    TensorGroupProfileId = tensorGroupProfileId,
                    AiModelHashId = model.Id,
                    ImatrixDefinitionId = imatrixDefinitionId,
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
                        ArchitectureFamilyId = architectureFamilyId,
                        TensorGroupProfileId = tensorGroupProfileId,
                        AiModelHashId = model.Id,
                        ImatrixDefinitionId = imatrixDefinitionId,
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

            await ReplaceBenchmarkLearnedSourcesAsync(db, bench, combo, architectureFamilyId, tensorGroupProfileId);

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


    private static async Task ReplaceBenchmarkLearnedSourcesAsync(
        MagicQuantContext db,
        AiBenchmark bench,
        TensorCombo combo,
        int architectureFamilyId,
        int tensorGroupProfileId)
    {
        await db.AiBenchmarkLearnedSources
            .Where(x => x.AiBenchmarkId == bench.Id)
            .ExecuteDeleteAsync();

        var groupSlots = new (byte GroupId, byte StoredValue)[]
        {
            (TReg.Embeddings.UniqueId, combo.Embeddings),
            (TReg.LmHead.UniqueId, combo.LmHead),
            (TReg.AttnQ.UniqueId, combo.AttnQ),
            (TReg.AttnKV.UniqueId, combo.AttnKV),
            (TReg.AttnOutput.UniqueId, combo.AttnOutput),
            (TReg.FfnUpGate.UniqueId, combo.FfnUpGate),
            (TReg.FfnDown.UniqueId, combo.FfnDown),
            (TReg.MoeExperts.UniqueId, combo.MoeExperts),
            (TReg.MoeRouter.UniqueId, combo.MoeRouter)
        };

        foreach (var (groupId, storedValue) in groupSlots)
        {
            if (storedValue == 0)
                continue;

            var runtimeBaselineId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedValue);
            if (runtimeBaselineId == BaselineQuants.BF16_Hybrid.UniqueId ||
                runtimeBaselineId == BaselineQuants.F16_Hybrid.UniqueId ||
                runtimeBaselineId == BaselineQuants.NativeSourceUniqueId)
                continue;

            var definition = await db.BaselineQuantDefinitions
                .AsNoTracking()
                .Where(x => (x.ArchitectureFamilyId == architectureFamilyId || x.ArchitectureFamilyId == null) &&
                            x.RuntimeBaselineId == runtimeBaselineId)
                .OrderByDescending(x => x.ArchitectureFamilyId.HasValue)
                .FirstOrDefaultAsync();

            if (definition == null)
                continue;

            db.AiBenchmarkLearnedSources.Add(new AiBenchmarkLearnedSource
            {
                Id = Guid.NewGuid(),
                AiBenchmarkId = bench.Id,
                ArchitectureFamilyId = architectureFamilyId,
                TensorGroupProfileId = tensorGroupProfileId,
                TensorComboId = combo.Id,
                TensorGroupId = groupId,
                BaselineQuantDefinitionId = definition.Id,
                SourceLearningBenchmarkId = null,
                BaselineCanonicalKey = definition.CanonicalKey,
                CreatedUtc = DateTime.UtcNow
            });
        }

        await db.SaveChangesAsync();
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
        int? imatrixDefinitionId,
        byte category,
        DateTime startedUtc,
        DateTime completedUtc,
        string error)
    {
        db.BenchmarkRuns.Add(new BenchmarkRun
        {
            Id = Guid.NewGuid(),
            ArchitectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId(),
            TensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId(),
            AiModelHashId = aiModelHashId,
            ImatrixDefinitionId = imatrixDefinitionId,
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
            ? $"\"{_bins.Bench}\" -m \"{modelPath}\" -p 8 -t 16 -ngl {fixedNgl}{BuildTensorSplitArgs(slot, LlamaGpuTool.LlamaBench)} -o md"
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
            ? $"\"{_bins.Ppl}\" -m \"{modelPath}\" -ngl {fixedNgl}{BuildTensorSplitArgs(slot, LlamaGpuTool.CommonCli)} -t 4 -c 2048 --file \"{corpusPath}\" {kldArgs}"
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

    private async Task<PplMetrics> RunPplBenchmarkWithNglFallbackAsync(
        string modelPath,
        string benchDir,
        string domain,
        string corpusPath,
        BenchmarkSlot slot,
        RuntimeNglState runtimeNgl,
        string? klLogitsDir,
        bool saveLogits)
    {
        var fallbackNgls = BuildNglFallbackList(runtimeNgl.CurrentNgl);
        Exception? lastException = null;

        for (int i = 0; i < fallbackNgls.Count; i++)
        {
            int ngl = fallbackNgls[i];
            try
            {
                var metrics = await RunPplBenchmarkAsync(
                    modelPath: modelPath,
                    benchDir: benchDir,
                    domain: domain,
                    corpusPath: corpusPath,
                    fixedNgl: ngl,
                    slot: slot,
                    klLogitsDir: klLogitsDir,
                    saveLogits: saveLogits);

                runtimeNgl.CurrentNgl = ngl;
                runtimeNgl.LastSuccessfulNgl = ngl;
                return metrics;
            }
            catch (Exception ex)
            {
                lastException = ex;
                string content = ex.ToString();
                bool retryable = LooksLikeRetryableGpuFailure(content);
                if (!retryable || i == fallbackNgls.Count - 1)
                    throw;

                int next = fallbackNgls[i + 1];
                AnsiConsole.MarkupLine(
                    $"[yellow]GPU failure at ngl={ngl} for model {Markup.Escape(Path.GetFileName(modelPath))}; retrying at ngl={next}.[/]");
            }
        }

        throw lastException ?? new InvalidOperationException("Perplexity benchmark failed after NGL fallbacks.");
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
                int retryNgl = ExtractNglFromCommand(cmd);
                AnsiConsole.MarkupLine(
                    $"[yellow]Transient benchmark failure detected at ngl={retryNgl} on slot {slot.SlotId} ({Markup.Escape(slot.DisplayName)}); retrying same NGL once before fallback.[/]");
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

    private static int ExtractNglFromCommand(string cmd)
    {
        var match = Regex.Match(cmd, @"(?:\s-ngl\s+)(\d+)", RegexOptions.IgnoreCase);
        if (match.Success && int.TryParse(match.Groups[1].Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int ngl))
            return ngl;

        return 0;
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
        if (tokenTarget <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokenTarget), "Token target must be greater than zero.");

        if (IsPplCorpusUsable(outPath, tokenTarget))
            return;

        AnsiConsole.MarkupLine($"[grey]Generating corpus for domain: {domain}[/]");

        string pyScript = $@"
import sys
from datasets import load_dataset

domain = '{domain}'
out_path = r'{outPath}'
max_chars = {tokenTarget} * {PplCharsPerTokenEstimate}

def get_sources(d):
    if d == 'general': return [('{GeneralPplDatasetId}', 'wikitext-103-raw-v1', 'test', 'text'), ('{GeneralPplDatasetId}', 'wikitext-2-raw-v1', 'test', 'text')]
    if d == 'code': return [('codeparrot/codeparrot-clean', None, 'train', 'content')]
    if d == 'math': return [('{MathPplDatasetId}', 'main', 'test', 'question')]
    return []

parts = []
total = 0
source_errors = []
for ds, conf, split, field in get_sources(domain):
    try:
        load_args = {{'split': split, 'streaming': True}}
        d = load_dataset(ds, conf, **load_args) if conf else load_dataset(ds, **load_args)
        for row in d:
            text = row.get(field)
            if not text or not isinstance(text, str):
                continue
            chunk = text.strip() + '\n'
            parts.append(chunk)
            total += len(chunk)
            if total >= max_chars:
                break
    except Exception as e:
        message = f'Error loading {{ds}}: {{e}}'
        source_errors.append(message)
        print(message, file=sys.stderr)
    if total >= max_chars:
        break

if total < max_chars:
    details = '; '.join(source_errors) or 'dataset sources returned insufficient text'
    raise RuntimeError(
        f'Failed to build corpus for domain {{domain!r}}: collected '
        f'{{total}} of {{max_chars}} required characters. {{details}}'
    )

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

        await _pyManager.RunPipInstallAsync("datasets");
        var generationResult = await RunShellCommandAsync(runner + " " + args, null);

        if (File.Exists(scriptPath))
            File.Delete(scriptPath);

        if (!generationResult.Success)
        {
            throw new InvalidOperationException(
                $"Failed to generate perplexity corpus for domain '{domain}'.\n\n{generationResult.LogOutput}");
        }

        if (!IsPplCorpusUsable(outPath, tokenTarget))
        {
            throw new InvalidOperationException(
                $"Generated perplexity corpus for domain '{domain}' did not contain the required " +
                $"{(long)tokenTarget * PplCharsPerTokenEstimate:N0} characters: {outPath}");
        }
    }

    internal static bool IsPplCorpusUsable(string path, int tokenTarget)
    {
        if (tokenTarget <= 0 || !File.Exists(path))
            return false;

        try
        {
            long minimumCharacters = (long)tokenTarget * PplCharsPerTokenEstimate;
            using var reader = new StreamReader(path);
            var buffer = new char[4096];
            long totalCharacters = 0;

            while (totalCharacters < minimumCharacters)
            {
                int read = reader.Read(buffer, 0, buffer.Length);
                if (read == 0)
                    break;

                totalCharacters += read;
            }

            return totalCharacters >= minimumCharacters;
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
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

    private sealed record BenchmarkExecutionPlan(
        string PlanModelPath,
        int StaticNgl,
        bool UsesGpu,
        int GroupSize,
        IReadOnlyList<BenchmarkSlot> Slots,
        int ProbeSchemaVersion = DynamicProbeSchemaVersion,
        ulong Q8ModelSizeBytes = 0,
        int Q8StableNgl = 0,
        ulong NativeModelSizeBytes = 0,
        int NativeStableNgl = 0,
        string NativeQuantizationKey = "",
        int MaxCandidateNgl = 0,
        string GpuMemoryLimitsJson = "{}",
        string TensorSplitJson = "{}",
        IReadOnlyList<BenchmarkSlot>? IndependentSlots = null,
        ulong IndependentMaxModelSizeBytes = 0,
        double SharedMeasuredJobsPerSecond = 0,
        double SharedMeasuredSecondsPerPass = 0,
        double IndependentMeasuredJobsPerSecond = 0,
        double IndependentMeasuredSecondsPerPass = 0)
    {
        public BenchmarkTopologyProfile SharedProfile => new(
            "shared",
            Slots,
            SharedMeasuredJobsPerSecond,
            SharedMeasuredSecondsPerPass);

        public BenchmarkTopologyProfile IndependentProfile => new(
            "independent",
            IndependentSlots ?? [],
            IndependentMeasuredJobsPerSecond,
            IndependentMeasuredSecondsPerPass);

        public static BenchmarkExecutionPlan CreateCpuPlan(string q8ModelPath)
            => new(
                PlanModelPath: q8ModelPath,
                StaticNgl: 0,
                UsesGpu: false,
                GroupSize: 0,
                Slots: new List<BenchmarkSlot> { new(0, Array.Empty<int>()) },
                ProbeSchemaVersion: DynamicProbeSchemaVersion,
                Q8ModelSizeBytes: 0,
                Q8StableNgl: 0,
                NativeModelSizeBytes: 0,
                NativeStableNgl: 0,
                NativeQuantizationKey: string.Empty,
                MaxCandidateNgl: 0,
                GpuMemoryLimitsJson: SerializeGpuMemoryLimits(),
                TensorSplitJson: "{}");
    }

    public sealed class RuntimeNglState
    {
        public int CurrentNgl { get; set; }
        public int LastSuccessfulNgl { get; set; }
    }

    private sealed record ExecutionPlanCacheKey(
        string HardwareFingerprint,
        string QuantizedModelFingerprint,
        string QuantizationKey,
        int DiscoveryTokenTarget,
        string PlanModelPath);
}
