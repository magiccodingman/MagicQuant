using System.Text.Json;

namespace MagicQuant.Services;

internal sealed record GpuProbeSample(
    int Ngl,
    bool Success,
    double SecondsPerPass,
    double ElapsedSeconds);

internal sealed record BenchmarkSlot(
    int SlotId,
    string ProfileName,
    int[] DeviceIndices,
    int Q8StableNgl,
    IReadOnlyList<GpuProbeSample> ProbeSamples)
{
    public BenchmarkSlot(int slotId, int[] deviceIndices)
        : this(slotId, "default", deviceIndices, 0, [])
    {
    }

    public bool UsesGpu => DeviceIndices.Length > 0;
    public int DeviceCount => DeviceIndices.Length;

    public string DisplayName =>
        UsesGpu
            ? $"{ProfileName}:GPU[{string.Join(",", DeviceIndices)}]"
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

internal sealed record BenchmarkTopologyProfile(
    string Name,
    IReadOnlyList<BenchmarkSlot> Slots,
    double MeasuredJobsPerSecond,
    double MeasuredSecondsPerPass);

internal sealed class BenchmarkTopologyCacheEnvelope
{
    public int Version { get; set; } = 1;
    public int MaxOffloadNgl { get; set; }
    public ulong IndependentMaxModelSizeBytes { get; set; }
    public BenchmarkTopologyCacheProfile Shared { get; set; } = new();
    public BenchmarkTopologyCacheProfile Independent { get; set; } = new();
}

internal sealed class BenchmarkTopologyCacheProfile
{
    public string Name { get; set; } = string.Empty;
    public double MeasuredJobsPerSecond { get; set; }
    public double MeasuredSecondsPerPass { get; set; }
    public List<BenchmarkTopologyCacheSlot> Slots { get; set; } = new();
}

internal sealed class BenchmarkTopologyCacheSlot
{
    public int SlotId { get; set; }
    public int[] DeviceIndices { get; set; } = [];
    public int Q8StableNgl { get; set; }
    public List<GpuProbeSample> ProbeSamples { get; set; } = new();
}

internal static class BenchmarkTopologyCacheCodec
{
    public static string Serialize(
        int maxOffloadNgl,
        ulong independentMaxModelSizeBytes,
        BenchmarkTopologyProfile shared,
        BenchmarkTopologyProfile independent)
    {
        var envelope = new BenchmarkTopologyCacheEnvelope
        {
            MaxOffloadNgl = maxOffloadNgl,
            IndependentMaxModelSizeBytes = independentMaxModelSizeBytes,
            Shared = ToCacheProfile(shared),
            Independent = ToCacheProfile(independent)
        };

        return JsonSerializer.Serialize(envelope);
    }

    public static bool TryDeserialize(
        string json,
        out int maxOffloadNgl,
        out ulong independentMaxModelSizeBytes,
        out BenchmarkTopologyProfile? shared,
        out BenchmarkTopologyProfile? independent)
    {
        maxOffloadNgl = 0;
        independentMaxModelSizeBytes = 0;
        shared = null;
        independent = null;

        try
        {
            var envelope = JsonSerializer.Deserialize<BenchmarkTopologyCacheEnvelope>(json);
            if (envelope == null ||
                envelope.Version != 1 ||
                envelope.MaxOffloadNgl <= 0 ||
                envelope.Shared.Slots.Count == 0)
            {
                return false;
            }

            maxOffloadNgl = envelope.MaxOffloadNgl;
            independentMaxModelSizeBytes = envelope.IndependentMaxModelSizeBytes;
            shared = FromCacheProfile(envelope.Shared);
            independent = FromCacheProfile(envelope.Independent);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static BenchmarkTopologyCacheProfile ToCacheProfile(BenchmarkTopologyProfile profile)
        => new()
        {
            Name = profile.Name,
            MeasuredJobsPerSecond = profile.MeasuredJobsPerSecond,
            MeasuredSecondsPerPass = profile.MeasuredSecondsPerPass,
            Slots = profile.Slots.Select(x => new BenchmarkTopologyCacheSlot
            {
                SlotId = x.SlotId,
                DeviceIndices = x.DeviceIndices,
                Q8StableNgl = x.Q8StableNgl,
                ProbeSamples = x.ProbeSamples.ToList()
            }).ToList()
        };

    private static BenchmarkTopologyProfile FromCacheProfile(BenchmarkTopologyCacheProfile profile)
        => new(
            string.IsNullOrWhiteSpace(profile.Name) ? "cached" : profile.Name,
            profile.Slots.Select(x => new BenchmarkSlot(
                x.SlotId,
                string.IsNullOrWhiteSpace(profile.Name) ? "cached" : profile.Name,
                x.DeviceIndices ?? [],
                x.Q8StableNgl,
                x.ProbeSamples ?? [])).ToList(),
            profile.MeasuredJobsPerSecond,
            profile.MeasuredSecondsPerPass);
}

internal static class BenchmarkGpuPlanner
{
    internal const double DefaultIndependentSpeedupMargin = 1.10d;
    internal const int NearFullOffloadToleranceLayers = 1;

    public static bool ShouldUseIndependentTopology(
        ulong modelSizeBytes,
        ulong independentMaxModelSizeBytes,
        int independentSlotCount,
        bool allowIndependentTopology)
        => allowIndependentTopology &&
           independentMaxModelSizeBytes > 0 &&
           modelSizeBytes > 0 &&
           modelSizeBytes <= independentMaxModelSizeBytes &&
           independentSlotCount > 0;

    public static int ResolveNglForModel(
        ulong q8ModelSizeBytes,
        int q8StableNgl,
        int maxOffloadNgl,
        ulong modelSizeBytes)
    {
        if (q8StableNgl <= 0 || maxOffloadNgl <= 0)
            return 0;
        if (q8ModelSizeBytes == 0 || modelSizeBytes == 0)
            return Math.Min(q8StableNgl, maxOffloadNgl);

        double scaled = Math.Floor(q8StableNgl * (q8ModelSizeBytes / (double)modelSizeBytes));
        return (int)Math.Clamp(scaled, 0d, maxOffloadNgl);
    }

    public static IReadOnlyList<BenchmarkSlot> RankIndependentSlotsForModel(
        IReadOnlyList<BenchmarkSlot> slots,
        ulong q8ModelSizeBytes,
        int maxOffloadNgl,
        ulong modelSizeBytes,
        int nearFullOffloadToleranceLayers = NearFullOffloadToleranceLayers)
    {
        ArgumentNullException.ThrowIfNull(slots);
        if (slots.Count <= 1 || q8ModelSizeBytes == 0 || modelSizeBytes == 0 || maxOffloadNgl <= 0)
            return slots;

        int nearFullThreshold = Math.Max(0, maxOffloadNgl - Math.Max(0, nearFullOffloadToleranceLayers));
        var ranked = slots
            .Select(slot => new
            {
                Slot = slot,
                Ngl = ResolveNglForModel(
                    q8ModelSizeBytes,
                    slot.Q8StableNgl,
                    maxOffloadNgl,
                    modelSizeBytes)
            })
            .ToArray();

        // A weaker device that is at most one layer shy of full offload is the best fit:
        // using it preserves the stronger device for a larger concurrent model. When only
        // one device is close to full offload, prefer that device. Otherwise use the device
        // that can offload the most layers and accept the unavoidable partial-offload tail.
        if (ranked.Any(x => x.Ngl >= nearFullThreshold))
        {
            return ranked
                .OrderByDescending(x => x.Ngl >= nearFullThreshold)
                .ThenBy(x => x.Ngl >= nearFullThreshold ? x.Slot.Q8StableNgl : int.MaxValue)
                .ThenByDescending(x => x.Ngl)
                .ThenBy(x => x.Slot.SlotId)
                .Select(x => x.Slot)
                .ToArray();
        }

        return ranked
            .OrderByDescending(x => x.Ngl)
            .ThenByDescending(x => x.Slot.Q8StableNgl)
            .ThenBy(x => x.Slot.SlotId)
            .Select(x => x.Slot)
            .ToArray();
    }

    public static ulong EstimateIndependentCrossoverBytes(
        ulong q8ModelSizeBytes,
        int maxOffloadNgl,
        double sharedSecondsPerPass,
        IReadOnlyList<BenchmarkSlot> independentSlots,
        double requiredSpeedup = DefaultIndependentSpeedupMargin)
    {
        if (q8ModelSizeBytes == 0 ||
            maxOffloadNgl <= 0 ||
            sharedSecondsPerPass <= 0 ||
            independentSlots.Count < 2 ||
            requiredSpeedup < 1d)
        {
            return 0;
        }

        var models = independentSlots
            .Select(BuildPassTimeModel)
            .ToArray();

        if (models.Any(x => x == null))
            return 0;

        double requiredJobsPerSecond = (1d / sharedSecondsPerPass) * requiredSpeedup;
        const int scanSteps = 1000;

        // Search from largest to smallest so the returned boundary is the first model size
        // where independent workers have a meaningful, not merely noise-level, advantage.
        for (int step = 0; step <= scanSteps; step++)
        {
            double sizeRatio = 1d - (0.8d * step / scanSteps);
            ulong modelSize = (ulong)Math.Max(1d, Math.Floor(q8ModelSizeBytes * sizeRatio));
            double aggregateJobsPerSecond = 0d;

            for (int i = 0; i < independentSlots.Count; i++)
            {
                var slot = independentSlots[i];
                int ngl = ResolveNglForModel(
                    q8ModelSizeBytes,
                    slot.Q8StableNgl,
                    maxOffloadNgl,
                    modelSize);

                double seconds = models[i]!.Value.EstimateSeconds(ngl);
                if (seconds <= 0 || double.IsNaN(seconds) || double.IsInfinity(seconds))
                {
                    aggregateJobsPerSecond = 0d;
                    break;
                }

                aggregateJobsPerSecond += 1d / seconds;
            }

            if (aggregateJobsPerSecond >= requiredJobsPerSecond)
                return modelSize;
        }

        return 0;
    }

    private static PassTimeModel? BuildPassTimeModel(BenchmarkSlot slot)
    {
        var successful = slot.ProbeSamples
            .Where(x => x.Success && x.Ngl > 0 && x.SecondsPerPass > 0)
            .GroupBy(x => x.Ngl)
            .Select(x => new
            {
                Ngl = x.Key,
                Seconds = x.Average(y => y.SecondsPerPass)
            })
            .OrderBy(x => x.Ngl)
            .ToArray();

        if (successful.Length < 2)
            return null;

        double meanX = successful.Average(x => (double)x.Ngl);
        double meanY = successful.Average(x => x.Seconds);
        double denominator = successful.Sum(x => Math.Pow(x.Ngl - meanX, 2));
        if (denominator <= double.Epsilon)
            return null;

        double slope = successful.Sum(x => (x.Ngl - meanX) * (x.Seconds - meanY)) / denominator;
        if (slope >= 0)
            return null;

        double intercept = meanY - slope * meanX;
        double minimumObserved = successful.Min(x => x.Seconds);
        return new PassTimeModel(intercept, slope, Math.Max(0.05d, minimumObserved * 0.55d));
    }

    private readonly record struct PassTimeModel(double Intercept, double Slope, double MinimumSeconds)
    {
        public double EstimateSeconds(int ngl) => Math.Max(MinimumSeconds, Intercept + Slope * ngl);
    }
}

internal sealed class GpuResourceScheduler
{
    private readonly object _sync = new();
    private readonly HashSet<int> _busyDevices = new();
    private readonly LinkedList<Waiter> _waiters = new();

    public ValueTask<GpuResourceLease> AcquireAsync(
        IReadOnlyList<BenchmarkSlot> candidates,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        if (candidates.Count == 0)
            throw new ArgumentException("At least one benchmark slot candidate is required.", nameof(candidates));

        lock (_sync)
        {
            if (_waiters.Count == 0 && TryReserveFirstAvailable(candidates, out var immediate))
                return ValueTask.FromResult(new GpuResourceLease(this, immediate!));

            var waiter = new Waiter(candidates);
            waiter.Node = _waiters.AddLast(waiter);

            if (ct.CanBeCanceled)
            {
                waiter.Cancellation = ct.Register(
                    static state =>
                    {
                        var pair = ((GpuResourceScheduler Scheduler, Waiter Waiter))state!;
                        pair.Scheduler.Cancel(pair.Waiter);
                    },
                    (this, waiter));
            }

            return new ValueTask<GpuResourceLease>(waiter.Completion.Task);
        }
    }

    private void Cancel(Waiter waiter)
    {
        lock (_sync)
        {
            if (waiter.Node?.List == null)
                return;

            _waiters.Remove(waiter.Node);
            waiter.Node = null;
            waiter.Cancellation.Dispose();
            waiter.Completion.TrySetCanceled();
        }
    }

    private bool TryReserveFirstAvailable(
        IReadOnlyList<BenchmarkSlot> candidates,
        out BenchmarkSlot? selected)
    {
        selected = candidates.FirstOrDefault(slot =>
            ReservationKeys(slot).All(device => !_busyDevices.Contains(device)));

        if (selected == null)
            return false;

        foreach (int device in ReservationKeys(selected))
            _busyDevices.Add(device);

        return true;
    }

    private void Release(BenchmarkSlot slot)
    {
        List<(Waiter Waiter, BenchmarkSlot Slot)> ready = new();

        lock (_sync)
        {
            foreach (int device in ReservationKeys(slot))
                _busyDevices.Remove(device);

            while (_waiters.First != null)
            {
                var waiter = _waiters.First.Value;
                if (!TryReserveFirstAvailable(waiter.Candidates, out var selected))
                    break;

                _waiters.RemoveFirst();
                waiter.Node = null;
                waiter.Cancellation.Dispose();
                ready.Add((waiter, selected!));
            }
        }

        foreach (var item in ready)
            item.Waiter.Completion.TrySetResult(new GpuResourceLease(this, item.Slot));
    }

    private static IEnumerable<int> ReservationKeys(BenchmarkSlot slot) =>
        slot.DeviceIndices.Length == 0 ? [-1] : slot.DeviceIndices;

    private sealed class Waiter
    {
        public IReadOnlyList<BenchmarkSlot> Candidates { get; }
        public TaskCompletionSource<GpuResourceLease> Completion { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        public LinkedListNode<Waiter>? Node { get; set; }
        public CancellationTokenRegistration Cancellation { get; set; }

        public Waiter(IReadOnlyList<BenchmarkSlot> candidates)
        {
            Candidates = candidates;
        }
    }

    internal sealed class GpuResourceLease : IAsyncDisposable
    {
        private GpuResourceScheduler? _owner;
        public BenchmarkSlot Slot { get; }

        internal GpuResourceLease(GpuResourceScheduler owner, BenchmarkSlot slot)
        {
            _owner = owner;
            Slot = slot;
        }

        public ValueTask DisposeAsync()
        {
            Interlocked.Exchange(ref _owner, null)?.Release(Slot);
            return ValueTask.CompletedTask;
        }
    }
}
