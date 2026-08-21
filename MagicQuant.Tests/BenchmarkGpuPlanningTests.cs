using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public class BenchmarkGpuPlanningTests
{
    private const ulong Q8Size = 29_047_084_736UL;

    [Theory]
    [InlineData(29_047_084_736UL, 44)]
    [InlineData(21_998_012_096UL, 58)]
    [InlineData(17_624_999_616UL, 66)]
    public void ResolveNglForModel_ScalesAndClampsAtFullOffload(ulong size, int expected)
    {
        int result = BenchmarkGpuPlanner.ResolveNglForModel(Q8Size, 44, 66, size);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void EstimateIndependentCrossover_UsesMeasuredPerSlotScaling()
    {
        var slots = new[]
        {
            Slot(0, 44, (35, 6.59), (41, 5.83), (44, 5.45)),
            Slot(1, 57, (48, 4.38), (56, 3.28), (57, 3.17))
        };

        ulong crossover = BenchmarkGpuPlanner.EstimateIndependentCrossoverBytes(
            Q8Size,
            maxOffloadNgl: 66,
            sharedSecondsPerPass: 1.60,
            slots);

        double gib = crossover / 1024d / 1024d / 1024d;
        Assert.InRange(gib, 22d, 27d);
    }

    [Theory]
    [InlineData(true, 20_000_000_000UL, true)]
    [InlineData(false, 20_000_000_000UL, false)]
    [InlineData(true, 26_000_000_000UL, false)]
    public void ShouldUseIndependentTopology_RequiresConcurrentBatchIntent(
        bool allowIndependentTopology,
        ulong modelSizeBytes,
        bool expected)
    {
        bool result = BenchmarkGpuPlanner.ShouldUseIndependentTopology(
            modelSizeBytes,
            independentMaxModelSizeBytes: 25_000_000_000UL,
            independentSlotCount: 2,
            allowIndependentTopology);

        Assert.Equal(expected, result);
    }

    [Fact]
    public void RankIndependentSlots_NearFullCandidateUsesWeakerDevice()
    {
        var slots = new[]
        {
            Slot(0, 44, (35, 6.5), (44, 5.4)),
            Slot(1, 57, (48, 4.3), (57, 3.1))
        };

        var ranked = BenchmarkGpuPlanner.RankIndependentSlotsForModel(
            slots,
            Q8Size,
            maxOffloadNgl: 66,
            modelSizeBytes: 19_500_000_000UL);

        Assert.Equal(0, ranked[0].DeviceIndices[0]);
        Assert.Equal(65, BenchmarkGpuPlanner.ResolveNglForModel(Q8Size, 44, 66, 19_500_000_000UL));
    }

    [Fact]
    public void RankIndependentSlots_LargeCandidateUsesStrongerDevice()
    {
        var slots = new[]
        {
            Slot(0, 44, (35, 6.5), (44, 5.4)),
            Slot(1, 57, (48, 4.3), (57, 3.1))
        };

        var ranked = BenchmarkGpuPlanner.RankIndependentSlotsForModel(
            slots,
            Q8Size,
            maxOffloadNgl: 66,
            modelSizeBytes: 22_900_000_000UL);

        Assert.Equal(1, ranked[0].DeviceIndices[0]);
        Assert.True(
            BenchmarkGpuPlanner.ResolveNglForModel(Q8Size, 44, 66, 22_900_000_000UL) <
            BenchmarkGpuPlanner.ResolveNglForModel(Q8Size, 57, 66, 22_900_000_000UL));
    }

    [Fact]
    public async Task ResourceScheduler_UsesCandidateSpecificSlotRanking()
    {
        var scheduler = new GpuResourceScheduler();
        var slots = new[]
        {
            Slot(0, 44, (35, 6.5), (44, 5.4)),
            Slot(1, 57, (48, 4.3), (57, 3.1))
        };
        var smaller = BenchmarkGpuPlanner.RankIndependentSlotsForModel(
            slots, Q8Size, 66, 19_500_000_000UL);
        var larger = BenchmarkGpuPlanner.RankIndependentSlotsForModel(
            slots, Q8Size, 66, 22_900_000_000UL);

        await using var first = await scheduler.AcquireAsync(smaller);
        await using var second = await scheduler.AcquireAsync(larger);

        Assert.Equal(0, first.Slot.DeviceIndices[0]);
        Assert.Equal(1, second.Slot.DeviceIndices[0]);
    }

    [Fact]
    public async Task ResourceScheduler_ReservesDisjointSingleGpuSlotsConcurrently()
    {
        var scheduler = new GpuResourceScheduler();
        var slots = new[] { Slot(0, 44, (35, 6.5), (44, 5.4)), Slot(1, 57, (48, 4.3), (57, 3.1)) };

        await using var first = await scheduler.AcquireAsync(slots);
        await using var second = await scheduler.AcquireAsync(slots);

        Assert.NotEqual(first.Slot.DeviceIndices[0], second.Slot.DeviceIndices[0]);
    }

    [Fact]
    public async Task ResourceScheduler_SharedSlotWaitsUntilAllDevicesAreFree()
    {
        var scheduler = new GpuResourceScheduler();
        var singleSlots = new[] { Slot(0, 44, (35, 6.5), (44, 5.4)), Slot(1, 57, (48, 4.3), (57, 3.1)) };
        var sharedSlots = new[] { new BenchmarkSlot(2, "shared", [0, 1], 66, []) };

        var first = await scheduler.AcquireAsync(singleSlots);
        var sharedTask = scheduler.AcquireAsync(sharedSlots).AsTask();

        Assert.False(sharedTask.IsCompleted);
        await first.DisposeAsync();

        await using var shared = await sharedTask.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal([0, 1], shared.Slot.DeviceIndices);
    }

    [Fact]
    public void TopologyCacheCodec_RoundTripsMeasuredProfiles()
    {
        var independentSlots = new[]
        {
            Slot(0, 44, (35, 6.59), (44, 5.45)),
            Slot(1, 57, (48, 4.38), (57, 3.17))
        };
        var shared = new BenchmarkTopologyProfile(
            "shared", [new BenchmarkSlot(2, "shared", [0, 1], 66, [])], 0.1, 1.6);
        var independent = new BenchmarkTopologyProfile("independent", independentSlots, 0.08, 0);

        string json = BenchmarkTopologyCacheCodec.Serialize(66, 26_000_000_000UL, shared, independent);
        bool ok = BenchmarkTopologyCacheCodec.TryDeserialize(
            json, out int maxNgl, out ulong crossover, out var loadedShared, out var loadedIndependent);

        Assert.True(ok);
        Assert.Equal(66, maxNgl);
        Assert.Equal(26_000_000_000UL, crossover);
        Assert.Equal([0, 1], loadedShared!.Slots.Single().DeviceIndices);
        Assert.Equal([44, 57], loadedIndependent!.Slots.Select(x => x.Q8StableNgl).ToArray());
    }

    private static BenchmarkSlot Slot(
        int device,
        int stableNgl,
        params (int Ngl, double Seconds)[] samples)
        => new(
            SlotId: device,
            ProfileName: "independent",
            DeviceIndices: [device],
            Q8StableNgl: stableNgl,
            ProbeSamples: samples
                .Select(x => new GpuProbeSample(x.Ngl, true, x.Seconds, x.Seconds * 3d))
                .ToArray());
}
