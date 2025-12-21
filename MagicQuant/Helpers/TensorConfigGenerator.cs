using MagicQuant.Models;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class TensorConfigGenerator
{
    public static List<HybridQuant> GenerateRequiredDataSampleCombos(List<TensorGroup>? MissingTensorGroup = null)
    {
        var allowedBaselines = BaselineQuants.All.Where(x => x.BaseConversionBase != null).ToList();
        var hybridQuants = new List<HybridQuant>();

        // Fast lookup for missing groups
        var missingIds = MissingTensorGroup?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<sbyte>();

        // ---------------------------------------------------------
        // 1. BASELINE CONTROLS (One pure sample per allowed baseline)
        // ---------------------------------------------------------
        int baseTestsRequired = 0;
        foreach (var baseline in allowedBaselines)
        {
            baseTestsRequired++;
            var hq = new HybridQuant
            {
                BaseQuant = baseline,
                Tensors = TReg.All
                    .Select(g => new HybridTensor
                    {
                        TGroup = g,
                        // If missing, mark NULL. Else default to BF16.
                        TensorType = missingIds.Contains(g.UniqueId)
                            ? TensorWeightScheme.NULL
                            : TensorWeightScheme.BF16_F16
                    })
                    .ToList()
            };

            hybridQuants.Add(hq);
        }

        AnsiConsole.MarkupLine($"[bold green]Required BF16 base hybrid tests:[/] {baseTestsRequired:N0}");

        // ---------------------------------------------------------
        // 2. ISOLATION SAMPLES (Always BF16 Base, isolate one tensor at a time)
        // ---------------------------------------------------------
        var tensorWeights = TensorWeightScheme.All
            .Where(x => x != TensorWeightScheme.NULL && x != TensorWeightScheme.BF16_F16)
            .ToList();

        int isolatedSamplesRequired = 0;

        // We always use the BF16 baseline for isolation tests
        var isolationBase = BaselineQuants.GetBF16Quant();

        foreach (var weight in tensorWeights)
        {
            // Get valid targets: Start with All, remove Banned by Scheme, remove Missing by User
            var validTargets = TReg.All.Where(x => !weight.BannedGroups.Contains(x)).ToList();

            if (missingIds.Count > 0)
            {
                validTargets.RemoveAll(x => missingIds.Contains(x.UniqueId));
            }

            foreach (var group in validTargets)
            {
                isolatedSamplesRequired++;

                // Create fresh list with default logic
                var tensors = TReg.All.Select(g => new HybridTensor
                {
                    TGroup = g,
                    TensorType = missingIds.Contains(g.UniqueId)
                        ? TensorWeightScheme.NULL
                        : TensorWeightScheme.BF16_F16
                }).ToList();

                // Set the isolated target
                var foundQuant = tensors.First(x => x.TGroup == group);
                foundQuant.TensorType = weight;

                var hq = new HybridQuant
                {
                    BaseQuant = isolationBase,
                    Tensors = tensors
                };

                hybridQuants.Add(hq);
            }
        }

        AnsiConsole.MarkupLine($"[bold green]Isolated Samples Required:[/] {isolatedSamplesRequired:N0}");
        AnsiConsole.MarkupLine($"[bold green]Total Samples Required:[/] {hybridQuants.Count:N0}");

        return hybridQuants;
    }

    public static IEnumerable<List<TensorConfig>> GenerateTensorConfigBatches(
        BaselineQuants baseQuant,
        int batchSize = 10_000_000,
        CancellationToken ct = default)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

        // ---------------------------
        // Diagnostics / invariants
        // ---------------------------
        if (TReg.All.IsDefault)
            throw new InvalidOperationException("TensorRegistry.All is default (uninitialized).");

        var allowed = ComboLogic.GetAllowedSchemeIdsPerGroup(baseQuant);

        if (allowed.IsDefault)
            throw new InvalidOperationException("Allowed scheme array is default (uninitialized).");

        if (allowed.Length == 0)
            yield break;

        for (int i = 0; i < allowed.Length; i++)
        {
            if (allowed[i] == null)
                throw new InvalidOperationException(
                    $"Allowed[{i}] is null for base {string.Join("/", baseQuant.Names)}.");

            if (allowed[i].Length == 0)
                throw new InvalidOperationException(
                    $"Allowed[{i}] is empty for base {string.Join("/", baseQuant.Names)}.");
        }

        int dims = allowed.Length;

        // ---------------------------
        // Threading setup
        // ---------------------------
        int dop = ComputeWorkerThreads(GetThreadCountSafe());

        // Cache baseQuant.UniqueId once (perf)
        sbyte baseId = baseQuant.UniqueId;

        var queue = new BlockingCollection<List<TensorConfig>>(
            boundedCapacity: Math.Max(2, dop * 2));

        // ---------------------------
        // Producer
        // ---------------------------
        var producer = Task.Run(() =>
        {
            try
            {
                Parallel.ForEach(
                    Partitioner.Create(0, allowed[0].Length),
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = dop,
                        CancellationToken = ct
                    },
                    range =>
                    {
                        var batch = new List<TensorConfig>(Math.Min(batchSize, 250_000));
                        var idx = new int[dims];

                        // Hot-path aliases (perf)
                        // NOTE: This assumes group count is stable at 9 (Embeddings..MoeRouter),
                        // which matches your TensorConfig mapping.
                        var d0 = allowed[0];
                        var d1 = allowed[1];
                        var d2 = allowed[2];
                        var d3 = allowed[3];
                        var d4 = allowed[4];
                        var d5 = allowed[5];
                        var d6 = allowed[6];
                        var d7 = allowed[7];
                        var d8 = allowed[8];

                        for (int i0 = range.Item1; i0 < range.Item2; i0++)
                        {
                            ct.ThrowIfCancellationRequested();

                            idx[0] = i0;
                            Array.Clear(idx, 1, dims - 1);

                            while (true)
                            {
                                // Inline-build (perf): avoids helper call overhead and repeated bounds checks
                                batch.Add(new TensorConfig(
                                    baseQuant: baseId,
                                    embeddings: d0[idx[0]],
                                    lmHead: d1[idx[1]],
                                    attnQ: d2[idx[2]],
                                    attnKV: d3[idx[3]],
                                    attnOutput: d4[idx[4]],
                                    ffnUpGate: d5[idx[5]],
                                    ffnDown: d6[idx[6]],
                                    moeExperts: d7[idx[7]],
                                    moeRouter: d8[idx[8]]
                                ));

                                if (batch.Count >= batchSize)
                                {
                                    queue.Add(batch, ct);
                                    batch = new List<TensorConfig>(Math.Min(batchSize, 250_000));
                                }

                                // Mixed-radix increment (dims-1 → 1)
                                int d = dims - 1;
                                while (d >= 1)
                                {
                                    idx[d]++;
                                    if (idx[d] < allowed[d].Length)
                                        break;

                                    idx[d] = 0;
                                    d--;
                                }

                                if (d < 1)
                                    break;
                            }
                        }

                        if (batch.Count > 0)
                            queue.Add(batch, ct);
                    });
            }
            finally
            {
                queue.CompleteAdding();
            }
        }, ct);

        // ---------------------------
        // Consumer (yield batches)
        // ---------------------------
        foreach (var batch in queue.GetConsumingEnumerable(ct))
            yield return batch;

        producer.GetAwaiter().GetResult();
    }

    private static int GetThreadCountSafe()
    {
        // Cache.SysInfo might not be initialized this early; fall back safely
        int tc = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        return Math.Max(1, tc);
    }

    private static int ComputeWorkerThreads(int threadCount)
    {
        if (threadCount <= 1)
            return 1;

        int workers =
            threadCount < 16
                ? threadCount - 1
                : (int)Math.Floor(threadCount * 0.90);

        // Always leave at least 1 thread free
        return Math.Clamp(workers, 1, Math.Max(1, threadCount - 1));
    }
}