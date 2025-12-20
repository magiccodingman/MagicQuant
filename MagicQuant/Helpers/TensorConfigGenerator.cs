using MagicQuant.Models;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Numerics;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class TensorConfigGenerator
{
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

        var queue = new BlockingCollection<List<TensorConfig>>(
            boundedCapacity: Math.Max(2, dop * 2));

        // ---------------------------
        // Producer
        // ---------------------------
        var producer = Task.Run(() =>
        {
            try
            {
                // Partition on first dimension
                Parallel.ForEach(
                    Partitioner.Create(0, allowed[0].Length),
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = dop,
                        CancellationToken = ct
                    },
                    range =>
                    {
                        var batch = new List<TensorConfig>(
                            Math.Min(batchSize, 250_000));

                        var idx = new int[dims];

                        for (int i0 = range.Item1; i0 < range.Item2; i0++)
                        {
                            ct.ThrowIfCancellationRequested();

                            idx[0] = i0;
                            Array.Clear(idx, 1, dims - 1);

                            while (true)
                            {
                                batch.Add(BuildTensorConfig(allowed, idx));

                                if (batch.Count >= batchSize)
                                {
                                    queue.Add(batch, ct);
                                    batch = new List<TensorConfig>(
                                        Math.Min(batchSize, 250_000));
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

    private static TensorConfig BuildTensorConfig(
        ImmutableArray<sbyte[]> allowed,
        int[] idx)
    {
        // Order MUST match ComboLogic.GroupsOrdered
        return new TensorConfig(
            embeddings: allowed[0][idx[0]],
            lmHead: allowed[1][idx[1]],
            attnQ: allowed[2][idx[2]],
            attnKV: allowed[3][idx[3]],
            attnOutput: allowed[4][idx[4]],
            ffnUpGate: allowed[5][idx[5]],
            ffnDown: allowed[6][idx[6]],
            moeExperts: allowed[7][idx[7]],
            moeRouter: allowed[8][idx[8]]
        );
    }

    private static int GetThreadCountSafe()
    {
        // Cache.SysInfo might not be initialized this early; fall back safely
        var tc = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        return Math.Max(1, tc);
    }

    private static int ComputeWorkerThreads(int threadCount)
    {
        if (threadCount <= 1)
            return 1;

        int workers;
        if (threadCount < 16)
            workers = threadCount - 1;
        else
            workers = (int)Math.Floor(threadCount * 0.90);

        // Always leave at least 1 thread free
        workers = Math.Clamp(workers, 1, Math.Max(1, threadCount - 1));
        return workers;
    }
}