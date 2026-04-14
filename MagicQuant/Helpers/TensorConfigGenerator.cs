using MQ.DB.Models;
using Spectre.Console;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using MQ.DB;

namespace MagicQuant.Helpers;

public static class TensorConfigGenerator
{
    public static RequiredSampleGenerationResult GenerateRequiredSamplePlan(
        List<TensorGroup>? missingTensorGroups = null)
    {
        if (missingTensorGroups != null && !missingTensorGroups.Any())
            missingTensorGroups = null;

        var skippedIds = missingTensorGroups?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<byte>();
        var activeGroups = TReg.All
            .Where(x => !skippedIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = new RequiredSampleGenerationResult();

        // ---------------------------------------------------------
        // 1. Pure baselines
        // ---------------------------------------------------------
        foreach (var baseline in BaselineQuants.All.OrderBy(x => x.UniqueId))
        {
            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.PureBaseline,
                Key = $"pure:{baseline.UniqueId}",
                Description = $"Pure baseline build for {string.Join("/", baseline.Names)}",
                Quant = HybridQuant.CreatePureBaseline(baseline),
                TestedBaselineId = baseline.UniqueId
            });

            result.PureBaselineCount++;
        }

        // ---------------------------------------------------------
        // 2. Base-only isolation for actual combo baselines
        //    This tells you whether uncovered tensors alone justify the baseline.
        // ---------------------------------------------------------
        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            var quant = HybridQuant.CreateBlanket(
                baseQuant: baseline,
                groups: activeGroups,
                blanketScheme: TensorWeightScheme.BF16_F16);

            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.BaseOnlyIsolation,
                Key = $"baseonly:{baseline.UniqueId}",
                Description =
                    $"Base-only isolation for {string.Join("/", baseline.Names)} with all known groups forced native.",
                Quant = quant,
                TestedBaselineId = baseline.UniqueId
            });

            result.BaseOnlyIsolationCount++;
        }

        // ---------------------------------------------------------
        // 3. Carrier base-only isolation for tensor-group probing
        //    Q8_0 is used as the carrier because llama-quantize actually applies
        //    mixed tensor overrides correctly on a real quantized output.
        // ---------------------------------------------------------
        var groupIsolationCarrier = BaselineQuants.Q8_0;

        var carrierBaseOnly = HybridQuant.CreateBlanket(
            baseQuant: groupIsolationCarrier,
            groups: activeGroups,
            blanketScheme: TensorWeightScheme.BF16_F16);

        result.Plans.Add(new RequiredSamplePlan
        {
            Kind = RequiredSampleKind.BaseOnlyIsolation,
            Key = $"carrier-baseonly:{groupIsolationCarrier.UniqueId}",
            Description =
                $"Carrier base-only isolation for {string.Join("/", groupIsolationCarrier.Names)} with all known groups forced native.",
            Quant = carrierBaseOnly,
            TestedBaselineId = groupIsolationCarrier.UniqueId
        });

        result.BaseOnlyIsolationCount++;

        // ---------------------------------------------------------
        // 4. Tensor-group isolation using the Q8_0 carrier
        // ---------------------------------------------------------
        var probeSchemes = TensorWeightScheme.All
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .OrderBy(x => x.UniqueId)
            .ToList();

        foreach (var group in activeGroups)
        {
            foreach (var scheme in probeSchemes)
            {
                if (scheme.IsBannedFor(group))
                    continue;

                var quant = HybridQuant.CreateBlanket(
                    baseQuant: groupIsolationCarrier,
                    groups: activeGroups,
                    blanketScheme: TensorWeightScheme.BF16_F16);

                var target = quant.Tensors.First(x => x.TGroup.UniqueId == group.UniqueId);
                target.TensorType = scheme;

                result.Plans.Add(new RequiredSamplePlan
                {
                    Kind = RequiredSampleKind.GroupIsolation,
                    Key = $"group:{groupIsolationCarrier.UniqueId}:{group.UniqueId}:{scheme.UniqueId}",
                    Description =
                        $"Carrier-based isolation for group '{group.Name}' using scheme '{scheme.Names[0]}' on base '{groupIsolationCarrier.Names[0]}'.",
                    Quant = quant,
                    TargetGroupId = group.UniqueId,
                    TestedSchemeId = scheme.UniqueId,
                    TestedBaselineId = groupIsolationCarrier.UniqueId
                });

                result.GroupIsolationCount++;
            }
        }

        AnsiConsole.MarkupLine($"[bold green]Pure baselines required:[/] {result.PureBaselineCount:N0}");
        AnsiConsole.MarkupLine(
            $"[bold green]Base-only isolation samples required:[/] {result.BaseOnlyIsolationCount:N0}");
        AnsiConsole.MarkupLine(
            $"[bold green]Tensor-group isolation samples required:[/] {result.GroupIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Total required samples:[/] {result.TotalCount:N0}");

        return result;
    }

    public static List<HybridQuant> GenerateRequiredDataSampleCombos(List<TensorGroup>? missingTensorGroups = null)
    {
        return GenerateRequiredSamplePlan(missingTensorGroups)
            .Plans
            .Select(x => x.Quant)
            .ToList();
    }

    public static IEnumerable<List<TensorConfig>> GenerateTensorConfigBatches(
        BaselineQuants baseQuant,
        int batchSize = 10_000_000,
        CancellationToken ct = default)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));

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
        int dop = ComputeWorkerThreads(GetThreadCountSafe());
        byte baseId = baseQuant.UniqueId;

        var queue = new BlockingCollection<List<TensorConfig>>(
            boundedCapacity: Math.Max(2, dop * 2));

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

        foreach (var batch in queue.GetConsumingEnumerable(ct))
            yield return batch;

        producer.GetAwaiter().GetResult();
    }

    private static int GetThreadCountSafe()
    {
        return Cache.SysInfo?.ThreadCount > 0
            ? Cache.SysInfo.ThreadCount
            : Environment.ProcessorCount;
    }

    private static int ComputeWorkerThreads(int threadCount)
    {
        if (threadCount <= 4)
            return 1;

        if (threadCount <= 12)
            return 2;

        return Math.Max(2, threadCount / 6);
    }
}