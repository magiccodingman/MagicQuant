using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Numerics;

namespace MagicQuant.Helpers;

public static class TensorConfigGenerator
{
    public static RequiredSampleGenerationResult GenerateRequiredSamplePlan(
        List<TensorGroup>? missingTensorGroups = null)
    {
        TensorWeightScheme.ValidateSmallestConfiguration();

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
        // 4. Smallest-first tensor-group isolation probes
        // ---------------------------------------------------------
        foreach (var group in activeGroups)
        {
            var firstProbe = GetInitialIsolationProbeScheme(group);
            if (firstProbe == null)
                continue;

            var quant = HybridQuant.CreateBlanket(
                baseQuant: groupIsolationCarrier,
                groups: activeGroups,
                blanketScheme: TensorWeightScheme.BF16_F16);

            var target = quant.Tensors.First(x => x.TGroup.UniqueId == group.UniqueId);
            target.TensorType = firstProbe;

            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.GroupIsolation,
                Key = $"probefirst:{groupIsolationCarrier.UniqueId}:{group.UniqueId}:{firstProbe.UniqueId}",
                Description =
                    $"Initial isolation probe for group '{group.Name}' using scheme '{firstProbe.Names[0]}' on carrier '{groupIsolationCarrier.Names[0]}'.",
                Quant = quant,
                TargetGroupId = group.UniqueId,
                TestedSchemeId = firstProbe.UniqueId,
                TestedBaselineId = groupIsolationCarrier.UniqueId
            });

            result.GroupIsolationCount++;
        }

        AnsiConsole.MarkupLine($"[bold green]Initial pure baseline samples required:[/] {result.PureBaselineCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Initial base-only isolation samples required:[/] {result.BaseOnlyIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Initial smallest-probe isolation samples required:[/] {result.GroupIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Initial total samples required:[/] {result.TotalCount:N0}");

        return result;
    }

    public static RequiredSampleGenerationResult GenerateContinuationIsolationPlan(
        IEnumerable<byte> groupIdsToContinue,
        List<TensorGroup>? missingTensorGroups = null)
    {
        if (groupIdsToContinue == null)
            throw new ArgumentNullException(nameof(groupIdsToContinue));

        TensorWeightScheme.ValidateSmallestConfiguration();

        if (missingTensorGroups != null && !missingTensorGroups.Any())
            missingTensorGroups = null;

        var skippedIds = missingTensorGroups?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<byte>();
        var continueIds = groupIdsToContinue.ToHashSet();
        var activeGroups = TReg.All
            .Where(x => !skippedIds.Contains(x.UniqueId))
            .Where(x => continueIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = new RequiredSampleGenerationResult();
        var groupIsolationCarrier = BaselineQuants.Q8_0;

        foreach (var group in activeGroups)
        {
            var allCandidates = GetOrderedIsolationCandidateSchemes(group);
            var firstProbe = GetInitialIsolationProbeScheme(group);

            foreach (var scheme in allCandidates)
            {
                if (firstProbe != null && scheme.UniqueId == firstProbe.UniqueId)
                    continue;

                var quant = HybridQuant.CreateBlanket(
                    baseQuant: groupIsolationCarrier,
                    groups: TReg.All.Where(x => !skippedIds.Contains(x.UniqueId)).OrderBy(x => x.UniqueId),
                    blanketScheme: TensorWeightScheme.BF16_F16);

                var target = quant.Tensors.First(x => x.TGroup.UniqueId == group.UniqueId);
                target.TensorType = scheme;

                result.Plans.Add(new RequiredSamplePlan
                {
                    Kind = RequiredSampleKind.GroupIsolation,
                    Key = $"group:{groupIsolationCarrier.UniqueId}:{group.UniqueId}:{scheme.UniqueId}",
                    Description =
                        $"Follow-up isolation sample for group '{group.Name}' using scheme '{scheme.Names[0]}' on carrier '{groupIsolationCarrier.Names[0]}'.",
                    Quant = quant,
                    TargetGroupId = group.UniqueId,
                    TestedSchemeId = scheme.UniqueId,
                    TestedBaselineId = groupIsolationCarrier.UniqueId
                });

                result.GroupIsolationCount++;
            }
        }

        AnsiConsole.MarkupLine($"[bold green]Continuation isolation samples required:[/] {result.GroupIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Continuation total samples required:[/] {result.TotalCount:N0}");

        return result;
    }

    private static TensorWeightScheme? GetInitialIsolationProbeScheme(TensorGroup group)
    {
        var ordered = GetOrderedIsolationCandidateSchemes(group);

        return ordered.FirstOrDefault(x => !x.RequiresImatrix) ?? ordered.FirstOrDefault();
    }

    private static List<TensorWeightScheme> GetOrderedIsolationCandidateSchemes(TensorGroup group)
    {
        return TensorWeightScheme.All
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .Where(x => !x.IsBannedFor(group))
            .OrderByDescending(x => !x.RequiresImatrix && x.IsSmallest)
            .ThenBy(x => x.RequiresImatrix ? 1 : 0)
            .ThenByDescending(x => x.UniqueId)
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

        BigInteger total = BigInteger.One;
        for (int i = 0; i < dims; i++)
            total *= allowed[i].Length;

        if (total == BigInteger.Zero)
            yield break;

        var buffer = new ConcurrentQueue<TensorConfig>();
        var produced = 0L;

        Parallel.ForEach(
            Partitioner.Create(0L, (long)total),
            new ParallelOptions { MaxDegreeOfParallelism = dop, CancellationToken = ct },
            range =>
            {
                var local = new List<TensorConfig>(Math.Min(batchSize, 8192));

                for (long flat = range.Item1; flat < range.Item2; flat++)
                {
                    ct.ThrowIfCancellationRequested();

                    long n = flat;
                    Span<byte> chosen = stackalloc byte[dims];

                    for (int d = dims - 1; d >= 0; d--)
                    {
                        var arr = allowed[d];
                        int len = arr.Length;
                        int idx = (int)(n % len);
                        chosen[d] = arr[idx];
                        n /= len;
                    }

                    local.Add(new TensorConfig(
                        baseId,
                        chosen[0],
                        chosen[1],
                        chosen[2],
                        chosen[3],
                        chosen[4],
                        chosen[5],
                        chosen[6],
                        chosen[7],
                        chosen[8]));

                    if (local.Count >= batchSize)
                    {
                        foreach (var item in local)
                            buffer.Enqueue(item);

                        local.Clear();
                    }
                }

                foreach (var item in local)
                    buffer.Enqueue(item);
            });

        while (!buffer.IsEmpty)
        {
            var batch = new List<TensorConfig>(batchSize);

            while (batch.Count < batchSize && buffer.TryDequeue(out var cfg))
                batch.Add(cfg);

            produced += batch.Count;
            if (batch.Count > 0)
                yield return batch;
        }
    }

    private static int GetThreadCountSafe()
    {
        int tc = Cache.SysInfo?.ThreadCount ?? Environment.ProcessorCount;
        return Math.Max(1, tc);
    }

    private static int ComputeWorkerThreads(int logicalThreads)
    {
        if (logicalThreads <= 2) return 1;
        if (logicalThreads <= 4) return 2;
        if (logicalThreads <= 8) return 4;
        return Math.Max(4, logicalThreads / 2);
    }
}