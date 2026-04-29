using System.Collections.Concurrent;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class TensorConfigGenerator
{
    public static RequiredSampleGenerationResult GenerateInitialIsolationSamplePlan(List<TensorGroup>? missingTensorGroups = null)
    {
        if (missingTensorGroups != null && !missingTensorGroups.Any())
            missingTensorGroups = null;

        var missingIds = missingTensorGroups?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<byte>();
        var activeGroups = TReg.All
            .Where(x => !missingIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = new RequiredSampleGenerationResult();
        var nativeExactScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();

        var alreadyAddedPureBaselineIds = new HashSet<byte>();

        void AddPureBaselinePlan(BaselineQuants baseline)
        {
            if (!alreadyAddedPureBaselineIds.Add(baseline.UniqueId))
                return;

            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.PureBaseline,
                Key = $"pure:{baseline.UniqueId}",
                Description = $"Pure baseline build for {string.Join("/", baseline.Names)}",
                Quant = HybridQuant.CreatePureBaseline(baseline),
                TestedBaselineId = baseline.UniqueId,
                TestedBaselineCanonicalKey = baseline.CanonicalKey
            });

            result.PureBaselineCount++;
        }

        foreach (var baseline in BaselineQuants.GetLearningBaselines(RuntimeSearchSpace.HasUsableImatrix()))
            AddPureBaselinePlan(baseline);

        // Q8 remains a required system anchor even when the user disables standard baselines.
        AddPureBaselinePlan(BaselineQuants.Q8_0);

        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.BaseOnlyIsolation,
                Key = $"baseonly:{baseline.UniqueId}",
                Description = $"Base-only isolation for {string.Join("/", baseline.Names)} with all active groups forced native.",
                Quant = HybridQuant.CreateExactBlanket(
                    baseQuant: baseline,
                    groups: activeGroups,
                    exactScheme: nativeExactScheme),
                TestedBaselineId = baseline.UniqueId,
                TestedBaselineCanonicalKey = baseline.CanonicalKey
            });

            result.BaseOnlyIsolationCount++;
        }

        var carrier = BaselineQuants.Q8_0;

        result.Plans.Add(new RequiredSamplePlan
        {
            Kind = RequiredSampleKind.BaseOnlyIsolation,
            Key = $"carrier-baseonly:{carrier.UniqueId}",
            Description = "Carrier base-only isolation on Q8 with all active groups forced native.",
            Quant = HybridQuant.CreateExactBlanket(
                baseQuant: carrier,
                groups: activeGroups,
                exactScheme: nativeExactScheme),
            TestedBaselineId = carrier.UniqueId,
            TestedBaselineCanonicalKey = carrier.CanonicalKey
        });

        result.BaseOnlyIsolationCount++;

        foreach (var group in activeGroups)
        {
            var smallest = GetSmallestAllowedProbeCandidateForGroup(group);
            if (smallest == null)
                continue;

            var quant = HybridQuant.CreateExactBlanket(
                baseQuant: carrier,
                groups: activeGroups,
                exactScheme: nativeExactScheme);

            quant.SetLearnedCandidateOverride(group, smallest);

            result.Plans.Add(new RequiredSamplePlan
            {
                Kind = RequiredSampleKind.GroupIsolationProbe,
                Key = $"probe:{carrier.UniqueId}:{group.UniqueId}:{smallest.UniqueId}",
                Description = $"Smallest-first probe for group '{group.Name}' using '{smallest.Names[0]}'.",
                Quant = quant,
                TargetGroupId = group.UniqueId,
                TestedCandidateId = smallest.UniqueId,
                TestedCandidateCanonicalKey = smallest.CanonicalKey,
                TestedBaselineId = carrier.UniqueId,
                TestedBaselineCanonicalKey = carrier.CanonicalKey,
                IsSmallestProbe = true
            });

            result.GroupIsolationCount++;
        }

        AnsiConsole.MarkupLine($"[bold green]Pure baselines required:[/] {result.PureBaselineCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Base-only isolation samples required:[/] {result.BaseOnlyIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Smallest-probe isolation samples required:[/] {result.GroupIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Total initial startup samples:[/] {result.TotalCount:N0}");
        EmitSamplePlanDiagnostics("initial", result.Plans);

        return result;
    }

    public static RequiredSampleGenerationResult GenerateContinuationIsolationSamplePlan(
        IEnumerable<byte> groupIdsToContinue,
        List<TensorGroup>? missingTensorGroups = null)
    {
        if (missingTensorGroups != null && !missingTensorGroups.Any())
            missingTensorGroups = null;

        var continueIds = groupIdsToContinue.Distinct().ToHashSet();
        var missingIds = missingTensorGroups?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<byte>();

        var activeGroups = TReg.All
            .Where(x => continueIds.Contains(x.UniqueId))
            .Where(x => !missingIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = BuildIsolationCoverageContinuationPlan(activeGroups, missingIds);

        AnsiConsole.MarkupLine($"[bold green]Continuation isolation samples required:[/] {result.GroupIsolationCount:N0}");
        EmitSamplePlanDiagnostics("continuation", result.Plans);
        return result;
    }

    /// <summary>
    /// Builds archival-only isolation coverage for any continuation-style samples that were not part
    /// of the live startup+continuation pruning plan. This is intentionally kept separate from the
    /// current run's pruning inputs so search-space behavior stays unchanged while the database still
    /// gains full isolated-sample coverage for future prediction/reporting flows.
    /// </summary>
    public static RequiredSampleGenerationResult GenerateArchivalIsolationCoverageSamplePlan(
        IEnumerable<byte>? groupIdsToArchive = null,
        IEnumerable<string>? existingPlanKeys = null,
        List<TensorGroup>? missingTensorGroups = null)
    {
        if (missingTensorGroups != null && !missingTensorGroups.Any())
            missingTensorGroups = null;

        var missingIds = missingTensorGroups?.Select(x => x.UniqueId).ToHashSet() ?? new HashSet<byte>();
        var archiveIds = groupIdsToArchive?
            .Distinct()
            .ToHashSet()
            ?? new HashSet<byte>();

        var existingKeys = existingPlanKeys?
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .ToHashSet(StringComparer.Ordinal)
            ?? new HashSet<string>(StringComparer.Ordinal);

        var activeGroups = TReg.All
            .Where(x => !missingIds.Contains(x.UniqueId))
            .Where(x => archiveIds.Count == 0 || archiveIds.Contains(x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = BuildIsolationCoverageContinuationPlan(activeGroups, missingIds);

        if (existingKeys.Count > 0)
        {
            result.Plans = result.Plans
                .Where(x => !existingKeys.Contains(x.Key))
                .ToList();
        }

        result.GroupIsolationCount = result.Plans.Count(x =>
            x.Kind == RequiredSampleKind.GroupIsolationProbe ||
            x.Kind == RequiredSampleKind.GroupIsolationContinuation);

        return result;
    }

    private static RequiredSampleGenerationResult BuildIsolationCoverageContinuationPlan(
        IReadOnlyCollection<TensorGroup> activeGroups,
        HashSet<byte> missingIds)
    {
        var result = new RequiredSampleGenerationResult();
        if (activeGroups.Count == 0)
            return result;

        var carrier = BaselineQuants.Q8_0;
        var nativeExactScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();
        var blanketGroups = TReg.All
            .Where(x => !missingIds.Contains(x.UniqueId))
            .ToList();

        var candidates = BaselineQuants.GetGroupCombinationCandidatesSmallestFirst(
                RuntimeSearchSpace.HasUsableImatrix(),
                allowHighPrecisionHybrids: false)
            .ToList();

        foreach (var group in activeGroups)
        {
            var smallest = GetSmallestAllowedProbeCandidateForGroup(group);

            foreach (var candidate in candidates)
            {
                if (smallest != null && candidate.UniqueId == smallest.UniqueId)
                    continue;

                var quant = HybridQuant.CreateExactBlanket(
                    baseQuant: carrier,
                    groups: blanketGroups,
                    exactScheme: nativeExactScheme);

                quant.SetLearnedCandidateOverride(group, candidate);

                result.Plans.Add(new RequiredSamplePlan
                {
                    Kind = RequiredSampleKind.GroupIsolationContinuation,
                    Key = $"cont:{carrier.UniqueId}:{group.UniqueId}:{candidate.UniqueId}",
                    Description = $"Continuation isolation for group '{group.Name}' using '{candidate.Names[0]}'.",
                    Quant = quant,
                    TargetGroupId = group.UniqueId,
                    TestedCandidateId = candidate.UniqueId,
                    TestedCandidateCanonicalKey = candidate.CanonicalKey,
                    TestedBaselineId = carrier.UniqueId,
                    TestedBaselineCanonicalKey = carrier.CanonicalKey
                });

                result.GroupIsolationCount++;
            }
        }

        return result;
    }

    private static void EmitSamplePlanDiagnostics(string phase, IReadOnlyCollection<RequiredSamplePlan> plans)
    {
        if (!MagicQuantDiagnostics.VerboseIsolationPruning)
            return;

        var byGroup = plans.Where(x => x.TargetGroupId.HasValue).GroupBy(x => x.TargetGroupId!.Value);
        foreach (var set in byGroup)
        {
            var group = TReg.All.First(x => x.UniqueId == set.Key);
            if (!MagicQuantDiagnostics.ShouldLogGroup(group))
                continue;
            var planned = set.Select(x => BaselineQuants.FromId(x.TestedCandidateId!.Value)).ToList();
            var smallest = GetSmallestAllowedProbeCandidateForGroup(group);
            var allowed = RuntimeSearchSpace.GetAllowedRealExplicitCombinationCandidatesForGroup(group);
            var raw = RuntimeSearchSpace.GetRealExplicitCombinationCandidatesForGroup(group);
            var staticBanned = BaselineQuants.GetGroupCombinationCandidates(RuntimeSearchSpace.HasUsableImatrix(), false)
                .Where(x => x.BannedGroupIds.Contains(group.UniqueId)).ToList();
            var runtimeBanned = raw.Where(x => RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, x)).ToList();
            var notAllowed = planned.Where(x => allowed.All(a => a.UniqueId != x.UniqueId)).ToList();
            MagicQuantDiagnostics.Log("sample-plan", $"phase={phase} group={group.Name}(id={group.UniqueId}) plannedCount={planned.Count} smallestProbe={(smallest == null ? "<none>" : MagicQuantDiagnostics.CandidateLabel(smallest))}");
            MagicQuantDiagnostics.Log("sample-plan", $"planned={string.Join(", ", planned.Select(MagicQuantDiagnostics.CandidateLabel))}");
            MagicQuantDiagnostics.Log("sample-plan", $"allowedAtPlan={string.Join(", ", allowed.Select(MagicQuantDiagnostics.CandidateLabel))}");
            MagicQuantDiagnostics.Log("sample-plan", $"staticBanned={string.Join(", ", staticBanned.Select(MagicQuantDiagnostics.CandidateLabel))}");
            MagicQuantDiagnostics.Log("sample-plan", $"runtimeBanned={string.Join(", ", runtimeBanned.Select(MagicQuantDiagnostics.CandidateLabel))}");
            MagicQuantDiagnostics.Log("sample-plan", $"plannedButNotAllowed={string.Join(", ", notAllowed.Select(MagicQuantDiagnostics.CandidateLabel))}");
        }
    }

    public static List<HybridQuant> GenerateRequiredDataSampleCombos(List<TensorGroup>? missingTensorGroups = null)
    {
        return GenerateInitialIsolationSamplePlan(missingTensorGroups)
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

        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseQuant);

        if (allowed.IsDefault)
            throw new InvalidOperationException("Allowed candidate array is default (uninitialized).");

        if (allowed.Length == 0)
            yield break;

        for (int i = 0; i < allowed.Length; i++)
        {
            if (allowed[i] == null)
                throw new InvalidOperationException($"Allowed[{i}] is null for base {string.Join("/", baseQuant.Names)}.");

            if (allowed[i].Length == 0)
                throw new InvalidOperationException($"Allowed[{i}] is empty for base {string.Join("/", baseQuant.Names)}.");
        }

        int dims = allowed.Length;
        int dop = ComputeWorkerThreads(GetThreadCountSafe());
        byte baseId = baseQuant.UniqueId;

        var queue = new BlockingCollection<List<TensorConfig>>(boundedCapacity: Math.Max(2, dop * 2));

        var producer = Task.Run(() =>
        {
            try
            {
                Parallel.ForEach(
                    Partitioner.Create(0, allowed[0].Length),
                    new ParallelOptions { MaxDegreeOfParallelism = dop, CancellationToken = ct },
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
                                    baseId,
                                    d0[idx[0]], d1[idx[1]], d2[idx[2]], d3[idx[3]], d4[idx[4]],
                                    d5[idx[5]], d6[idx[6]], d7[idx[7]], d8[idx[8]]));

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

    private static BaselineQuants? GetSmallestAllowedProbeCandidateForGroup(TensorGroup group)
    {
        foreach (var candidate in BaselineQuants.GetGroupCombinationCandidatesSmallestFirst(
                     RuntimeSearchSpace.HasUsableImatrix(),
                     allowHighPrecisionHybrids: false))
        {
            return candidate;
        }

        return null;
    }

    private static int GetThreadCountSafe()
    {
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

        return Math.Clamp(workers, 1, Math.Max(1, threadCount - 1));
    }
}
