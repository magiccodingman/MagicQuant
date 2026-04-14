using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;

namespace MagicQuant.Services;

public sealed class IsolationOptimizationOptions
{
    public double MinMeaningfulBaseOnlyReductionRatio { get; set; } = IsolationRules.MinimumMeaningfulBaseOnlyReductionRatio;
}

public sealed class IsolationSamplingGateDecision
{
    public string GroupName { get; set; } = string.Empty;
    public string ProbeScheme { get; set; } = string.Empty;
    public double ReductionRatio { get; set; }
    public bool ContinueSampling { get; set; }
    public string Reason { get; set; } = string.Empty;
    public ulong? ProbeSizeBytes { get; set; }
}

public sealed class IsolationSamplingGateResult
{
    public int GroupsStoppedEarly { get; set; }
    public List<byte> GroupIdsToContinue { get; set; } = new();
    public List<string> Notes { get; set; } = new();
    public List<IsolationSamplingGateDecision> GroupDetails { get; set; } = new();
}

public sealed class IsolationGroupDecision
{
    public string GroupName { get; set; } = string.Empty;
    public bool StoppedEarly { get; set; }
    public string? WinningScheme { get; set; }
    public ulong? WinningSizeBytes { get; set; }
    public double? WinningKld { get; set; }
    public double? WinningPplDelta { get; set; }
    public List<string> Candidates { get; set; } = new();
    public List<string> Eliminations { get; set; } = new();
}

public sealed class IsolationOptimizationResult
{
    public int GroupsStoppedEarly { get; set; }
    public int HardDamageEliminations { get; set; }
    public int DominatedGroupSchemesBanned { get; set; }
    public int DisabledBaselines { get; set; }
    public List<string> Notes { get; set; } = new();
    public List<IsolationGroupDecision> GroupDetails { get; set; } = new();
}

public class IsolationOptimizationService
{
    public async Task<IsolationSamplingGateResult> ApplyInitialSamplingGateAsync(
        RequiredSampleGenerationResult initialPlan,
        CancellationToken ct = default)
    {
        if (initialPlan == null)
            throw new ArgumentNullException(nameof(initialPlan));

        var result = new IsolationSamplingGateResult();

        var carrierBaselineId = BaselineQuants.Q8_0.UniqueId;

        var carrierBaseOnlyPlan = initialPlan.Plans.FirstOrDefault(x =>
            x.Kind == RequiredSampleKind.BaseOnlyIsolation &&
            x.Key.StartsWith("carrier-baseonly:", StringComparison.Ordinal) &&
            x.TestedBaselineId == carrierBaselineId);

        if (carrierBaseOnlyPlan == null)
            throw new InvalidOperationException("Carrier base-only isolation plan was not found.");

        var carrierBaseOnly = await LoadSnapshotAsync(carrierBaseOnlyPlan.Quant, ct)
            ?? throw new InvalidOperationException("Carrier base-only isolation benchmark was not found in SQLite.");

        var probePlans = initialPlan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolation)
            .Where(x => x.Key.StartsWith("probefirst:", StringComparison.Ordinal))
            .OrderBy(x => x.TargetGroupId)
            .ToList();

        foreach (var plan in probePlans)
        {
            var group = TReg.All.First(x => x.UniqueId == plan.TargetGroupId!.Value);
            var scheme = TensorWeightScheme.All.First(x => x.UniqueId == plan.TestedSchemeId!.Value);

            var snapshot = await LoadSnapshotAsync(plan.Quant, ct)
                ?? throw new InvalidOperationException(
                    $"Initial isolation probe benchmark was missing for group '{group.Name}' and scheme '{scheme.Names[0]}'.");

            double reduction = ComputeReductionRatio(carrierBaseOnly.SizeBytes, snapshot.SizeBytes);

            var decision = new IsolationSamplingGateDecision
            {
                GroupName = group.Name,
                ProbeScheme = scheme.Names[0],
                ReductionRatio = reduction,
                ProbeSizeBytes = snapshot.SizeBytes
            };

            if (reduction < IsolationRules.MinimumIsolationReductionToContinue)
            {
                RuntimeSearchSpace.BanAllExplicitTensorSchemesForGroup(group);
                result.GroupsStoppedEarly++;

                decision.ContinueSampling = false;
                decision.Reason =
                    $"Stopped early because smallest explicit probe reduction was only {reduction:P2}, below the {IsolationRules.MinimumIsolationReductionToContinue:P2} gate.";

                result.Notes.Add(
                    $"Stopped isolated explicit sampling for '{group.Name}' because smallest probe '{scheme.Names[0]}' only reduced the model by {reduction:P2}.");
            }
            else
            {
                result.GroupIdsToContinue.Add(group.UniqueId);
                decision.ContinueSampling = true;
                decision.Reason =
                    $"Continuing sampling because smallest explicit probe reduction was {reduction:P2}.";
            }

            result.GroupDetails.Add(decision);
        }

        return result;
    }

    public async Task<IsolationOptimizationResult> AnalyzeAndApplyAsync(
        RequiredSampleGenerationResult plan,
        IsolationSamplingGateResult? gateResult = null,
        IsolationOptimizationOptions? options = null,
        CancellationToken ct = default)
    {
        if (plan == null)
            throw new ArgumentNullException(nameof(plan));

        options ??= new IsolationOptimizationOptions();

        var result = new IsolationOptimizationResult
        {
            GroupsStoppedEarly = gateResult?.GroupsStoppedEarly ?? 0
        };

        var nativeBaseline = await LoadSnapshotAsync(
            HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()), ct)
            ?? throw new InvalidOperationException("Native BF16/F16/F32 baseline benchmark was not found in SQLite.");

        // ============================
        // GROUP ISOLATION ANALYSIS
        // ============================
        var groupPlans = plan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolation)
            .Where(x => x.TestedBaselineId == BaselineQuants.Q8_0.UniqueId)
            .GroupBy(x => x.TargetGroupId!.Value)
            .OrderBy(x => x.Key)
            .ToList();

        var stoppedEarlyIds = gateResult?.GroupIdsToContinue == null
            ? new HashSet<byte>()
            : TReg.All.Select(x => x.UniqueId).Except(gateResult.GroupIdsToContinue).ToHashSet();

        foreach (var groupSet in groupPlans)
        {
            var group = TReg.All.First(x => x.UniqueId == groupSet.Key);
            var decision = new IsolationGroupDecision
            {
                GroupName = group.Name,
                StoppedEarly = gateResult?.GroupDetails.Any(x => x.GroupName == group.Name && !x.ContinueSampling) == true
            };

            var candidates = new List<CandidateEvaluation>();

            foreach (var item in groupSet)
            {
                var snap = await LoadSnapshotAsync(item.Quant, ct);
                if (snap == null)
                    continue;

                var scheme = TensorWeightScheme.All.First(x => x.UniqueId == item.TestedSchemeId!.Value);
                var avgKld = GetAggregateKld(snap);
                var pplDelta = GetAggregatePplDelta(snap, nativeBaseline);

                var candidate = new CandidateEvaluation
                {
                    Plan = item,
                    Scheme = scheme,
                    Snapshot = snap,
                    AggregateKld = avgKld,
                    AggregatePplDelta = pplDelta
                };

                decision.Candidates.Add(
                    $"{scheme.Names[0]} | size={(snap.SizeBytes / 1024.0 / 1024.0):F2} MiB | avgKLD={avgKld:G6} | avgΔPPL={pplDelta:P4}");

                candidates.Add(candidate);
            }

            if (candidates.Count == 0)
            {
                result.GroupDetails.Add(decision);
                continue;
            }

            // ------------------------------------------------------
            // Hard damage elimination
            // ------------------------------------------------------
            foreach (var candidate in candidates)
            {
                if (candidate.Eliminated)
                    continue;

                bool pplTooHigh = candidate.AggregatePplDelta >= IsolationRules.MaximumIsolationPplDeltaRatio;
                bool kldTooHigh = candidate.AggregateKld >= IsolationRules.MaximumIsolationKld;

                if (!pplTooHigh && !kldTooHigh)
                    continue;

                candidate.Eliminated = true;
                candidate.EliminationReason = pplTooHigh && kldTooHigh
                    ? $"hard-damage: avgΔPPL={candidate.AggregatePplDelta:P4} and avgKLD={candidate.AggregateKld:G6} exceeded thresholds"
                    : pplTooHigh
                        ? $"hard-damage: avgΔPPL={candidate.AggregatePplDelta:P4} exceeded {IsolationRules.MaximumIsolationPplDeltaRatio:P2}"
                        : $"hard-damage: avgKLD={candidate.AggregateKld:G6} exceeded {IsolationRules.MaximumIsolationKld:G6}";

                if (RuntimeSearchSpace.BanSchemeForGroup(group, candidate.Scheme))
                    result.HardDamageEliminations++;

                decision.Eliminations.Add($"{candidate.Scheme.Names[0]} -> {candidate.EliminationReason}");
                result.Notes.Add($"Eliminated '{candidate.Scheme.Names[0]}' for '{group.Name}' due to {candidate.EliminationReason}.");
            }

            // ------------------------------------------------------
            // Dominance elimination (non-imatrix only)
            // ------------------------------------------------------
            var nonImatrixSurvivors = candidates
                .Where(x => !x.Eliminated)
                .Where(x => !x.Scheme.RequiresImatrix)
                .ToList();

            for (int i = 0; i < nonImatrixSurvivors.Count; i++)
            {
                var a = nonImatrixSurvivors[i];
                if (a.Eliminated)
                    continue;

                for (int j = 0; j < nonImatrixSurvivors.Count; j++)
                {
                    if (i == j)
                        continue;

                    var b = nonImatrixSurvivors[j];
                    if (b.Eliminated)
                        continue;

                    if (Dominates(a, b))
                    {
                        b.Eliminated = true;
                        b.EliminationReason =
                            $"dominance: {a.Scheme.Names[0]} was same size or smaller and no worse on KLD/PPL with at least one strict win";

                        if (RuntimeSearchSpace.BanSchemeForGroup(group, b.Scheme))
                            result.DominatedGroupSchemesBanned++;

                        decision.Eliminations.Add($"{b.Scheme.Names[0]} -> {b.EliminationReason}");
                        result.Notes.Add($"Eliminated '{b.Scheme.Names[0]}' for '{group.Name}' because '{a.Scheme.Names[0]}' clearly dominated it.");
                    }
                }
            }

            var survivors = candidates
                .Where(x => !x.Eliminated)
                .OrderBy(x => x.AggregateKld)
                .ThenBy(x => x.AggregatePplDelta)
                .ThenBy(x => x.Snapshot.SizeBytes)
                .ToList();

            if (survivors.Count > 0)
            {
                var winner = survivors[0];
                decision.WinningScheme = winner.Scheme.Names[0];
                decision.WinningSizeBytes = winner.Snapshot.SizeBytes;
                decision.WinningKld = winner.AggregateKld;
                decision.WinningPplDelta = winner.AggregatePplDelta;
            }

            result.GroupDetails.Add(decision);
        }

        // ============================
        // BASELINE PRUNING
        // ============================
        var baseOnlyPlans = plan.Plans
            .Where(x => x.Kind == RequiredSampleKind.BaseOnlyIsolation)
            .Where(x => x.Key.StartsWith("baseonly:", StringComparison.Ordinal))
            .ToList();

        foreach (var item in baseOnlyPlans)
        {
            var snap = await LoadSnapshotAsync(item.Quant, ct);
            if (snap == null)
                continue;

            double reduction = ComputeReductionRatio(nativeBaseline.SizeBytes, snap.SizeBytes);

            if (reduction < options.MinMeaningfulBaseOnlyReductionRatio)
            {
                var baseline = BaselineQuants.FromId(item.TestedBaselineId!.Value);

                if (RuntimeSearchSpace.DisableCombinationBaseline(baseline))
                {
                    result.DisabledBaselines++;
                    result.Notes.Add(
                        $"Disabled baseline '{baseline.Names[0]}' because uncovered-tensor reduction was only {reduction:P2}.");
                }
            }
        }

        return result;
    }

    private static bool Dominates(CandidateEvaluation a, CandidateEvaluation b)
    {
        if (a.Scheme.RequiresImatrix || b.Scheme.RequiresImatrix)
            return false;

        bool sizeNoWorse = a.Snapshot.SizeBytes <= b.Snapshot.SizeBytes;
        bool kldNoWorse = a.AggregateKld <= b.AggregateKld + IsolationRules.MetricComparisonEpsilon;
        bool pplNoWorse = a.AggregatePplDelta <= b.AggregatePplDelta + IsolationRules.MetricComparisonEpsilon;

        bool strictlyBetter =
            a.Snapshot.SizeBytes < b.Snapshot.SizeBytes ||
            a.AggregateKld + IsolationRules.MetricComparisonEpsilon < b.AggregateKld ||
            a.AggregatePplDelta + IsolationRules.MetricComparisonEpsilon < b.AggregatePplDelta;

        return sizeNoWorse && kldNoWorse && pplNoWorse && strictlyBetter;
    }

    private async Task<BenchmarkSnapshot?> LoadSnapshotAsync(HybridQuant quant, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return null;

        var lookup = BuildLookup(quant);

        var row = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .Join(db.TensorCombos,
                b => b.TensorComboId,
                c => c.Id,
                (b, c) => new { b, c })
            .FirstOrDefaultAsync(x =>
                x.b.AiModelHashId == model.Id &&
                x.c.BaseQuant == lookup.BaseQuant &&
                x.c.Embeddings == lookup.Embeddings &&
                x.c.LmHead == lookup.LmHead &&
                x.c.AttnQ == lookup.AttnQ &&
                x.c.AttnKV == lookup.AttnKV &&
                x.c.AttnOutput == lookup.AttnOutput &&
                x.c.FfnUpGate == lookup.FfnUpGate &&
                x.c.FfnDown == lookup.FfnDown &&
                x.c.MoeExperts == lookup.MoeExperts &&
                x.c.MoeRouter == lookup.MoeRouter,
                ct);

        if (row == null)
            return null;

        var snapshot = new BenchmarkSnapshot { SizeBytes = row.b.SizeBytes };

        foreach (var cat in row.b.CategorBenchmarks)
        {
            snapshot.Domains[cat.Category.ToString()] = new BenchmarkDomainSnapshot
            {
                Kld = cat.Kld,
                Ppl = cat.Ppl,
                PplError = cat.PplError
            };
        }

        return snapshot;
    }

    private static double ComputeReductionRatio(ulong nativeSize, ulong candidateSize)
        => (nativeSize == 0 || candidateSize >= nativeSize)
            ? 0d
            : (nativeSize - candidateSize) / (double)nativeSize;

    private static double GetAggregateKld(BenchmarkSnapshot s)
        => s.Domains.Values
            .Where(x => x.Kld.HasValue)
            .Select(x => x.Kld!.Value)
            .DefaultIfEmpty(double.MaxValue)
            .Average();

    private static double GetAggregatePplDelta(BenchmarkSnapshot s, BenchmarkSnapshot n)
    {
        var list = new List<double>();

        foreach (var kv in s.Domains)
        {
            if (!n.Domains.TryGetValue(kv.Key, out var native))
                continue;

            if (native.Ppl <= 0)
                continue;

            list.Add(Math.Abs(kv.Value.Ppl - native.Ppl) / native.Ppl);
        }

        return list.Count == 0 ? double.MaxValue : list.Average();
    }

    private static TensorLookup BuildLookup(HybridQuant quant)
    {
        byte Get(TensorGroup g) =>
            quant.Tensors?.FirstOrDefault(x => x.TGroup.UniqueId == g.UniqueId)?.TensorType.UniqueId ?? (byte)0;

        return new TensorLookup
        {
            BaseQuant = quant.BaseQuant.UniqueId,
            Embeddings = Get(TReg.Embeddings),
            LmHead = Get(TReg.LmHead),
            AttnQ = Get(TReg.AttnQ),
            AttnKV = Get(TReg.AttnKV),
            AttnOutput = Get(TReg.AttnOutput),
            FfnUpGate = Get(TReg.FfnUpGate),
            FfnDown = Get(TReg.FfnDown),
            MoeExperts = Get(TReg.MoeExperts),
            MoeRouter = Get(TReg.MoeRouter)
        };
    }

    private sealed class CandidateEvaluation
    {
        public RequiredSamplePlan Plan { get; set; } = default!;
        public TensorWeightScheme Scheme { get; set; } = default!;
        public BenchmarkSnapshot Snapshot { get; set; } = default!;
        public double AggregateKld { get; set; }
        public double AggregatePplDelta { get; set; }
        public bool Eliminated { get; set; }
        public string? EliminationReason { get; set; }
    }

    private sealed class TensorLookup
    {
        public byte BaseQuant;
        public byte Embeddings;
        public byte LmHead;
        public byte AttnQ;
        public byte AttnKV;
        public byte AttnOutput;
        public byte FfnUpGate;
        public byte FfnDown;
        public byte MoeExperts;
        public byte MoeRouter;
    }

    private sealed class BenchmarkSnapshot
    {
        public ulong SizeBytes;
        public Dictionary<string, BenchmarkDomainSnapshot> Domains = new();
    }

    private sealed class BenchmarkDomainSnapshot
    {
        public double? Kld;
        public double Ppl;
        public double PplError;
    }
}