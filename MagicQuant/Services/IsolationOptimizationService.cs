using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;

namespace MagicQuant.Services;

public sealed class IsolationOptimizationOptions
{
    public double MinMeaningfulGroupReductionRatio { get; set; } = 0.10d;
    public double MinMeaningfulBaseOnlyReductionRatio { get; set; } = 0.01d;
}

public sealed class IsolationGroupDecision
{
    public string GroupName { get; set; } = string.Empty;
    public double BestReductionRatio { get; set; }
    public bool LockedToNative { get; set; }

    public string? WinningScheme { get; set; }
    public ulong? WinningSizeBytes { get; set; }
    public double? WinningKld { get; set; }
    public double? WinningPplDelta { get; set; }

    public List<string> Candidates { get; set; } = new();
}

public sealed class IsolationOptimizationResult
{
    public int NativeLockedGroups { get; set; }
    public int DominatedGroupSchemesBanned { get; set; }
    public int DisabledBaselines { get; set; }

    public List<string> Notes { get; set; } = new();
    public List<IsolationGroupDecision> GroupDetails { get; set; } = new();
}

public class IsolationOptimizationService
{
    public async Task<IsolationOptimizationResult> AnalyzeAndApplyAsync(
        RequiredSampleGenerationResult plan,
        IsolationOptimizationOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new IsolationOptimizationOptions();

        var result = new IsolationOptimizationResult();

        var nativeBaseline = await LoadSnapshotAsync(
            HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()), ct);

        var carrierBaselineId = BaselineQuants.Q8_0.UniqueId;

        var carrierBaseOnlyPlan = plan.Plans.First(x =>
            x.Kind == RequiredSampleKind.BaseOnlyIsolation &&
            x.TestedBaselineId == carrierBaselineId);

        var carrierBaseOnly = await LoadSnapshotAsync(carrierBaseOnlyPlan.Quant, ct);

        // ============================
        // GROUP ISOLATION ANALYSIS
        // ============================
        var groupPlans = plan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolation)
            .Where(x => x.TestedBaselineId == carrierBaselineId)
            .GroupBy(x => x.TargetGroupId!.Value)
            .ToList();

        foreach (var groupSet in groupPlans)
        {
            var group = TReg.All.First(x => x.UniqueId == groupSet.Key);

            var snapshots = new List<(RequiredSamplePlan Plan, BenchmarkSnapshot Snapshot)>();

            foreach (var item in groupSet)
            {
                var snap = await LoadSnapshotAsync(item.Quant, ct);
                if (snap != null)
                    snapshots.Add((item, snap));
            }

            if (snapshots.Count == 0)
                continue;

            var decision = new IsolationGroupDecision
            {
                GroupName = group.Name
            };

            // compute best reduction vs carrier
            double maxReduction = snapshots
                .Select(x => ComputeReductionRatio(carrierBaseOnly.SizeBytes, x.Snapshot.SizeBytes))
                .Max();

            decision.BestReductionRatio = maxReduction;

            // build candidate debug list
            foreach (var snap in snapshots.OrderBy(x => x.Snapshot.SizeBytes))
            {
                var scheme = TensorWeightScheme.All
                    .FirstOrDefault(x => x.UniqueId == snap.Plan.TestedSchemeId);

                var reduction = ComputeReductionRatio(carrierBaseOnly.SizeBytes, snap.Snapshot.SizeBytes);
                var kld = GetAggregateKld(snap.Snapshot);
                var pplDelta = GetAggregatePplDelta(snap.Snapshot, nativeBaseline);

                decision.Candidates.Add(
                    $"{scheme?.Names[0] ?? "Unknown"} | size={(snap.Snapshot.SizeBytes / 1024.0 / 1024.0):F2}MB | reduction={reduction:P2} | kld={kld:G6} | pplΔ={pplDelta:P4}");
            }

            // LOCK LOGIC
            if (maxReduction < options.MinMeaningfulGroupReductionRatio)
            {
                RuntimeSearchSpace.LockGroupToNative(group);

                decision.LockedToNative = true;
                result.NativeLockedGroups++;

                result.Notes.Add(
                    $"Locked '{group.Name}' to native (max reduction {maxReduction:P2})");

                result.GroupDetails.Add(decision);
                continue;
            }

            decision.LockedToNative = false;

            // FIND WINNER
            var ordered = snapshots
                .OrderBy(x => GetAggregateKld(x.Snapshot))
                .ThenBy(x => GetAggregatePplDelta(x.Snapshot, nativeBaseline))
                .ToList();

            var winner = ordered.First();

            var winnerScheme = TensorWeightScheme.All
                .FirstOrDefault(x => x.UniqueId == winner.Plan.TestedSchemeId);

            decision.WinningScheme = winnerScheme?.Names[0];
            decision.WinningSizeBytes = winner.Snapshot.SizeBytes;
            decision.WinningKld = GetAggregateKld(winner.Snapshot);
            decision.WinningPplDelta = GetAggregatePplDelta(winner.Snapshot, nativeBaseline);

            // BAN LOSERS WITH SAME SIZE
            foreach (var sizeBucket in snapshots.GroupBy(x => x.Snapshot.SizeBytes))
            {
                if (sizeBucket.Count() <= 1)
                    continue;

                var sorted = sizeBucket
                    .OrderBy(x => GetAggregateKld(x.Snapshot))
                    .ThenBy(x => GetAggregatePplDelta(x.Snapshot, nativeBaseline))
                    .ToList();

                foreach (var loser in sorted.Skip(1))
                {
                    var scheme = TensorWeightScheme.All
                        .FirstOrDefault(x => x.UniqueId == loser.Plan.TestedSchemeId);

                    if (scheme == null)
                        continue;

                    if (!scheme.BannedGroups.Any(x => x.UniqueId == group.UniqueId))
                    {
                        scheme.BannedGroups.Add(group);
                        result.DominatedGroupSchemesBanned++;

                        result.Notes.Add(
                            $"Banned {scheme.Names[0]} for {group.Name} (same size, worse fidelity)");
                    }
                }
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
            if (snap == null) continue;

            double reduction = ComputeReductionRatio(nativeBaseline.SizeBytes, snap.SizeBytes);

            if (reduction < options.MinMeaningfulBaseOnlyReductionRatio)
            {
                var baseline = BaselineQuants.FromId(item.TestedBaselineId!.Value);

                if (RuntimeSearchSpace.DisableCombinationBaseline(baseline))
                {
                    result.DisabledBaselines++;
                    result.Notes.Add($"Disabled baseline {baseline.Names[0]} (reduction {reduction:P2})");
                }
            }
        }

        return result;
    }

    // ============================
    // HELPERS (unchanged)
    // ============================

    private async Task<BenchmarkSnapshot?> LoadSnapshotAsync(HybridQuant quant, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null) return null;

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

        if (row == null) return null;

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
        => s.Domains.Values.Where(x => x.Kld.HasValue).Select(x => x.Kld!.Value).DefaultIfEmpty(double.MaxValue).Average();

    private static double GetAggregatePplDelta(BenchmarkSnapshot s, BenchmarkSnapshot n)
    {
        var list = new List<double>();

        foreach (var kv in s.Domains)
        {
            if (!n.Domains.TryGetValue(kv.Key, out var native)) continue;
            if (native.Ppl <= 0) continue;

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