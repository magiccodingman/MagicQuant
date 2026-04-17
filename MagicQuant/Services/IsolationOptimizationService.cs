using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;

namespace MagicQuant.Services;

public sealed class IsolationOptimizationOptions
{
    public double MinMeaningfulGroupReductionRatio { get; set; } =
        IsolationPruningConfig.MinimumIsolationReductionToContinueRatio;

    public double MinMeaningfulBaseOnlyReductionRatio { get; set; } =
        IsolationPruningConfig.MinimumMeaningfulBaseOnlyReductionRatio;
}

public sealed class IsolationGroupDecision
{
    public string GroupName { get; set; } = string.Empty;
    public double BestReductionRatio { get; set; }

    public bool ExplicitQuantBanned { get; set; }
    public bool Bf16Suppressed { get; set; }

    public string? WinningScheme { get; set; }
    public ulong? WinningSizeBytes { get; set; }
    public double? WinningKld { get; set; }
    public double? WinningPplDelta { get; set; }

    public List<string> Candidates { get; set; } = new();
}

public sealed class InitialIsolationAnalysisResult
{
    public List<byte> GroupsToContinue { get; set; } = new();
    public List<string> Notes { get; set; } = new();
    public List<IsolationGroupDecision> GroupDetails { get; set; } = new();
}

public sealed class IsolationOptimizationResult
{
    public int ExplicitQuantBannedGroups { get; set; }
    public int DominatedGroupSchemesBanned { get; set; }
    public int HardDamageEliminations { get; set; }
    public int BadTradeEliminations { get; set; }
    public int DisabledBaselines { get; set; }
    public int Bf16SuppressedGroups { get; set; }

    public List<string> Notes { get; set; } = new();
    public List<IsolationGroupDecision> GroupDetails { get; set; } = new();
}

public class IsolationOptimizationService
{
    public async Task<InitialIsolationAnalysisResult> AnalyzeInitialIsolationProbesAsync(
        RequiredSampleGenerationResult plan,
        IsolationOptimizationOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new IsolationOptimizationOptions();

        var result = new InitialIsolationAnalysisResult();

        var nativeBaseline = await LoadSnapshotAsync(
                                 HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()), ct)
                             ?? throw new InvalidOperationException("Native BF16 baseline benchmark was not found.");

        var carrierBaselineId = BaselineQuants.Q8_0.UniqueId;

        var carrierBaseOnlyPlan = plan.Plans.First(x =>
            x.Kind == RequiredSampleKind.BaseOnlyIsolation &&
            x.TestedBaselineId == carrierBaselineId &&
            x.Key.StartsWith("carrier-baseonly:", StringComparison.Ordinal));

        var carrierBaseOnly = await LoadSnapshotAsync(carrierBaseOnlyPlan.Quant, ct)
                              ?? throw new InvalidOperationException("Carrier base-only benchmark was not found.");

        var groupPlans = plan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolationProbe)
            .Where(x => x.TestedBaselineId == carrierBaselineId)
            .GroupBy(x => x.TargetGroupId!.Value)
            .OrderBy(x => x.Key)
            .ToList();

        foreach (var groupSet in groupPlans)
        {
            var group = TReg.All.First(x => x.UniqueId == groupSet.Key);

            var item = groupSet.Single();
            var snap = await LoadSnapshotAsync(item.Quant, ct);
            if (snap == null)
                continue;

            var scheme = TensorWeightScheme.All_Allowed_Hybrid_Quants.First(x => x.UniqueId == item.TestedSchemeId);
            var reduction = ComputeReductionRatio(carrierBaseOnly.SizeBytes, snap.SizeBytes);
            var kld = GetAggregateKld(snap);
            var pplDelta = GetAggregatePplDeltaPercent(snap, nativeBaseline);

            var decision = new IsolationGroupDecision
            {
                GroupName = group.Name,
                BestReductionRatio = reduction,
                WinningScheme = scheme.Names[0],
                WinningSizeBytes = snap.SizeBytes,
                WinningKld = kld,
                WinningPplDelta = pplDelta
            };

            decision.Candidates.Add(
                $"{scheme.Names[0]} | size={(snap.SizeBytes / 1024.0 / 1024.0):F2}MB | savings={reduction:P2} | kld={kld:G6} | pplΔ={pplDelta:F4}%");

            if (reduction < options.MinMeaningfulGroupReductionRatio)
            {
                RuntimeSearchSpace.BanAllExplicitTensorSchemesForGroup(group);
                decision.ExplicitQuantBanned = true;

                result.Notes.Add(
                    $"Early stop for '{group.Name}': smallest non-imatrix '{scheme.Names[0]}' only saved {reduction:P2}, below {options.MinMeaningfulGroupReductionRatio:P2}. Explicit tensor quant exploration removed for this group.");

                result.GroupDetails.Add(decision);
                continue;
            }

            result.GroupsToContinue.Add(group.UniqueId);

            if (reduction >= IsolationPruningConfig.MinimumIsolationReductionToSuppressBf16Ratio)
            {
                RuntimeSearchSpace.SuppressBf16TensorChoice(group);
                decision.Bf16Suppressed = true;

                result.Notes.Add(
                    $"Suppressed BF16 tensor-choice for '{group.Name}' because smallest probe already saved {reduction:P2}.");
            }

            result.GroupDetails.Add(decision);
        }

        return result;
    }

    public async Task<IsolationOptimizationResult> AnalyzeAndApplyFinalAsync(
        RequiredSampleGenerationResult fullPlan,
        IsolationOptimizationOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new IsolationOptimizationOptions();

        var result = new IsolationOptimizationResult();

        var nativeBaseline = await LoadSnapshotAsync(
                                 HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()), ct)
                             ?? throw new InvalidOperationException("Native BF16 baseline benchmark was not found.");

        var carrierBaselineId = BaselineQuants.Q8_0.UniqueId;

        var carrierBaseOnlyPlan = fullPlan.Plans.First(x =>
            x.Kind == RequiredSampleKind.BaseOnlyIsolation &&
            x.TestedBaselineId == carrierBaselineId &&
            x.Key.StartsWith("carrier-baseonly:", StringComparison.Ordinal));

        var carrierBaseOnly = await LoadSnapshotAsync(carrierBaseOnlyPlan.Quant, ct)
                              ?? throw new InvalidOperationException("Carrier base-only benchmark was not found.");

        var groupPlans = fullPlan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolationProbe ||
                        x.Kind == RequiredSampleKind.GroupIsolationContinuation)
            .Where(x => x.TestedBaselineId == carrierBaselineId)
            .GroupBy(x => x.TargetGroupId!.Value)
            .OrderBy(x => x.Key)
            .ToList();

        foreach (var groupSet in groupPlans)
        {
            var group = TReg.All.First(x => x.UniqueId == groupSet.Key);

            var candidates = new List<GroupCandidate>();

            foreach (var item in groupSet)
            {
                var snap = await LoadSnapshotAsync(item.Quant, ct);
                if (snap == null)
                    continue;

                var scheme = TensorWeightScheme.All_Allowed_Hybrid_Quants.First(x => x.UniqueId == item.TestedSchemeId);

                candidates.Add(new GroupCandidate
                {
                    Group = group,
                    Scheme = scheme,
                    SizeBytes = snap.SizeBytes,
                    SavingsRatio = ComputeReductionRatio(carrierBaseOnly.SizeBytes, snap.SizeBytes),
                    Kld = GetAggregateKld(snap),
                    PplDeltaPercent = GetAggregatePplDeltaPercent(snap, nativeBaseline)
                });
            }

            if (candidates.Count == 0)
                continue;

            var decision = new IsolationGroupDecision
            {
                GroupName = group.Name,
                BestReductionRatio = candidates.Max(x => x.SavingsRatio)
            };

            foreach (var candidate in candidates.ToList())
            {
                if (candidate.Scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId)
                    continue;

                if (candidate.Scheme.RequiresImatrix)
                    continue;

                bool hardFail =
                    candidate.PplDeltaPercent >= IsolationPruningConfig.MaximumIsolationPplDeltaPercent ||
                    candidate.Kld >= IsolationPruningConfig.MaximumIsolationKld;

                if (!hardFail)
                    continue;

                RuntimeSearchSpace.BanSchemeForGroup(group, candidate.Scheme);
                result.HardDamageEliminations++;

                result.Notes.Add(
                    $"Hard damage elimination: '{candidate.Scheme.Names[0]}' removed for '{group.Name}' " +
                    $"(savings={candidate.SavingsRatio:P2}, KLD={candidate.Kld:G6}, PPLΔ={candidate.PplDeltaPercent:F4}%).");
            }

            candidates = FilterSurvivors(group, candidates);

            ApplyDominanceElimination(group, candidates, result);

            candidates = FilterSurvivors(group, candidates);

            ApplyBadTradeElimination(group, candidates, result);

            candidates = FilterSurvivors(group, candidates)
                .OrderBy(x => x.Kld)
                .ThenBy(x => x.PplDeltaPercent)
                .ThenByDescending(x => x.SavingsRatio)
                .ToList();

            if (candidates.Count == 0)
            {
                decision.ExplicitQuantBanned = RuntimeSearchSpace.IsGroupExplicitQuantBanned(group);
                decision.Bf16Suppressed = RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group);
                result.GroupDetails.Add(decision);
                continue;
            }

            var winner = candidates.First();

            decision.WinningScheme = winner.Scheme.Names[0];
            decision.WinningSizeBytes = winner.SizeBytes;
            decision.WinningKld = winner.Kld;
            decision.WinningPplDelta = winner.PplDeltaPercent;
            decision.ExplicitQuantBanned = RuntimeSearchSpace.IsGroupExplicitQuantBanned(group);
            decision.Bf16Suppressed = RuntimeSearchSpace.IsBf16TensorChoiceSuppressed(group);

            foreach (var candidate in candidates.OrderBy(x => x.SizeBytes))
            {
                decision.Candidates.Add(
                    $"{candidate.Scheme.Names[0]} | size={(candidate.SizeBytes / 1024.0 / 1024.0):F2}MB | savings={candidate.SavingsRatio:P2} | kld={candidate.Kld:G6} | pplΔ={candidate.PplDeltaPercent:F4}%");
            }

            foreach (var banInfo in RuntimeSearchSpace.GetLearnedBaselineMissingPrunedSchemesForGroup(group))
            {
                var sourceBaselines = string.Join(
                    ", ",
                    banInfo.MissingBaselines.Select(x => x.Names[0]));

                decision.Candidates.Add(
                    $"[pruned-early] {banInfo.Scheme.Names[0]} removed by learned-baseline mapping for this group (no matching tensor weights in baseline(s): {sourceBaselines}).");
            }

            result.GroupDetails.Add(decision);
        }

        var baseOnlyPlans = fullPlan.Plans
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
                        $"Disabled combination baseline '{baseline.Names[0]}' because uncovered-tensor reduction was only {reduction:P2}.");
                }
            }
        }

        result.ExplicitQuantBannedGroups = RuntimeSearchSpace.GetGroupsWithExplicitQuantBanned().Count;
        result.Bf16SuppressedGroups = RuntimeSearchSpace.GetBf16SuppressedGroups().Count;

        return result;
    }

    private static List<GroupCandidate> FilterSurvivors(TensorGroup group, List<GroupCandidate> candidates)
    {
        return candidates
            .Where(x => x.Scheme.UniqueId == TensorWeightScheme.BF16_F16.UniqueId ||
                        !RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(group, x.Scheme))
            .ToList();
    }

    private static void ApplyDominanceElimination(
        TensorGroup group,
        List<GroupCandidate> candidates,
        IsolationOptimizationResult result)
    {
        var explicitCandidates = candidates
            .Where(x => x.Scheme.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .Where(x => !x.Scheme.RequiresImatrix)
            .ToList();

        for (int i = 0; i < explicitCandidates.Count; i++)
        {
            for (int j = 0; j < explicitCandidates.Count; j++)
            {
                if (i == j)
                    continue;

                var a = explicitCandidates[i];
                var b = explicitCandidates[j];

                bool sameOrSmaller = a.SizeBytes <= b.SizeBytes;
                bool kldNoWorse = a.Kld <= b.Kld + IsolationPruningConfig.FloatingPointEpsilon;
                bool pplNoWorse = a.PplDeltaPercent <= b.PplDeltaPercent + IsolationPruningConfig.FloatingPointEpsilon;

                bool strictlyBetter =
                    a.Kld + IsolationPruningConfig.FloatingPointEpsilon < b.Kld ||
                    a.PplDeltaPercent + IsolationPruningConfig.FloatingPointEpsilon < b.PplDeltaPercent ||
                    a.SizeBytes < b.SizeBytes;

                if (sameOrSmaller && kldNoWorse && pplNoWorse && strictlyBetter)
                {
                    if (!RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(group, b.Scheme))
                    {
                        RuntimeSearchSpace.BanSchemeForGroup(group, b.Scheme);
                        result.DominatedGroupSchemesBanned++;

                        result.Notes.Add(
                            $"Dominance elimination: '{b.Scheme.Names[0]}' removed for '{group.Name}' because '{a.Scheme.Names[0]}' was same-size-or-smaller and no worse on KLD/PPL.");
                    }
                }
            }
        }
    }

    private static void ApplyBadTradeElimination(
        TensorGroup group,
        List<GroupCandidate> candidates,
        IsolationOptimizationResult result)
    {
        var explicitCandidates = candidates
            .Where(x => x.Scheme.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .Where(x => !x.Scheme.RequiresImatrix)
            .OrderBy(x => x.SizeBytes)
            .ToList();

        // Compare only to nearby larger neighbors, not the whole ladder.
        for (int i = 0; i < explicitCandidates.Count; i++)
        {
            var smaller = explicitCandidates[i];

            for (int j = i + 1; j < explicitCandidates.Count && j <= i + 2; j++)
            {
                var larger = explicitCandidates[j];

                double sizeDeltaPercent =
                    ((double)larger.SizeBytes - smaller.SizeBytes) / larger.SizeBytes * 100.0;

                if (sizeDeltaPercent > IsolationPruningConfig.BadTradeMaxSizeDeltaPercent)
                    continue;

                double smallerPplAbs = Math.Abs(smaller.PplDeltaPercent);
                double largerPplAbs = Math.Abs(larger.PplDeltaPercent);

                double kldRatio = larger.Kld <= IsolationPruningConfig.FloatingPointEpsilon
                    ? double.PositiveInfinity
                    : smaller.Kld / larger.Kld;

                double pplRatio = largerPplAbs <= IsolationPruningConfig.FloatingPointEpsilon
                    ? double.PositiveInfinity
                    : smallerPplAbs / largerPplAbs;

                bool kldBadTrade =
                    smaller.Kld > larger.Kld * IsolationPruningConfig.BadTradeKldMultiplier;

                bool pplBadTrade =
                    smallerPplAbs > largerPplAbs * IsolationPruningConfig.BadTradePplMultiplier;

                bool smallerMeaningfullyBetterKld =
                    smaller.Kld + IsolationPruningConfig.FloatingPointEpsilon < larger.Kld * 0.90;

                bool smallerMeaningfullyBetterPpl =
                    smallerPplAbs + IsolationPruningConfig.FloatingPointEpsilon < largerPplAbs * 0.90;

                bool mixedTradeoff =
                    (kldBadTrade && smallerMeaningfullyBetterPpl) ||
                    (pplBadTrade && smallerMeaningfullyBetterKld);

                if (mixedTradeoff)
                    continue;

                if (!kldBadTrade && !pplBadTrade)
                    continue;

                if (!RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(group, smaller.Scheme))
                {
                    RuntimeSearchSpace.BanSchemeForGroup(group, smaller.Scheme);
                    result.BadTradeEliminations++;

                    result.Notes.Add(
                        $"Bad trade elimination: '{smaller.Scheme.Names[0]}' removed vs '{larger.Scheme.Names[0]}' for '{group.Name}'. " +
                        $"Reason: small size gain ({sizeDeltaPercent:F2}%) but disproportionate damage " +
                        $"(KLD x{kldRatio:F2}, |PPL| x{pplRatio:F2}).");
                }

                break;
            }
        }
    }

    private async Task<BenchmarkSnapshot?> LoadSnapshotAsync(HybridQuant quant, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return null;

        var lookup = (TensorConfig)quant;

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

        return new BenchmarkSnapshot
        {
            SizeBytes = row.b.SizeBytes,
            Benchmarks = row.b.CategorBenchmarks
                .Select(x => new CategorySnapshot
                {
                    Category = x.Category,
                    Kld = x.Kld,
                    Ppl = x.Ppl,
                    PplError = x.PplError
                })
                .ToList()
        };
    }

    private static double ComputeReductionRatio(ulong baselineBytes, ulong candidateBytes)
    {
        if (baselineBytes == 0)
            return 0d;

        double delta = baselineBytes - candidateBytes;
        return delta / baselineBytes;
    }

    private static double GetAggregateKld(BenchmarkSnapshot snapshot)
    {
        return snapshot.Benchmarks
            .Where(x => x.Kld.HasValue)
            .Select(x => x.Kld!.Value)
            .DefaultIfEmpty(double.PositiveInfinity)
            .Average();
    }

    private static double GetAggregatePplDeltaPercent(BenchmarkSnapshot snapshot, BenchmarkSnapshot nativeBaseline)
    {
        var nativeMap = nativeBaseline.Benchmarks.ToDictionary(x => x.Category);
        var deltas = new List<double>();

        foreach (var bench in snapshot.Benchmarks)
        {
            if (!nativeMap.TryGetValue(bench.Category, out var native))
                continue;

            if (native.Ppl <= IsolationPruningConfig.FloatingPointEpsilon)
                continue;

            double deltaPercent = ((bench.Ppl - native.Ppl) / native.Ppl) * 100.0;
            deltas.Add(deltaPercent);
        }

        return deltas.Count == 0 ? double.PositiveInfinity : deltas.Average();
    }

    private sealed class GroupCandidate
    {
        public TensorGroup Group { get; set; } = default!;
        public TensorWeightScheme Scheme { get; set; } = default!;
        public ulong SizeBytes { get; set; }
        public double SavingsRatio { get; set; }
        public double Kld { get; set; }
        public double PplDeltaPercent { get; set; }
    }

    private sealed class BenchmarkSnapshot
    {
        public ulong SizeBytes { get; set; }
        public List<CategorySnapshot> Benchmarks { get; set; } = new();
    }

    private sealed class CategorySnapshot
    {
        public byte Category { get; set; }
        public double? Kld { get; set; }
        public double Ppl { get; set; }
        public double PplError { get; set; }
    }
}
