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

    public string? WinningCandidate { get; set; }
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
    public int DominatedGroupCandidatesBanned { get; set; }
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
            var probe = groupSet.Single();

            var snap = await LoadSnapshotAsync(probe.Quant, ct);
            if (snap == null)
                continue;

            var candidate = BaselineQuants.FromId(probe.TestedCandidateId!.Value);
            var reduction = ComputeReductionRatio(carrierBaseOnly.SizeBytes, snap.SizeBytes);
            var kld = GetAggregateKld(snap);
            var pplDelta = GetAggregatePplDeltaPercent(snap, nativeBaseline);

            var decision = new IsolationGroupDecision
            {
                GroupName = group.Name,
                BestReductionRatio = reduction,
                WinningCandidate = candidate.Names[0],
                WinningSizeBytes = snap.SizeBytes,
                WinningKld = kld,
                WinningPplDelta = pplDelta
            };

            decision.Candidates.Add(
                $"{candidate.Names[0]} | size={(snap.SizeBytes / 1024.0 / 1024.0):F2}MB | savings={reduction:P2} | kld={kld:G6} | pplΔ={pplDelta:F4}%");

            if (reduction < options.MinMeaningfulGroupReductionRatio)
            {
                RuntimeSearchSpace.BanAllExplicitCombinationCandidatesForGroup(group);
                decision.ExplicitQuantBanned = true;

                result.Notes.Add(
                    $"Early stop for '{group.Name}': smallest baseline-candidate probe '{candidate.Names[0]}' only saved {reduction:P2}, below {options.MinMeaningfulGroupReductionRatio:P2}. Explicit baseline-candidate exploration removed for this group.");

                result.GroupDetails.Add(decision);
                continue;
            }

            result.GroupsToContinue.Add(group.UniqueId);

            if (reduction >= IsolationPruningConfig.MinimumIsolationReductionToSuppressBf16Ratio)
            {
                RuntimeSearchSpace.SuppressBf16TensorChoice(group);
                decision.Bf16Suppressed = true;

                result.Notes.Add(
                    $"Suppressed BF16 explicit candidate for '{group.Name}' because smallest baseline-candidate probe already saved {reduction:P2}.");
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
            var candidates = new List<GroupCandidateEvaluation>();

            foreach (var item in groupSet)
            {
                var snap = await LoadSnapshotAsync(item.Quant, ct);
                if (snap == null)
                    continue;

                var candidateBaseline = BaselineQuants.FromId(item.TestedCandidateId!.Value);
                RuntimeSearchSpace.ClearLearnedBaselinePruneForGroupCandidate(group, candidateBaseline);

                candidates.Add(new GroupCandidateEvaluation
                {
                    Group = group,
                    CandidateBaseline = candidateBaseline,
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
                if (IsHighPrecisionCandidate(candidate.CandidateBaseline))
                    continue;

                bool hardFail =
                    candidate.PplDeltaPercent >= IsolationPruningConfig.MaximumIsolationPplDeltaPercent ||
                    candidate.Kld >= IsolationPruningConfig.MaximumIsolationKld;

                if (!hardFail)
                    continue;

                RuntimeSearchSpace.BanCombinationCandidateForGroup(group, candidate.CandidateBaseline);
                result.HardDamageEliminations++;

                result.Notes.Add(
                    $"Hard damage elimination: '{candidate.CandidateBaseline.Names[0]}' removed for '{group.Name}' " +
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
                PopulateFinalGroupFlags(group, decision, result);
                result.GroupDetails.Add(decision);
                continue;
            }

            var winner = candidates.First();
            decision.WinningCandidate = winner.CandidateBaseline.Names[0];
            decision.WinningSizeBytes = winner.SizeBytes;
            decision.WinningKld = winner.Kld;
            decision.WinningPplDelta = winner.PplDeltaPercent;
            PopulateFinalGroupFlags(group, decision, result);

            foreach (var candidate in candidates.OrderBy(x => x.SizeBytes))
            {
                decision.Candidates.Add(
                    $"{candidate.CandidateBaseline.Names[0]} | size={(candidate.SizeBytes / 1024.0 / 1024.0):F2}MB | savings={candidate.SavingsRatio:P2} | kld={candidate.Kld:G6} | pplΔ={candidate.PplDeltaPercent:F4}%");
            }

            var survivorIds = candidates.Select(x => x.CandidateBaseline.UniqueId).ToHashSet();
            foreach (var banInfo in RuntimeSearchSpace.GetLearnedBaselineMissingPrunedCandidatesForGroup(group))
            {
                if (survivorIds.Contains(banInfo.Candidate.UniqueId))
                    continue;

                string expected = banInfo.ExpectedTensorWeightSchemeIds.Count == 0
                    ? "<none>"
                    : string.Join(", ", banInfo.ExpectedTensorWeightSchemeIds);
                string matched = banInfo.MatchedTensorWeightSchemeIds.Count == 0
                    ? "<none>"
                    : string.Join(", ", banInfo.MatchedTensorWeightSchemeIds);
                string missing = banInfo.MissingTensorWeightSchemeIds.Count == 0
                    ? "<none>"
                    : string.Join(", ", banInfo.MissingTensorWeightSchemeIds);

                decision.Candidates.Add(
                    $"[pruned-early] {banInfo.Candidate.Names[0]} removed by learned candidate/group scheme matching " +
                    $"(expected schemes: {expected}; matched: {matched}; missing: {missing}).");
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
        result.Bf16SuppressedGroups = result.GroupDetails.Count(x => x.Bf16Suppressed);

        return result;
    }

    private static bool IsHighPrecisionCandidate(BaselineQuants candidate)
        => candidate.UniqueId == BaselineQuants.BF16_Hybrid.UniqueId || candidate.UniqueId == BaselineQuants.F16_Hybrid.UniqueId;

    private static void PopulateFinalGroupFlags(TensorGroup group, IsolationGroupDecision decision, IsolationOptimizationResult result)
    {
        var (explicitAllowed, bf16Allowed) = RuntimeSearchSpace.GetFinalAllowedQuantFamiliesForGroup(group);

        decision.ExplicitQuantBanned = !explicitAllowed;
        decision.Bf16Suppressed = !bf16Allowed;

        if (!explicitAllowed && !bf16Allowed)
        {
            result.Notes.Add(
                $"[invariant-warning] Invalid final quant-family state for '{group.Name}': neither explicit candidate nor BF16 is allowed.");
        }
    }

    private static List<GroupCandidateEvaluation> FilterSurvivors(TensorGroup group, List<GroupCandidateEvaluation> candidates)
    {
        return candidates
            .Where(x => IsHighPrecisionCandidate(x.CandidateBaseline) ||
                        !RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, x.CandidateBaseline))
            .ToList();
    }

    private static void ApplyDominanceElimination(TensorGroup group, List<GroupCandidateEvaluation> candidates, IsolationOptimizationResult result)
    {
        var explicitCandidates = GetActiveExplicitCandidates(group, candidates);

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
                bool pplNoWorse = Math.Abs(a.PplDeltaPercent) <= Math.Abs(b.PplDeltaPercent) + IsolationPruningConfig.FloatingPointEpsilon;

                bool strictlyBetter =
                    a.Kld + IsolationPruningConfig.FloatingPointEpsilon < b.Kld ||
                    Math.Abs(a.PplDeltaPercent) + IsolationPruningConfig.FloatingPointEpsilon < Math.Abs(b.PplDeltaPercent) ||
                    a.SizeBytes < b.SizeBytes;

                if (sameOrSmaller && kldNoWorse && pplNoWorse && strictlyBetter)
                {
                    if (!RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, b.CandidateBaseline))
                    {
                        RuntimeSearchSpace.BanCombinationCandidateForGroup(group, b.CandidateBaseline);
                        result.DominatedGroupCandidatesBanned++;

                        result.Notes.Add(
                            $"Dominance elimination: '{b.CandidateBaseline.Names[0]}' removed for '{group.Name}' because '{a.CandidateBaseline.Names[0]}' was same-size-or-smaller and no worse on KLD/PPL.");
                    }
                }
            }
        }
    }

    private static void ApplyBadTradeElimination(TensorGroup group, List<GroupCandidateEvaluation> candidates, IsolationOptimizationResult result)
    {
        var activeCandidates = GetActiveExplicitCandidates(group, candidates);
        if (activeCandidates.Count <= 1)
            return;

        var sizeBuckets = BuildSizeBuckets(activeCandidates);
        if (sizeBuckets.Count == 0)
            return;

        var acceptedAnchor = SelectBestBucketSurvivor(sizeBuckets[0]);
        if (acceptedAnchor == null)
            return;

        for (int i = 1; i < sizeBuckets.Count; i++)
        {
            var bucketSurvivors = new List<GroupCandidateEvaluation>();

            foreach (var candidate in sizeBuckets[i])
            {
                if (RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate.CandidateBaseline))
                    continue;

                if (ShouldEliminateAsBadTrade(acceptedAnchor, candidate, out var reason))
                {
                    RuntimeSearchSpace.BanCombinationCandidateForGroup(group, candidate.CandidateBaseline);
                    result.BadTradeEliminations++;
                    result.Notes.Add(
                        $"Bad trade elimination: '{candidate.CandidateBaseline.Names[0]}' removed vs accepted anchor '{acceptedAnchor.CandidateBaseline.Names[0]}' for '{group.Name}'. {reason}");
                    continue;
                }

                bucketSurvivors.Add(candidate);
            }

            var promotedAnchor = SelectBestBucketSurvivor(bucketSurvivors);
            if (promotedAnchor != null)
                acceptedAnchor = promotedAnchor;
        }
    }

    private static List<GroupCandidateEvaluation> GetActiveExplicitCandidates(TensorGroup group, List<GroupCandidateEvaluation> candidates)
    {
        return candidates
            .Where(x => !IsHighPrecisionCandidate(x.CandidateBaseline))
            .Where(x => !RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, x.CandidateBaseline))
            .ToList();
    }

    private static List<List<GroupCandidateEvaluation>> BuildSizeBuckets(List<GroupCandidateEvaluation> candidates)
    {
        return candidates
            .GroupBy(x => x.SizeBytes)
            .OrderByDescending(x => x.Key)
            .Select(x => x
                .OrderBy(c => c.Kld)
                .ThenBy(c => Math.Abs(c.PplDeltaPercent))
                .ThenByDescending(c => GetCandidateSafetyScore(c.CandidateBaseline))
                .ThenBy(c => c.CandidateBaseline.Names[0], StringComparer.Ordinal)
                .ToList())
            .ToList();
    }

    private static bool ShouldEliminateAsBadTrade(GroupCandidateEvaluation anchor, GroupCandidateEvaluation candidate, out string reason)
    {
        reason = string.Empty;

        if (anchor.SizeBytes <= candidate.SizeBytes)
            return false;

        double sizeDeltaPercent = ((double)anchor.SizeBytes - candidate.SizeBytes) / anchor.SizeBytes * 100.0;
        if (sizeDeltaPercent > IsolationPruningConfig.BadTradeMaxSizeDeltaPercent)
            return false;

        double anchorPplAbs = Math.Abs(anchor.PplDeltaPercent);
        double candidatePplAbs = Math.Abs(candidate.PplDeltaPercent);

        double kldRatio = anchor.Kld <= IsolationPruningConfig.FloatingPointEpsilon ? double.PositiveInfinity : candidate.Kld / anchor.Kld;
        double pplRatio = anchorPplAbs <= IsolationPruningConfig.FloatingPointEpsilon ? double.PositiveInfinity : candidatePplAbs / anchorPplAbs;

        bool kldBadTrade = candidate.Kld > anchor.Kld * IsolationPruningConfig.BadTradeKldMultiplier;
        bool pplBadTrade = candidatePplAbs > anchorPplAbs * IsolationPruningConfig.BadTradePplMultiplier;

        bool candidateMeaningfullyBetterKld =
            candidate.Kld + IsolationPruningConfig.FloatingPointEpsilon < anchor.Kld * 0.90;

        bool candidateMeaningfullyBetterPpl =
            candidatePplAbs + IsolationPruningConfig.FloatingPointEpsilon < anchorPplAbs * 0.90;

        bool mixedTradeoff =
            (kldBadTrade && candidateMeaningfullyBetterPpl) ||
            (pplBadTrade && candidateMeaningfullyBetterKld);

        if (mixedTradeoff || (!kldBadTrade && !pplBadTrade))
            return false;

        reason =
            $"Reason: small size gain ({sizeDeltaPercent:F2}%) but disproportionate damage (KLD x{kldRatio:F2}, |PPL| x{pplRatio:F2}).";

        return true;
    }

    private static GroupCandidateEvaluation? SelectBestBucketSurvivor(List<GroupCandidateEvaluation> survivors)
    {
        return survivors
            .OrderBy(x => x.Kld)
            .ThenBy(x => Math.Abs(x.PplDeltaPercent))
            .ThenByDescending(x => GetCandidateSafetyScore(x.CandidateBaseline))
            .ThenBy(x => x.CandidateBaseline.Names[0], StringComparer.Ordinal)
            .FirstOrDefault();
    }

    private static int GetCandidateSafetyScore(BaselineQuants candidate)
    {
        string canonical = candidate.Names[0];

        for (int i = 0; i < canonical.Length - 1; i++)
        {
            if ((canonical[i] == 'q' || canonical[i] == 'Q') && char.IsDigit(canonical[i + 1]))
                return canonical[i + 1] - '0';
        }

        return 0;
    }

    private async Task<BenchmarkSnapshot?> LoadSnapshotAsync(HybridQuant quant, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (model == null)
            return null;

        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, model.Id, createIfMissing: false, ct);
        var lookup = (TensorConfig)quant;

        var row = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .Join(db.TensorCombos,
                b => b.TensorComboId,
                c => c.Id,
                (b, c) => new { b, c })
            .FirstOrDefaultAsync(x =>
                    x.b.AiModelHashId == model.Id &&
                    x.b.ImatrixDefinitionId == imatrixDefinitionId &&
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

    private sealed class GroupCandidateEvaluation
    {
        public TensorGroup Group { get; set; } = default!;
        public BaselineQuants CandidateBaseline { get; set; } = default!;
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
