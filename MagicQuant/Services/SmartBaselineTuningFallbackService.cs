using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Conservative, non-DuckDB fallback used only after the normal prediction-guided
/// selector fails to validate a candidate for a strict/premium/interior phase.
///
/// The service starts from the real benchmarked pure/uniform baseline anchor, keeps
/// that blanket baseline available for every group even if the baseline was pruned
/// from a group, and only swaps in group candidates that survived isolation pruning.
/// It does not create the normal prediction-space gremlin trades: shrinking is only
/// allowed when the isolated group sample is same-size-or-smaller and measurably lower
/// KLD than the blanket state; higher-fidelity protection is bounded by config and
/// must fit the target real-size window exactly.
/// </summary>
public sealed class SmartBaselineTuningFallbackService
{
    private const double KldEpsilon = 1e-12d;

    private readonly HybridBenchmarkRepository _repository;
    private RankSafeKldPredictionService.RankSafePredictionModel? _context;

    public SmartBaselineTuningFallbackService(HybridBenchmarkRepository repository)
    {
        _repository = repository;
    }

    public async Task<IReadOnlyList<HybridSelectionCandidate>> BuildStrictDominanceCandidatesAsync(
        BenchmarkSnapshotRecord anchor,
        int phaseWindowIndex,
        int phaseWindowCount,
        CancellationToken ct = default)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return Array.Empty<HybridSelectionCandidate>();

        return await BuildCandidatesAsync(new SmartFallbackRequest
        {
            Reason = HybridSelectionReason.SmartStrictDominanceFallback,
            WindowLabel = $"smart strict baseline tuning vs {anchor.DisplayName}",
            BaselineAnchor = anchor,
            HigherDamageAnchor = anchor,
            LowerDamageAnchor = anchor,
            WindowMinSizeBytes = 0,
            WindowMaxSizeBytes = anchor.SizeBytes,
            StrictDominance = true,
            PhaseWindowIndex = phaseWindowIndex,
            PhaseWindowCount = phaseWindowCount
        }, ct);
    }

    public async Task<IReadOnlyList<HybridSelectionCandidate>> BuildNearBaselineCandidatesAsync(
        BenchmarkSnapshotRecord lowerSizeHigherDamage,
        BenchmarkSnapshotRecord upperSizeLowerDamage,
        ulong realMinSizeBytes,
        ulong realMaxSizeBytes,
        int phaseWindowIndex,
        int phaseWindowCount,
        CancellationToken ct = default)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return Array.Empty<HybridSelectionCandidate>();

        return await BuildCandidatesAsync(new SmartFallbackRequest
        {
            Reason = HybridSelectionReason.SmartNearBaselineFallback,
            WindowLabel = $"smart near-baseline tuning {lowerSizeHigherDamage.DisplayName} → {upperSizeLowerDamage.DisplayName}",
            BaselineAnchor = lowerSizeHigherDamage,
            HigherDamageAnchor = lowerSizeHigherDamage,
            LowerDamageAnchor = upperSizeLowerDamage,
            WindowMinSizeBytes = realMinSizeBytes,
            WindowMaxSizeBytes = realMaxSizeBytes,
            StrictDominance = false,
            PhaseWindowIndex = phaseWindowIndex,
            PhaseWindowCount = phaseWindowCount
        }, ct);
    }

    public async Task<IReadOnlyList<HybridSelectionCandidate>> BuildInteriorCandidatesAsync(
        BenchmarkSnapshotRecord lowerSizeHigherDamage,
        BenchmarkSnapshotRecord upperSizeLowerDamage,
        ulong realMinSizeBytes,
        ulong realMaxSizeBytes,
        string windowLabel,
        int phaseWindowIndex,
        int phaseWindowCount,
        CancellationToken ct = default)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return Array.Empty<HybridSelectionCandidate>();

        return await BuildCandidatesAsync(new SmartFallbackRequest
        {
            Reason = HybridSelectionReason.SmartInteriorSubspaceFallback,
            WindowLabel = $"smart interior tuning {windowLabel}",
            BaselineAnchor = lowerSizeHigherDamage,
            HigherDamageAnchor = lowerSizeHigherDamage,
            LowerDamageAnchor = upperSizeLowerDamage,
            WindowMinSizeBytes = realMinSizeBytes,
            WindowMaxSizeBytes = realMaxSizeBytes,
            StrictDominance = false,
            PhaseWindowIndex = phaseWindowIndex,
            PhaseWindowCount = phaseWindowCount
        }, ct);
    }

    private async Task<IReadOnlyList<HybridSelectionCandidate>> BuildCandidatesAsync(
        SmartFallbackRequest request,
        CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();

        if (!TryResolveBaselineBlanket(request.BaselineAnchor, out var blanketBaseline, out var skipReason))
        {
            AnsiConsole.MarkupLine($"[grey]Smart fallback skipped:[/] {Markup.Escape(skipReason)}");
            return Array.Empty<HybridSelectionCandidate>();
        }

        var context = await GetContextAsync(ct);
        var blanketAnchor = ResolveSmartBlanketAnchor(request.BaselineAnchor, blanketBaseline, context);
        ValidateBlanketIsolationCoverage(blanketBaseline, context);

        var baseSize = blanketAnchor.SizeBytes;
        var baseKld = Math.Max(0d, blanketAnchor.Kld);

        if (request.StrictDominance && baseSize > request.WindowMaxSizeBytes)
        {
            AnsiConsole.MarkupLine($"[grey]Smart strict fallback skipped:[/] {Markup.Escape(blanketBaseline.Names[0])} blanket is larger than the strict anchor.");
            return Array.Empty<HybridSelectionCandidate>();
        }

        var options = BuildGroupOptions(blanketBaseline, context)
            .Where(x => request.StrictDominance ? x.SizeDeltaBytes <= 0 : true)
            .ToList();

        if (options.Count == 0)
        {
            AnsiConsole.MarkupLine($"[grey]Smart fallback found no isolated group trades for[/] [cyan]{Markup.Escape(blanketBaseline.Names[0])}[/].");
            return Array.Empty<HybridSelectionCandidate>();
        }

        var plans = BuildPlans(request, blanketBaseline, blanketAnchor, baseSize, baseKld, options, context);
        if (plans.Count == 0)
        {
            AnsiConsole.MarkupLine($"[grey]Smart fallback found no size-safe plans for[/] [cyan]{Markup.Escape(blanketBaseline.Names[0])}[/] in window {Markup.Escape(request.WindowLabel)}.");
            return Array.Empty<HybridSelectionCandidate>();
        }

        var orderedPlans = request.StrictDominance
            ? plans.OrderByDescending(x => x.TotalKldGain).ThenBy(x => x.PredictedSizeBytes).ThenByDescending(x => x.Score).ToList()
            : plans.OrderByDescending(x => x.Score).ThenByDescending(x => x.TotalKldGain).ThenByDescending(x => x.PredictedSizeBytes).ToList();

        int limit = Config.SelectionSmartFallbackAttemptsPerFailure;
        var selected = orderedPlans
            .Take(limit)
            .Select((plan, index) => ToCandidate(request, blanketBaseline, plan, index + 1, options.Count, orderedPlans.Count, orderedPlans.Count(x => x.PredictedGainOverLine > 0d)))
            .ToList();

        AnsiConsole.MarkupLine(
            $"[yellow]Smart baseline fallback staged:[/] [cyan]{selected.Count:N0}[/] candidate(s) for {Markup.Escape(request.WindowLabel)} from [cyan]{Markup.Escape(blanketBaseline.Names[0])}[/] blanket.");

        foreach (var candidate in selected)
        {
            var swaps = string.Join(", ", candidate.CandidateSelectionNotes.Where(x => x.StartsWith("swap ", StringComparison.Ordinal)).Take(4));
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(candidate.Prediction.Quant))}[/] size={ToGiB(candidate.Prediction.PredictedSizeBytes):0.00}GiB additiveKLD={candidate.Prediction.PredictedKld:0.000000} {Markup.Escape(swaps)}");
        }

        return selected;
    }

    private static bool TryResolveBaselineBlanket(
        BenchmarkSnapshotRecord anchor,
        out BaselineQuants baseline,
        out string reason)
    {
        baseline = anchor.Quant.BaseQuant;
        reason = string.Empty;

        if (BaselineQuants.IsNativeExactAlias(baseline.UniqueId))
        {
            reason = $"anchor '{anchor.DisplayName}' uses native/exact precision and cannot be used as a learned baseline blanket.";
            return false;
        }

        if (TensorConfigIdentity.IsPureBaseline(anchor.Config))
            return true;

        foreach (var (group, storedValue) in TensorConfigIdentity.EnumerateGroupSlots(anchor.Config))
        {
            if (Cache.UnusedTensorGroups.Any(x => x.UniqueId == group.UniqueId))
                continue;

            if (BaselineQuants.IsNullTensorConfigGroupSlot(storedValue))
                continue;

            var decoded = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedValue);
            if (decoded != baseline.UniqueId)
            {
                reason = $"anchor '{anchor.DisplayName}' is already a non-uniform hybrid; smart fallback only starts from pure/uniform baseline blankets.";
                return false;
            }
        }

        return true;
    }

    private static BenchmarkSnapshotRecord ResolveSmartBlanketAnchor(
        BenchmarkSnapshotRecord anchor,
        BaselineQuants blanketBaseline,
        RankSafeKldPredictionService.RankSafePredictionModel context)
    {
        if (TensorConfigIdentity.IsPureBaseline(anchor.Config))
        {
            if (!context.PureSnapshotsByBaselineId.TryGetValue(blanketBaseline.UniqueId, out _))
            {
                throw new InvalidOperationException(
                    $"Smart baseline fallback critical truth error: pure benchmark snapshot for '{blanketBaseline.Names[0]}' (id {blanketBaseline.UniqueId}) was not loaded, " +
                    $"but strict/near/interior fallback is trying to tune from anchor '{anchor.DisplayName}'. " +
                    "This is not a soft skip; the original baseline anchor is missing from the prediction context.");
            }

            /*
             * Preserve the exact anchor that triggered the fallback. The dictionary check above is a
             * consistency guard proving the pure baseline truth exists in the loaded context; using
             * request.BaselineAnchor keeps size/KLD aligned with the active strict/near/interior frontier.
             */
            return anchor;
        }

        if (!IsUniformLearnedBlanket(anchor, blanketBaseline))
        {
            throw new InvalidOperationException(
                $"Smart baseline fallback critical truth error: anchor '{anchor.DisplayName}' resolved to blanket '{blanketBaseline.Names[0]}', " +
                "but the anchor is not a pure baseline and not a uniform learned-candidate blanket. This should have been rejected before anchor resolution.");
        }

        return anchor;
    }

    private static bool IsUniformLearnedBlanket(BenchmarkSnapshotRecord anchor, BaselineQuants blanketBaseline)
    {
        if (TensorConfigIdentity.IsPureBaseline(anchor.Config))
            return true;

        foreach (var (group, storedValue) in TensorConfigIdentity.EnumerateGroupSlots(anchor.Config))
        {
            if (Cache.UnusedTensorGroups.Any(x => x.UniqueId == group.UniqueId))
                continue;

            if (BaselineQuants.IsNullTensorConfigGroupSlot(storedValue))
                continue;

            var decoded = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedValue);
            if (decoded != blanketBaseline.UniqueId)
                return false;
        }

        return true;
    }

    private static void ValidateBlanketIsolationCoverage(
        BaselineQuants blanketBaseline,
        RankSafeKldPredictionService.RankSafePredictionModel context)
    {
        var missingGroups = context.ActiveGroups
            .Where(group => !context.IsolationByGroupAndBaseline.ContainsKey((group.UniqueId, blanketBaseline.UniqueId)))
            .Select(group => group.Name)
            .ToList();

        if (missingGroups.Count == 0)
            return;

        throw new InvalidOperationException(
            $"Smart baseline fallback critical truth error: baseline '{blanketBaseline.Names[0]}' (id {blanketBaseline.UniqueId}) is present as a fallback anchor, " +
            $"but isolated group truth is missing for {missingGroups.Count:N0}/{context.ActiveGroups.Count:N0} active group(s): {string.Join(", ", missingGroups)}. " +
            "Smart fallback must not silently skip groups when tuning from a real baseline anchor.");
    }

    private async Task<RankSafeKldPredictionService.RankSafePredictionModel> GetContextAsync(CancellationToken ct)
    {
        if (_context != null)
            return _context;

        var activeGroups = TReg.All
            .Where(x => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        if (activeGroups.Count == 0)
            throw new InvalidOperationException("No active tensor groups were available for smart baseline fallback.");

        var notes = new List<string>();
        var pureSnapshots = await _repository.LoadPureBaselineSnapshotsAsync(ct);
        var pureByBaselineId = pureSnapshots
            .GroupBy(x => x.Quant.BaseQuant.UniqueId)
            .ToDictionary(g => g.Key, g => g.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes).First());

        if (!pureByBaselineId.TryGetValue(BaselineQuants.Q8_0.UniqueId, out var pureQ8))
            throw new InvalidOperationException("Smart baseline fallback requires a pure Q8_0 benchmark snapshot.");

        var nativeExactScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();
        var q8BaseOnlyQuant = HybridQuant.CreateExactBlanket(
            baseQuant: BaselineQuants.Q8_0,
            groups: activeGroups,
            exactScheme: nativeExactScheme);

        var q8BaseOnly = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)q8BaseOnlyQuant, ct)
            ?? throw new InvalidOperationException("Smart baseline fallback requires the Q8_0 native-exact base-only anchor.");

        var baseOnlyByBaselineId = new Dictionary<byte, BenchmarkSnapshotRecord>
        {
            [BaselineQuants.Q8_0.UniqueId] = q8BaseOnly
        };

        foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines()
                     .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
                     .OrderBy(x => x.UniqueId))
        {
            if (baseOnlyByBaselineId.ContainsKey(baseline.UniqueId))
                continue;

            var directBaseOnlyQuant = HybridQuant.CreateExactBlanket(
                baseQuant: baseline,
                groups: activeGroups,
                exactScheme: nativeExactScheme);

            var directBaseOnlySnapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)directBaseOnlyQuant, ct);
            if (directBaseOnlySnapshot != null)
                baseOnlyByBaselineId[baseline.UniqueId] = directBaseOnlySnapshot;
        }

        var isolationByGroupAndBaseline = new Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord>();

        foreach (var group in activeGroups)
        {
            foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines()
                         .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
                         .OrderBy(x => x.UniqueId))
            {
                if (isolationByGroupAndBaseline.ContainsKey((group.UniqueId, baseline.UniqueId)))
                    continue;

                var isolationQuant = HybridQuant.CreateExactBlanket(
                    baseQuant: BaselineQuants.Q8_0,
                    groups: activeGroups,
                    exactScheme: nativeExactScheme);

                isolationQuant.SetLearnedCandidateOverride(group, baseline);
                var snapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)isolationQuant, ct);
                if (snapshot != null)
                    isolationByGroupAndBaseline[(group.UniqueId, baseline.UniqueId)] = snapshot;
            }
        }

        notes.Add($"Smart fallback isolation context loaded: activeGroups={activeGroups.Count:N0}, baseOnlyAnchors={baseOnlyByBaselineId.Count:N0}, groupIsolations={isolationByGroupAndBaseline.Count:N0}.");

        _context = new RankSafeKldPredictionService.RankSafePredictionModel(
            activeGroups: activeGroups,
            pureQ8: pureQ8,
            q8BaseOnly: q8BaseOnly,
            pureSnapshotsByBaselineId: pureByBaselineId,
            baseOnlySnapshotsByBaselineId: baseOnlyByBaselineId,
            isolationByGroupAndBaseline: isolationByGroupAndBaseline,
            isolationDominanceBitTruthByGroupAndBaseline: new Dictionary<(byte GroupId, byte BaselineId), double>(),
            notes: notes);

        return _context;
    }

    private static IReadOnlyList<SmartGroupOption> BuildGroupOptions(
        BaselineQuants blanketBaseline,
        RankSafeKldPredictionService.RankSafePredictionModel context)
    {
        var result = new List<SmartGroupOption>();

        foreach (var group in context.ActiveGroups)
        {
            if (!context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, blanketBaseline.UniqueId), out var baseIsolation))
                continue;

            long baseContributionBytes = (long)baseIsolation.SizeBytes - (long)context.Q8BaseOnly.SizeBytes;

            var allowed = RuntimeSearchSpace.GetAllowedRealExplicitCombinationCandidatesForGroup(group)
                .Where(x => x.UniqueId != blanketBaseline.UniqueId)
                .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
                .OrderBy(x => x.ExplicitCandidateSortOrder)
                .ThenBy(x => x.UniqueId)
                .ToList();

            foreach (var candidate in allowed)
            {
                if (!context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, candidate.UniqueId), out var candidateIsolation))
                    continue;

                double kldGain = baseIsolation.Kld - candidateIsolation.Kld;
                if (kldGain <= Config.SelectionMinimumKldImprovementEpsilon + KldEpsilon)
                    continue;

                long candidateContributionBytes = (long)candidateIsolation.SizeBytes - (long)context.Q8BaseOnly.SizeBytes;
                long sizeDeltaBytes = candidateContributionBytes - baseContributionBytes;

                int fidelitySteps = CountHigherFidelitySteps(blanketBaseline, candidate);
                if (fidelitySteps > Config.SelectionSmartFallbackMaxHigherFidelitySteps)
                    continue;

                bool isShrink = sizeDeltaBytes < 0;
                bool isFreeOrBetter = sizeDeltaBytes <= 0;

                if (isShrink && candidateIsolation.Kld + Config.SelectionMinimumKldImprovementEpsilon >= baseIsolation.Kld)
                    continue;

                double damageAvoidedPerMiB = kldGain / Math.Max(1d, Math.Abs(sizeDeltaBytes) / 1024d / 1024d);
                double sensitivityScore = kldGain * Math.Log2(2d + Math.Max(0d, baseIsolation.Kld) / Math.Max(candidateIsolation.Kld, 1e-12d));
                double score = sensitivityScore + damageAvoidedPerMiB;
                if (isFreeOrBetter)
                    score += kldGain * 1000d;

                result.Add(new SmartGroupOption
                {
                    Group = group,
                    CandidateBaseline = candidate,
                    BaseIsolation = baseIsolation,
                    CandidateIsolation = candidateIsolation,
                    SizeDeltaBytes = sizeDeltaBytes,
                    KldGain = kldGain,
                    Score = score,
                    HigherFidelitySteps = fidelitySteps
                });
            }
        }

        return result;
    }

    private static List<SmartCandidatePlan> BuildPlans(
        SmartFallbackRequest request,
        BaselineQuants blanketBaseline,
        BenchmarkSnapshotRecord blanketAnchor,
        ulong baseSize,
        double baseKld,
        IReadOnlyList<SmartGroupOption> options,
        RankSafeKldPredictionService.RankSafePredictionModel context)
    {
        var plans = new List<SmartCandidatePlan>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        void TryAddPlan(IEnumerable<SmartGroupOption> selected, string strategy)
        {
            var chosen = selected
                .GroupBy(x => x.Group.UniqueId)
                .Select(g => g.OrderByDescending(x => x.Score).ThenBy(x => x.SizeDeltaBytes).First())
                .OrderBy(x => x.Group.UniqueId)
                .ToList();

            if (chosen.Count == 0)
                return;

            var quant = HybridQuant.CreateLearnedCandidateBlanket(
                baseQuant: blanketBaseline,
                groups: context.ActiveGroups,
                candidateBaseline: blanketBaseline);

            foreach (var option in chosen)
                quant.SetLearnedCandidateOverride(option.Group, option.CandidateBaseline);

            var config = (TensorConfig)quant;
            string key = TensorConfigIdentity.ToKey(config);
            if (!seen.Add(key))
                return;

            if (!TryPredictSize(config, blanketBaseline, blanketAnchor, context, out var predictedSize, out _) ||
                !TryComputeAdditiveKld(config, blanketBaseline, blanketAnchor, context, out var predictedKld, out _))
                return;

            if (request.StrictDominance)
            {
                if (predictedSize > request.WindowMaxSizeBytes)
                    return;
            }
            else if (predictedSize < request.WindowMinSizeBytes || predictedSize > request.WindowMaxSizeBytes)
            {
                return;
            }

            double expectedLine = request.StrictDominance
                ? request.HigherDamageAnchor.Kld
                : InterpolateKldLine(predictedSize, request.HigherDamageAnchor, request.LowerDamageAnchor);
            double gainOverLine = expectedLine - predictedKld;
            double totalKldGain = baseKld - predictedKld;
            long totalSizeDelta = (long)predictedSize - (long)baseSize;

            if (request.StrictDominance && totalKldGain <= Config.SelectionMinimumKldImprovementEpsilon)
                return;

            double score = chosen.Sum(x => x.Score)
                           + Math.Max(0d, totalKldGain) * 100d
                           + Math.Max(0d, gainOverLine) * 25d;

            if (!request.StrictDominance)
            {
                // Premium/interior fallback is allowed to gamble, but it should still
                // prefer plans that protect the most isolated damage per byte spent.
                double budgetUsedFraction = request.WindowMaxSizeBytes <= request.WindowMinSizeBytes
                    ? 0d
                    : ((double)predictedSize - request.WindowMinSizeBytes) / Math.Max(1d, request.WindowMaxSizeBytes - request.WindowMinSizeBytes);
                score += Math.Clamp(budgetUsedFraction, 0d, 1d) * Math.Max(0d, totalKldGain) * 50d;
            }

            plans.Add(new SmartCandidatePlan
            {
                Quant = quant,
                Config = config,
                Strategy = strategy,
                Options = chosen,
                PredictedSizeBytes = predictedSize,
                PredictedAdditiveKld = predictedKld,
                LinearExpectedKld = expectedLine,
                PredictedGainOverLine = gainOverLine,
                TotalKldGain = totalKldGain,
                TotalSizeDeltaBytes = totalSizeDelta,
                Score = score
            });
        }

        var freeLunch = options
            .Where(x => x.SizeDeltaBytes <= 0)
            .GroupBy(x => x.Group.UniqueId)
            .Select(g => g.OrderByDescending(x => x.KldGain).ThenBy(x => x.SizeDeltaBytes).First())
            .ToList();

        TryAddPlan(freeLunch, "free-lunch-same-or-smaller");

        foreach (var single in options.OrderByDescending(x => x.Score).ThenBy(x => x.SizeDeltaBytes).Take(Math.Max(12, Config.SelectionSmartFallbackAttemptsPerFailure * 4)))
            TryAddPlan(new[] { single }, "single-sensitive-group");

        if (!request.StrictDominance)
        {
            var protectedSet = new List<SmartGroupOption>();
            protectedSet.AddRange(freeLunch);

            foreach (var option in options
                         .Where(x => x.SizeDeltaBytes > 0)
                         .OrderByDescending(x => x.Score)
                         .ThenBy(x => x.SizeDeltaBytes))
            {
                var trial = protectedSet
                    .Where(x => x.Group.UniqueId != option.Group.UniqueId)
                    .Concat(new[] { option })
                    .ToList();

                var trialQuant = HybridQuant.CreateLearnedCandidateBlanket(
                    baseQuant: blanketBaseline,
                    groups: context.ActiveGroups,
                    candidateBaseline: blanketBaseline);
                foreach (var selected in trial)
                    trialQuant.SetLearnedCandidateOverride(selected.Group, selected.CandidateBaseline);

                if (!TryPredictSize((TensorConfig)trialQuant, blanketBaseline, blanketAnchor, context, out var trialSize, out _))
                    continue;

                if (trialSize <= request.WindowMaxSizeBytes)
                    protectedSet = trial;
            }

            TryAddPlan(protectedSet, "balanced-brain-protection");

            foreach (var groupedBySensitivity in options
                         .Where(x => x.SizeDeltaBytes >= 0)
                         .OrderByDescending(x => x.KldGain)
                         .ThenBy(x => x.SizeDeltaBytes)
                         .Take(Math.Max(8, Config.SelectionSmartFallbackAttemptsPerFailure * 3)))
            {
                var blend = freeLunch
                    .Where(x => x.Group.UniqueId != groupedBySensitivity.Group.UniqueId)
                    .Concat(new[] { groupedBySensitivity });
                TryAddPlan(blend, "sensitivity-first-blend");
            }
        }

        return plans;
    }

    private static HybridSelectionCandidate ToCandidate(
        SmartFallbackRequest request,
        BaselineQuants blanketBaseline,
        SmartCandidatePlan plan,
        int attemptOrder,
        int optionCount,
        int planCount,
        int lineBeatingCount)
    {
        var notes = new List<string>
        {
            "smartFallback=sqlite-isolation-truth; not selected from DuckDB prediction rows",
            $"blanket={blanketBaseline.Names[0]}",
            $"strategy={plan.Strategy}",
            $"exactSizeWindow={ToGiB(request.WindowMinSizeBytes):0.00}..{ToGiB(request.WindowMaxSizeBytes):0.00}GiB",
            $"smartAttemptsLimit={Config.SelectionSmartFallbackAttemptsPerFailure}",
            $"maxHigherFidelitySteps={Config.SelectionSmartFallbackMaxHigherFidelitySteps}"
        };

        notes.AddRange(plan.Options.Select(option =>
            $"swap {option.Group.Name}: {blanketBaseline.Names[0]} -> {option.CandidateBaseline.Names[0]} " +
            $"sizeDelta={option.SizeDeltaBytes:N0}B isolatedKldGain={option.KldGain:0.000000}"));

        var row = new RankSafePredictionRow
        {
            Config = plan.Config,
            Quant = plan.Quant,
            PredictedSizeBytes = plan.PredictedSizeBytes,
            IsSizePredictable = true,
            AdditiveKld = plan.PredictedAdditiveKld,
            InteractionKld = plan.PredictedAdditiveKld,
            PredictedKld = plan.PredictedAdditiveKld,
            PredictionConfidence = 0.50d,
            PredictedPpl = 0d,
            CrossTerm = 0d,
            IsPureBaseline = false,
            IsPredictable = true,
            HasUnknownMappings = false,
            EffectiveStateKey = $"smart-fallback:{TensorConfigIdentity.ToKey(plan.Config)}",
            PredictedRank = null,
            Notes = notes
        };

        return new HybridSelectionCandidate
        {
            Prediction = row,
            Reason = request.Reason,
            LowerDamageAnchor = request.LowerDamageAnchor,
            HigherDamageAnchor = request.HigherDamageAnchor,
            WindowMinSizeBytes = request.WindowMinSizeBytes,
            WindowMaxSizeBytes = request.WindowMaxSizeBytes,
            PredictionWindowMinSizeBytes = request.WindowMinSizeBytes,
            PredictionWindowMaxSizeBytes = request.WindowMaxSizeBytes,
            LinearExpectedKld = plan.LinearExpectedKld,
            PredictedGainOverLine = plan.PredictedGainOverLine,
            AttemptOrder = attemptOrder,
            WindowLabel = request.WindowLabel,
            CandidatePoolSize = optionCount,
            WindowCandidateCount = planCount,
            LineBeatingCandidateCount = lineBeatingCount,
            FetchedCandidateCount = planCount,
            CandidatesAfterBrutalityCount = planCount,
            CandidateAttemptLimit = Config.SelectionSmartFallbackAttemptsPerFailure,
            PhaseWindowIndex = request.PhaseWindowIndex,
            PhaseWindowCount = request.PhaseWindowCount,
            RawSelectionRank = attemptOrder,
            CandidateTheoryFamilyKey = "smart-baseline-tuning",
            CandidateTheoryFamilyRank = 1,
            CandidateTheoryFamilyMemberRank = attemptOrder,
            CandidateTheoryFamilyDisplay = "smart baseline tuning",
            DiversityMode = "sqlite-isolation-fallback",
            CandidateSelectionNotes = notes
        };
    }

    private static bool TryPredictSize(
        TensorConfig config,
        BaselineQuants blanketBaseline,
        BenchmarkSnapshotRecord blanketAnchor,
        RankSafeKldPredictionService.RankSafePredictionModel context,
        out ulong sizeBytes,
        out IReadOnlyList<string> notes)
    {
        var localNotes = new List<string>();
        notes = localNotes;
        sizeBytes = 0;

        long total = (long)blanketAnchor.SizeBytes;

        foreach (var (group, effectiveBaselineId) in RankSafeKldPredictionService.EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (effectiveBaselineId == blanketBaseline.UniqueId)
                continue;

            if (!TryResolveIsolationSnapshot(group, blanketBaseline.UniqueId, context, localNotes, "baseline size", out var baseSnapshot) ||
                !TryResolveIsolationSnapshot(group, effectiveBaselineId, context, localNotes, "candidate size", out var candidateSnapshot))
            {
                return false;
            }

            total += (long)candidateSnapshot.SizeBytes - (long)baseSnapshot.SizeBytes;
        }

        if (total <= 0)
        {
            localNotes.Add($"Predicted size collapsed to {total:N0} bytes.");
            return false;
        }

        sizeBytes = (ulong)total;
        return true;
    }

    private static bool TryComputeAdditiveKld(
        TensorConfig config,
        BaselineQuants blanketBaseline,
        BenchmarkSnapshotRecord blanketAnchor,
        RankSafeKldPredictionService.RankSafePredictionModel context,
        out double additiveKld,
        out IReadOnlyList<string> notes)
    {
        var localNotes = new List<string>();
        notes = localNotes;
        additiveKld = Math.Max(0d, blanketAnchor.Kld);

        foreach (var (group, effectiveBaselineId) in RankSafeKldPredictionService.EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (effectiveBaselineId == blanketBaseline.UniqueId)
                continue;

            if (!TryResolveIsolationSnapshot(group, blanketBaseline.UniqueId, context, localNotes, "baseline KLD", out var baseSnapshot) ||
                !TryResolveIsolationSnapshot(group, effectiveBaselineId, context, localNotes, "candidate KLD", out var candidateSnapshot))
            {
                return false;
            }

            additiveKld += Math.Max(0d, candidateSnapshot.Kld) - Math.Max(0d, baseSnapshot.Kld);
        }

        additiveKld = Math.Max(0d, additiveKld);
        return true;
    }

    private static bool TryResolveIsolationSnapshot(
        TensorGroup group,
        byte baselineId,
        RankSafeKldPredictionService.RankSafePredictionModel context,
        List<string> notes,
        string role,
        out BenchmarkSnapshotRecord snapshot)
    {
        if (BaselineQuants.IsNativeExactAlias(baselineId))
        {
            snapshot = context.Q8BaseOnly;
            return true;
        }

        if (context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, baselineId), out var foundSnapshot))
        {
            snapshot = foundSnapshot;
            return true;
        }

        var name = BaselineQuants.FromId(baselineId).Names[0];
        var message = $"Missing {role} isolation anchor for group '{group.Name}' and baseline '{name}' (id {baselineId}).";
        notes.Add(message);

        if (BaselineQuants.FromId(baselineId).IsExternalRepositoryBaseline)
        {
            throw new InvalidOperationException(
                $"Smart baseline fallback critical truth error: {message} " +
                "External/custom fallback must use exact isolated truth and must not silently collapse or skip.");
        }

        snapshot = default!;
        return false;
    }

    private static int CountHigherFidelitySteps(BaselineQuants baseBaseline, BaselineQuants candidate)
    {
        if (candidate.BitRange <= baseBaseline.BitRange)
            return 0;

        var ladder = BaselineQuants.GetAllRecognizedBaselines()
            .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
            .GroupBy(x => x.BitRange)
            .Select(g => g.Key)
            .OrderBy(x => x)
            .ToList();

        int baseIndex = ladder.IndexOf(baseBaseline.BitRange);
        int candidateIndex = ladder.IndexOf(candidate.BitRange);
        if (baseIndex < 0 || candidateIndex < 0)
            return candidate.BitRange > baseBaseline.BitRange ? 1 : 0;

        return Math.Max(0, candidateIndex - baseIndex);
    }

    private static double InterpolateKldLine(
        ulong sizeBytes,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger)
    {
        if (lowerDamageLarger.SizeBytes <= higherDamageSmaller.SizeBytes)
            return Math.Min(higherDamageSmaller.Kld, lowerDamageLarger.Kld);

        double t = ((double)sizeBytes - higherDamageSmaller.SizeBytes) /
                   (lowerDamageLarger.SizeBytes - higherDamageSmaller.SizeBytes);
        t = Math.Clamp(t, 0d, 1d);
        return higherDamageSmaller.Kld + (lowerDamageLarger.Kld - higherDamageSmaller.Kld) * t;
    }

    private static double ToGiB(ulong bytes) => bytes / 1024d / 1024d / 1024d;

    private sealed class SmartFallbackRequest
    {
        public HybridSelectionReason Reason { get; init; }
        public string WindowLabel { get; init; } = string.Empty;
        public BenchmarkSnapshotRecord BaselineAnchor { get; init; } = default!;
        public BenchmarkSnapshotRecord HigherDamageAnchor { get; init; } = default!;
        public BenchmarkSnapshotRecord LowerDamageAnchor { get; init; } = default!;
        public ulong WindowMinSizeBytes { get; init; }
        public ulong WindowMaxSizeBytes { get; init; }
        public bool StrictDominance { get; init; }
        public int PhaseWindowIndex { get; init; }
        public int PhaseWindowCount { get; init; }
    }

    private sealed class SmartGroupOption
    {
        public TensorGroup Group { get; init; } = default!;
        public BaselineQuants CandidateBaseline { get; init; } = default!;
        public BenchmarkSnapshotRecord BaseIsolation { get; init; } = default!;
        public BenchmarkSnapshotRecord CandidateIsolation { get; init; } = default!;
        public long SizeDeltaBytes { get; init; }
        public double KldGain { get; init; }
        public double Score { get; init; }
        public int HigherFidelitySteps { get; init; }
    }

    private sealed class SmartCandidatePlan
    {
        public HybridQuant Quant { get; init; } = default!;
        public TensorConfig Config { get; init; }
        public string Strategy { get; init; } = string.Empty;
        public IReadOnlyList<SmartGroupOption> Options { get; init; } = Array.Empty<SmartGroupOption>();
        public ulong PredictedSizeBytes { get; init; }
        public double PredictedAdditiveKld { get; init; }
        public double LinearExpectedKld { get; init; }
        public double PredictedGainOverLine { get; init; }
        public double TotalKldGain { get; init; }
        public long TotalSizeDeltaBytes { get; init; }
        public double Score { get; init; }
    }
}