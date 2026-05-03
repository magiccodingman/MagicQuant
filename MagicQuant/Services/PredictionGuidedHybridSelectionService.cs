using System.Text.Json;
using MagicQuant.Models;
using MagicQuant.Services.Progress;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models.DbModels;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Final hybrid chooser driven by the rank-safe isolation prediction engine.
///
/// This service intentionally does not brute-force the whole remaining DuckDB space.
/// It only validates candidates whose predicted outcome proves one of the user-defined
/// survival claims:
/// 1. strict dominance over a pure/current anchor: lower KLD at same-or-smaller size
/// 2. near-baseline replacement: <= configured small size premium and better-than-linear KLD
/// 3. interior subspace discovery: better-than-linear KLD inside configurable size windows
/// </summary>
public sealed class PredictionGuidedHybridSelectionService
{
    private const int DiagnosticPreviewLimit = 25;
    private const int DiagnosticPreviewDisplayCount = 8;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly QuantizationService _quantizationService;
    private readonly HybridBenchmarkRepository _repository;
    private readonly FinalRealBenchmarkEliminationService _finalEliminator;
    private readonly RemainingCombinationStore _predictedStore;

    public PredictionGuidedHybridSelectionService(
        QuantizationService quantizationService,
        HybridBenchmarkRepository repository,
        FinalRealBenchmarkEliminationService finalEliminator,
        RemainingCombinationStore predictedStore)
    {
        _quantizationService = quantizationService;
        _repository = repository;
        _finalEliminator = finalEliminator;
        _predictedStore = predictedStore;
    }

    public async Task<PredictionGuidedSelectionResult> RunAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        CancellationToken ct = default)
    {
        var eliminationRecords = new List<BaselineEliminationRecord>();
        var validationFailures = new List<CandidateValidationResult>();
        var validationAttempts = new List<CandidateValidationResult>();
        var phaseDiagnostics = new List<SelectionPhaseDiagnostic>();

        var current = _finalEliminator.Eliminate(pureBaselineSnapshots).Survivors.ToList();
        AnsiConsole.MarkupLine($"[green]Pure/current anchor survivors after dominance:[/] [cyan]{current.Count:N0}[/]");
        PrintAnchorFrontier(current, "Initial anchor frontier after dominance");

        var strict = await RunStrictDominanceReplacementAsync(current, eliminationRecords, validationFailures, validationAttempts, phaseDiagnostics, ct);
        current = MergeAndDominanceFilter(current, strict.AcceptedSnapshots, eliminationRecords, "strict predicted hybrid dominance validated by real benchmark");

        var near = await RunNearBaselineReplacementAsync(current, eliminationRecords, validationFailures, validationAttempts, phaseDiagnostics, ct);
        current = MergeAndDominanceFilter(current, near.AcceptedSnapshots, eliminationRecords, "near-baseline size-premium replacement validated by real benchmark");

        var interior = await RunInteriorSubspaceDiscoveryAsync(current, validationFailures, validationAttempts, phaseDiagnostics, ct);
        current = MergeAndDominanceFilter(current, interior.AcceptedSnapshots, eliminationRecords, "interior subspace discovery dominated by real benchmark truth");

        var bestConfirmedAnomaly = await LoadBestConfirmedBeneficialAnomalySnapshotAsync(ct);
        if (bestConfirmedAnomaly != null && current.All(x => TensorConfigIdentity.ToKey(x.Config) != TensorConfigIdentity.ToKey(bestConfirmedAnomaly.Config)))
        {
            AnsiConsole.MarkupLine($"[yellow]Best confirmed anomaly reconciliation:[/] adding probe-confirmed anomaly to final frontier consideration: [cyan]{Markup.Escape(bestConfirmedAnomaly.DisplayName)}[/]");
            current = MergeAndDominanceFilter(current, new[] { bestConfirmedAnomaly }, eliminationRecords, "best confirmed beneficial anomaly included for final reconciliation");
        }

        current = ApplyMeaningfulSpacing(current, eliminationRecords);

        var finalDominance = _finalEliminator.Eliminate(current);
        foreach (var eliminated in finalDominance.Eliminated)
        {
            var eliminator = finalDominance.Survivors
                .FirstOrDefault(x => Dominates(x, eliminated));

            if (eliminator != null)
            {
                eliminationRecords.Add(new BaselineEliminationRecord
                {
                    Eliminated = eliminated,
                    Eliminator = eliminator,
                    Reason = "final dominance pass"
                });
            }
        }

        await WriteAnomalySelectionReconciliationAsync(finalDominance.Survivors, bestConfirmedAnomaly, ct);
        await WriteSelectionPhaseDiagnosticsAsync(phaseDiagnostics, validationFailures, validationAttempts, ct);

        return new PredictionGuidedSelectionResult
        {
            Survivors = finalDominance.Survivors.ToList(),
            Eliminations = eliminationRecords
                .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{NormalizePublicEliminationReason(x.Reason)}")
                .ToList(),
            ValidationFailures = validationFailures
        };
    }

    private async Task<PhaseValidationResult> RunStrictDominanceReplacementAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        List<SelectionPhaseDiagnostic> phaseDiagnostics,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 1: Strict Hybrid Dominance[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[grey]Strict dominance retry policy:[/] max attempts per anchor=[cyan]{Config.SelectionMaxFallbackAttemptsPerAnchor:N0}[/], epsilon=[cyan]{Config.SelectionMinimumKldImprovementEpsilon:0.########}[/]");

        var accepted = new List<BenchmarkSnapshotRecord>();

        foreach (var anchor in currentAnchors.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes))
        {
            if (ShouldSkipAnchorReplacement(anchor))
            {
                AnsiConsole.MarkupLine($"[grey]Skipping 8-bit anchor replacement attempts:[/] {Markup.Escape(anchor.DisplayName)}");
                phaseDiagnostics.Add(new SelectionPhaseDiagnostic
                {
                    Phase = "StrictDominanceReplacement",
                    WindowLabel = $"strict <= {anchor.DisplayName}",
                    HigherDamageSmaller = ToAnchorLog(anchor),
                    LowerDamageLarger = ToAnchorLog(anchor),
                    WindowMinSizeBytes = 0,
                    WindowMaxSizeBytes = anchor.SizeBytes,
                    CandidateAttemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor,
                    Notes = ["Skipped because SelectionAllowEightBitAnchorReplacements=false and anchor is an 8-bit/non-exact anchor."]
                });
                continue;
            }

            long poolCount = await _predictedStore.CountStrictDominanceCandidatesAsync(anchor, ct);
            var strictRows = await _predictedStore.QueryStrictDominanceCandidatesAsync(anchor, Config.SelectionMaxFallbackAttemptsPerAnchor, ct);
            var candidates = strictRows.Select((x, i) => new HybridSelectionCandidate
            {
                Prediction = x,
                Reason = HybridSelectionReason.StrictDominanceReplacement,
                LowerDamageAnchor = anchor,
                HigherDamageAnchor = anchor,
                WindowMinSizeBytes = 0,
                WindowMaxSizeBytes = anchor.SizeBytes,
                LinearExpectedKld = anchor.Kld,
                PredictedGainOverLine = anchor.Kld - x.PredictedKld,
                AttemptOrder = i + 1,
                WindowLabel = $"strict <= {anchor.DisplayName}",
                CandidatePoolSize = poolCount,
                WindowCandidateCount = poolCount,
                LineBeatingCandidateCount = poolCount,
                FetchedCandidateCount = strictRows.Count,
                CandidatesAfterBrutalityCount = strictRows.Count,
                CandidateAttemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor,
                PhaseWindowIndex = 1,
                PhaseWindowCount = 1,
                CandidateSelectionNotes = ["Strict query requires predicted size <= anchor size and predicted KLD + epsilon < anchor KLD."]
            }).ToList();

            var strictNotes = new List<string> { "Strict query requires predicted size <= anchor size and predicted KLD + epsilon < anchor KLD." };
            bool anomalyStrictMode = IsQ8Anchor(anchor) || candidates.Any(x => Math.Abs(x.Prediction.AnomalyAdjustmentKld) > 1e-12);
            if (anomalyStrictMode)
                strictNotes.Add("Q8/anomaly strict mode: validate all fetched candidates up to the configured attempt limit before choosing by actual KLD/size truth.");

            var diag = new SelectionPhaseDiagnostic
            {
                Phase = "StrictDominanceReplacement",
                WindowLabel = $"strict <= {anchor.DisplayName}",
                HigherDamageSmaller = ToAnchorLog(anchor),
                LowerDamageLarger = ToAnchorLog(anchor),
                WindowMinSizeBytes = 0,
                WindowMaxSizeBytes = anchor.SizeBytes,
                CandidatePoolSize = poolCount,
                WindowCandidateCount = poolCount,
                LineBeatingCandidateCount = poolCount,
                FetchedCandidateCount = strictRows.Count,
                CandidatesAfterBrutalityCount = strictRows.Count,
                SelectedForValidationCount = candidates.Count,
                CandidateAttemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor,
                TopCandidates = candidates.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                Notes = strictNotes
            };
            phaseDiagnostics.Add(diag);

            AnsiConsole.MarkupLine($"[grey]Strict candidates for {Markup.Escape(anchor.DisplayName)}:[/] pool={poolCount:N0}, selected={candidates.Count:N0}/{Config.SelectionMaxFallbackAttemptsPerAnchor:N0}, q8/anomaly-mode={anomalyStrictMode}");

            if (candidates.Count == 0)
                continue;

            var acceptedForAnchor = new List<CandidateValidationResult>();
            foreach (var candidate in candidates)
            {
                var validation = await BuildAndValidateSingleAsync(
                    candidate,
                    snapshot => snapshot.SizeBytes <= anchor.SizeBytes &&
                                snapshot.Kld + Config.SelectionMinimumKldImprovementEpsilon < anchor.Kld,
                    $"must be <= {anchor.SizeBytes:N0} bytes and lower KLD than {anchor.DisplayName}",
                    ct);

                validationAttempts.Add(validation);

                if (validation.Accepted && validation.Snapshot != null)
                {
                    acceptedForAnchor.Add(validation);
                    if (!anomalyStrictMode)
                        break;

                    continue;
                }

                validationFailures.Add(validation);
            }

            if (acceptedForAnchor.Count == 0)
            {
                AnsiConsole.MarkupLine($"[grey]No strict predicted replacement validated for anchor:[/] {Markup.Escape(anchor.DisplayName)}");
                continue;
            }

            var chosen = ChooseBestStrictDominanceCandidate(anchor, acceptedForAnchor);
            accepted.Add(chosen.Snapshot!);
            eliminations.Add(new BaselineEliminationRecord
            {
                Eliminated = anchor,
                Eliminator = chosen.Snapshot!,
                Reason = "strict hybrid dominance: best accepted actual KLD at same-or-smaller real size"
            });

            var nonChosen = acceptedForAnchor
                .Where(x => !ReferenceEquals(x, chosen))
                .Select(x => new
                {
                    candidate = x.Snapshot!.DisplayName,
                    actualKld = x.Snapshot.Kld,
                    actualSizeBytes = x.Snapshot.SizeBytes,
                    reasonLost = ExplainStrictAcceptedLoss(anchor, chosen.Snapshot!, x.Snapshot)
                })
                .ToList();

            strictNotes.Add($"validated candidates={validationAttempts.Count(v => v.Candidate.WindowLabel == $"strict <= {anchor.DisplayName}")}; accepted candidates={acceptedForAnchor.Count}; chosen={chosen.Snapshot!.DisplayName}");
            foreach (var loss in nonChosen)
                strictNotes.Add($"accepted-but-not-chosen: {loss.candidate} lost because {loss.reasonLost}");

            AnsiConsole.MarkupLine("[green]Best strict dominance candidate selected:[/]");
            AnsiConsole.MarkupLine($"[grey]  anchor=[/] [cyan]{Markup.Escape(anchor.DisplayName)}[/]");
            AnsiConsole.MarkupLine($"[grey]  chosen=[/] [cyan]{Markup.Escape(chosen.Snapshot!.DisplayName)}[/]");
            AnsiConsole.MarkupLine($"[grey]  actualKld=[/] [cyan]{chosen.Snapshot.Kld:0.000000}[/]");
            AnsiConsole.MarkupLine($"[grey]  actualSizeBytes=[/] [cyan]{chosen.Snapshot.SizeBytes:N0}[/]");
            AnsiConsole.MarkupLine($"[grey]  gainVsAnchor=[/] [cyan]{anchor.Kld - chosen.Snapshot.Kld:0.000000}[/]");
            AnsiConsole.MarkupLine($"[grey]  acceptedCandidateCount=[/] [cyan]{acceptedForAnchor.Count:N0}[/]");
            AnsiConsole.MarkupLine($"[grey]  reason=[/] [cyan]{Markup.Escape(ResolveStrictChosenReason(anchor, chosen.Snapshot!))}[/]");
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }



    private async Task<BenchmarkSnapshotRecord?> LoadBestConfirmedBeneficialAnomalySnapshotAsync(CancellationToken ct)
    {
        if (!Config.AnomalyDetection.Enabled)
            return null;

        await using var db = new MagicQuantContext();
        var modelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (modelHashId == null)
            return null;

        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();
        int? imatrixId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, modelHashId.Value, createIfMissing: false, ct);

        // SQLite cannot translate ulong ordering expressions. Keep the database query
        // to filtering/include only, then rank the tiny scoped anomaly observation set
        // in LINQ-to-Objects. This preserves the intended ordering without tripping
        // Microsoft.Data.Sqlite on SizeSavingsBytes.
        var observations = await db.AnomalyProbeObservations
            .AsNoTracking()
            .Include(x => x.ProbeTensorCombo)
            .Where(x => x.ArchitectureFamilyId == architectureFamilyId)
            .Where(x => x.TensorGroupProfileId == tensorGroupProfileId)
            .Where(x => x.AiModelHashId == modelHashId.Value)
            .Where(x => x.ImatrixDefinitionId == imatrixId)
            .Where(x => x.BenchmarkCategory == (byte)BenchmarkCategory.General)
            .Where(x => x.RuleDirection == AnomalyRuleDirection.Beneficial.ToString())
            .Where(x => x.Accepted)
            .Where(x => x.IsContextualAnomalyProbe && !x.OldBf16Isolation && x.AllActiveGroupsExplicit)
            .Where(x => x.ProbeTensorCombo != null)
            .ToListAsync(ct);

        var observation = observations
            .OrderByDescending(x => x.ActualGainVsTwin)
            .ThenBy(x => x.ActualKld)
            .ThenByDescending(x => x.SizeSavingsBytes)
            .FirstOrDefault();

        if (observation?.ProbeTensorCombo == null)
            return null;

        var combo = observation.ProbeTensorCombo;
        var config = new TensorConfig(combo.BaseQuant, combo.Embeddings, combo.LmHead, combo.AttnQ, combo.AttnKV, combo.AttnOutput, combo.FfnUpGate, combo.FfnDown, combo.MoeExperts, combo.MoeRouter);
        return await _repository.LoadBenchmarkSnapshotAsync(config, ct);
    }

    private async Task WriteAnomalySelectionReconciliationAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> survivors,
        BenchmarkSnapshotRecord? bestAnomaly,
        CancellationToken ct)
    {
        if (!Config.AnomalyDetection.Enabled)
            return;

        object payload;
        if (bestAnomaly == null)
        {
            payload = new
            {
                generatedAtUtc = DateTime.UtcNow,
                anomalyModeEnabled = true,
                bestConfirmedAnomaly = (object?)null,
                selectedAnomalyDerivedSurvivor = (object?)null,
                bestAnomalyWasSelected = false,
                reasonNotSelected = "no confirmed beneficial anomaly observation was available"
            };
        }
        else
        {
            string bestKey = TensorConfigIdentity.ToKey(bestAnomaly.Config);
            bool selected = survivors.Any(x => TensorConfigIdentity.ToKey(x.Config) == bestKey);
            string reason = selected ? string.Empty : ExplainBestAnomalyNotSelected(bestAnomaly, survivors);
            payload = new
            {
                generatedAtUtc = DateTime.UtcNow,
                anomalyModeEnabled = true,
                bestConfirmedAnomaly = ToAnchorLog(bestAnomaly),
                selectedAnomalyDerivedSurvivor = selected ? ToAnchorLog(bestAnomaly) : null,
                bestAnomalyWasSelected = selected,
                reasonNotSelected = reason,
                survivorKeys = survivors.Select(x => new { key = TensorConfigIdentity.ToKey(x.Config), x.DisplayName, x.Kld, x.SizeBytes }).ToList()
            };

            AnsiConsole.MarkupLine("[yellow]Best confirmed beneficial anomaly:[/]");
            AnsiConsole.MarkupLine($"[grey]  candidate=[/] [cyan]{Markup.Escape(bestAnomaly.DisplayName)}[/]");
            AnsiConsole.MarkupLine($"[grey]  actualCandidateKld=[/] [cyan]{bestAnomaly.Kld:0.000000}[/]");
            AnsiConsole.MarkupLine($"[grey]  actualCandidateSizeBytes=[/] [cyan]{bestAnomaly.SizeBytes:N0}[/]");
            AnsiConsole.MarkupLine($"[grey]  selectedAsSurvivor=[/] [cyan]{selected}[/]");
            if (!selected)
                AnsiConsole.MarkupLine($"[grey]  reasonNotSelected=[/] [yellow]{Markup.Escape(reason)}[/]");
        }

        if (!string.IsNullOrWhiteSpace(Cache.OutputDirectory))
        {
            string manifestDir = Path.Combine(Cache.OutputDirectory!, "magicquant-manifest");
            Directory.CreateDirectory(manifestDir);
            await File.WriteAllTextAsync(Path.Combine(manifestDir, "magicquant.anomaly-selection-reconciliation.json"), JsonSerializer.Serialize(payload, JsonOptions), ct);
        }
    }

    private static string ExplainBestAnomalyNotSelected(BenchmarkSnapshotRecord bestAnomaly, IReadOnlyList<BenchmarkSnapshotRecord> survivors)
    {
        var dominator = survivors.FirstOrDefault(x => x.SizeBytes <= bestAnomaly.SizeBytes && x.Kld <= bestAnomaly.Kld && (x.SizeBytes < bestAnomaly.SizeBytes || x.Kld < bestAnomaly.Kld));
        if (dominator != null)
            return $"dominated by survivor {dominator.DisplayName}";

        var lower = survivors.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes).FirstOrDefault();
        if (lower != null && lower.Kld < bestAnomaly.Kld)
            return $"survivor {lower.DisplayName} has lower actual KLD; spacing/final frontier kept that candidate";

        return "not selected after spacing/final dominance; no direct dominator found";
    }

    private static bool IsQ8Anchor(BenchmarkSnapshotRecord anchor)
    {
        try
        {
            return BaselineQuants.FromId(anchor.Config.BaseQuant).Names.Any(x => x.Contains("Q8", StringComparison.OrdinalIgnoreCase));
        }
        catch
        {
            return anchor.DisplayName.Contains("Q8", StringComparison.OrdinalIgnoreCase);
        }
    }

    private static CandidateValidationResult ChooseBestStrictDominanceCandidate(
        BenchmarkSnapshotRecord anchor,
        IReadOnlyList<CandidateValidationResult> accepted)
    {
        return accepted
            .Where(x => x.Snapshot != null)
            .OrderBy(x => x.Snapshot!.Kld)
            .ThenBy(x => x.Snapshot!.SizeBytes)
            .ThenByDescending(x => anchor.Kld - x.Snapshot!.Kld)
            .ThenBy(x => x.Candidate.Prediction.PredictedRank ?? ulong.MaxValue)
            .ThenByDescending(x => x.Candidate.Prediction.PredictionConfidence)
            .First();
    }

    private static string ResolveStrictChosenReason(BenchmarkSnapshotRecord anchor, BenchmarkSnapshotRecord chosen)
        => $"lowest actual KLD among accepted strict dominance candidates, then smaller actual size, gainVsAnchor={anchor.Kld - chosen.Kld:0.000000}";

    private static string ExplainStrictAcceptedLoss(
        BenchmarkSnapshotRecord anchor,
        BenchmarkSnapshotRecord chosen,
        BenchmarkSnapshotRecord loser)
    {
        if (loser.Kld > chosen.Kld)
            return $"higher actual KLD ({loser.Kld:0.000000} > {chosen.Kld:0.000000})";

        if (Math.Abs(loser.Kld - chosen.Kld) < 1e-12 && loser.SizeBytes > chosen.SizeBytes)
            return $"same actual KLD but larger actual size ({loser.SizeBytes:N0} > {chosen.SizeBytes:N0})";

        double chosenGain = anchor.Kld - chosen.Kld;
        double loserGain = anchor.Kld - loser.Kld;
        if (Math.Abs(loser.Kld - chosen.Kld) < 1e-12 && loser.SizeBytes == chosen.SizeBytes && loserGain < chosenGain)
            return $"weaker gain over anchor ({loserGain:0.000000} < {chosenGain:0.000000})";

        return "lost by prediction rank/confidence tie-breaker after actual KLD and size were equivalent";
    }

    private async Task<PhaseValidationResult> RunNearBaselineReplacementAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        List<SelectionPhaseDiagnostic> phaseDiagnostics,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 2: Near-Baseline Replacement[/]") { Justification = Justify.Left });

        var accepted = new List<BenchmarkSnapshotRecord>();
        var pairs = BuildAdjacentPairs(currentAnchors);
        int attemptLimit = Math.Max(1, Config.SelectionMaxFallbackAttemptsPerAnchor);
        int fetchLimit = Math.Max(DiagnosticPreviewLimit, attemptLimit * 3);

        AnsiConsole.MarkupLine($"[grey]Near-baseline neighbor pairs:[/] [cyan]{pairs.Count:N0}[/] | size premium=[cyan]{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}%[/] | fetch limit=[cyan]{fetchLimit:N0}[/] | validation attempts/window=[cyan]{attemptLimit:N0}[/]");

        for (int pairIndex = 0; pairIndex < pairs.Count; pairIndex++)
        {
            var pair = pairs[pairIndex];
            var lowerSizeHigherDamage = pair.HigherDamageSmaller;
            var upperSizeLowerDamage = pair.LowerDamageLarger;
            string windowLabel = $"near-baseline +{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}% {lowerSizeHigherDamage.DisplayName}";

            if (ShouldSkipAnchorReplacement(lowerSizeHigherDamage))
            {
                AnsiConsole.MarkupLine($"[grey]Skipping near-baseline lower anchor replacement:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)}");
                phaseDiagnostics.Add(new SelectionPhaseDiagnostic
                {
                    Phase = "NearBaselineReplacement",
                    WindowLabel = windowLabel,
                    PhaseWindowIndex = pairIndex + 1,
                    PhaseWindowCount = pairs.Count,
                    HigherDamageSmaller = ToAnchorLog(lowerSizeHigherDamage),
                    LowerDamageLarger = ToAnchorLog(upperSizeLowerDamage),
                    Notes = ["Skipped because the smaller/higher-damage anchor is an 8-bit/non-exact anchor and SelectionAllowEightBitAnchorReplacements=false."]
                });
                continue;
            }

            ulong min = lowerSizeHigherDamage.SizeBytes;
            ulong max = AddPercent(min, Config.SelectionNearBaselineMaxSizeGrowthPercent);

            if (max > upperSizeLowerDamage.SizeBytes)
                max = upperSizeLowerDamage.SizeBytes;

            long windowRows = await _predictedStore.CountPredictedHybridCandidatesInSizeWindowAsync(min, max, ct);
            long lineBeaters = await _predictedStore.CountBetterThanLinearCandidatesAsync(lowerSizeHigherDamage, upperSizeLowerDamage, min, max, ct);
            var rawCandidates = (await _predictedStore.QueryBetterThanLinearCandidatesAsync(
                lowerSizeHigherDamage,
                upperSizeLowerDamage,
                min,
                max,
                HybridSelectionReason.NearBaselineOnePercentReplacement,
                windowLabel,
                fetchLimit,
                ct)).ToList();

            var brutalityAnalyses = rawCandidates
                .Select(x => new { Candidate = x, Brutality = AnalyzeNearLowerAnchorBrutality(x) })
                .ToList();

            var candidates = brutalityAnalyses
                .Where(x => x.Brutality.Passed)
                .Select(x => AttachSelectionDiagnostics(
                    x.Candidate,
                    poolSize: lineBeaters,
                    windowCandidateCount: windowRows,
                    lineBeatingCandidateCount: lineBeaters,
                    fetchedCandidateCount: rawCandidates.Count,
                    candidatesAfterBrutalityCount: brutalityAnalyses.Count(y => y.Brutality.Passed),
                    candidateAttemptLimit: attemptLimit,
                    phaseWindowIndex: pairIndex + 1,
                    phaseWindowCount: pairs.Count,
                    notes: [x.Brutality.Explanation]))
                .Take(attemptLimit)
                .ToList();

            var rejectedByBrutality = brutalityAnalyses
                .Where(x => !x.Brutality.Passed)
                .Take(DiagnosticPreviewDisplayCount)
                .Select(x => ToCandidatePreviewLog(x.Candidate, x.Brutality))
                .ToList();

            var diag = new SelectionPhaseDiagnostic
            {
                Phase = "NearBaselineReplacement",
                WindowLabel = windowLabel,
                PhaseWindowIndex = pairIndex + 1,
                PhaseWindowCount = pairs.Count,
                HigherDamageSmaller = ToAnchorLog(lowerSizeHigherDamage),
                LowerDamageLarger = ToAnchorLog(upperSizeLowerDamage),
                WindowMinSizeBytes = min,
                WindowMaxSizeBytes = max,
                WindowSizeGiB = ToGiB(max > min ? max - min : 0),
                CandidatePoolSize = lineBeaters,
                WindowCandidateCount = windowRows,
                LineBeatingCandidateCount = lineBeaters,
                FetchedCandidateCount = rawCandidates.Count,
                CandidatesAfterBrutalityCount = brutalityAnalyses.Count(x => x.Brutality.Passed),
                SelectedForValidationCount = candidates.Count,
                CandidateAttemptLimit = attemptLimit,
                QueryFetchLimit = fetchLimit,
                TopCandidates = candidates.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                RejectedByBrutalityPreview = rejectedByBrutality,
                Notes = [
                    "Near-baseline first counts predicted hybrids inside the near-size window, then counts candidates predicted to beat the local line, then applies near-lower-anchor brutality, then caps validation attempts.",
                    $"Brutal zone fraction={Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###}; required gain fraction of pair KLD gap={Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap:0.###}."
                ]
            };
            phaseDiagnostics.Add(diag);

            AnsiConsole.MarkupLine(
                $"[grey]Near-baseline window {pairIndex + 1:N0}/{pairs.Count:N0}:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)} -> {Markup.Escape(upperSizeLowerDamage.DisplayName)} " +
                $"| rows-in-window={windowRows:N0}, beat-line={lineBeaters:N0}, fetched={rawCandidates.Count:N0}, after-brutality={diag.CandidatesAfterBrutalityCount:N0}, selected={candidates.Count:N0}/{attemptLimit:N0}");

            if (rejectedByBrutality.Count > 0)
                AnsiConsole.MarkupLine($"[grey]  rejected by near-lower-anchor brutality preview:[/] [cyan]{rejectedByBrutality.Count:N0}[/] (see magicquant-selection-phase-diagnostics.json)");

            if (candidates.Count == 0)
                continue;

            foreach (var candidate in candidates)
            {
                var validation = await BuildAndValidateSingleAsync(
                    candidate,
                    snapshot => snapshot.SizeBytes >= min &&
                                snapshot.SizeBytes <= max &&
                                BeatsLinearKldLine(snapshot.SizeBytes, snapshot.Kld, lowerSizeHigherDamage, upperSizeLowerDamage),
                    $"must land inside {min:N0}..{max:N0} bytes and beat the real linear KLD line",
                    ct);

                validationAttempts.Add(validation);

                if (validation.Accepted && validation.Snapshot != null)
                {
                    accepted.Add(validation.Snapshot);
                    eliminations.Add(new BaselineEliminationRecord
                    {
                        Eliminated = lowerSizeHigherDamage,
                        Eliminator = validation.Snapshot,
                        Reason = $"near-baseline replacement within +{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}% size premium"
                    });
                    break;
                }

                validationFailures.Add(validation);
            }
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }

    private async Task<PhaseValidationResult> RunInteriorSubspaceDiscoveryAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        List<SelectionPhaseDiagnostic> phaseDiagnostics,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 3: Interior Subspace Discovery[/]") { Justification = Justify.Left });

        var accepted = new List<BenchmarkSnapshotRecord>();
        var pairs = BuildAdjacentPairs(currentAnchors);
        var fractions = Config.SelectionInteriorWindowFractions.ToList();

        int interiorAttemptLimit = Math.Max(1, Math.Max(Config.SelectionMaxCandidatesPerInteriorWindow, Config.SelectionMaxFallbackAttemptsPerAnchor));
        int interiorFetchLimit = Math.Max(interiorAttemptLimit, DiagnosticPreviewLimit);

        AnsiConsole.MarkupLine($"[grey]Interior neighbor pairs:[/] [cyan]{pairs.Count:N0}[/] | window fractions=[cyan]{Markup.Escape(string.Join(", ", fractions.Select(x => x.ToString("0.###"))))}[/] | candidates/window=[cyan]{Config.SelectionMaxCandidatesPerInteriorWindow:N0}[/] | fallback attempts/window=[cyan]{Config.SelectionMaxFallbackAttemptsPerAnchor:N0}[/] | validation attempts/window=[cyan]{interiorAttemptLimit:N0}[/] | fetch preview/window=[cyan]{interiorFetchLimit:N0}[/]");

        var allCandidates = new List<HybridSelectionCandidate>();
        int globalWindowIndex = 0;
        int estimatedWindowCount = pairs.Sum(pair => EstimateInteriorWindowCount(pair, fractions));

        for (int pairIndex = 0; pairIndex < pairs.Count; pairIndex++)
        {
            var pair = pairs[pairIndex];
            ulong lowSize = pair.HigherDamageSmaller.SizeBytes;
            ulong highSize = pair.LowerDamageLarger.SizeBytes;

            if (highSize <= lowSize)
                continue;

            ulong span = highSize - lowSize;
            ulong cursor = lowSize;

            for (int i = 0; i < fractions.Count; i++)
            {
                double fraction = fractions[i];
                if (fraction <= 0d)
                    continue;

                ulong width = (ulong)Math.Round(span * fraction, MidpointRounding.AwayFromZero);
                if (width == 0)
                    continue;

                ulong min = cursor;
                ulong max = i == fractions.Count - 1
                    ? Math.Min(highSize, cursor + width)
                    : Math.Min(highSize, cursor + width);

                if (max <= min)
                    continue;

                globalWindowIndex++;
                string windowLabel = $"interior {i + 1}: {pair.HigherDamageSmaller.DisplayName} -> {pair.LowerDamageLarger.DisplayName}";
                long windowRows = await _predictedStore.CountPredictedHybridCandidatesInSizeWindowAsync(min, max, ct);
                long lineBeaters = await _predictedStore.CountBetterThanLinearCandidatesAsync(pair.HigherDamageSmaller, pair.LowerDamageLarger, min, max, ct);

                var rawCandidates = (await _predictedStore.QueryBetterThanLinearCandidatesAsync(
                    pair.HigherDamageSmaller,
                    pair.LowerDamageLarger,
                    min,
                    max,
                    HybridSelectionReason.InteriorSubspaceDiscovery,
                    windowLabel,
                    interiorFetchLimit,
                    ct)).ToList();

                var brutalityAnalyses = rawCandidates
                    .Select(x => new { Candidate = x, Brutality = AnalyzeNearLowerAnchorBrutality(x) })
                    .ToList();

                int afterBrutalityCount = brutalityAnalyses.Count(y => y.Brutality.Passed);

                var kept = brutalityAnalyses
                    .Where(x => x.Brutality.Passed)
                    .Select(x => AttachSelectionDiagnostics(
                        x.Candidate,
                        poolSize: lineBeaters,
                        windowCandidateCount: windowRows,
                        lineBeatingCandidateCount: lineBeaters,
                        fetchedCandidateCount: rawCandidates.Count,
                        candidatesAfterBrutalityCount: afterBrutalityCount,
                        candidateAttemptLimit: interiorAttemptLimit,
                        phaseWindowIndex: globalWindowIndex,
                        phaseWindowCount: estimatedWindowCount,
                        notes: [x.Brutality.Explanation]))
                    .Take(interiorAttemptLimit)
                    .ToList();

                allCandidates.AddRange(kept);

                var rejectedByBrutality = brutalityAnalyses
                    .Where(x => !x.Brutality.Passed)
                    .Take(DiagnosticPreviewDisplayCount)
                    .Select(x => ToCandidatePreviewLog(x.Candidate, x.Brutality))
                    .ToList();

                phaseDiagnostics.Add(new SelectionPhaseDiagnostic
                {
                    Phase = "InteriorSubspaceDiscovery",
                    WindowLabel = windowLabel,
                    PhaseWindowIndex = globalWindowIndex,
                    PhaseWindowCount = estimatedWindowCount,
                    HigherDamageSmaller = ToAnchorLog(pair.HigherDamageSmaller),
                    LowerDamageLarger = ToAnchorLog(pair.LowerDamageLarger),
                    WindowMinSizeBytes = min,
                    WindowMaxSizeBytes = max,
                    WindowSizeGiB = ToGiB(max > min ? max - min : 0),
                    CandidatePoolSize = lineBeaters,
                    WindowCandidateCount = windowRows,
                    LineBeatingCandidateCount = lineBeaters,
                    FetchedCandidateCount = rawCandidates.Count,
                    CandidatesAfterBrutalityCount = afterBrutalityCount,
                    SelectedForValidationCount = kept.Count,
                    CandidateAttemptLimit = interiorAttemptLimit,
                    QueryFetchLimit = interiorFetchLimit,
                    TopCandidates = kept.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                    RejectedByBrutalityPreview = rejectedByBrutality,
                    Notes = ["Interior candidates are gathered per window, then globally deduped by tensor config before batch validation."]
                });

                AnsiConsole.MarkupLine(
                    $"[grey]Interior window {globalWindowIndex:N0}/{Math.Max(estimatedWindowCount, globalWindowIndex):N0}:[/] {Markup.Escape(pair.HigherDamageSmaller.DisplayName)} -> {Markup.Escape(pair.LowerDamageLarger.DisplayName)} " +
                    $"| rows-in-window={windowRows:N0}, beat-line={lineBeaters:N0}, fetched={rawCandidates.Count:N0}, after-brutality={afterBrutalityCount:N0}, selected={kept.Count:N0}/{interiorAttemptLimit:N0}");

                cursor = max;

                if (cursor >= highSize)
                    break;
            }
        }

        var deduped = allCandidates
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Prediction.Config), StringComparer.Ordinal)
            .Select(g => g.OrderByDescending(x => x.PredictedGainOverLine).ThenBy(x => x.Prediction.PredictedSizeBytes).First())
            .OrderByDescending(x => x.PredictedGainOverLine)
            .ThenBy(x => x.Prediction.PredictedSizeBytes)
            .ToList();

        int duplicateCount = Math.Max(0, allCandidates.Count - deduped.Count);
        AnsiConsole.MarkupLine($"[grey]Interior candidate rollup:[/] raw-after-brutality={allCandidates.Count:N0}, duplicate-configs-removed={duplicateCount:N0}, selected-for-batch={deduped.Count:N0}");

        if (deduped.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No predicted interior candidates beat their local linear KLD lines after window/brutality filtering.[/]");
            return new PhaseValidationResult();
        }

        AnsiConsole.MarkupLine($"[grey]Interior candidates selected for batch validation:[/] [cyan]{deduped.Count:N0}[/]");
        PrintCandidatePreviewTable(deduped, "Interior selected candidates preview");

        var quantBatch = deduped.Select(x => x.Prediction.Quant).DistinctBy(x => TensorConfigIdentity.ToKey((TensorConfig)x)).ToList();
        var summary = await _quantizationService.ProcessHybridBatchAsync(
            quantBatch,
            new StageProgressOptions
            {
                StageName = "Interior candidate validation batch",
                Total = quantBatch.Count,
                MinimumNonSkippedSamplesBeforeEta = 2,
                ShowEta = true,
                CountSkippedForEta = false
            },
            ct);
        AnsiConsole.MarkupLine($"[grey]Interior validation batch:[/] requested={summary.Requested:N0} completed={summary.Completed:N0} skipped={summary.Skipped:N0} failed={summary.Failed:N0}");

        foreach (var candidate in deduped)
        {
            var snapshot = await _repository.LoadBenchmarkSnapshotAsync(candidate.Prediction.Config, ct);
            bool acceptedCandidate = snapshot != null &&
                                     snapshot.SizeBytes >= candidate.WindowMinSizeBytes &&
                                     snapshot.SizeBytes <= candidate.WindowMaxSizeBytes &&
                                     BeatsLinearKldLine(snapshot.SizeBytes, snapshot.Kld, candidate.HigherDamageAnchor, candidate.LowerDamageAnchor);

            var validation = new CandidateValidationResult
            {
                Candidate = candidate,
                Snapshot = snapshot,
                Accepted = acceptedCandidate,
                FailureCode = acceptedCandidate ? string.Empty : snapshot == null ? "SNAPSHOT_MISSING_AFTER_BATCH" : "REAL_BENCHMARK_DID_NOT_BEAT_LINE",
                Message = acceptedCandidate
                    ? "validated interior candidate"
                    : snapshot == null
                        ? $"benchmark snapshot was not found after batch build (requested={summary.Requested}, completed={summary.Completed}, skipped={summary.Skipped}, failed={summary.Failed})"
                        : BuildDetailedFailureMessage(candidate, snapshot, "real benchmark did not beat the local linear KLD line inside the requested size window")
            };
            validationAttempts.Add(validation);

            if (acceptedCandidate && snapshot != null)
            {
                accepted.Add(snapshot);
                PrintCandidateValidationOutcome(candidate, snapshot, accepted: true, "validated interior candidate");
                continue;
            }

            if (snapshot != null)
                PrintCandidateValidationOutcome(candidate, snapshot, accepted: false, validation.Message);
            else
                AnsiConsole.MarkupLine($"[yellow]Rejected predicted candidate:[/] {Markup.Escape(validation.Message)}");

            validationFailures.Add(validation);
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }

    private async Task<CandidateValidationResult> BuildAndValidateSingleAsync(
        HybridSelectionCandidate candidate,
        Func<BenchmarkSnapshotRecord, bool> accept,
        string expectation,
        CancellationToken ct)
    {
        AnsiConsole.MarkupLine(
            $"[grey]Validating candidate:[/] {Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(candidate.Prediction.Quant))} " +
            $"[grey]| reason=[/] {candidate.Reason} [grey]| attempt=[/] {candidate.AttemptOrder:N0}/{Math.Max(candidate.CandidateAttemptLimit, candidate.AttemptOrder):N0} " +
            $"[grey]| window=[/] {Markup.Escape(candidate.WindowLabel)}");
        PrintCandidatePredictionLine(candidate);

        var summary = await _quantizationService.ProcessHybridBatchAsync(new[] { candidate.Prediction.Quant }, ct);
        var snapshot = await _repository.LoadBenchmarkSnapshotAsync(candidate.Prediction.Config, ct);

        bool accepted = snapshot != null && accept(snapshot);
        string message = accepted
            ? "validated"
            : snapshot == null
                ? $"no benchmark snapshot was available after build attempt (completed={summary.Completed}, skipped={summary.Skipped}, failed={summary.Failed})"
                : BuildDetailedFailureMessage(candidate, snapshot, $"failed expectation: {expectation}");

        if (accepted && snapshot != null)
        {
            PrintCandidateValidationOutcome(candidate, snapshot, accepted: true, "validated");
        }
        else if (snapshot != null)
        {
            PrintCandidateValidationOutcome(candidate, snapshot, accepted: false, message);
        }
        else
        {
            AnsiConsole.MarkupLine($"[yellow]Rejected predicted candidate:[/] {Markup.Escape(message)}");
        }

        return new CandidateValidationResult
        {
            Candidate = candidate,
            Snapshot = snapshot,
            Accepted = accepted,
            FailureCode = accepted ? string.Empty : snapshot == null ? "SNAPSHOT_MISSING_AFTER_BUILD" : "REAL_BENCHMARK_FAILED_EXPECTATION",
            Message = message
        };
    }

    private static HybridSelectionCandidate AttachSelectionDiagnostics(
        HybridSelectionCandidate candidate,
        long poolSize,
        long windowCandidateCount,
        long lineBeatingCandidateCount,
        int fetchedCandidateCount,
        int candidatesAfterBrutalityCount,
        int candidateAttemptLimit,
        int phaseWindowIndex,
        int phaseWindowCount,
        IReadOnlyList<string> notes)
    {
        return new HybridSelectionCandidate
        {
            Prediction = candidate.Prediction,
            Reason = candidate.Reason,
            LowerDamageAnchor = candidate.LowerDamageAnchor,
            HigherDamageAnchor = candidate.HigherDamageAnchor,
            WindowMinSizeBytes = candidate.WindowMinSizeBytes,
            WindowMaxSizeBytes = candidate.WindowMaxSizeBytes,
            LinearExpectedKld = candidate.LinearExpectedKld,
            PredictedGainOverLine = candidate.PredictedGainOverLine,
            AttemptOrder = candidate.AttemptOrder,
            WindowLabel = candidate.WindowLabel,
            CandidatePoolSize = poolSize,
            WindowCandidateCount = windowCandidateCount,
            LineBeatingCandidateCount = lineBeatingCandidateCount,
            FetchedCandidateCount = fetchedCandidateCount,
            CandidatesAfterBrutalityCount = candidatesAfterBrutalityCount,
            CandidateAttemptLimit = candidateAttemptLimit,
            PhaseWindowIndex = phaseWindowIndex,
            PhaseWindowCount = phaseWindowCount,
            CandidateSelectionNotes = notes
        };
    }

    private static BrutalityAnalysis AnalyzeNearLowerAnchorBrutality(HybridSelectionCandidate candidate)
    {
        ulong span = candidate.LowerDamageAnchor.SizeBytes > candidate.HigherDamageAnchor.SizeBytes
            ? candidate.LowerDamageAnchor.SizeBytes - candidate.HigherDamageAnchor.SizeBytes
            : 0;

        if (span == 0)
        {
            return new BrutalityAnalysis
            {
                Passed = true,
                FractionFromSmallAnchor = 1d,
                RequiredGain = Config.SelectionMinimumKldImprovementEpsilon,
                Explanation = "Brutality passed because anchor span is zero."
            };
        }

        ulong distanceFromSmall = candidate.Prediction.PredictedSizeBytes > candidate.HigherDamageAnchor.SizeBytes
            ? candidate.Prediction.PredictedSizeBytes - candidate.HigherDamageAnchor.SizeBytes
            : 0;

        double fraction = distanceFromSmall / (double)span;
        double requiredGain = Math.Max(
            Config.SelectionMinimumKldImprovementEpsilon,
            Math.Abs(candidate.HigherDamageAnchor.Kld - candidate.LowerDamageAnchor.Kld) *
            Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap);

        bool passed = fraction > Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan ||
                      candidate.PredictedGainOverLine >= requiredGain;

        string explanation = passed
            ? fraction > Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan
                ? $"Brutality passed because candidate is outside brutal zone (fraction={fraction:0.###} > {Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###})."
                : $"Brutality passed because predicted gain {candidate.PredictedGainOverLine:0.########} >= required gain {requiredGain:0.########}."
            : $"Brutality rejected because candidate is inside brutal zone (fraction={fraction:0.###} <= {Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###}) and predicted gain {candidate.PredictedGainOverLine:0.########} < required gain {requiredGain:0.########}.";

        return new BrutalityAnalysis
        {
            Passed = passed,
            FractionFromSmallAnchor = fraction,
            RequiredGain = requiredGain,
            Explanation = explanation
        };
    }

    private static bool PassesNearLowerAnchorBrutality(HybridSelectionCandidate candidate) =>
        AnalyzeNearLowerAnchorBrutality(candidate).Passed;

    private List<BenchmarkSnapshotRecord> ApplyMeaningfulSpacing(
        IReadOnlyList<BenchmarkSnapshotRecord> snapshots,
        List<BaselineEliminationRecord> eliminations)
    {
        if (snapshots.Count <= 2)
            return snapshots.ToList();

        var ordered = snapshots
            .OrderBy(x => x.SizeBytes)
            .ThenBy(x => x.Kld)
            .ToList();

        ulong minSize = ordered.Min(x => x.SizeBytes);
        ulong maxSize = ordered.Max(x => x.SizeBytes);
        ulong globalSpan = maxSize > minSize ? maxSize - minSize : 0;

        if (globalSpan == 0)
            return _finalEliminator.Eliminate(ordered).Survivors.ToList();

        ulong minGap = (ulong)Math.Round(globalSpan * Config.SelectionMinimumNeighborGapFractionOfGlobalSpan, MidpointRounding.AwayFromZero);
        if (minGap == 0)
            return _finalEliminator.Eliminate(ordered).Survivors.ToList();

        var kept = new List<BenchmarkSnapshotRecord>();

        foreach (var snap in ordered)
        {
            var tooClose = kept
                .Where(x => Distance(x.SizeBytes, snap.SizeBytes) < minGap)
                .OrderBy(x => Distance(x.SizeBytes, snap.SizeBytes))
                .FirstOrDefault();

            if (tooClose == null)
            {
                kept.Add(snap);
                continue;
            }

            var winner = ChooseSpacingWinner(tooClose, snap);
            var loser = ReferenceEquals(winner, tooClose) ? snap : tooClose;

            if (!ReferenceEquals(winner, tooClose))
            {
                kept.Remove(tooClose);
                kept.Add(winner);
            }

            eliminations.Add(new BaselineEliminationRecord
            {
                Eliminated = loser,
                Eliminator = winner,
                Reason = $"meaningful spacing collapse; size gap below {minGap:N0} bytes"
            });
        }

        return _finalEliminator.Eliminate(kept).Survivors.ToList();
    }

    private static BenchmarkSnapshotRecord ChooseSpacingWinner(BenchmarkSnapshotRecord left, BenchmarkSnapshotRecord right)
    {
        if (Dominates(left, right))
            return left;

        if (Dominates(right, left))
            return right;

        return left.Kld.CompareTo(right.Kld) switch
        {
            < 0 => left,
            > 0 => right,
            _ => left.SizeBytes <= right.SizeBytes ? left : right
        };
    }

    private List<BenchmarkSnapshotRecord> MergeAndDominanceFilter(
        IReadOnlyList<BenchmarkSnapshotRecord> current,
        IReadOnlyList<BenchmarkSnapshotRecord> additions,
        List<BaselineEliminationRecord> eliminations,
        string reason)
    {
        if (additions.Count == 0)
            return current.ToList();

        var merged = current
            .Concat(additions)
            .DistinctBy(x => TensorConfigIdentity.ToKey(x.Config))
            .ToList();

        var result = _finalEliminator.Eliminate(merged);

        foreach (var eliminated in result.Eliminated)
        {
            var eliminator = result.Survivors.FirstOrDefault(x => Dominates(x, eliminated));
            if (eliminator == null)
                continue;

            eliminations.Add(new BaselineEliminationRecord
            {
                Eliminated = eliminated,
                Eliminator = eliminator,
                Reason = reason
            });
        }

        return result.Survivors.ToList();
    }

    private static bool ShouldSkipAnchorReplacement(BenchmarkSnapshotRecord anchor)
    {
        return !Config.SelectionAllowEightBitAnchorReplacements &&
               anchor.Quant.BaseQuant.BitRange >= 8 &&
               !anchor.Quant.BaseQuant.IsHighPrecisionExactAlias;
    }

    private static List<AdjacentAnchorPair> BuildAdjacentPairs(IReadOnlyList<BenchmarkSnapshotRecord> anchors)
    {
        var ordered = anchors
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ToList();

        var result = new List<AdjacentAnchorPair>();

        for (int i = 0; i < ordered.Count - 1; i++)
        {
            var lowerDamage = ordered[i];
            var higherDamage = ordered[i + 1];

            if (higherDamage.SizeBytes >= lowerDamage.SizeBytes)
                continue;

            result.Add(new AdjacentAnchorPair
            {
                LowerDamageLarger = lowerDamage,
                HigherDamageSmaller = higherDamage
            });
        }

        return result;
    }

    private static int EstimateInteriorWindowCount(AdjacentAnchorPair pair, IReadOnlyList<double> fractions)
    {
        if (pair.LowerDamageLarger.SizeBytes <= pair.HigherDamageSmaller.SizeBytes)
            return 0;

        ulong span = pair.LowerDamageLarger.SizeBytes - pair.HigherDamageSmaller.SizeBytes;
        ulong cursor = pair.HigherDamageSmaller.SizeBytes;
        int count = 0;

        for (int i = 0; i < fractions.Count; i++)
        {
            double fraction = fractions[i];
            if (fraction <= 0d)
                continue;

            ulong width = (ulong)Math.Round(span * fraction, MidpointRounding.AwayFromZero);
            if (width == 0)
                continue;

            ulong max = Math.Min(pair.LowerDamageLarger.SizeBytes, cursor + width);
            if (max <= cursor)
                continue;

            count++;
            cursor = max;
            if (cursor >= pair.LowerDamageLarger.SizeBytes)
                break;
        }

        return count;
    }

    private static bool BeatsLinearKldLine(
        ulong candidateSize,
        double candidateKld,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger)
    {
        double expected = InterpolateKldLine(candidateSize, higherDamageSmaller, lowerDamageLarger);
        return candidateKld + Config.SelectionMinimumKldImprovementEpsilon < expected;
    }

    private static double InterpolateKldLine(
        ulong candidateSize,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger)
    {
        ulong smallSize = higherDamageSmaller.SizeBytes;
        ulong largeSize = lowerDamageLarger.SizeBytes;

        if (largeSize <= smallSize)
            return Math.Min(higherDamageSmaller.Kld, lowerDamageLarger.Kld);

        double t = Math.Clamp((candidateSize - smallSize) / (double)(largeSize - smallSize), 0d, 1d);
        return higherDamageSmaller.Kld + ((lowerDamageLarger.Kld - higherDamageSmaller.Kld) * t);
    }

    private static ValidationMetrics ComputeValidationMetrics(HybridSelectionCandidate candidate, BenchmarkSnapshotRecord snapshot)
    {
        double line = InterpolateKldLine(snapshot.SizeBytes, candidate.HigherDamageAnchor, candidate.LowerDamageAnchor);
        double gain = line - snapshot.Kld;
        bool insideWindow = snapshot.SizeBytes >= candidate.WindowMinSizeBytes && snapshot.SizeBytes <= candidate.WindowMaxSizeBytes;
        bool beatsLine = snapshot.Kld + Config.SelectionMinimumKldImprovementEpsilon < line;
        long sizeMissBytes = 0;

        if (snapshot.SizeBytes < candidate.WindowMinSizeBytes)
            sizeMissBytes = (long)candidate.WindowMinSizeBytes - (long)snapshot.SizeBytes;
        else if (snapshot.SizeBytes > candidate.WindowMaxSizeBytes)
            sizeMissBytes = (long)snapshot.SizeBytes - (long)candidate.WindowMaxSizeBytes;

        return new ValidationMetrics
        {
            ActualLineKld = line,
            ActualGainOverLine = gain,
            KldMiss = snapshot.Kld + Config.SelectionMinimumKldImprovementEpsilon - line,
            SizeMissBytes = sizeMissBytes,
            InsideWindow = insideWindow,
            BeatsLine = beatsLine
        };
    }

    private static string BuildDetailedFailureMessage(HybridSelectionCandidate candidate, BenchmarkSnapshotRecord snapshot, string prefix)
    {
        var metrics = ComputeValidationMetrics(candidate, snapshot);
        string sizeText = metrics.SizeMissBytes == 0 ? "inside size window" : $"missed size window by {metrics.SizeMissBytes:N0} bytes";
        string kldText = metrics.KldMiss <= 0 ? "beat required KLD line" : $"missed KLD line by {metrics.KldMiss:0.000000}";

        return $"{prefix}; actual size={snapshot.SizeBytes:N0} ({ToGiB(snapshot.SizeBytes):0.00} GiB), actual KLD={snapshot.Kld:0.000000}, line={metrics.ActualLineKld:0.000000}, gain={metrics.ActualGainOverLine:0.000000}, {sizeText}, {kldText}";
    }

    private static void PrintCandidatePredictionLine(HybridSelectionCandidate candidate)
    {
        AnsiConsole.MarkupLine(
            $"[grey]  predicted:[/] size={candidate.Prediction.PredictedSizeBytes:N0} bytes ({ToGiB(candidate.Prediction.PredictedSizeBytes):0.00} GiB), " +
            $"kld={candidate.Prediction.PredictedKld:0.000000}, line={candidate.LinearExpectedKld:0.000000}, gain={candidate.PredictedGainOverLine:0.000000}, " +
            $"rank={candidate.Prediction.PredictedRank}, confidence={candidate.Prediction.PredictionConfidence:0.###}");
        AnsiConsole.MarkupLine(
            $"[grey]  selection context:[/] pool={candidate.CandidatePoolSize:N0}, windowRows={candidate.WindowCandidateCount:N0}, lineBeat={candidate.LineBeatingCandidateCount:N0}, " +
            $"fetched={candidate.FetchedCandidateCount:N0}, afterBrutality={candidate.CandidatesAfterBrutalityCount:N0}, attemptLimit={candidate.CandidateAttemptLimit:N0}");
        AnsiConsole.MarkupLine($"[grey]  bit space:[/] {Markup.Escape(DescribeBitSpace(candidate.Prediction.Config))}");
    }

    private static void PrintCandidateValidationOutcome(HybridSelectionCandidate candidate, BenchmarkSnapshotRecord snapshot, bool accepted, string message)
    {
        var metrics = ComputeValidationMetrics(candidate, snapshot);
        string status = accepted ? "[green]Validated[/]" : "[yellow]Rejected predicted candidate[/]";
        AnsiConsole.MarkupLine(
            $"{status}: {Markup.Escape(snapshot.DisplayName)} | actual size={snapshot.SizeBytes:N0} ({ToGiB(snapshot.SizeBytes):0.00} GiB), " +
            $"actual KLD={snapshot.Kld:0.000000}, line={metrics.ActualLineKld:0.000000}, gain={metrics.ActualGainOverLine:0.000000}, " +
            $"sizeMiss={metrics.SizeMissBytes:N0}, kldMiss={Math.Max(0d, metrics.KldMiss):0.000000}");

        if (!accepted)
            AnsiConsole.MarkupLine($"[yellow]  reason:[/] {Markup.Escape(message)}");
    }

    private static void PrintAnchorFrontier(IReadOnlyList<BenchmarkSnapshotRecord> anchors, string title)
    {
        if (anchors.Count == 0)
            return;

        var table = new Table().RoundedBorder().BorderColor(Color.Grey);
        table.Title = new TableTitle(Markup.Escape(title));
        table.AddColumn("Order");
        table.AddColumn("Anchor");
        table.AddColumn("Provider");
        table.AddColumn("KLD");
        table.AddColumn("Size GiB");

        int i = 0;
        foreach (var anchor in anchors.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes))
        {
            table.AddRow(
                (++i).ToString("N0"),
                Markup.Escape(anchor.DisplayName),
                Markup.Escape(anchor.ProviderName),
                anchor.Kld.ToString("0.000000"),
                ToGiB(anchor.SizeBytes).ToString("0.00"));
        }

        AnsiConsole.Write(table);
    }

    private static void PrintCandidatePreviewTable(IReadOnlyList<HybridSelectionCandidate> candidates, string title)
    {
        if (candidates.Count == 0)
            return;

        var table = new Table().RoundedBorder().BorderColor(Color.Grey);
        table.Title = new TableTitle(Markup.Escape(title));
        table.AddColumn("Attempt");
        table.AddColumn("Candidate");
        table.AddColumn("Pred KLD");
        table.AddColumn("Line");
        table.AddColumn("Gain");
        table.AddColumn("Size GiB");
        table.AddColumn("Rank");
        table.AddColumn("Bit Space");

        foreach (var c in candidates.Take(DiagnosticPreviewDisplayCount))
        {
            table.AddRow(
                c.AttemptOrder.ToString("N0"),
                Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(c.Prediction.Quant)),
                c.Prediction.PredictedKld.ToString("0.000000"),
                c.LinearExpectedKld.ToString("0.000000"),
                c.PredictedGainOverLine.ToString("0.000000"),
                ToGiB(c.Prediction.PredictedSizeBytes).ToString("0.00"),
                c.Prediction.PredictedRank?.ToString("N0") ?? "n/a",
                Markup.Escape(DescribeBitSpace(c.Prediction.Config)));
        }

        AnsiConsole.Write(table);
    }

    private static CandidatePreviewLog ToCandidatePreviewLog(HybridSelectionCandidate candidate) =>
        ToCandidatePreviewLog(candidate, AnalyzeNearLowerAnchorBrutality(candidate));

    private static CandidatePreviewLog ToCandidatePreviewLog(HybridSelectionCandidate candidate, BrutalityAnalysis brutality)
    {
        return new CandidatePreviewLog
        {
            AttemptOrder = candidate.AttemptOrder,
            Key = TensorConfigIdentity.ToKey(candidate.Prediction.Config),
            DisplayName = HybridBenchmarkRepository.BuildDisplayName(candidate.Prediction.Quant),
            PredictedSizeBytes = candidate.Prediction.PredictedSizeBytes,
            PredictedSizeGiB = ToGiB(candidate.Prediction.PredictedSizeBytes),
            PredictedKld = candidate.Prediction.PredictedKld,
            LinearExpectedKld = candidate.LinearExpectedKld,
            PredictedGainOverLine = candidate.PredictedGainOverLine,
            PredictionConfidence = candidate.Prediction.PredictionConfidence,
            PredictionRank = candidate.Prediction.PredictedRank,
            BaseQuant = candidate.Prediction.Quant.BaseQuant.Names[0],
            BaseBitRange = candidate.Prediction.Quant.BaseQuant.BitRange,
            BitSpace = DescribeBitSpace(candidate.Prediction.Config),
            OverrideSummary = DescribeOverrides(candidate.Prediction.Config),
            BrutalityPassed = brutality.Passed,
            BrutalityFractionFromSmallAnchor = brutality.FractionFromSmallAnchor,
            BrutalityRequiredGain = brutality.RequiredGain,
            BrutalityExplanation = brutality.Explanation
        };
    }

    private static object ToAnchorLog(BenchmarkSnapshotRecord anchor)
    {
        return new
        {
            key = TensorConfigIdentity.ToKey(anchor.Config),
            displayName = anchor.DisplayName,
            provider = anchor.ProviderName,
            baselineFamily = anchor.BaselineFamily,
            sizeBytes = anchor.SizeBytes,
            sizeGiB = ToGiB(anchor.SizeBytes),
            kld = anchor.Kld,
            ppl = anchor.Ppl,
            bitRange = anchor.Quant.BaseQuant.BitRange,
            quantizeBase = anchor.Quant.BaseQuant.QuantizeBaseArgumentName
        };
    }

    private static async Task WriteSelectionPhaseDiagnosticsAsync(
        IReadOnlyList<SelectionPhaseDiagnostic> phaseDiagnostics,
        IReadOnlyList<CandidateValidationResult> validationFailures,
        IReadOnlyList<CandidateValidationResult> validationAttempts,
        CancellationToken ct)
    {
        string directory = ResolveGgufDirectory();
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "magicquant-selection-phase-diagnostics.json");
        string attemptsPath = Path.Combine(directory, "magicquant-selection-validation-attempts.json");
        string missesPath = Path.Combine(directory, "magicquant-selection-validation-misses.json");

        var payload = new
        {
            generatedUtc = DateTime.UtcNow,
            config = new
            {
                nearBaselineMaxSizeGrowthPercent = Config.SelectionNearBaselineMaxSizeGrowthPercent,
                interiorWindowFractions = Config.SelectionInteriorWindowFractions,
                maxCandidatesPerInteriorWindow = Config.SelectionMaxCandidatesPerInteriorWindow,
                maxFallbackAttemptsPerAnchor = Config.SelectionMaxFallbackAttemptsPerAnchor,
                minimumKldImprovementEpsilon = Config.SelectionMinimumKldImprovementEpsilon,
                nearLowerAnchorBrutalZoneFractionOfPairSpan = Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan,
                nearAnchorRequiredKldGainFractionOfPairGap = Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap,
                allowEightBitAnchorReplacements = Config.SelectionAllowEightBitAnchorReplacements
            },
            totals = new
            {
                phaseWindowCount = phaseDiagnostics.Count,
                selectedForValidation = phaseDiagnostics.Sum(x => x.SelectedForValidationCount),
                validationAttempts = validationAttempts.Count,
                validationAccepted = validationAttempts.Count(x => x.Accepted),
                validationMisses = validationFailures.Count(x => !x.Accepted)
            },
            windows = phaseDiagnostics
        };

        var attemptsPayload = validationAttempts.Select(ToValidationAttemptLog).ToList();
        var missesPayload = validationFailures.Where(x => !x.Accepted).Select(ToValidationAttemptLog).ToList();

        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(payload, JsonOptions), ct);
        await File.WriteAllTextAsync(attemptsPath, JsonSerializer.Serialize(attemptsPayload, JsonOptions), ct);
        await File.WriteAllTextAsync(missesPath, JsonSerializer.Serialize(missesPayload, JsonOptions), ct);
        AnsiConsole.MarkupLine($"[green]Selection phase diagnostics log:[/] {Markup.Escape(path)}");
        AnsiConsole.MarkupLine($"[green]Selection validation attempts log:[/] {Markup.Escape(attemptsPath)}");
        AnsiConsole.MarkupLine($"[green]Selection validation miss log:[/] {Markup.Escape(missesPath)}");
    }


    private static object ToValidationAttemptLog(CandidateValidationResult attempt)
    {
        var c = attempt.Candidate;
        var snap = attempt.Snapshot;
        double? actualLine = null;
        double? actualGainOverLine = null;
        long? sizeMissBytes = null;
        double? kldMiss = null;
        bool? actualInsideSizeWindow = null;
        bool? actualBeatLine = null;

        if (snap != null)
        {
            actualLine = InterpolateKldLine(snap.SizeBytes, c.HigherDamageAnchor, c.LowerDamageAnchor);
            actualGainOverLine = actualLine.Value - snap.Kld;
            actualInsideSizeWindow = snap.SizeBytes >= c.WindowMinSizeBytes && snap.SizeBytes <= c.WindowMaxSizeBytes;
            actualBeatLine = snap.Kld + Config.SelectionMinimumKldImprovementEpsilon < actualLine.Value;

            if (snap.SizeBytes < c.WindowMinSizeBytes)
                sizeMissBytes = (long)c.WindowMinSizeBytes - (long)snap.SizeBytes;
            else if (snap.SizeBytes > c.WindowMaxSizeBytes)
                sizeMissBytes = (long)snap.SizeBytes - (long)c.WindowMaxSizeBytes;
            else
                sizeMissBytes = 0;

            kldMiss = snap.Kld + Config.SelectionMinimumKldImprovementEpsilon - actualLine.Value;
        }

        return new
        {
            accepted = attempt.Accepted,
            failureCode = attempt.FailureCode,
            message = attempt.Message,
            reason = c.Reason.ToString(),
            attemptOrder = c.AttemptOrder,
            attemptLimit = c.CandidateAttemptLimit,
            windowLabel = c.WindowLabel,
            phaseWindowIndex = c.PhaseWindowIndex,
            phaseWindowCount = c.PhaseWindowCount,
            candidateKey = TensorConfigIdentity.ToKey(c.Prediction.Config),
            candidateInternalName = HybridBenchmarkRepository.BuildDisplayName(c.Prediction.Quant),
            bitSpace = DescribeBitSpace(c.Prediction.Config),
            overrideSummary = DescribeOverrides(c.Prediction.Config),
            baseQuant = c.Prediction.Quant.BaseQuant.Names[0],
            baseBitRange = c.Prediction.Quant.BaseQuant.BitRange,
            predicted = new
            {
                sizeBytes = c.Prediction.PredictedSizeBytes,
                sizeGiB = ToGiB(c.Prediction.PredictedSizeBytes),
                kld = c.Prediction.PredictedKld,
                lineKldAtPredictedSize = c.LinearExpectedKld,
                gainOverLine = c.PredictedGainOverLine,
                confidence = c.Prediction.PredictionConfidence,
                rank = c.Prediction.PredictedRank
            },
            selectionContext = new
            {
                candidatePoolSize = c.CandidatePoolSize,
                windowCandidateCount = c.WindowCandidateCount,
                lineBeatingCandidateCount = c.LineBeatingCandidateCount,
                fetchedCandidateCount = c.FetchedCandidateCount,
                candidatesAfterBrutalityCount = c.CandidatesAfterBrutalityCount,
                candidateAttemptLimit = c.CandidateAttemptLimit,
                notes = c.CandidateSelectionNotes
            },
            actual = snap == null
                ? null
                : new
                {
                    displayName = snap.DisplayName,
                    sizeBytes = snap.SizeBytes,
                    sizeGiB = ToGiB(snap.SizeBytes),
                    kld = snap.Kld,
                    ppl = snap.Ppl,
                    lineKldAtActualSize = actualLine,
                    gainOverLine = actualGainOverLine,
                    insideSizeWindow = actualInsideSizeWindow,
                    beatLine = actualBeatLine,
                    sizeMissBytes,
                    kldMiss,
                    positiveKldShortfall = kldMiss.HasValue ? Math.Max(0d, kldMiss.Value) : (double?)null
                },
            anchors = new
            {
                higherDamageSmaller = ToAnchorLog(c.HigherDamageAnchor),
                lowerDamageLarger = ToAnchorLog(c.LowerDamageAnchor)
            }
        };
    }

    private static string DescribeBitSpace(TensorConfig config)
    {
        var baseQuant = BaselineQuants.FromId(config.BaseQuant);
        var overrides = DescribeOverrides(config);
        return string.IsNullOrWhiteSpace(overrides)
            ? $"base={baseQuant.Names[0]}({baseQuant.BitRange}b); overrides=inherit-all"
            : $"base={baseQuant.Names[0]}({baseQuant.BitRange}b); overrides={overrides}";
    }

    private static string DescribeOverrides(TensorConfig config)
    {
        var parts = new List<string>();
        AddOverride(parts, "E", config.Embeddings);
        AddOverride(parts, "H", config.LmHead);
        AddOverride(parts, "Q", config.AttnQ);
        AddOverride(parts, "K", config.AttnKV);
        AddOverride(parts, "O", config.AttnOutput);
        AddOverride(parts, "U", config.FfnUpGate);
        AddOverride(parts, "D", config.FfnDown);
        AddOverride(parts, "X", config.MoeExperts);
        AddOverride(parts, "R", config.MoeRouter);
        return string.Join(", ", parts);
    }

    private static void AddOverride(List<string> parts, string groupToken, byte storedSlot)
    {
        if (BaselineQuants.IsNullTensorConfigGroupSlot(storedSlot))
            return;

        var baseline = BaselineQuants.DecodeTensorConfigGroupSlotToBaseline(storedSlot);
        parts.Add($"{groupToken}:{baseline.Names[0]}({baseline.BitRange}b)");
    }

    private static string ResolveGgufDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Path.Combine(Cache.ModelMagicQuantDirectory!, "GGUF");

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Path.Combine(Cache.MagicQuantDirectory!, "GGUF");

        return Path.Combine(Directory.GetCurrentDirectory(), "GGUF");
    }

    private static ulong AddPercent(ulong bytes, double percent)
    {
        if (percent <= 0d)
            return bytes;

        double multiplier = 1d + (percent / 100d);
        double result = bytes * multiplier;
        if (result >= ulong.MaxValue)
            return ulong.MaxValue;

        return (ulong)Math.Round(result, MidpointRounding.AwayFromZero);
    }

    private static ulong Distance(ulong left, ulong right) => left >= right ? left - right : right - left;

    private static string NormalizePublicEliminationReason(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            return string.Empty;

        if (reason.Contains("dominance", StringComparison.OrdinalIgnoreCase))
            return "dominance";

        if (reason.Contains("spacing", StringComparison.OrdinalIgnoreCase))
            return "spacing";

        if (reason.Contains("strict", StringComparison.OrdinalIgnoreCase))
            return "strict-dominance";

        return reason.Trim().ToLowerInvariant();
    }

    private static bool Dominates(BenchmarkSnapshotRecord better, BenchmarkSnapshotRecord worse)
    {
        bool sameOrSmaller = better.SizeBytes <= worse.SizeBytes;
        bool strictlyLowerKld = better.Kld + Config.SelectionMinimumKldImprovementEpsilon < worse.Kld;
        return sameOrSmaller && strictlyLowerKld;
    }

    private static double ToGiB(ulong bytes) => bytes / 1024d / 1024d / 1024d;

    private sealed class AdjacentAnchorPair
    {
        public BenchmarkSnapshotRecord LowerDamageLarger { get; init; } = default!;
        public BenchmarkSnapshotRecord HigherDamageSmaller { get; init; } = default!;
    }

    private sealed class BrutalityAnalysis
    {
        public bool Passed { get; init; }
        public double FractionFromSmallAnchor { get; init; }
        public double RequiredGain { get; init; }
        public string Explanation { get; init; } = string.Empty;
    }

    private sealed class ValidationMetrics
    {
        public double ActualLineKld { get; init; }
        public double ActualGainOverLine { get; init; }
        public double KldMiss { get; init; }
        public long SizeMissBytes { get; init; }
        public bool InsideWindow { get; init; }
        public bool BeatsLine { get; init; }
    }

    private sealed class SelectionPhaseDiagnostic
    {
        public string Phase { get; init; } = string.Empty;
        public string WindowLabel { get; init; } = string.Empty;
        public int PhaseWindowIndex { get; init; }
        public int PhaseWindowCount { get; init; }
        public object? HigherDamageSmaller { get; init; }
        public object? LowerDamageLarger { get; init; }
        public ulong WindowMinSizeBytes { get; init; }
        public ulong WindowMaxSizeBytes { get; init; }
        public double WindowSizeGiB { get; init; }
        public long CandidatePoolSize { get; init; }
        public long WindowCandidateCount { get; init; }
        public long LineBeatingCandidateCount { get; init; }
        public int FetchedCandidateCount { get; init; }
        public int CandidatesAfterBrutalityCount { get; init; }
        public int SelectedForValidationCount { get; init; }
        public int CandidateAttemptLimit { get; init; }
        public int QueryFetchLimit { get; init; }
        public IReadOnlyList<CandidatePreviewLog> TopCandidates { get; init; } = Array.Empty<CandidatePreviewLog>();
        public IReadOnlyList<CandidatePreviewLog> RejectedByBrutalityPreview { get; init; } = Array.Empty<CandidatePreviewLog>();
        public IReadOnlyList<string> Notes { get; init; } = Array.Empty<string>();
    }

    private sealed class CandidatePreviewLog
    {
        public int AttemptOrder { get; init; }
        public string Key { get; init; } = string.Empty;
        public string DisplayName { get; init; } = string.Empty;
        public ulong PredictedSizeBytes { get; init; }
        public double PredictedSizeGiB { get; init; }
        public double PredictedKld { get; init; }
        public double LinearExpectedKld { get; init; }
        public double PredictedGainOverLine { get; init; }
        public double PredictionConfidence { get; init; }
        public ulong? PredictionRank { get; init; }
        public string BaseQuant { get; init; } = string.Empty;
        public byte BaseBitRange { get; init; }
        public string BitSpace { get; init; } = string.Empty;
        public string OverrideSummary { get; init; } = string.Empty;
        public bool BrutalityPassed { get; init; }
        public double BrutalityFractionFromSmallAnchor { get; init; }
        public double BrutalityRequiredGain { get; init; }
        public string BrutalityExplanation { get; init; } = string.Empty;
    }
}