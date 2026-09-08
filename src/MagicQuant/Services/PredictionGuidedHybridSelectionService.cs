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
    private readonly SmartBaselineTuningFallbackService _smartFallbackService;

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
        _smartFallbackService = new SmartBaselineTuningFallbackService(repository);
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

        var predictedAnchors = await _predictedStore.GetPredictedAnchorRowsAsync(ct);
        PrintPredictionAnchorFrontier(predictedAnchors, current, "Prediction Anchor Frontier");

        var strict = await RunStrictDominanceReplacementAsync(current, predictedAnchors, eliminationRecords, validationFailures, validationAttempts, phaseDiagnostics, ct);
        current = MergeAndDominanceFilter(current, strict.AcceptedSnapshots, eliminationRecords, "strict predicted hybrid dominance validated by real benchmark");

        predictedAnchors = AugmentPredictedAnchorsWithAcceptedValidationRows(predictedAnchors, validationAttempts);

        var near = await RunNearBaselineReplacementAsync(current, predictedAnchors, eliminationRecords, validationFailures, validationAttempts, phaseDiagnostics, ct);
        current = MergeAndDominanceFilter(current, near.AcceptedSnapshots, eliminationRecords, "near-baseline size-premium replacement validated by real benchmark");

        predictedAnchors = AugmentPredictedAnchorsWithAcceptedValidationRows(predictedAnchors, validationAttempts);

        var interior = await RunInteriorSubspaceDiscoveryAsync(current, predictedAnchors, validationFailures, validationAttempts, phaseDiagnostics, ct);
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
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        List<SelectionPhaseDiagnostic> phaseDiagnostics,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 1: Strict Hybrid Dominance[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"[grey]Strict dominance retry policy:[/] max attempts per anchor=[cyan]{Config.SelectionMaxFallbackAttemptsPerAnchor:N0}[/], epsilon=[cyan]{Config.SelectionMinimumKldImprovementEpsilon:0.########}[/], validate all anomaly/Q8 top-N after first success=[cyan]{Config.SelectionValidateAllAnomalyStrictCandidatesAfterSuccess}[/]");

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

            var predictedAnchor = await _predictedStore.FindPredictedAnchorForRealAnchorAsync(anchor, predictedAnchors, ct);
            if (predictedAnchor == null)
            {
                AnsiConsole.MarkupLine($"[yellow]Skipping strict prediction-space discovery:[/] no predicted virtual anchor matched real anchor [cyan]{Markup.Escape(anchor.DisplayName)}[/].");
                phaseDiagnostics.Add(new SelectionPhaseDiagnostic
                {
                    Phase = "StrictDominanceReplacement",
                    WindowLabel = $"strict <= {anchor.DisplayName}",
                    HigherDamageSmaller = ToAnchorLog(anchor),
                    LowerDamageLarger = ToAnchorLog(anchor),
                    WindowMinSizeBytes = 0,
                    WindowMaxSizeBytes = anchor.SizeBytes,
                    CandidateAttemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor + Config.SelectionSmartFallbackAttemptsPerFailure,
                    Notes = ["Skipped because no predicted virtual anchor row was available. DuckDB preselection intentionally does not fall back to real anchor KLD/size. Smart baseline fallback may still inspect SQLite isolation truth."]
                });

                await TryRunSmartStrictFallbackAsync(anchor, accepted, eliminations, validationFailures, validationAttempts, ct);
                continue;
            }

            int attemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor;
            ulong predictedAnchorSizeBytes = predictedAnchor.PredictedSizeBytes;
            ulong realAnchorSizeBytes = anchor.SizeBytes;
            ulong effectiveStrictMaxSizeBytes = Math.Min(predictedAnchorSizeBytes, realAnchorSizeBytes);

            long predictedPoolCount = await _predictedStore.CountStrictDominanceCandidatesAsync(predictedAnchor, predictedAnchorSizeBytes, ct);
            long poolCount = await _predictedStore.CountStrictDominanceCandidatesAsync(predictedAnchor, effectiveStrictMaxSizeBytes, ct);
            long deterministicEligibleCount = poolCount;
            long rejectedByRealStrictSizeCeiling = Math.Max(0, predictedPoolCount - deterministicEligibleCount);
            long rejectedByOtherPhaseDeterministicRules = 0;

            bool diversityEligible = ShouldUseDiversityForWindow(anchor, anchor, predictedAnchor, predictedAnchor);
            int strictScanLimit = ResolveValidationScanLimit(attemptLimit, poolCount, diversityEligible);
            var strictRows = await _predictedStore.QueryStrictDominanceCandidatesAsync(predictedAnchor, effectiveStrictMaxSizeBytes, strictScanLimit, ct);
            var rankedStrictCandidates = strictRows.Select((x, i) => new HybridSelectionCandidate
            {
                Prediction = x,
                Reason = HybridSelectionReason.StrictDominanceReplacement,
                LowerDamageAnchor = anchor,
                HigherDamageAnchor = anchor,
                LowerDamagePredictionAnchor = predictedAnchor,
                HigherDamagePredictionAnchor = predictedAnchor,
                PredictionWindowMinSizeBytes = 0,
                PredictionWindowMaxSizeBytes = effectiveStrictMaxSizeBytes,
                WindowMinSizeBytes = 0,
                WindowMaxSizeBytes = effectiveStrictMaxSizeBytes,
                LinearExpectedKld = predictedAnchor.PredictedKld,
                PredictedGainOverLine = predictedAnchor.PredictedKld - x.PredictedKld,
                AttemptOrder = i + 1,
                WindowLabel = $"strict <= {anchor.DisplayName}",
                CandidatePoolSize = poolCount,
                WindowCandidateCount = poolCount,
                LineBeatingCandidateCount = poolCount,
                FetchedCandidateCount = strictRows.Count,
                CandidatesAfterBrutalityCount = strictRows.Count,
                CandidateAttemptLimit = attemptLimit,
                PhaseWindowIndex = 1,
                PhaseWindowCount = 1,
                RawSelectionRank = i + 1,
                CandidateSelectionNotes =
                [
                    "Strict DuckDB query uses prediction-space KLD, but predicted-size eligibility is capped by the real anchor size because MagicQuant size prediction is trusted/exact.",
                    $"predictedAnchorSizeBytes={predictedAnchorSizeBytes:N0}; realAnchorSizeBytes={realAnchorSizeBytes:N0}; effectiveStrictMaxSizeBytes={effectiveStrictMaxSizeBytes:N0}; deterministicEligibleCount={deterministicEligibleCount:N0}; rejectedByRealStrictSizeCeiling={rejectedByRealStrictSizeCeiling:N0}; rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}."
                ]
            }).ToList();

            var selection = SelectValidationCandidates(
                rankedStrictCandidates,
                attemptLimit,
                anchor,
                anchor,
                predictedAnchor,
                predictedAnchor,
                "StrictDominanceReplacement",
                diversityEligible);
            var candidates = selection.Candidates.ToList();

            var strictNotes = new List<string>
            {
                "Strict KLD eligibility remains prediction-space, but strict predicted-size eligibility is capped by the real anchor size because MagicQuant size prediction is trusted/exact.",
                $"Prediction anchor={predictedAnchor.DisplayName}; predictedKld={predictedAnchor.PredictedKld:0.000000}; predictedAnchorSizeBytes={predictedAnchorSizeBytes:N0}; realKld={anchor.Kld:0.000000}; realAnchorSizeBytes={realAnchorSizeBytes:N0}; effectiveStrictMaxSizeBytes={effectiveStrictMaxSizeBytes:N0}; deterministicEligibleCount={deterministicEligibleCount:N0}; rejectedByRealStrictSizeCeiling={rejectedByRealStrictSizeCeiling:N0}; rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}."
            };
            bool anomalyStrictMode = IsQ8Anchor(anchor) || candidates.Any(x => Math.Abs(x.Prediction.AnomalyAdjustmentKld) > 1e-12);
            bool validateAllAfterSuccess = anomalyStrictMode && Config.SelectionValidateAllAnomalyStrictCandidatesAfterSuccess;
            if (anomalyStrictMode)
            {
                strictNotes.Add(validateAllAfterSuccess
                    ? "Q8/anomaly strict mode: legacy validate-all-after-success is enabled, so all fetched candidates up to the configured attempt limit may be built before choosing by actual KLD/size truth."
                    : "Q8/anomaly strict mode: stop after the first candidate validates for this anchor. Set candidate_selection.validate_all_anomaly_strict_candidates_after_success=true to restore legacy top-N validation.");
            }

            var diag = new SelectionPhaseDiagnostic
            {
                Phase = "StrictDominanceReplacement",
                WindowLabel = $"strict <= {anchor.DisplayName}",
                HigherDamageSmaller = ToAnchorLog(anchor),
                LowerDamageLarger = ToAnchorLog(anchor),
                PredictionHigherDamageSmaller = ToPredictionAnchorLog(predictedAnchor),
                PredictionLowerDamageLarger = ToPredictionAnchorLog(predictedAnchor),
                PredictionWindowMinSizeBytes = 0,
                PredictionWindowMaxSizeBytes = effectiveStrictMaxSizeBytes,
                WindowMinSizeBytes = 0,
                WindowMaxSizeBytes = effectiveStrictMaxSizeBytes,
                CandidatePoolSize = poolCount,
                WindowCandidateCount = poolCount,
                LineBeatingCandidateCount = poolCount,
                FetchedCandidateCount = strictRows.Count,
                CandidatesAfterBrutalityCount = strictRows.Count,
                SelectedForValidationCount = candidates.Count,
                CandidateAttemptLimit = attemptLimit,
                QueryFetchLimit = strictScanLimit,
                DiversityEnabled = selection.DiversityEnabled,
                DiversityMode = selection.Mode,
                DiversityScanLimit = strictScanLimit,
                DiversityScanFetched = strictRows.Count,
                CandidateFamilyCount = selection.CandidateFamilyCount,
                SelectedFamilyCount = selection.SelectedFamilyCount,
                SelectedFamilyKeys = selection.SelectedFamilyKeys,
                DiversitySelectionStrategy = selection.SelectionStrategy,
                DiversityOverflowCount = selection.OverflowCount,
                DiversitySizeFloorStartBytes = selection.SizeFloorStartBytes,
                DiversitySizeFloorEndBytes = selection.SizeFloorEndBytes,
                TopCandidates = candidates.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                Notes = strictNotes.Concat(selection.Notes).ToList()
            };
            phaseDiagnostics.Add(diag);

            AnsiConsole.MarkupLine($"[grey]Strict candidates for {Markup.Escape(anchor.DisplayName)}:[/] pool={poolCount:N0}, predictedPoolCount={predictedPoolCount:N0}, deterministicEligibleCount={deterministicEligibleCount:N0}, scanLimit={strictScanLimit:N0}, scanFetched={strictRows.Count:N0}, afterBrutality={strictRows.Count:N0}, diversity={Markup.Escape(selection.Mode)}, selectionStrategy={Markup.Escape(selection.SelectionStrategy)}, candidateFamilies={selection.CandidateFamilyCount:N0}, selectedFamilies={selection.SelectedFamilyCount:N0}, selected={candidates.Count:N0}/{attemptLimit:N0}, overflowCount={selection.OverflowCount:N0}, sizeFloorStart={selection.SizeFloorStartBytes?.ToString("N0") ?? "n/a"}, predictedAnchorSizeBytes={predictedAnchorSizeBytes:N0}, realAnchorSizeBytes={realAnchorSizeBytes:N0}, effectiveStrictMaxSizeBytes={effectiveStrictMaxSizeBytes:N0}, rejectedByRealStrictSizeCeiling={rejectedByRealStrictSizeCeiling:N0}, rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}, q8/anomaly-mode={anomalyStrictMode}, validate-all-after-success={validateAllAfterSuccess}");
            PrintSelectedCandidateFamilySummary(candidates);
            PrintSelectionLadderNotes(selection.Notes);

            if (deterministicEligibleCount == 0)
                AnsiConsole.MarkupLine($"[yellow]Strict dominance skipped builds for {Markup.Escape(anchor.DisplayName)}:[/] no physically eligible predicted candidates remained after deterministic size/KLD filters.");

            if (candidates.Count == 0)
            {
                await TryRunSmartStrictFallbackAsync(anchor, accepted, eliminations, validationFailures, validationAttempts, ct);
                continue;
            }

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
                    if (!validateAllAfterSuccess)
                        break;

                    continue;
                }

                validationFailures.Add(validation);
            }

            if (acceptedForAnchor.Count == 0)
            {
                AnsiConsole.MarkupLine($"[grey]No strict predicted replacement validated for anchor:[/] {Markup.Escape(anchor.DisplayName)}");
                await TryRunSmartStrictFallbackAsync(anchor, accepted, eliminations, validationFailures, validationAttempts, ct);
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


    private async Task<bool> TryRunSmartStrictFallbackAsync(
        BenchmarkSnapshotRecord anchor,
        List<BenchmarkSnapshotRecord> accepted,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        CancellationToken ct)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return false;

        var smartCandidates = (await _smartFallbackService.BuildStrictDominanceCandidatesAsync(anchor, 1, 1, ct)).ToList();
        if (smartCandidates.Count == 0)
            return false;

        var acceptedForAnchor = new List<CandidateValidationResult>();

        foreach (var candidate in smartCandidates)
        {
            var validation = await BuildAndValidateSingleAsync(
                candidate,
                snapshot => snapshot.SizeBytes <= anchor.SizeBytes &&
                            snapshot.Kld + Config.SelectionMinimumKldImprovementEpsilon < anchor.Kld,
                $"smart fallback must be <= {anchor.SizeBytes:N0} bytes and lower KLD than {anchor.DisplayName}",
                ct);

            validationAttempts.Add(validation);

            if (validation.Accepted && validation.Snapshot != null)
            {
                acceptedForAnchor.Add(validation);
                break;
            }

            validationFailures.Add(validation);
        }

        if (acceptedForAnchor.Count == 0)
        {
            AnsiConsole.MarkupLine($"[grey]Smart strict fallback found no validated replacement for anchor:[/] {Markup.Escape(anchor.DisplayName)}");
            return false;
        }

        var chosen = ChooseBestStrictDominanceCandidate(anchor, acceptedForAnchor);
        accepted.Add(chosen.Snapshot!);
        eliminations.Add(new BaselineEliminationRecord
        {
            Eliminated = anchor,
            Eliminator = chosen.Snapshot!,
            Reason = "smart baseline-tuning strict dominance fallback: real benchmark validated lower KLD at same-or-smaller size"
        });

        AnsiConsole.MarkupLine("[green]Smart strict fallback candidate selected:[/]");
        AnsiConsole.MarkupLine($"[grey]  anchor=[/] [cyan]{Markup.Escape(anchor.DisplayName)}[/]");
        AnsiConsole.MarkupLine($"[grey]  chosen=[/] [cyan]{Markup.Escape(chosen.Snapshot!.DisplayName)}[/]");
        AnsiConsole.MarkupLine($"[grey]  actualKld=[/] [cyan]{chosen.Snapshot.Kld:0.000000}[/]");
        AnsiConsole.MarkupLine($"[grey]  actualSizeBytes=[/] [cyan]{chosen.Snapshot.SizeBytes:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  gainVsAnchor=[/] [cyan]{anchor.Kld - chosen.Snapshot.Kld:0.000000}[/]");
        return true;
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

    private static IReadOnlyList<PredictedAnchorRow> AugmentPredictedAnchorsWithAcceptedValidationRows(
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
        IReadOnlyList<CandidateValidationResult> validationAttempts)
    {
        if (validationAttempts.Count == 0)
            return predictedAnchors;

        var result = new List<PredictedAnchorRow>(predictedAnchors);
        var knownKeys = result
            .Select(x => x.ConfigKey)
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .ToHashSet(StringComparer.Ordinal);

        int added = 0;

        foreach (var validation in validationAttempts)
        {
            if (!validation.Accepted || validation.Snapshot == null)
                continue;

            var prediction = validation.Candidate.Prediction;
            if (!prediction.IsPredictable || !prediction.IsSizePredictable)
                continue;

            var predictionSpaceConfig = CanonicalizeSelectionConfigForPredictionSpace(prediction.Config);
            string key = TensorConfigIdentity.ToKey(predictionSpaceConfig);
            if (!knownKeys.Add(key))
                continue;

            var sourceBaseline = HybridBenchmarkRepository.ResolveSourceBaselineForProvider(validation.Snapshot.Quant);

            result.Add(new PredictedAnchorRow
            {
                Config = predictionSpaceConfig,
                ConfigKey = key,
                DisplayName = validation.Snapshot.DisplayName,
                BaselineCanonicalKey = sourceBaseline.CanonicalKey,
                RuntimeBaselineId = sourceBaseline.UniqueId,
                PredictedKld = prediction.PredictedKld,
                PredictedSizeBytes = prediction.PredictedSizeBytes,
                PredictionConfidence = prediction.PredictionConfidence,
                PredictionRank = prediction.PredictedRank ?? ulong.MaxValue,
                IsVirtualPredictionAnchor = false
            });

            added++;
        }

        if (added > 0)
        {
            AnsiConsole.MarkupLine(
                $"[grey]Prediction anchor frontier augmented from accepted validation rows:[/] [cyan]{added:N0}[/] phase-local anchor(s) added for smart-fallback / accepted hybrid coordinates outside the pruned DuckDB row set.");
        }

        return added == 0 ? predictedAnchors : result;
    }

    private static TensorConfig CanonicalizeSelectionConfigForPredictionSpace(TensorConfig config)
    {
        if (config.BaseQuant == BaselineQuants.Q8_0.UniqueId)
            return config;

        var baseBaseline = BaselineQuants.FromId(config.BaseQuant);
        byte inheritedBaseSlot = BaselineQuants.EncodeTensorConfigGroupSlot(baseBaseline);

        return new TensorConfig(
            baseQuant: BaselineQuants.Q8_0.UniqueId,
            embeddings: CanonicalizeSelectionPredictionSlot(TReg.Embeddings, config.Embeddings, inheritedBaseSlot),
            lmHead: CanonicalizeSelectionPredictionSlot(TReg.LmHead, config.LmHead, inheritedBaseSlot),
            attnQ: CanonicalizeSelectionPredictionSlot(TReg.AttnQ, config.AttnQ, inheritedBaseSlot),
            attnKV: CanonicalizeSelectionPredictionSlot(TReg.AttnKV, config.AttnKV, inheritedBaseSlot),
            attnOutput: CanonicalizeSelectionPredictionSlot(TReg.AttnOutput, config.AttnOutput, inheritedBaseSlot),
            ffnUpGate: CanonicalizeSelectionPredictionSlot(TReg.FfnUpGate, config.FfnUpGate, inheritedBaseSlot),
            ffnDown: CanonicalizeSelectionPredictionSlot(TReg.FfnDown, config.FfnDown, inheritedBaseSlot),
            moeExperts: CanonicalizeSelectionPredictionSlot(TReg.MoeExperts, config.MoeExperts, inheritedBaseSlot),
            moeRouter: CanonicalizeSelectionPredictionSlot(TReg.MoeRouter, config.MoeRouter, inheritedBaseSlot));
    }

    private static byte CanonicalizeSelectionPredictionSlot(TensorGroup group, byte storedValue, byte inheritedBaseSlot)
    {
        if (Cache.UnusedTensorGroups.Any(x => x.UniqueId == group.UniqueId))
            return BaselineQuants.TensorConfigNullSlotValue;

        return BaselineQuants.IsNullTensorConfigGroupSlot(storedValue)
            ? inheritedBaseSlot
            : storedValue;
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
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
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
        int fetchLimit = ResolveValidationScanLimit(attemptLimit, long.MaxValue, diversityEligible: Config.SelectionDiversifyValidationCandidates);

        AnsiConsole.MarkupLine($"[grey]Near-baseline neighbor pairs:[/] [cyan]{pairs.Count:N0}[/] | size premium=[cyan]{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}%[/] | max scan/window=[cyan]{fetchLimit:N0}[/] | validation attempts/window=[cyan]{attemptLimit:N0}[/]");

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

            ulong realMin = lowerSizeHigherDamage.SizeBytes;
            ulong realMax = AddPercent(realMin, Config.SelectionNearBaselineMaxSizeGrowthPercent);

            if (realMax > upperSizeLowerDamage.SizeBytes)
                realMax = upperSizeLowerDamage.SizeBytes;

            if (realMax <= realMin)
            {
                AnsiConsole.MarkupLine($"[grey]Skipping near-baseline pair with empty real window:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)} -> {Markup.Escape(upperSizeLowerDamage.DisplayName)}");
                continue;
            }

            var predictedLowerSizeHigherDamage = await _predictedStore.FindPredictedAnchorForRealAnchorAsync(lowerSizeHigherDamage, predictedAnchors, ct);
            var predictedUpperSizeLowerDamage = await _predictedStore.FindPredictedAnchorForRealAnchorAsync(upperSizeLowerDamage, predictedAnchors, ct);
            if (predictedLowerSizeHigherDamage == null || predictedUpperSizeLowerDamage == null)
            {
                AnsiConsole.MarkupLine($"[yellow]Skipping near-baseline prediction-space discovery:[/] missing predicted anchor for pair [cyan]{Markup.Escape(lowerSizeHigherDamage.DisplayName)}[/] -> [cyan]{Markup.Escape(upperSizeLowerDamage.DisplayName)}[/].");
                phaseDiagnostics.Add(new SelectionPhaseDiagnostic
                {
                    Phase = "NearBaselineReplacement",
                    WindowLabel = windowLabel,
                    PhaseWindowIndex = pairIndex + 1,
                    PhaseWindowCount = pairs.Count,
                    HigherDamageSmaller = ToAnchorLog(lowerSizeHigherDamage),
                    LowerDamageLarger = ToAnchorLog(upperSizeLowerDamage),
                    PredictionHigherDamageSmaller = ToPredictionAnchorLog(predictedLowerSizeHigherDamage),
                    PredictionLowerDamageLarger = ToPredictionAnchorLog(predictedUpperSizeLowerDamage),
                    WindowMinSizeBytes = realMin,
                    WindowMaxSizeBytes = realMax,
                    CandidateAttemptLimit = Config.SelectionMaxFallbackAttemptsPerAnchor + Config.SelectionSmartFallbackAttemptsPerFailure,
                    Notes = ["Skipped because one or both predicted virtual anchor rows were unavailable. DuckDB preselection intentionally does not fall back to real anchor KLD/size. Smart baseline fallback may still inspect SQLite isolation truth."]
                });

                await TryRunSmartNearFallbackAsync(lowerSizeHigherDamage, upperSizeLowerDamage, realMin, realMax, pairIndex + 1, pairs.Count, accepted, eliminations, validationFailures, validationAttempts, ct);
                continue;
            }

            ulong predictionMin = predictedLowerSizeHigherDamage.PredictedSizeBytes;
            ulong predictionMax = AddPercent(predictionMin, Config.SelectionNearBaselineMaxSizeGrowthPercent);

            if (predictionMax > predictedUpperSizeLowerDamage.PredictedSizeBytes)
                predictionMax = predictedUpperSizeLowerDamage.PredictedSizeBytes;

            if (predictionMax <= predictionMin || realMax <= realMin)
            {
                AnsiConsole.MarkupLine($"[grey]Skipping near-baseline pair with empty prediction/real window:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)} -> {Markup.Escape(upperSizeLowerDamage.DisplayName)}");
                await TryRunSmartNearFallbackAsync(lowerSizeHigherDamage, upperSizeLowerDamage, realMin, realMax, pairIndex + 1, pairs.Count, accepted, eliminations, validationFailures, validationAttempts, ct);
                continue;
            }

            LogPredictionAndRealPairLines(lowerSizeHigherDamage, upperSizeLowerDamage, predictedLowerSizeHigherDamage, predictedUpperSizeLowerDamage, "Near-baseline pair");

            long predictedWindowRows = await _predictedStore.CountPredictedHybridCandidatesInSizeWindowAsync(predictionMin, predictionMax, ct);
            long predictedPoolCount = await _predictedStore.CountBetterThanLinearCandidatesAsync(predictedLowerSizeHigherDamage, predictedUpperSizeLowerDamage, predictionMin, predictionMax, ct);
            long deterministicWindowRows = await CountPredictedRowsInIntersectedSizeWindowAsync(predictionMin, predictionMax, realMin, realMax, ct);
            long phaseSizeEligiblePool = await _predictedStore.CountBetterThanLinearCandidatesAsync(
                predictedLowerSizeHigherDamage,
                predictedUpperSizeLowerDamage,
                predictionMin,
                predictionMax,
                realMin,
                realMax,
                ct);
            long rejectedByRealSizeWindow = Math.Max(0, predictedPoolCount - phaseSizeEligiblePool);
            bool diversityEligible = ShouldUseDiversityForWindow(lowerSizeHigherDamage, upperSizeLowerDamage, predictedLowerSizeHigherDamage, predictedUpperSizeLowerDamage);
            fetchLimit = ResolveValidationScanLimit(attemptLimit, phaseSizeEligiblePool, diversityEligible);
            var rawCandidates = (await _predictedStore.QueryBetterThanLinearCandidatesAsync(
                lowerSizeHigherDamage,
                upperSizeLowerDamage,
                predictedLowerSizeHigherDamage,
                predictedUpperSizeLowerDamage,
                predictionMin,
                predictionMax,
                realMin,
                realMax,
                HybridSelectionReason.NearBaselineOnePercentReplacement,
                windowLabel,
                fetchLimit,
                ct)).ToList();

            var brutalityAnalyses = rawCandidates
                .Select((x, rawIndex) => new { Candidate = x, Brutality = AnalyzeNearLowerAnchorBrutality(x), RawRank = rawIndex + 1 })
                .ToList();

            int afterBrutalityCount = brutalityAnalyses.Count(y => y.Brutality.Passed);
            long deterministicEligibleCount = afterBrutalityCount;
            long rejectedByOtherPhaseDeterministicRules = Math.Max(0, rawCandidates.Count - afterBrutalityCount);
            var rankedCandidates = brutalityAnalyses
                .Where(x => x.Brutality.Passed)
                .Select(x => AttachSelectionDiagnostics(
                    x.Candidate,
                    poolSize: phaseSizeEligiblePool,
                    windowCandidateCount: deterministicWindowRows,
                    lineBeatingCandidateCount: phaseSizeEligiblePool,
                    fetchedCandidateCount: rawCandidates.Count,
                    candidatesAfterBrutalityCount: afterBrutalityCount,
                    candidateAttemptLimit: attemptLimit,
                    phaseWindowIndex: pairIndex + 1,
                    phaseWindowCount: pairs.Count,
                    notes: [x.Brutality.Explanation],
                    rawSelectionRank: x.RawRank))
                .ToList();

            var selection = SelectValidationCandidates(
                rankedCandidates,
                attemptLimit,
                lowerSizeHigherDamage,
                upperSizeLowerDamage,
                predictedLowerSizeHigherDamage,
                predictedUpperSizeLowerDamage,
                "NearBaselineReplacement",
                diversityEligible);
            var candidates = selection.Candidates.ToList();

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
                PredictionHigherDamageSmaller = ToPredictionAnchorLog(predictedLowerSizeHigherDamage),
                PredictionLowerDamageLarger = ToPredictionAnchorLog(predictedUpperSizeLowerDamage),
                PredictionWindowMinSizeBytes = predictionMin,
                PredictionWindowMaxSizeBytes = predictionMax,
                WindowMinSizeBytes = realMin,
                WindowMaxSizeBytes = realMax,
                WindowSizeGiB = ToGiB(realMax > realMin ? realMax - realMin : 0),
                CandidatePoolSize = phaseSizeEligiblePool,
                PredictedPoolCount = predictedPoolCount,
                DeterministicEligibleCount = deterministicEligibleCount,
                RejectedByRealSizeWindow = rejectedByRealSizeWindow,
                RejectedByOtherPhaseDeterministicRules = rejectedByOtherPhaseDeterministicRules,
                WindowCandidateCount = deterministicWindowRows,
                LineBeatingCandidateCount = phaseSizeEligiblePool,
                FetchedCandidateCount = rawCandidates.Count,
                CandidatesAfterBrutalityCount = afterBrutalityCount,
                SelectedForValidationCount = candidates.Count,
                CandidateAttemptLimit = attemptLimit,
                QueryFetchLimit = fetchLimit,
                DiversityEnabled = selection.DiversityEnabled,
                DiversityMode = selection.Mode,
                DiversityScanLimit = fetchLimit,
                DiversityScanFetched = rawCandidates.Count,
                CandidateFamilyCount = selection.CandidateFamilyCount,
                SelectedFamilyCount = selection.SelectedFamilyCount,
                SelectedFamilyKeys = selection.SelectedFamilyKeys,
                DiversitySelectionStrategy = selection.SelectionStrategy,
                DiversityOverflowCount = selection.OverflowCount,
                DiversitySizeFloorStartBytes = selection.SizeFloorStartBytes,
                DiversitySizeFloorEndBytes = selection.SizeFloorEndBytes,
                TopCandidates = candidates.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                RejectedByBrutalityPreview = rejectedByBrutality,
                Notes = new[]
                {
                    "Near-baseline DuckDB discovery uses the predicted virtual KLD line, then deterministically caps candidate size to the real validation window before diversity/ladder selection. Real KLD is still used only after benchmark validation.",
                    $"predictedPoolCount={predictedPoolCount:N0}; phaseSizeEligiblePool={phaseSizeEligiblePool:N0}; deterministicEligibleCount={deterministicEligibleCount:N0}; rejectedByRealSizeWindow={rejectedByRealSizeWindow:N0}; rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}.",
                    $"Brutal zone fraction={Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###}; required gain fraction of pair KLD gap={Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap:0.###}."
                }.Concat(selection.Notes).ToList()
            };
            phaseDiagnostics.Add(diag);

            AnsiConsole.MarkupLine(
                $"[grey]Near-baseline window {pairIndex + 1:N0}/{pairs.Count:N0}:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)} -> {Markup.Escape(upperSizeLowerDamage.DisplayName)} " +
                $"| pred-window={predictionMin:N0}..{predictionMax:N0}, real-window={realMin:N0}..{realMax:N0}, predictedPoolCount={predictedPoolCount:N0}, phaseSizeEligiblePool={phaseSizeEligiblePool:N0}, deterministicEligibleCount={deterministicEligibleCount:N0}, rejectedByRealSizeWindow={rejectedByRealSizeWindow:N0}, rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}, rows-in-window={deterministicWindowRows:N0}, beat-line={phaseSizeEligiblePool:N0}, scanLimit={fetchLimit:N0}, scanFetched={rawCandidates.Count:N0}, after-brutality={afterBrutalityCount:N0}, diversity={Markup.Escape(selection.Mode)}, selectionStrategy={Markup.Escape(selection.SelectionStrategy)}, candidateFamilies={selection.CandidateFamilyCount:N0}, selectedFamilies={selection.SelectedFamilyCount:N0}, selected={candidates.Count:N0}/{attemptLimit:N0}, overflowCount={selection.OverflowCount:N0}, sizeFloorStart={selection.SizeFloorStartBytes?.ToString("N0") ?? "n/a"}");
            PrintSelectedCandidateFamilySummary(candidates);
            PrintSelectionLadderNotes(selection.Notes);

            if (deterministicEligibleCount == 0)
                AnsiConsole.MarkupLine($"[yellow]Near-baseline window skipped builds:[/] no physically eligible candidates remained after predicted line, real size window, and deterministic brutality filters.");

            if (rejectedByBrutality.Count > 0)
                AnsiConsole.MarkupLine($"[grey]  rejected by near-lower-anchor brutality preview:[/] [cyan]{rejectedByBrutality.Count:N0}[/] (see magicquant-selection-phase-diagnostics.json)");

            if (candidates.Count == 0)
            {
                await TryRunSmartNearFallbackAsync(lowerSizeHigherDamage, upperSizeLowerDamage, realMin, realMax, pairIndex + 1, pairs.Count, accepted, eliminations, validationFailures, validationAttempts, ct);
                continue;
            }

            bool acceptedThisPair = false;
            foreach (var candidate in candidates)
            {
                var validation = await BuildAndValidateSingleAsync(
                    candidate,
                    snapshot => snapshot.SizeBytes >= realMin &&
                                snapshot.SizeBytes <= realMax &&
                                BeatsLinearKldLine(snapshot.SizeBytes, snapshot.Kld, lowerSizeHigherDamage, upperSizeLowerDamage),
                    $"must land inside {realMin:N0}..{realMax:N0} bytes and beat the real linear KLD line",
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
                    acceptedThisPair = true;
                    break;
                }

                validationFailures.Add(validation);
            }

            if (!acceptedThisPair)
            {
                await TryRunSmartNearFallbackAsync(lowerSizeHigherDamage, upperSizeLowerDamage, realMin, realMax, pairIndex + 1, pairs.Count, accepted, eliminations, validationFailures, validationAttempts, ct);
            }
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }


    private async Task<bool> TryRunSmartNearFallbackAsync(
        BenchmarkSnapshotRecord lowerSizeHigherDamage,
        BenchmarkSnapshotRecord upperSizeLowerDamage,
        ulong realMin,
        ulong realMax,
        int phaseWindowIndex,
        int phaseWindowCount,
        List<BenchmarkSnapshotRecord> accepted,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        CancellationToken ct)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return false;

        var smartCandidates = (await _smartFallbackService.BuildNearBaselineCandidatesAsync(
            lowerSizeHigherDamage,
            upperSizeLowerDamage,
            realMin,
            realMax,
            phaseWindowIndex,
            phaseWindowCount,
            ct)).ToList();

        if (smartCandidates.Count == 0)
            return false;

        foreach (var candidate in smartCandidates)
        {
            var validation = await BuildAndValidateSingleAsync(
                candidate,
                snapshot => snapshot.SizeBytes >= realMin &&
                            snapshot.SizeBytes <= realMax &&
                            BeatsLinearKldLine(snapshot.SizeBytes, snapshot.Kld, lowerSizeHigherDamage, upperSizeLowerDamage),
                $"smart fallback must land inside {realMin:N0}..{realMax:N0} bytes and beat the real linear KLD line",
                ct);

            validationAttempts.Add(validation);

            if (validation.Accepted && validation.Snapshot != null)
            {
                accepted.Add(validation.Snapshot);
                eliminations.Add(new BaselineEliminationRecord
                {
                    Eliminated = lowerSizeHigherDamage,
                    Eliminator = validation.Snapshot,
                    Reason = $"smart baseline-tuning near-baseline fallback within +{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}% size premium"
                });

                AnsiConsole.MarkupLine("[green]Smart near-baseline fallback candidate selected:[/]");
                AnsiConsole.MarkupLine($"[grey]  lower anchor=[/] [cyan]{Markup.Escape(lowerSizeHigherDamage.DisplayName)}[/]");
                AnsiConsole.MarkupLine($"[grey]  upper anchor=[/] [cyan]{Markup.Escape(upperSizeLowerDamage.DisplayName)}[/]");
                AnsiConsole.MarkupLine($"[grey]  chosen=[/] [cyan]{Markup.Escape(validation.Snapshot.DisplayName)}[/]");
                AnsiConsole.MarkupLine($"[grey]  actualKld=[/] [cyan]{validation.Snapshot.Kld:0.000000}[/]");
                AnsiConsole.MarkupLine($"[grey]  actualSizeBytes=[/] [cyan]{validation.Snapshot.SizeBytes:N0}[/]");
                return true;
            }

            validationFailures.Add(validation);
        }

        AnsiConsole.MarkupLine($"[grey]Smart near-baseline fallback found no validated candidate for:[/] {Markup.Escape(lowerSizeHigherDamage.DisplayName)} -> {Markup.Escape(upperSizeLowerDamage.DisplayName)}");
        return false;
    }


    private async Task<PhaseValidationResult> RunInteriorSubspaceDiscoveryAsync(
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
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
        int interiorFetchLimit = ResolveValidationScanLimit(interiorAttemptLimit, long.MaxValue, diversityEligible: Config.SelectionDiversifyValidationCandidates);

        AnsiConsole.MarkupLine($"[grey]Interior neighbor pairs:[/] [cyan]{pairs.Count:N0}[/] | window fractions=[cyan]{Markup.Escape(string.Join(", ", fractions.Select(x => x.ToString("0.###"))))}[/] | candidates/window=[cyan]{Config.SelectionMaxCandidatesPerInteriorWindow:N0}[/] | fallback attempts/window=[cyan]{Config.SelectionMaxFallbackAttemptsPerAnchor:N0}[/] | validation attempts/window=[cyan]{interiorAttemptLimit:N0}[/] | max scan/window=[cyan]{interiorFetchLimit:N0}[/]");

        var allCandidates = new List<HybridSelectionCandidate>();
        int globalWindowIndex = 0;
        int estimatedWindowCount = pairs.Sum(pair => EstimateInteriorWindowCount(pair, fractions));

        for (int pairIndex = 0; pairIndex < pairs.Count; pairIndex++)
        {
            var pair = pairs[pairIndex];
            var predictedHigherDamageSmaller = await _predictedStore.FindPredictedAnchorForRealAnchorAsync(pair.HigherDamageSmaller, predictedAnchors, ct);
            var predictedLowerDamageLarger = await _predictedStore.FindPredictedAnchorForRealAnchorAsync(pair.LowerDamageLarger, predictedAnchors, ct);
            if (predictedHigherDamageSmaller == null || predictedLowerDamageLarger == null)
            {
                AnsiConsole.MarkupLine($"[yellow]Skipping interior prediction-space discovery:[/] missing predicted anchor for pair [cyan]{Markup.Escape(pair.HigherDamageSmaller.DisplayName)}[/] -> [cyan]{Markup.Escape(pair.LowerDamageLarger.DisplayName)}[/].");
                continue;
            }

            ulong realLowSize = pair.HigherDamageSmaller.SizeBytes;
            ulong realHighSize = pair.LowerDamageLarger.SizeBytes;
            ulong predictionLowSize = predictedHigherDamageSmaller.PredictedSizeBytes;
            ulong predictionHighSize = predictedLowerDamageLarger.PredictedSizeBytes;

            if (realHighSize <= realLowSize || predictionHighSize <= predictionLowSize)
                continue;

            LogPredictionAndRealPairLines(pair.HigherDamageSmaller, pair.LowerDamageLarger, predictedHigherDamageSmaller, predictedLowerDamageLarger, "Interior pair");

            ulong realSpan = realHighSize - realLowSize;
            ulong predictionSpan = predictionHighSize - predictionLowSize;
            ulong realCursor = realLowSize;
            ulong predictionCursor = predictionLowSize;

            for (int i = 0; i < fractions.Count; i++)
            {
                double fraction = fractions[i];
                if (fraction <= 0d)
                    continue;

                ulong realWidth = (ulong)Math.Round(realSpan * fraction, MidpointRounding.AwayFromZero);
                ulong predictionWidth = (ulong)Math.Round(predictionSpan * fraction, MidpointRounding.AwayFromZero);
                if (realWidth == 0 || predictionWidth == 0)
                    continue;

                ulong realMin = realCursor;
                ulong realMax = Math.Min(realHighSize, realCursor + realWidth);
                ulong predictionMin = predictionCursor;
                ulong predictionMax = Math.Min(predictionHighSize, predictionCursor + predictionWidth);

                if (realMax <= realMin || predictionMax <= predictionMin)
                    continue;

                globalWindowIndex++;
                string windowLabel = $"interior {i + 1}: {pair.HigherDamageSmaller.DisplayName} -> {pair.LowerDamageLarger.DisplayName}";
                long predictedWindowRows = await _predictedStore.CountPredictedHybridCandidatesInSizeWindowAsync(predictionMin, predictionMax, ct);
                long predictedPoolCount = await _predictedStore.CountBetterThanLinearCandidatesAsync(predictedHigherDamageSmaller, predictedLowerDamageLarger, predictionMin, predictionMax, ct);
                long deterministicWindowRows = await CountPredictedRowsInIntersectedSizeWindowAsync(predictionMin, predictionMax, realMin, realMax, ct);
                long phaseSizeEligiblePool = await _predictedStore.CountBetterThanLinearCandidatesAsync(
                    predictedHigherDamageSmaller,
                    predictedLowerDamageLarger,
                    predictionMin,
                    predictionMax,
                    realMin,
                    realMax,
                    ct);
                long rejectedByRealSizeWindow = Math.Max(0, predictedPoolCount - phaseSizeEligiblePool);
                bool diversityEligible = ShouldUseDiversityForWindow(pair.HigherDamageSmaller, pair.LowerDamageLarger, predictedHigherDamageSmaller, predictedLowerDamageLarger);
                interiorFetchLimit = ResolveValidationScanLimit(interiorAttemptLimit, phaseSizeEligiblePool, diversityEligible);

                var rawCandidates = (await _predictedStore.QueryBetterThanLinearCandidatesAsync(
                    pair.HigherDamageSmaller,
                    pair.LowerDamageLarger,
                    predictedHigherDamageSmaller,
                    predictedLowerDamageLarger,
                    predictionMin,
                    predictionMax,
                    realMin,
                    realMax,
                    HybridSelectionReason.InteriorSubspaceDiscovery,
                    windowLabel,
                    interiorFetchLimit,
                    ct)).ToList();

                var brutalityAnalyses = rawCandidates
                    .Select((x, rawIndex) => new { Candidate = x, Brutality = AnalyzeNearLowerAnchorBrutality(x), RawRank = rawIndex + 1 })
                    .ToList();

                int afterBrutalityCount = brutalityAnalyses.Count(y => y.Brutality.Passed);
                long deterministicEligibleCount = afterBrutalityCount;
                long rejectedByOtherPhaseDeterministicRules = Math.Max(0, rawCandidates.Count - afterBrutalityCount);

                var rankedCandidates = brutalityAnalyses
                    .Where(x => x.Brutality.Passed)
                    .Select(x => AttachSelectionDiagnostics(
                        x.Candidate,
                        poolSize: phaseSizeEligiblePool,
                        windowCandidateCount: deterministicWindowRows,
                        lineBeatingCandidateCount: phaseSizeEligiblePool,
                        fetchedCandidateCount: rawCandidates.Count,
                        candidatesAfterBrutalityCount: afterBrutalityCount,
                        candidateAttemptLimit: interiorAttemptLimit,
                        phaseWindowIndex: globalWindowIndex,
                        phaseWindowCount: estimatedWindowCount,
                        notes: [x.Brutality.Explanation],
                        rawSelectionRank: x.RawRank))
                    .ToList();

                var selection = SelectValidationCandidates(
                    rankedCandidates,
                    interiorAttemptLimit,
                    pair.HigherDamageSmaller,
                    pair.LowerDamageLarger,
                    predictedHigherDamageSmaller,
                    predictedLowerDamageLarger,
                    "InteriorSubspaceDiscovery",
                    diversityEligible);
                var kept = selection.Candidates.ToList();

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
                    PredictionHigherDamageSmaller = ToPredictionAnchorLog(predictedHigherDamageSmaller),
                    PredictionLowerDamageLarger = ToPredictionAnchorLog(predictedLowerDamageLarger),
                    PredictionWindowMinSizeBytes = predictionMin,
                    PredictionWindowMaxSizeBytes = predictionMax,
                    WindowMinSizeBytes = realMin,
                    WindowMaxSizeBytes = realMax,
                    WindowSizeGiB = ToGiB(realMax > realMin ? realMax - realMin : 0),
                    CandidatePoolSize = phaseSizeEligiblePool,
                    PredictedPoolCount = predictedPoolCount,
                    DeterministicEligibleCount = deterministicEligibleCount,
                    RejectedByRealSizeWindow = rejectedByRealSizeWindow,
                    RejectedByOtherPhaseDeterministicRules = rejectedByOtherPhaseDeterministicRules,
                    WindowCandidateCount = deterministicWindowRows,
                    LineBeatingCandidateCount = phaseSizeEligiblePool,
                    FetchedCandidateCount = rawCandidates.Count,
                    CandidatesAfterBrutalityCount = afterBrutalityCount,
                    SelectedForValidationCount = kept.Count,
                    CandidateAttemptLimit = interiorAttemptLimit,
                    QueryFetchLimit = interiorFetchLimit,
                    DiversityEnabled = selection.DiversityEnabled,
                    DiversityMode = selection.Mode,
                    DiversityScanLimit = interiorFetchLimit,
                    DiversityScanFetched = rawCandidates.Count,
                    CandidateFamilyCount = selection.CandidateFamilyCount,
                    SelectedFamilyCount = selection.SelectedFamilyCount,
                    SelectedFamilyKeys = selection.SelectedFamilyKeys,
                    DiversitySelectionStrategy = selection.SelectionStrategy,
                    DiversityOverflowCount = selection.OverflowCount,
                    DiversitySizeFloorStartBytes = selection.SizeFloorStartBytes,
                    DiversitySizeFloorEndBytes = selection.SizeFloorEndBytes,
                    TopCandidates = kept.Take(DiagnosticPreviewDisplayCount).Select(ToCandidatePreviewLog).ToList(),
                    RejectedByBrutalityPreview = rejectedByBrutality,
                    Notes = new[]
                    {
                        "Interior DuckDB discovery uses the predicted virtual nonlinear KLD line/window, then deterministically caps candidate size to the real interior slice before diversity/ladder selection. Real KLD is still used only after benchmark validation.",
                        $"predictedPoolCount={predictedPoolCount:N0}; phaseSizeEligiblePool={phaseSizeEligiblePool:N0}; deterministicEligibleCount={deterministicEligibleCount:N0}; rejectedByRealSizeWindow={rejectedByRealSizeWindow:N0}; rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}."
                    }.Concat(selection.Notes).ToList()
                });

                AnsiConsole.MarkupLine(
                    $"[grey]Interior window {globalWindowIndex:N0}/{Math.Max(estimatedWindowCount, globalWindowIndex):N0}:[/] {Markup.Escape(pair.HigherDamageSmaller.DisplayName)} -> {Markup.Escape(pair.LowerDamageLarger.DisplayName)} " +
                    $"| pred-window={predictionMin:N0}..{predictionMax:N0}, real-window={realMin:N0}..{realMax:N0}, predictedPoolCount={predictedPoolCount:N0}, phaseSizeEligiblePool={phaseSizeEligiblePool:N0}, deterministicEligibleCount={deterministicEligibleCount:N0}, rejectedByRealSizeWindow={rejectedByRealSizeWindow:N0}, rejectedByOtherPhaseDeterministicRules={rejectedByOtherPhaseDeterministicRules:N0}, rows-in-window={deterministicWindowRows:N0}, beat-line={phaseSizeEligiblePool:N0}, scanLimit={interiorFetchLimit:N0}, scanFetched={rawCandidates.Count:N0}, after-brutality={afterBrutalityCount:N0}, diversity={Markup.Escape(selection.Mode)}, selectionStrategy={Markup.Escape(selection.SelectionStrategy)}, candidateFamilies={selection.CandidateFamilyCount:N0}, selectedFamilies={selection.SelectedFamilyCount:N0}, selected={kept.Count:N0}/{interiorAttemptLimit:N0}, overflowCount={selection.OverflowCount:N0}, sizeFloorStart={selection.SizeFloorStartBytes?.ToString("N0") ?? "n/a"}");
                PrintSelectedCandidateFamilySummary(kept);
                PrintSelectionLadderNotes(selection.Notes);

                if (deterministicEligibleCount == 0)
                    AnsiConsole.MarkupLine($"[yellow]Interior window skipped builds:[/] no physically eligible candidates remained after predicted nonlinear line, real size slice, and deterministic brutality filters.");

                realCursor = realMax;
                predictionCursor = predictionMax;

                if (realCursor >= realHighSize || predictionCursor >= predictionHighSize)
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
            var smartOnly = await RunSmartInteriorFallbackAsync(pairs, fractions, validationFailures, validationAttempts, ct);
            return new PhaseValidationResult { AcceptedSnapshots = smartOnly };
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

        if (accepted.Count == 0)
        {
            var smartAccepted = await RunSmartInteriorFallbackAsync(pairs, fractions, validationFailures, validationAttempts, ct);
            accepted.AddRange(smartAccepted);
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }

    private async Task<IReadOnlyList<BenchmarkSnapshotRecord>> RunSmartInteriorFallbackAsync(
        IReadOnlyList<AdjacentAnchorPair> pairs,
        IReadOnlyList<double> fractions,
        List<CandidateValidationResult> validationFailures,
        List<CandidateValidationResult> validationAttempts,
        CancellationToken ct)
    {
        if (!Config.SelectionSmartFallbackEnabled)
            return Array.Empty<BenchmarkSnapshotRecord>();

        var accepted = new List<BenchmarkSnapshotRecord>();
        int estimatedWindowCount = pairs.Sum(pair => EstimateInteriorWindowCount(pair, fractions));
        int globalWindowIndex = 0;

        foreach (var pair in pairs)
        {
            ulong realLowSize = pair.HigherDamageSmaller.SizeBytes;
            ulong realHighSize = pair.LowerDamageLarger.SizeBytes;
            if (realHighSize <= realLowSize)
                continue;

            ulong realSpan = realHighSize - realLowSize;
            ulong realCursor = realLowSize;

            for (int i = 0; i < fractions.Count; i++)
            {
                double fraction = fractions[i];
                if (fraction <= 0d)
                    continue;

                ulong realWidth = (ulong)Math.Max(1d, Math.Round(realSpan * Math.Clamp(fraction, 0d, 1d)));
                ulong realMin = realCursor;
                ulong realMax = i == fractions.Count - 1
                    ? realHighSize
                    : Math.Min(realHighSize, realCursor + realWidth);

                if (realMax <= realMin)
                    continue;

                globalWindowIndex++;
                string windowLabel = $"smart interior {globalWindowIndex:N0}: {pair.HigherDamageSmaller.DisplayName} -> {pair.LowerDamageLarger.DisplayName}";

                var smartCandidates = (await _smartFallbackService.BuildInteriorCandidatesAsync(
                    pair.HigherDamageSmaller,
                    pair.LowerDamageLarger,
                    realMin,
                    realMax,
                    windowLabel,
                    globalWindowIndex,
                    Math.Max(estimatedWindowCount, globalWindowIndex),
                    ct)).ToList();

                foreach (var candidate in smartCandidates)
                {
                    var validation = await BuildAndValidateSingleAsync(
                        candidate,
                        snapshot => snapshot.SizeBytes >= realMin &&
                                    snapshot.SizeBytes <= realMax &&
                                    BeatsLinearKldLine(snapshot.SizeBytes, snapshot.Kld, pair.HigherDamageSmaller, pair.LowerDamageLarger),
                        $"smart fallback must land inside {realMin:N0}..{realMax:N0} bytes and beat the real interior linear KLD line",
                        ct);

                    validationAttempts.Add(validation);

                    if (validation.Accepted && validation.Snapshot != null)
                    {
                        accepted.Add(validation.Snapshot);
                        AnsiConsole.MarkupLine("[green]Smart interior fallback candidate selected:[/]");
                        AnsiConsole.MarkupLine($"[grey]  lower anchor=[/] [cyan]{Markup.Escape(pair.HigherDamageSmaller.DisplayName)}[/]");
                        AnsiConsole.MarkupLine($"[grey]  upper anchor=[/] [cyan]{Markup.Escape(pair.LowerDamageLarger.DisplayName)}[/]");
                        AnsiConsole.MarkupLine($"[grey]  chosen=[/] [cyan]{Markup.Escape(validation.Snapshot.DisplayName)}[/]");
                        AnsiConsole.MarkupLine($"[grey]  actualKld=[/] [cyan]{validation.Snapshot.Kld:0.000000}[/]");
                        AnsiConsole.MarkupLine($"[grey]  actualSizeBytes=[/] [cyan]{validation.Snapshot.SizeBytes:N0}[/]");
                        return accepted;
                    }

                    validationFailures.Add(validation);
                }

                realCursor = realMax;
                if (realCursor >= realHighSize)
                    break;
            }
        }

        AnsiConsole.MarkupLine("[grey]Smart interior fallback found no validated candidates.[/]");
        return accepted;
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
        IReadOnlyList<string> notes,
        int? rawSelectionRank = null)
    {
        return CloneCandidateWithSelectionMetadata(
            candidate,
            attemptOrder: candidate.AttemptOrder,
            rawSelectionRank: rawSelectionRank ?? candidate.RawSelectionRank,
            familyKey: candidate.CandidateTheoryFamilyKey,
            familyDisplay: candidate.CandidateTheoryFamilyDisplay,
            familyRank: candidate.CandidateTheoryFamilyRank,
            familyMemberRank: candidate.CandidateTheoryFamilyMemberRank,
            diversityMode: candidate.DiversityMode,
            notes: notes,
            poolSize: poolSize,
            windowCandidateCount: windowCandidateCount,
            lineBeatingCandidateCount: lineBeatingCandidateCount,
            fetchedCandidateCount: fetchedCandidateCount,
            candidatesAfterBrutalityCount: candidatesAfterBrutalityCount,
            candidateAttemptLimit: candidateAttemptLimit,
            phaseWindowIndex: phaseWindowIndex,
            phaseWindowCount: phaseWindowCount);
    }

    private static int ResolveValidationScanLimit(int attemptLimit, long candidatePoolSize, bool diversityEligible)
    {
        attemptLimit = Math.Max(1, attemptLimit);

        if (!Config.SelectionDiversifyValidationCandidates || !diversityEligible)
            return attemptLimit;

        if (candidatePoolSize > 0 && candidatePoolSize <= attemptLimit)
            return attemptLimit;

        long requested = (long)attemptLimit * Config.SelectionDiversityScanMultiplier;
        int min = Math.Max(attemptLimit, Config.SelectionDiversityScanMinCandidates);
        int max = Math.Max(min, Config.SelectionDiversityScanMaxCandidates);
        long clamped = Math.Clamp(requested, min, max);

        if (candidatePoolSize > 0 && candidatePoolSize < clamped)
            clamped = candidatePoolSize;

        return checked((int)Math.Max(attemptLimit, clamped));
    }

    private static bool ShouldUseDiversityForWindow(
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        PredictedAnchorRow? higherDamagePredictionAnchor,
        PredictedAnchorRow? lowerDamagePredictionAnchor)
    {
        if (!Config.SelectionDiversifyValidationCandidates)
            return false;

        if (!Config.SelectionDiversityLowBitOnly)
            return true;

        return IsQ4ishOrBelow(higherDamageSmaller.Quant.BaseQuant) ||
               IsQ4ishOrBelow(lowerDamageLarger.Quant.BaseQuant) ||
               IsQ4ishOrBelow(higherDamagePredictionAnchor?.RuntimeBaselineId) ||
               IsQ4ishOrBelow(lowerDamagePredictionAnchor?.RuntimeBaselineId);
    }

    private static bool IsQ4ishOrBelow(byte? baselineId)
    {
        if (!baselineId.HasValue)
            return false;

        try
        {
            return BaselineQuants.FromId(baselineId.Value).BitRange <= 4;
        }
        catch
        {
            return false;
        }
    }

    private static bool IsQ4ishOrBelow(BaselineQuants baseline) => baseline.BitRange <= 4;

    private static ValidationCandidateSelectionResult SelectValidationCandidates(
        IReadOnlyList<HybridSelectionCandidate> rankedCandidates,
        int attemptLimit,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        PredictedAnchorRow? higherDamagePredictionAnchor,
        PredictedAnchorRow? lowerDamagePredictionAnchor,
        string phaseName,
        bool diversityEligible)
    {
        attemptLimit = Math.Max(1, attemptLimit);
        if (rankedCandidates.Count == 0)
        {
            return new ValidationCandidateSelectionResult
            {
                Candidates = Array.Empty<HybridSelectionCandidate>(),
                Mode = Config.SelectionDiversifyValidationCandidates ? diversityEligible ? "enabled-empty" : "disabled-low-bit-only" : "disabled",
                DiversityEnabled = Config.SelectionDiversifyValidationCandidates && diversityEligible,
                SelectionStrategy = "none",
                Notes = ["No deterministic-eligible candidates survived the prediction/brutality filters for this window; no phase-original primary candidate exists."]
            };
        }

        var activeGroups = GetActiveTensorGroups();
        var entries = rankedCandidates
            .Select((candidate, rawIndex) =>
            {
                var signature = BuildCandidateTheorySignature(candidate, higherDamageSmaller, lowerDamageLarger, higherDamagePredictionAnchor, lowerDamagePredictionAnchor, activeGroups);
                return new CandidateFamilyEntry
                {
                    Candidate = candidate,
                    Signature = signature,
                    RawRank = candidate.RawSelectionRank > 0 ? candidate.RawSelectionRank : rawIndex + 1
                };
            })
            .ToList();

        var families = BuildCandidateTheoryFamilies(entries);
        var originalPrimary = entries[0];
        string originalPrimaryConfigKey = TensorConfigIdentity.ToKey(originalPrimary.Candidate.Prediction.Config);

        var originalOrderNotes = BuildOriginalPhaseOrderPreviewNotes(phaseName, rankedCandidates, previewLimit: 10);
        PrintOriginalPhaseOrderPreview(phaseName, higherDamageSmaller, lowerDamageLarger, rankedCandidates, previewLimit: 10);

        bool canDiversify = Config.SelectionDiversifyValidationCandidates && diversityEligible && rankedCandidates.Count > attemptLimit;
        if (!canDiversify || attemptLimit == 1)
        {
            string mode = Config.SelectionDiversifyValidationCandidates
                ? diversityEligible
                    ? rankedCandidates.Count <= attemptLimit ? "not-needed" : "not-needed"
                    : "disabled-low-bit-only"
                : "disabled";

            var selectedWithoutDiversity = entries
                .Take(attemptLimit)
                .Select((entry, index) => DecorateSelectedCandidate(
                    entry,
                    families,
                    index + 1,
                    index == 0 ? "primary-original-phase-order" : mode))
                .ToList();

            AssertPrimaryPreserved(phaseName, originalPrimaryConfigKey, selectedWithoutDiversity);

            var nonDiversityNotes = new List<string>
            {
                BuildDiversityNote(mode, phaseName, rankedCandidates.Count, attemptLimit, families.Count, selectedWithoutDiversity.Count, overflowCount: 0),
                $"primarySelectionSource=original-phase-order; primaryRawRankBeforeDiversity={originalPrimary.RawRank:N0}; primaryConfigKey={originalPrimaryConfigKey}; primaryPredictedSizeBytes={originalPrimary.Candidate.Prediction.PredictedSizeBytes:N0}; primaryEffectivePredictedKld={GetEffectivePredictedKld(originalPrimary.Candidate):0.000000}; primaryPredictionRank={FormatNullableRank(originalPrimary.Candidate.Prediction.PredictedRank)}; primaryWasExcluded=false.",
                "fallbackSelectionStrategy=not-applied; diversity ladder did not run because the candidate count did not exceed the attempt limit, diversity was disabled, or only one attempt was allowed."
            };
            nonDiversityNotes.AddRange(originalOrderNotes);

            return new ValidationCandidateSelectionResult
            {
                Candidates = selectedWithoutDiversity,
                Mode = mode,
                DiversityEnabled = false,
                SelectionStrategy = "original-phase-order",
                CandidateFamilyCount = families.Count,
                SelectedFamilyCount = selectedWithoutDiversity.Select(x => x.CandidateTheoryFamilyKey).Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.Ordinal).Count(),
                SelectedFamilyKeys = selectedWithoutDiversity.Select(x => x.CandidateTheoryFamilyKey).Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.Ordinal).ToList(),
                Notes = nonDiversityNotes
            };
        }

        var selectedEntries = new List<CandidateFamilyEntry>();
        var selectedConfigKeys = new HashSet<string>(StringComparer.Ordinal);
        var selectedFamilyKeys = new HashSet<string>(StringComparer.Ordinal);
        var notesForFallback = new List<string>();
        int overflowCount = 0;
        int familyExhaustionCount = 0;
        ulong? sizeFloorStart = null;
        ulong sizeFloor = 0;

        bool AddEntry(CandidateFamilyEntry entry, string selectionMode, ulong? previousSizeFloor)
        {
            string configKey = TensorConfigIdentity.ToKey(entry.Candidate.Prediction.Config);
            if (!selectedConfigKeys.Add(configKey))
                return false;

            entry.SelectionMode = selectionMode;
            entry.PreviousSizeFloorBytes = previousSizeFloor;
            entry.SizeDeltaVsFloorBytes = previousSizeFloor.HasValue
                ? unchecked((long)entry.Candidate.Prediction.PredictedSizeBytes - (long)previousSizeFloor.Value)
                : null;
            entry.EffectivePredictedKld = GetEffectivePredictedKld(entry.Candidate);

            selectedEntries.Add(entry);
            selectedFamilyKeys.Add(entry.Signature.Key);
            return true;
        }

        // Critical invariant: the phase's original deterministic-eligible primary owns attempt #1.
        // Diversity and the size ladder are fallback ordering only and must never re-rank this row.
        AddEntry(originalPrimary, "primary-original-phase-order", previousSizeFloor: null);
        sizeFloor = originalPrimary.Candidate.Prediction.PredictedSizeBytes;
        sizeFloorStart = sizeFloor;

        while (selectedEntries.Count < attemptLimit)
        {
            ulong previousFloor = sizeFloor;

            var unusedAtOrAbove = entries
                .Skip(1)
                .Where(x => IsSelectable(x, selectedConfigKeys) && !selectedFamilyKeys.Contains(x.Signature.Key) && x.Candidate.Prediction.PredictedSizeBytes >= sizeFloor)
                .OrderBy(GetEffectivePredictedKld)
                .ThenByDescending(x => x.Candidate.Prediction.PredictedSizeBytes)
                .ThenBy(x => x.RawRank)
                .ToList();

            var next = unusedAtOrAbove.FirstOrDefault();
            if (next != null && AddEntry(next, "fallback-ladder-unused-family", previousFloor))
            {
                sizeFloor = Math.Max(sizeFloor, next.Candidate.Prediction.PredictedSizeBytes);
                continue;
            }

            int unusedFamiliesRemaining = families.Count(f => !selectedFamilyKeys.Contains(f.Key) && f.Members.Any(m => IsSelectable(m, selectedConfigKeys)));
            int unusedBelowCount = entries.Skip(1).Count(x => IsSelectable(x, selectedConfigKeys) && !selectedFamilyKeys.Contains(x.Signature.Key) && x.Candidate.Prediction.PredictedSizeBytes < sizeFloor);

            if (unusedFamiliesRemaining > 0)
            {
                overflowCount++;
                notesForFallback.Add(
                    $"Diversity ladder overflow: attemptSlot={selectedEntries.Count + 1:N0}; sizeFloorBytes={sizeFloor:N0}; unusedFamiliesRemaining={unusedFamiliesRemaining:N0}; candidatesAtOrAboveFloor=0; overflowCandidatesBelowFloor={unusedBelowCount:N0}; reason=no-unused-family-candidate-at-or-above-size-floor.");

                next = entries
                    .Skip(1)
                    .Where(x => IsSelectable(x, selectedConfigKeys) && !selectedFamilyKeys.Contains(x.Signature.Key) && x.Candidate.Prediction.PredictedSizeBytes < sizeFloor)
                    .OrderByDescending(x => x.Candidate.Prediction.PredictedSizeBytes)
                    .ThenBy(GetEffectivePredictedKld)
                    .ThenBy(x => x.RawRank)
                    .FirstOrDefault();

                if (next != null && AddEntry(next, "fallback-overflow-unused-family", previousFloor))
                {
                    // Deliberately monotone: an overflow candidate below the floor cannot drag the safety floor down.
                    sizeFloor = Math.Max(sizeFloor, next.Candidate.Prediction.PredictedSizeBytes);
                    continue;
                }
            }

            familyExhaustionCount++;
            notesForFallback.Add(
                $"Diversity ladder family exhaustion: attemptSlot={selectedEntries.Count + 1:N0}; selectedFamilies={selectedFamilyKeys.Count:N0}; candidateFamilies={families.Count:N0}; reason=all-distinct-families-exhausted.");

            next = entries
                .Skip(1)
                .Where(x => IsSelectable(x, selectedConfigKeys) && x.Candidate.Prediction.PredictedSizeBytes >= sizeFloor)
                .OrderBy(GetEffectivePredictedKld)
                .ThenByDescending(x => x.Candidate.Prediction.PredictedSizeBytes)
                .ThenBy(x => x.RawRank)
                .FirstOrDefault();

            if (next != null && AddEntry(next, "fallback-ladder-used-family", previousFloor))
            {
                sizeFloor = Math.Max(sizeFloor, next.Candidate.Prediction.PredictedSizeBytes);
                continue;
            }

            int belowFloorCount = entries.Skip(1).Count(x => IsSelectable(x, selectedConfigKeys) && x.Candidate.Prediction.PredictedSizeBytes < sizeFloor);
            overflowCount++;
            notesForFallback.Add(
                $"Diversity ladder overflow: attemptSlot={selectedEntries.Count + 1:N0}; sizeFloorBytes={sizeFloor:N0}; unusedFamiliesRemaining=0; candidatesAtOrAboveFloor=0; overflowCandidatesBelowFloor={belowFloorCount:N0}; reason=no-remaining-candidate-at-or-above-size-floor.");

            next = entries
                .Skip(1)
                .Where(x => IsSelectable(x, selectedConfigKeys) && x.Candidate.Prediction.PredictedSizeBytes < sizeFloor)
                .OrderByDescending(x => x.Candidate.Prediction.PredictedSizeBytes)
                .ThenBy(GetEffectivePredictedKld)
                .ThenBy(x => x.RawRank)
                .FirstOrDefault();

            if (next != null && AddEntry(next, "fallback-overflow-used-family", previousFloor))
            {
                // Deliberately monotone: do not lower the floor after overflow.
                sizeFloor = Math.Max(sizeFloor, next.Candidate.Prediction.PredictedSizeBytes);
                continue;
            }

            break;
        }

        var selected = selectedEntries
            .Take(attemptLimit)
            .Select((entry, index) => DecorateSelectedCandidate(entry, families, index + 1, entry.SelectionMode))
            .ToList();

        AssertPrimaryPreserved(phaseName, originalPrimaryConfigKey, selected);

        bool exhaustedDistinctFamilies = selected.Count > selected.Select(x => x.CandidateTheoryFamilyKey).Distinct(StringComparer.Ordinal).Count();
        var notes = new List<string>
        {
            BuildDiversityNote("enabled", phaseName, rankedCandidates.Count, attemptLimit, families.Count, selected.Count, overflowCount),
            $"primarySelectionSource=original-phase-order; primaryRawRankBeforeDiversity={originalPrimary.RawRank:N0}; primaryConfigKey={originalPrimaryConfigKey}; primaryPredictedSizeBytes={originalPrimary.Candidate.Prediction.PredictedSizeBytes:N0}; primaryEffectivePredictedKld={GetEffectivePredictedKld(originalPrimary.Candidate):0.000000}; primaryPredictionRank={FormatNullableRank(originalPrimary.Candidate.Prediction.PredictedRank)}; primaryWasExcluded=false.",
            $"fallbackSelectionStrategy=family-size-ladder; fallbackCount={Math.Max(0, selected.Count - 1):N0}; sizeFloorStartBytes={sizeFloorStart.GetValueOrDefault():N0}; sizeFloorEndBytes={sizeFloor:N0}; overflowCount={overflowCount:N0}; familyExhaustionCount={familyExhaustionCount:N0}."
        };
        notes.AddRange(originalOrderNotes);
        notes.AddRange(notesForFallback);

        if (exhaustedDistinctFamilies)
            notes.Add("Distinct candidate theory families were exhausted before the attempt limit; remaining fallback slots were filled from already-selected families using the same monotone size-floor ladder.");
        if (overflowCount > 0)
            notes.Add("Diversity ladder overflow occurred: at least one selected retry was below the monotone size floor because no same/larger alternative was available in the preferred family bucket.");
        if (selected.Count > 0 && overflowCount >= Math.Max(1, selected.Count / 2))
            notes.Add("WARNING: Diversity ladder overflow selected many candidates below the size floor; candidate pool may not contain enough safer alternatives.");
        if (families.Count >= rankedCandidates.Count * 0.90 && rankedCandidates.Count >= 25)
            notes.Add("WARNING: Diversity warning: candidate family key may be too fine-grained; most scanned candidates formed unique families.");

        return new ValidationCandidateSelectionResult
        {
            Candidates = selected,
            Mode = "enabled",
            DiversityEnabled = true,
            SelectionStrategy = "original-primary-plus-family-size-ladder-fallbacks",
            CandidateFamilyCount = families.Count,
            SelectedFamilyCount = selected.Select(x => x.CandidateTheoryFamilyKey).Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.Ordinal).Count(),
            SelectedFamilyKeys = selected.Select(x => x.CandidateTheoryFamilyKey).Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.Ordinal).ToList(),
            OverflowCount = overflowCount,
            SizeFloorStartBytes = sizeFloorStart,
            SizeFloorEndBytes = sizeFloor,
            Notes = notes
        };
    }

    private static List<CandidateTheoryFamily> BuildCandidateTheoryFamilies(IReadOnlyList<CandidateFamilyEntry> entries)
    {
        var families = entries
            .GroupBy(x => x.Signature.Key, StringComparer.Ordinal)
            .Select(g => new CandidateTheoryFamily
            {
                Key = g.Key,
                Display = g.First().Signature.Display,
                Members = g.OrderBy(x => x.RawRank).ToList()
            })
            .OrderBy(x => x.Members[0].RawRank)
            .ToList();

        for (int familyIndex = 0; familyIndex < families.Count; familyIndex++)
        {
            families[familyIndex].Rank = familyIndex + 1;
            foreach (var member in families[familyIndex].Members)
                member.Family = families[familyIndex];

            for (int memberIndex = 0; memberIndex < families[familyIndex].Members.Count; memberIndex++)
                families[familyIndex].Members[memberIndex].MemberRank = memberIndex + 1;
        }

        return families;
    }

    private static void AssertPrimaryPreserved(
        string phaseName,
        string originalPrimaryConfigKey,
        IReadOnlyList<HybridSelectionCandidate> selected)
    {
        if (selected.Count == 0)
            return;

        string selectedPrimaryConfigKey = TensorConfigIdentity.ToKey(selected[0].Prediction.Config);
        if (string.Equals(originalPrimaryConfigKey, selectedPrimaryConfigKey, StringComparison.Ordinal))
            return;

        string message =
            $"CRITICAL SELECTION INVARIANT FAILED: {phaseName} fallback ordering changed attempt #1. " +
            $"originalPrimary={originalPrimaryConfigKey}; selectedPrimary={selectedPrimaryConfigKey}. " +
            "Family diversity / size ladder may only reorder fallback attempts after the phase-original primary candidate.";

        AnsiConsole.MarkupLine($"[red]{Markup.Escape(message)}[/]");
        throw new InvalidOperationException(message);
    }

    private static IReadOnlyList<string> BuildOriginalPhaseOrderPreviewNotes(
        string phaseName,
        IReadOnlyList<HybridSelectionCandidate> rankedCandidates,
        int previewLimit)
    {
        var notes = new List<string>
        {
            $"originalOrderPreview phase={phaseName}; showingTop={Math.Min(previewLimit, rankedCandidates.Count):N0}/{rankedCandidates.Count:N0}; these rows are in deterministic-eligible original phase order before fallback diversity."
        };

        int order = 0;
        foreach (var candidate in rankedCandidates.Take(previewLimit))
        {
            order++;
            notes.Add(
                $"originalOrderPreview rawOrder={order:N0}; config={TensorConfigIdentity.ToKey(candidate.Prediction.Config)}; size={candidate.Prediction.PredictedSizeBytes:N0}; predKld={GetEffectivePredictedKld(candidate):0.000000}; predictionRank={FormatNullableRank(candidate.Prediction.PredictedRank)}.");
        }

        return notes;
    }

    private static void PrintOriginalPhaseOrderPreview(
        string phaseName,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        IReadOnlyList<HybridSelectionCandidate> rankedCandidates,
        int previewLimit)
    {
        int count = Math.Min(previewLimit, rankedCandidates.Count);
        if (count == 0)
            return;

        string anchorText = ReferenceEquals(higherDamageSmaller, lowerDamageLarger) ||
                            TensorConfigIdentity.ToKey(higherDamageSmaller.Config) == TensorConfigIdentity.ToKey(lowerDamageLarger.Config)
            ? higherDamageSmaller.DisplayName
            : $"{higherDamageSmaller.DisplayName} -> {lowerDamageLarger.DisplayName}";

        AnsiConsole.MarkupLine($"[grey]Original phase order preview ({Markup.Escape(phaseName)}) for {Markup.Escape(anchorText)}:[/] [cyan]{count:N0}/{rankedCandidates.Count:N0}[/]");
        int order = 0;
        foreach (var candidate in rankedCandidates.Take(previewLimit))
        {
            order++;
            AnsiConsole.MarkupLine(
                $"[grey]  rawOrder={order:N0} config={Markup.Escape(TensorConfigIdentity.ToKey(candidate.Prediction.Config))} " +
                $"size={candidate.Prediction.PredictedSizeBytes:N0} predKld={GetEffectivePredictedKld(candidate):0.000000} predictionRank={Markup.Escape(FormatNullableRank(candidate.Prediction.PredictedRank))}[/]");
        }
    }

    private static string FormatNullableRank(ulong? rank) => rank.HasValue ? rank.Value.ToString("N0") : "n/a";

    private static bool IsSelectable(CandidateFamilyEntry entry, HashSet<string> selectedConfigKeys)
    {
        string key = TensorConfigIdentity.ToKey(entry.Candidate.Prediction.Config);
        return !selectedConfigKeys.Contains(key);
    }

    private static HybridSelectionCandidate DecorateSelectedCandidate(
        CandidateFamilyEntry entry,
        IReadOnlyList<CandidateTheoryFamily> families,
        int attemptOrder,
        string diversityMode)
    {
        var family = entry.Family ?? families.First(x => string.Equals(x.Key, entry.Signature.Key, StringComparison.Ordinal));
        int memberRank = entry.MemberRank > 0 ? entry.MemberRank : Math.Max(1, family.Members.FindIndex(x => ReferenceEquals(x, entry)) + 1);
        double effectivePredictedKld = double.IsNaN(entry.EffectivePredictedKld) ? GetEffectivePredictedKld(entry.Candidate) : entry.EffectivePredictedKld;
        string previousFloorText = entry.PreviousSizeFloorBytes.HasValue ? entry.PreviousSizeFloorBytes.Value.ToString("N0") : "n/a";
        string deltaText = entry.SizeDeltaVsFloorBytes.HasValue ? entry.SizeDeltaVsFloorBytes.Value.ToString("N0") : "n/a";

        string selectionStrategy = diversityMode switch
        {
            "primary-original-phase-order" => "original-phase-primary",
            "disabled" or "not-needed" or "disabled-low-bit-only" => "original-phase-order",
            _ => "family-size-ladder-fallback"
        };

        var notes = entry.Candidate.CandidateSelectionNotes
            .Concat(new[]
            {
                $"diversity={diversityMode}; selectionStrategy={selectionStrategy}; rawRank={entry.RawRank}; familyRank={family.Rank}; familyMemberRank={memberRank}; previousSizeFloorBytes={previousFloorText}; selectedSizeBytes={entry.Candidate.Prediction.PredictedSizeBytes:N0}; sizeDeltaVsFloorBytes={deltaText}; effectivePredictedKld={effectivePredictedKld:0.000000}; familyKey={entry.Signature.Key}; familyDisplay={entry.Signature.Display}"
            })
            .ToList();

        return CloneCandidateWithSelectionMetadata(
            entry.Candidate,
            attemptOrder,
            entry.RawRank,
            entry.Signature.Key,
            entry.Signature.Display,
            family.Rank,
            memberRank,
            diversityMode,
            notes);
    }

    private static HybridSelectionCandidate CloneCandidateWithSelectionMetadata(
        HybridSelectionCandidate candidate,
        int attemptOrder,
        int rawSelectionRank,
        string familyKey,
        string familyDisplay,
        int familyRank,
        int familyMemberRank,
        string diversityMode,
        IReadOnlyList<string> notes,
        long? poolSize = null,
        long? windowCandidateCount = null,
        long? lineBeatingCandidateCount = null,
        int? fetchedCandidateCount = null,
        int? candidatesAfterBrutalityCount = null,
        int? candidateAttemptLimit = null,
        int? phaseWindowIndex = null,
        int? phaseWindowCount = null)
    {
        return new HybridSelectionCandidate
        {
            Prediction = candidate.Prediction,
            Reason = candidate.Reason,
            LowerDamageAnchor = candidate.LowerDamageAnchor,
            HigherDamageAnchor = candidate.HigherDamageAnchor,
            LowerDamagePredictionAnchor = candidate.LowerDamagePredictionAnchor,
            HigherDamagePredictionAnchor = candidate.HigherDamagePredictionAnchor,
            PredictionWindowMinSizeBytes = candidate.PredictionWindowMinSizeBytes,
            PredictionWindowMaxSizeBytes = candidate.PredictionWindowMaxSizeBytes,
            WindowMinSizeBytes = candidate.WindowMinSizeBytes,
            WindowMaxSizeBytes = candidate.WindowMaxSizeBytes,
            LinearExpectedKld = candidate.LinearExpectedKld,
            PredictedGainOverLine = candidate.PredictedGainOverLine,
            AttemptOrder = attemptOrder,
            WindowLabel = candidate.WindowLabel,
            CandidatePoolSize = poolSize ?? candidate.CandidatePoolSize,
            WindowCandidateCount = windowCandidateCount ?? candidate.WindowCandidateCount,
            LineBeatingCandidateCount = lineBeatingCandidateCount ?? candidate.LineBeatingCandidateCount,
            FetchedCandidateCount = fetchedCandidateCount ?? candidate.FetchedCandidateCount,
            CandidatesAfterBrutalityCount = candidatesAfterBrutalityCount ?? candidate.CandidatesAfterBrutalityCount,
            CandidateAttemptLimit = candidateAttemptLimit ?? candidate.CandidateAttemptLimit,
            PhaseWindowIndex = phaseWindowIndex ?? candidate.PhaseWindowIndex,
            PhaseWindowCount = phaseWindowCount ?? candidate.PhaseWindowCount,
            RawSelectionRank = rawSelectionRank,
            CandidateTheoryFamilyKey = familyKey,
            CandidateTheoryFamilyDisplay = familyDisplay,
            CandidateTheoryFamilyRank = familyRank,
            CandidateTheoryFamilyMemberRank = familyMemberRank,
            DiversityMode = diversityMode,
            CandidateSelectionNotes = notes
        };
    }

    private static CandidateTheorySignature BuildCandidateTheorySignature(
        HybridSelectionCandidate candidate,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        PredictedAnchorRow? higherDamagePredictionAnchor,
        PredictedAnchorRow? lowerDamagePredictionAnchor,
        IReadOnlyList<TensorGroup> activeGroups)
    {
        var config = candidate.Prediction.Config;
        var baseQuant = BaselineQuants.FromId(config.BaseQuant);
        int anchorBit = Math.Min(higherDamageSmaller.Quant.BaseQuant.BitRange, lowerDamageLarger.Quant.BaseQuant.BitRange);

        var lowRisk = new List<string>();
        var protectedGroups = new List<string>();
        var external = new List<string>();
        var sensitivity = new List<string>();
        int sixPlus = 0;
        int five = 0;
        int four = 0;
        int threeOrLess = 0;

        foreach (var group in activeGroups.OrderBy(x => x.UniqueId))
        {
            var effective = GetEffectiveGroupBaseline(config, group);
            string placement = $"{group.ShortCode}={effective.Names[0]}";

            if (effective.BitRange >= 6)
                sixPlus++;
            else if (effective.BitRange == 5)
                five++;
            else if (effective.BitRange == 4)
                four++;
            else
                threeOrLess++;

            if (effective.BitRange <= anchorBit - 1 || effective.BitRange <= 3)
                lowRisk.Add(placement);

            if ((anchorBit <= 5 && effective.BitRange >= 6) || effective.BitRange >= anchorBit + 1)
                protectedGroups.Add(placement);

            if (effective.IsCustomBaseline || effective.IsExternalRepositoryBaseline)
                external.Add(placement);

            if (IsHighSensitivityGroup(group) && effective.UniqueId != baseQuant.UniqueId)
                sensitivity.Add(placement);
        }

        string anchorBand = $"anchor={higherDamagePredictionAnchor?.DisplayName ?? higherDamageSmaller.DisplayName}->{lowerDamagePredictionAnchor?.DisplayName ?? lowerDamageLarger.DisplayName}@{anchorBit}b";
        string bulk = $"bulk:6p={sixPlus},5={five},4={four},3m={threeOrLess}";

        var components = new List<string>
        {
            anchorBand,
            $"base={baseQuant.Names[0]}",
            bulk
        };

        if (lowRisk.Count > 0)
            components.Add("risk:" + string.Join(",", lowRisk.OrderBy(x => x, StringComparer.Ordinal)));
        if (protectedGroups.Count > 0)
            components.Add("protect:" + string.Join(",", protectedGroups.OrderBy(x => x, StringComparer.Ordinal)));
        if (external.Count > 0)
            components.Add("external:" + string.Join(",", external.OrderBy(x => x, StringComparer.Ordinal)));
        if (sensitivity.Count > 0)
            components.Add("sensitive:" + string.Join(",", sensitivity.OrderBy(x => x, StringComparer.Ordinal)));

        string key = string.Join("|", components);
        string display = string.Join("|", components.Where(x => !x.StartsWith("anchor=", StringComparison.Ordinal)));
        return new CandidateTheorySignature { Key = key, Display = display };
    }

    private static BaselineQuants GetEffectiveGroupBaseline(TensorConfig config, TensorGroup group)
    {
        byte stored = group.UniqueId switch
        {
            0 => config.Embeddings,
            1 => config.LmHead,
            2 => config.AttnQ,
            3 => config.AttnKV,
            4 => config.AttnOutput,
            5 => config.FfnUpGate,
            6 => config.FfnDown,
            7 => config.MoeExperts,
            8 => config.MoeRouter,
            _ => BaselineQuants.TensorConfigNullSlotValue
        };

        return BaselineQuants.IsNullTensorConfigGroupSlot(stored)
            ? BaselineQuants.FromId(config.BaseQuant)
            : BaselineQuants.DecodeTensorConfigGroupSlotToBaseline(stored);
    }

    private static IReadOnlyList<TensorGroup> GetActiveTensorGroups()
    {
        var unusedIds = Cache.UnusedTensorGroups.Select(x => x.UniqueId).ToHashSet();
        return TReg.All.Where(x => !unusedIds.Contains(x.UniqueId)).OrderBy(x => x.UniqueId).ToList();
    }

    private static bool IsHighSensitivityGroup(TensorGroup group) =>
        group.UniqueId == TReg.Embeddings.UniqueId ||
        group.UniqueId == TReg.LmHead.UniqueId ||
        group.UniqueId == TReg.AttnQ.UniqueId ||
        group.UniqueId == TReg.AttnKV.UniqueId ||
        group.UniqueId == TReg.FfnDown.UniqueId;


    private static double GetEffectivePredictedKld(CandidateFamilyEntry entry) => GetEffectivePredictedKld(entry.Candidate);

    private static double GetEffectivePredictedKld(HybridSelectionCandidate candidate) =>
        double.IsNaN(candidate.Prediction.PredictedKld) ? double.PositiveInfinity : candidate.Prediction.PredictedKld;

    private static string BuildDiversityNote(string mode, string phaseName, int candidateCount, int attemptLimit, int familyCount, int selectedCount, int overflowCount) =>
        mode switch
        {
            "enabled" => $"Diversity enabled for {phaseName}: selectionStrategy=family-size-ladder; selected {selectedCount:N0}/{attemptLimit:N0} validation attempts from {familyCount:N0} candidate theory families across {candidateCount:N0} filtered scan candidates; phase-original primary candidate is preserved as attempt 1; only fallback attempts prefer same/larger predicted size before explicit overflow; overflowCount={overflowCount:N0}.",
            "not-needed" => $"Diversity not needed for {phaseName}: filtered candidate count {candidateCount:N0} <= attempt limit {attemptLimit:N0}; candidates kept in raw predicted order.",
            "disabled-low-bit-only" => $"Diversity skipped for {phaseName}: candidate_selection.diversity_low_bit_only=true and this anchor/window was not Q4-ish or below.",
            _ => $"Diversity disabled for {phaseName}; candidates kept in raw predicted order."
        };

    private static void PrintSelectionLadderNotes(IReadOnlyList<string> notes)
    {
        foreach (var note in notes)
        {
            if (note.Contains("WARNING", StringComparison.OrdinalIgnoreCase) ||
                note.Contains("Diversity ladder overflow", StringComparison.OrdinalIgnoreCase) ||
                note.Contains("Diversity ladder family exhaustion", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine($"[yellow]  {Markup.Escape(note)}[/]");
            }
            else if (note.Contains("selectionStrategy=family-size-ladder", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine($"[grey]  {Markup.Escape(note)}[/]");
            }
        }
    }

    private static void PrintSelectedCandidateFamilySummary(IReadOnlyList<HybridSelectionCandidate> candidates)
    {
        foreach (var candidate in candidates.Take(DiagnosticPreviewDisplayCount))
        {
            if (string.IsNullOrWhiteSpace(candidate.CandidateTheoryFamilyDisplay))
                continue;

            string ladderNote = candidate.CandidateSelectionNotes.FirstOrDefault(x => x.Contains("selectionStrategy=family-size-ladder", StringComparison.Ordinal)) ?? string.Empty;
            string previousFloor = ExtractSelectionNoteValue(ladderNote, "previousSizeFloorBytes") ?? "n/a";
            string deltaVsFloor = ExtractSelectionNoteValue(ladderNote, "sizeDeltaVsFloorBytes") ?? "n/a";
            string effectiveKld = ExtractSelectionNoteValue(ladderNote, "effectivePredictedKld") ?? candidate.Prediction.PredictedKld.ToString("0.000000");

            AnsiConsole.MarkupLine(
                $"[grey]  selected attempt={candidate.AttemptOrder:N0}/{candidate.CandidateAttemptLimit:N0} rawRank={candidate.RawSelectionRank:N0} " +
                $"selectionMode={Markup.Escape(candidate.DiversityMode)} familyRank={candidate.CandidateTheoryFamilyRank:N0} memberRank={candidate.CandidateTheoryFamilyMemberRank:N0} " +
                $"previousSizeFloorBytes={Markup.Escape(previousFloor)} selectedSizeBytes={candidate.Prediction.PredictedSizeBytes:N0} sizeDeltaVsFloorBytes={Markup.Escape(deltaVsFloor)} effectivePredictedKld={Markup.Escape(effectiveKld)}[/]");
            AnsiConsole.MarkupLine($"[grey]    familyKey=[/][cyan]{Markup.Escape(candidate.CandidateTheoryFamilyKey)}[/]");
            AnsiConsole.MarkupLine($"[grey]    familyDisplay=[/][cyan]{Markup.Escape(candidate.CandidateTheoryFamilyDisplay)}[/]");
        }
    }

    private static string? ExtractSelectionNoteValue(string note, string key)
    {
        if (string.IsNullOrWhiteSpace(note))
            return null;

        string prefix = key + "=";
        int start = note.IndexOf(prefix, StringComparison.Ordinal);
        if (start < 0)
            return null;

        start += prefix.Length;
        int end = note.IndexOf(';', start);
        return end < 0 ? note[start..].Trim() : note[start..end].Trim();
    }

    private static BrutalityAnalysis AnalyzeNearLowerAnchorBrutality(HybridSelectionCandidate candidate)
    {
        var higherDamagePredictionAnchor = candidate.HigherDamagePredictionAnchor;
        var lowerDamagePredictionAnchor = candidate.LowerDamagePredictionAnchor;
        if (higherDamagePredictionAnchor == null || lowerDamagePredictionAnchor == null)
        {
            return new BrutalityAnalysis
            {
                Passed = true,
                FractionFromSmallAnchor = 1d,
                RequiredGain = Config.SelectionMinimumKldImprovementEpsilon,
                Explanation = "Brutality skipped because prediction anchor metadata is missing; no predicted-vs-real comparison was performed."
            };
        }

        ulong span = lowerDamagePredictionAnchor.PredictedSizeBytes > higherDamagePredictionAnchor.PredictedSizeBytes
            ? lowerDamagePredictionAnchor.PredictedSizeBytes - higherDamagePredictionAnchor.PredictedSizeBytes
            : 0;

        if (span == 0)
        {
            return new BrutalityAnalysis
            {
                Passed = true,
                FractionFromSmallAnchor = 1d,
                RequiredGain = Config.SelectionMinimumKldImprovementEpsilon,
                Explanation = "Brutality passed because prediction-anchor span is zero."
            };
        }

        ulong distanceFromSmall = candidate.Prediction.PredictedSizeBytes > higherDamagePredictionAnchor.PredictedSizeBytes
            ? candidate.Prediction.PredictedSizeBytes - higherDamagePredictionAnchor.PredictedSizeBytes
            : 0;

        double fraction = distanceFromSmall / (double)span;
        double requiredGain = Math.Max(
            Config.SelectionMinimumKldImprovementEpsilon,
            Math.Abs(higherDamagePredictionAnchor.PredictedKld - lowerDamagePredictionAnchor.PredictedKld) *
            Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap);

        bool passed = fraction > Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan ||
                      candidate.PredictedGainOverLine >= requiredGain;

        string explanation = passed
            ? fraction > Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan
                ? $"Brutality passed in prediction space because candidate is outside brutal zone (fraction={fraction:0.###} > {Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###})."
                : $"Brutality passed in prediction space because predicted gain {candidate.PredictedGainOverLine:0.########} >= required gain {requiredGain:0.########}."
            : $"Brutality rejected in prediction space because candidate is inside brutal zone (fraction={fraction:0.###} <= {Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan:0.###}) and predicted gain {candidate.PredictedGainOverLine:0.########} < required gain {requiredGain:0.########}.";

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
            $"[grey]  prediction anchors/window:[/] {Markup.Escape(candidate.HigherDamagePredictionAnchor?.DisplayName ?? "n/a")} -> {Markup.Escape(candidate.LowerDamagePredictionAnchor?.DisplayName ?? "n/a")}, " +
            $"predWindow={candidate.PredictionWindowMinSizeBytes:N0}..{candidate.PredictionWindowMaxSizeBytes:N0}, realWindow={candidate.WindowMinSizeBytes:N0}..{candidate.WindowMaxSizeBytes:N0}");
        AnsiConsole.MarkupLine(
            $"[grey]  selection context:[/] pool={candidate.CandidatePoolSize:N0}, windowRows={candidate.WindowCandidateCount:N0}, lineBeat={candidate.LineBeatingCandidateCount:N0}, " +
            $"fetched={candidate.FetchedCandidateCount:N0}, afterBrutality={candidate.CandidatesAfterBrutalityCount:N0}, attemptLimit={candidate.CandidateAttemptLimit:N0}");
        if (!string.IsNullOrWhiteSpace(candidate.CandidateTheoryFamilyDisplay))
        {
            string ladderNote = candidate.CandidateSelectionNotes.FirstOrDefault(x => x.Contains("selectionStrategy=family-size-ladder", StringComparison.Ordinal)) ?? string.Empty;
            string previousFloor = ExtractSelectionNoteValue(ladderNote, "previousSizeFloorBytes") ?? "n/a";
            string deltaVsFloor = ExtractSelectionNoteValue(ladderNote, "sizeDeltaVsFloorBytes") ?? "n/a";
            string effectiveKld = ExtractSelectionNoteValue(ladderNote, "effectivePredictedKld") ?? candidate.Prediction.PredictedKld.ToString("0.000000");
            AnsiConsole.MarkupLine(
                $"[grey]  diversity family:[/] selectionMode={Markup.Escape(candidate.DiversityMode)}, rawRank={candidate.RawSelectionRank:N0}, " +
                $"familyRank={candidate.CandidateTheoryFamilyRank:N0}, memberRank={candidate.CandidateTheoryFamilyMemberRank:N0}, " +
                $"previousSizeFloorBytes={Markup.Escape(previousFloor)}, selectedSizeBytes={candidate.Prediction.PredictedSizeBytes:N0}, sizeDeltaVsFloorBytes={Markup.Escape(deltaVsFloor)}, effectivePredictedKld={Markup.Escape(effectiveKld)}");
            AnsiConsole.MarkupLine($"[grey]  diversity familyKey:[/] {Markup.Escape(candidate.CandidateTheoryFamilyKey)}");
            AnsiConsole.MarkupLine($"[grey]  diversity familyDisplay:[/] {Markup.Escape(candidate.CandidateTheoryFamilyDisplay)}");
        }
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

    private static void PrintPredictionAnchorFrontier(
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
        IReadOnlyList<BenchmarkSnapshotRecord> realAnchors,
        string title)
    {
        if (predictedAnchors.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]Prediction Anchor Frontier:[/] no scored virtual prediction anchors were found. Final DuckDB preselection will skip anchor-based discovery rather than mix prediction and real spaces.");
            return;
        }

        var table = new Table().RoundedBorder().BorderColor(Color.Grey);
        table.Title = new TableTitle(Markup.Escape(title));
        table.AddColumn("Virtual Anchor");
        table.AddColumn("Canonical Key");
        table.AddColumn("Config Key");
        table.AddColumn("Pred KLD");
        table.AddColumn("Pred Size GiB");
        table.AddColumn("Rank");
        table.AddColumn("Conf");
        table.AddColumn("Isolation Source");
        table.AddColumn("Matching Real Anchor");
        table.AddColumn("Real KLD/Size GiB");

        foreach (var anchor in predictedAnchors.OrderBy(x => x.PredictedKld).ThenBy(x => x.PredictedSizeBytes))
        {
            var real = FindMatchingRealAnchor(anchor, realAnchors);
            table.AddRow(
                Markup.Escape(anchor.DisplayName),
                Markup.Escape(anchor.BaselineCanonicalKey),
                Markup.Escape(anchor.ConfigKey),
                anchor.PredictedKld.ToString("0.000000"),
                ToGiB(anchor.PredictedSizeBytes).ToString("0.00"),
                anchor.PredictionRank.ToString("N0"),
                anchor.PredictionConfidence.ToString("0.###"),
                Markup.Escape(DescribeVirtualAnchorIsolationSource(anchor)),
                real == null ? "[grey]none[/]" : Markup.Escape(real.DisplayName),
                real == null ? "[grey]n/a[/]" : $"{real.Kld:0.000000} / {ToGiB(real.SizeBytes):0.00}");
        }

        AnsiConsole.Write(table);

        var q8Anchor = predictedAnchors.FirstOrDefault(x =>
            x.RuntimeBaselineId == BaselineQuants.Q8_0.UniqueId ||
            string.Equals(NormalizeAnchorKey(x.BaselineCanonicalKey), NormalizeAnchorKey(BaselineQuants.Q8_0.CanonicalKey), StringComparison.Ordinal) ||
            string.Equals(x.DisplayName, BaselineQuants.Q8_0.Names[0], StringComparison.OrdinalIgnoreCase));

        if (q8Anchor != null && Math.Abs(q8Anchor.PredictedKld) <= 1e-12d)
        {
            AnsiConsole.MarkupLine("[yellow]WARNING:[/] Q8_0 virtual prediction anchor has zero predicted KLD. This usually means Q8_0 isolation rows were skipped or missing. Q8_0 must not be treated as native/exact truth in prediction space.");
        }
    }

    private static string DescribeVirtualAnchorIsolationSource(PredictedAnchorRow anchor)
    {
        try
        {
            var baseline = BaselineQuants.FromId(anchor.RuntimeBaselineId);
            if (baseline.IsExternalRepositoryBaseline)
                return "exact external; fallback disabled";

            if (BaselineQuants.IsNativeExactAlias(baseline.UniqueId))
                return "native exact";

            return "standard exact";
        }
        catch
        {
            return "unknown";
        }
    }

    private static BenchmarkSnapshotRecord? FindMatchingRealAnchor(
        PredictedAnchorRow predictedAnchor,
        IReadOnlyList<BenchmarkSnapshotRecord> realAnchors)
    {
        string predictedKey = NormalizeAnchorKey(predictedAnchor.BaselineCanonicalKey);
        var byCanonical = realAnchors.FirstOrDefault(x =>
            string.Equals(
                NormalizeAnchorKey(HybridBenchmarkRepository.ResolveSourceBaselineForProvider(x.Quant).CanonicalKey),
                predictedKey,
                StringComparison.Ordinal));

        if (byCanonical != null)
            return byCanonical;

        return realAnchors.FirstOrDefault(x =>
            HybridBenchmarkRepository.ResolveSourceBaselineForProvider(x.Quant).UniqueId == predictedAnchor.RuntimeBaselineId);
    }

    private static void LogPredictionAndRealPairLines(
        BenchmarkSnapshotRecord realHigherDamageSmaller,
        BenchmarkSnapshotRecord realLowerDamageLarger,
        PredictedAnchorRow predictedHigherDamageSmaller,
        PredictedAnchorRow predictedLowerDamageLarger,
        string label)
    {
        AnsiConsole.MarkupLine($"[grey]{Markup.Escape(label)}:[/] [cyan]{Markup.Escape(realHigherDamageSmaller.DisplayName)}[/] -> [cyan]{Markup.Escape(realLowerDamageLarger.DisplayName)}[/]");
        AnsiConsole.MarkupLine($"[grey]  Prediction line:[/] {Markup.Escape(predictedHigherDamageSmaller.DisplayName)} size={predictedHigherDamageSmaller.PredictedSizeBytes:N0} kld={predictedHigherDamageSmaller.PredictedKld:0.000000} -> {Markup.Escape(predictedLowerDamageLarger.DisplayName)} size={predictedLowerDamageLarger.PredictedSizeBytes:N0} kld={predictedLowerDamageLarger.PredictedKld:0.000000}");
        AnsiConsole.MarkupLine($"[grey]  Real validation line:[/] {Markup.Escape(realHigherDamageSmaller.DisplayName)} size={realHigherDamageSmaller.SizeBytes:N0} kld={realHigherDamageSmaller.Kld:0.000000} -> {Markup.Escape(realLowerDamageLarger.DisplayName)} size={realLowerDamageLarger.SizeBytes:N0} kld={realLowerDamageLarger.Kld:0.000000}");
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
        table.AddColumn("Family");
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
                Markup.Escape(string.IsNullOrWhiteSpace(c.CandidateTheoryFamilyDisplay) ? "n/a" : c.CandidateTheoryFamilyDisplay),
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
            RawSelectionRank = candidate.RawSelectionRank,
            CandidateTheoryFamilyKey = candidate.CandidateTheoryFamilyKey,
            CandidateTheoryFamilyDisplay = candidate.CandidateTheoryFamilyDisplay,
            CandidateTheoryFamilyRank = candidate.CandidateTheoryFamilyRank,
            CandidateTheoryFamilyMemberRank = candidate.CandidateTheoryFamilyMemberRank,
            DiversityMode = candidate.DiversityMode,
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

    private static object? ToPredictionAnchorLog(PredictedAnchorRow? anchor)
    {
        if (anchor == null)
            return null;

        return new
        {
            configKey = anchor.ConfigKey,
            displayName = anchor.DisplayName,
            baselineCanonicalKey = anchor.BaselineCanonicalKey,
            runtimeBaselineId = anchor.RuntimeBaselineId,
            predictedSizeBytes = anchor.PredictedSizeBytes,
            predictedSizeGiB = ToGiB(anchor.PredictedSizeBytes),
            predictedKld = anchor.PredictedKld,
            predictionRank = anchor.PredictionRank,
            predictionConfidence = anchor.PredictionConfidence,
            isolationSource = DescribeVirtualAnchorIsolationSource(anchor),
            isVirtualPredictionAnchor = anchor.IsVirtualPredictionAnchor
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
                allowEightBitAnchorReplacements = Config.SelectionAllowEightBitAnchorReplacements,
                diversifyValidationCandidates = Config.SelectionDiversifyValidationCandidates,
                diversityScanMultiplier = Config.SelectionDiversityScanMultiplier,
                diversityScanMinCandidates = Config.SelectionDiversityScanMinCandidates,
                diversityScanMaxCandidates = Config.SelectionDiversityScanMaxCandidates,
                diversityLowBitOnly = Config.SelectionDiversityLowBitOnly
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
                predictionWindowMinSizeBytes = c.PredictionWindowMinSizeBytes,
                predictionWindowMaxSizeBytes = c.PredictionWindowMaxSizeBytes,
                realValidationWindowMinSizeBytes = c.WindowMinSizeBytes,
                realValidationWindowMaxSizeBytes = c.WindowMaxSizeBytes,
                candidatePoolSize = c.CandidatePoolSize,
                windowCandidateCount = c.WindowCandidateCount,
                lineBeatingCandidateCount = c.LineBeatingCandidateCount,
                fetchedCandidateCount = c.FetchedCandidateCount,
                candidatesAfterBrutalityCount = c.CandidatesAfterBrutalityCount,
                candidateAttemptLimit = c.CandidateAttemptLimit,
                rawSelectionRank = c.RawSelectionRank,
                diversityMode = c.DiversityMode,
                selectionStrategy = c.CandidateSelectionNotes.FirstOrDefault(x => x.Contains("selectionStrategy=family-size-ladder", StringComparison.Ordinal)),
                candidateTheoryFamilyKey = c.CandidateTheoryFamilyKey,
                candidateTheoryFamilyDisplay = c.CandidateTheoryFamilyDisplay,
                candidateTheoryFamilyRank = c.CandidateTheoryFamilyRank,
                candidateTheoryFamilyMemberRank = c.CandidateTheoryFamilyMemberRank,
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
                realHigherDamageSmaller = ToAnchorLog(c.HigherDamageAnchor),
                realLowerDamageLarger = ToAnchorLog(c.LowerDamageAnchor),
                predictionHigherDamageSmaller = ToPredictionAnchorLog(c.HigherDamagePredictionAnchor),
                predictionLowerDamageLarger = ToPredictionAnchorLog(c.LowerDamagePredictionAnchor)
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

    private async Task<long> CountPredictedRowsInIntersectedSizeWindowAsync(
        ulong predictionMin,
        ulong predictionMax,
        ulong realMin,
        ulong realMax,
        CancellationToken ct)
    {
        var window = IntersectSizeWindows(predictionMin, predictionMax, realMin, realMax);
        if (window == null)
            return 0;

        return await _predictedStore.CountPredictedHybridCandidatesInSizeWindowAsync(window.Value.Min, window.Value.Max, ct);
    }

    private static (ulong Min, ulong Max)? IntersectSizeWindows(
        ulong firstMin,
        ulong firstMax,
        ulong secondMin,
        ulong secondMax)
    {
        ulong min = Math.Max(firstMin, secondMin);
        ulong max = Math.Min(firstMax, secondMax);
        return max < min ? null : (min, max);
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

    private static string NormalizeAnchorKey(string? value) => (value ?? string.Empty).Trim().ToLowerInvariant();

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

    private sealed class ValidationCandidateSelectionResult
    {
        public IReadOnlyList<HybridSelectionCandidate> Candidates { get; init; } = Array.Empty<HybridSelectionCandidate>();
        public string Mode { get; init; } = string.Empty;
        public bool DiversityEnabled { get; init; }
        public string SelectionStrategy { get; init; } = string.Empty;
        public int CandidateFamilyCount { get; init; }
        public int SelectedFamilyCount { get; init; }
        public IReadOnlyList<string> SelectedFamilyKeys { get; init; } = Array.Empty<string>();
        public string DiversitySelectionStrategy { get; init; } = string.Empty;
        public int DiversityOverflowCount { get; init; }
        public ulong? DiversitySizeFloorStartBytes { get; init; }
        public ulong? DiversitySizeFloorEndBytes { get; init; }
        public int OverflowCount { get; init; }
        public ulong? SizeFloorStartBytes { get; init; }
        public ulong? SizeFloorEndBytes { get; init; }
        public IReadOnlyList<string> Notes { get; init; } = Array.Empty<string>();
    }

    private sealed class CandidateTheorySignature
    {
        public string Key { get; init; } = string.Empty;
        public string Display { get; init; } = string.Empty;
    }

    private sealed class CandidateFamilyEntry
    {
        public HybridSelectionCandidate Candidate { get; init; } = default!;
        public CandidateTheorySignature Signature { get; init; } = new();
        public int RawRank { get; init; }
        public int MemberRank { get; set; }
        public CandidateTheoryFamily? Family { get; set; }
        public string SelectionMode { get; set; } = string.Empty;
        public ulong? PreviousSizeFloorBytes { get; set; }
        public long? SizeDeltaVsFloorBytes { get; set; }
        public double EffectivePredictedKld { get; set; } = double.NaN;
    }

    private sealed class CandidateTheoryFamily
    {
        public string Key { get; init; } = string.Empty;
        public string Display { get; init; } = string.Empty;
        public int Rank { get; set; }
        public List<CandidateFamilyEntry> Members { get; init; } = new();
    }

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
        public object? PredictionHigherDamageSmaller { get; init; }
        public object? PredictionLowerDamageLarger { get; init; }
        public ulong PredictionWindowMinSizeBytes { get; init; }
        public ulong PredictionWindowMaxSizeBytes { get; init; }
        public ulong WindowMinSizeBytes { get; init; }
        public ulong WindowMaxSizeBytes { get; init; }
        public double WindowSizeGiB { get; init; }
        public long CandidatePoolSize { get; init; }
        public long PredictedPoolCount { get; init; }
        public long DeterministicEligibleCount { get; init; }
        public long RejectedByRealSizeWindow { get; init; }
        public long RejectedByOtherPhaseDeterministicRules { get; init; }
        public long WindowCandidateCount { get; init; }
        public long LineBeatingCandidateCount { get; init; }
        public int FetchedCandidateCount { get; init; }
        public int CandidatesAfterBrutalityCount { get; init; }
        public int SelectedForValidationCount { get; init; }
        public int CandidateAttemptLimit { get; init; }
        public int QueryFetchLimit { get; init; }
        public bool DiversityEnabled { get; init; }
        public string DiversityMode { get; init; } = string.Empty;
        public int DiversityScanLimit { get; init; }
        public int DiversityScanFetched { get; init; }
        public int CandidateFamilyCount { get; init; }
        public int SelectedFamilyCount { get; init; }
        public IReadOnlyList<string> SelectedFamilyKeys { get; init; } = Array.Empty<string>();
        public string DiversitySelectionStrategy { get; init; } = string.Empty;
        public int DiversityOverflowCount { get; init; }
        public ulong? DiversitySizeFloorStartBytes { get; init; }
        public ulong? DiversitySizeFloorEndBytes { get; init; }
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
        public int RawSelectionRank { get; init; }
        public string CandidateTheoryFamilyKey { get; init; } = string.Empty;
        public string CandidateTheoryFamilyDisplay { get; init; } = string.Empty;
        public int CandidateTheoryFamilyRank { get; init; }
        public int CandidateTheoryFamilyMemberRank { get; init; }
        public string DiversityMode { get; init; } = string.Empty;
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