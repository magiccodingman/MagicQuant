using MagicQuant.Models;
using MagicQuant.Services.Progress;
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
    private readonly QuantizationService _quantizationService;
    private readonly HybridBenchmarkRepository _repository;
    private readonly FinalRealBenchmarkEliminationService _finalEliminator;

    public PredictionGuidedHybridSelectionService(
        QuantizationService quantizationService,
        HybridBenchmarkRepository repository,
        FinalRealBenchmarkEliminationService finalEliminator)
    {
        _quantizationService = quantizationService;
        _repository = repository;
        _finalEliminator = finalEliminator;
    }

    public async Task<PredictionGuidedSelectionResult> RunAsync(
        IReadOnlyList<RankSafePredictionRow> predictions,
        IReadOnlyList<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        CancellationToken ct = default)
    {
        var eliminationRecords = new List<BaselineEliminationRecord>();
        var validationFailures = new List<CandidateValidationResult>();

        var current = _finalEliminator.Eliminate(pureBaselineSnapshots).Survivors.ToList();
        AnsiConsole.MarkupLine($"[green]Pure/current anchor survivors after dominance:[/] [cyan]{current.Count:N0}[/]");

        var strict = await RunStrictDominanceReplacementAsync(predictions, current, eliminationRecords, validationFailures, ct);
        current = MergeAndDominanceFilter(current, strict.AcceptedSnapshots, eliminationRecords, "strict predicted hybrid dominance validated by real benchmark");

        var near = await RunNearBaselineReplacementAsync(predictions, current, eliminationRecords, validationFailures, ct);
        current = MergeAndDominanceFilter(current, near.AcceptedSnapshots, eliminationRecords, "near-baseline size-premium replacement validated by real benchmark");

        var interior = await RunInteriorSubspaceDiscoveryAsync(predictions, current, validationFailures, ct);
        current = MergeAndDominanceFilter(current, interior.AcceptedSnapshots, eliminationRecords, "interior subspace discovery dominated by real benchmark truth");

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

        return new PredictionGuidedSelectionResult
        {
            Survivors = finalDominance.Survivors.ToList(),
            Eliminations = eliminationRecords
                .DistinctBy(x => $"{TensorConfigIdentity.ToKey(x.Eliminated.Config)}::{TensorConfigIdentity.ToKey(x.Eliminator.Config)}::{x.Reason}")
                .ToList(),
            ValidationFailures = validationFailures
        };
    }

    private async Task<PhaseValidationResult> RunStrictDominanceReplacementAsync(
        IReadOnlyList<RankSafePredictionRow> predictions,
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 1: Strict Hybrid Dominance[/]") { Justification = Justify.Left });

        var accepted = new List<BenchmarkSnapshotRecord>();
        var hybridPredictions = predictions
            .Where(x => x.IsPredictable && x.IsSizePredictable && x.IsHybrid)
            .ToList();

        foreach (var anchor in currentAnchors.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes))
        {
            if (ShouldSkipAnchorReplacement(anchor))
            {
                AnsiConsole.MarkupLine($"[grey]Skipping 8-bit anchor replacement attempts:[/] {Markup.Escape(anchor.DisplayName)}");
                continue;
            }

            var candidates = hybridPredictions
                .Where(x => x.PredictedSizeBytes <= anchor.SizeBytes)
                .Where(x => x.PredictedKld + Config.SelectionMinimumKldImprovementEpsilon < anchor.Kld)
                .OrderBy(x => x.PredictedSizeBytes)
                .ThenBy(x => x.PredictedKld)
                .Take(Config.SelectionMaxFallbackAttemptsPerAnchor)
                .Select((x, i) => new HybridSelectionCandidate
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
                    WindowLabel = $"strict <= {anchor.DisplayName}"
                })
                .ToList();

            if (candidates.Count == 0)
                continue;

            CandidateValidationResult? acceptedForAnchor = null;
            foreach (var candidate in candidates)
            {
                var validation = await BuildAndValidateSingleAsync(
                    candidate,
                    snapshot => snapshot.SizeBytes <= anchor.SizeBytes &&
                                snapshot.Kld + Config.SelectionMinimumKldImprovementEpsilon < anchor.Kld,
                    $"must be <= {anchor.SizeBytes:N0} bytes and lower KLD than {anchor.DisplayName}",
                    ct);

                if (validation.Accepted && validation.Snapshot != null)
                {
                    acceptedForAnchor = validation;
                    accepted.Add(validation.Snapshot);
                    eliminations.Add(new BaselineEliminationRecord
                    {
                        Eliminated = anchor,
                        Eliminator = validation.Snapshot,
                        Reason = "strict hybrid dominance: lower KLD at same-or-smaller real size"
                    });
                    break;
                }

                validationFailures.Add(validation);
            }

            if (acceptedForAnchor == null)
            {
                AnsiConsole.MarkupLine($"[grey]No strict predicted replacement validated for anchor:[/] {Markup.Escape(anchor.DisplayName)}");
            }
        }

        return new PhaseValidationResult { AcceptedSnapshots = accepted };
    }

    private async Task<PhaseValidationResult> RunNearBaselineReplacementAsync(
        IReadOnlyList<RankSafePredictionRow> predictions,
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<BaselineEliminationRecord> eliminations,
        List<CandidateValidationResult> validationFailures,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 2: Near-Baseline Replacement[/]") { Justification = Justify.Left });

        var accepted = new List<BenchmarkSnapshotRecord>();
        var pairs = BuildAdjacentPairs(currentAnchors);

        foreach (var pair in pairs)
        {
            var lowerSizeHigherDamage = pair.HigherDamageSmaller;
            var upperSizeLowerDamage = pair.LowerDamageLarger;

            if (ShouldSkipAnchorReplacement(lowerSizeHigherDamage))
                continue;

            ulong min = lowerSizeHigherDamage.SizeBytes;
            ulong max = AddPercent(min, Config.SelectionNearBaselineMaxSizeGrowthPercent);

            if (max > upperSizeLowerDamage.SizeBytes)
                max = upperSizeLowerDamage.SizeBytes;

            var candidates = FindBetterThanLinearCandidates(
                    predictions,
                    lowerSizeHigherDamage,
                    upperSizeLowerDamage,
                    min,
                    max,
                    HybridSelectionReason.NearBaselineOnePercentReplacement,
                    $"near-baseline +{Config.SelectionNearBaselineMaxSizeGrowthPercent:0.###}% {lowerSizeHigherDamage.DisplayName}")
                .Take(Config.SelectionMaxFallbackAttemptsPerAnchor)
                .ToList();

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
        IReadOnlyList<RankSafePredictionRow> predictions,
        IReadOnlyList<BenchmarkSnapshotRecord> currentAnchors,
        List<CandidateValidationResult> validationFailures,
        CancellationToken ct)
    {
        AnsiConsole.Write(new Rule("[yellow]Prediction Phase 3: Interior Subspace Discovery[/]") { Justification = Justify.Left });

        var accepted = new List<BenchmarkSnapshotRecord>();
        var pairs = BuildAdjacentPairs(currentAnchors);

        var allCandidates = new List<HybridSelectionCandidate>();

        foreach (var pair in pairs)
        {
            ulong lowSize = pair.HigherDamageSmaller.SizeBytes;
            ulong highSize = pair.LowerDamageLarger.SizeBytes;

            if (highSize <= lowSize)
                continue;

            ulong span = highSize - lowSize;
            ulong cursor = lowSize;

            for (int i = 0; i < Config.SelectionInteriorWindowFractions.Count; i++)
            {
                double fraction = Config.SelectionInteriorWindowFractions[i];
                if (fraction <= 0d)
                    continue;

                ulong width = (ulong)Math.Round(span * fraction, MidpointRounding.AwayFromZero);
                if (width == 0)
                    continue;

                ulong min = cursor;
                ulong max = i == Config.SelectionInteriorWindowFractions.Count - 1
                    ? Math.Min(highSize, cursor + width)
                    : Math.Min(highSize, cursor + width);

                if (max <= min)
                    continue;

                allCandidates.AddRange(
                    FindBetterThanLinearCandidates(
                            predictions,
                            pair.HigherDamageSmaller,
                            pair.LowerDamageLarger,
                            min,
                            max,
                            HybridSelectionReason.InteriorSubspaceDiscovery,
                            $"interior {i + 1}: {pair.HigherDamageSmaller.DisplayName} -> {pair.LowerDamageLarger.DisplayName}")
                        .Take(Config.SelectionMaxCandidatesPerInteriorWindow));

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

        if (deduped.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]No predicted interior candidates beat their local linear KLD lines.[/]");
            return new PhaseValidationResult();
        }

        AnsiConsole.MarkupLine($"[grey]Interior candidates selected for batch validation:[/] [cyan]{deduped.Count:N0}[/]");

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

            if (acceptedCandidate && snapshot != null)
            {
                accepted.Add(snapshot);
                continue;
            }

            validationFailures.Add(new CandidateValidationResult
            {
                Candidate = candidate,
                Snapshot = snapshot,
                Accepted = false,
                Message = snapshot == null
                    ? "benchmark snapshot was not found after batch build"
                    : "real benchmark did not beat the local linear KLD line inside the requested size window"
            });
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
            $"[grey]| reason=[/] {candidate.Reason} [grey]| window=[/] {Markup.Escape(candidate.WindowLabel)}");

        var summary = await _quantizationService.ProcessHybridBatchAsync(
            new[] { candidate.Prediction.Quant },
            new StageProgressOptions
            {
                StageName = "Single candidate validation",
                Total = 1,
                ShowEta = false,
                MinimumPrintInterval = TimeSpan.FromSeconds(5)
            },
            ct);
        var snapshot = await _repository.LoadBenchmarkSnapshotAsync(candidate.Prediction.Config, ct);

        bool accepted = snapshot != null && accept(snapshot);
        string message = accepted
            ? "validated"
            : snapshot == null
                ? $"no benchmark snapshot was available after build attempt (completed={summary.Completed}, skipped={summary.Skipped}, failed={summary.Failed})"
                : $"failed expectation: {expectation}; actual size={snapshot.SizeBytes:N0}, actual KLD={snapshot.Kld:0.000000}";

        if (accepted && snapshot != null)
        {
            AnsiConsole.MarkupLine($"[green]Validated:[/] {Markup.Escape(snapshot.DisplayName)} size={snapshot.SizeBytes:N0} KLD={snapshot.Kld:0.000000}");
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
            Message = message
        };
    }

    private List<HybridSelectionCandidate> FindBetterThanLinearCandidates(
        IReadOnlyList<RankSafePredictionRow> predictions,
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        ulong minSize,
        ulong maxSize,
        HybridSelectionReason reason,
        string windowLabel)
    {
        if (maxSize < minSize)
            return new List<HybridSelectionCandidate>();

        var result = predictions
            .Where(x => x.IsPredictable && x.IsSizePredictable && x.IsHybrid)
            .Where(x => x.PredictedSizeBytes >= minSize && x.PredictedSizeBytes <= maxSize)
            .Select(x =>
            {
                double line = InterpolateKldLine(x.PredictedSizeBytes, higherDamageSmaller, lowerDamageLarger);
                double gain = line - x.PredictedKld;
                return new HybridSelectionCandidate
                {
                    Prediction = x,
                    Reason = reason,
                    HigherDamageAnchor = higherDamageSmaller,
                    LowerDamageAnchor = lowerDamageLarger,
                    WindowMinSizeBytes = minSize,
                    WindowMaxSizeBytes = maxSize,
                    LinearExpectedKld = line,
                    PredictedGainOverLine = gain,
                    WindowLabel = windowLabel
                };
            })
            .Where(x => x.PredictedGainOverLine > Config.SelectionMinimumKldImprovementEpsilon)
            .Where(x => PassesNearLowerAnchorBrutality(x))
            .OrderByDescending(x => x.PredictedGainOverLine)
            .ThenBy(x => x.Prediction.PredictedSizeBytes)
            .ThenBy(x => x.Prediction.PredictedKld)
            .Select((x, i) =>
            {
                x = new HybridSelectionCandidate
                {
                    Prediction = x.Prediction,
                    Reason = x.Reason,
                    HigherDamageAnchor = x.HigherDamageAnchor,
                    LowerDamageAnchor = x.LowerDamageAnchor,
                    WindowMinSizeBytes = x.WindowMinSizeBytes,
                    WindowMaxSizeBytes = x.WindowMaxSizeBytes,
                    LinearExpectedKld = x.LinearExpectedKld,
                    PredictedGainOverLine = x.PredictedGainOverLine,
                    WindowLabel = x.WindowLabel,
                    AttemptOrder = i + 1
                };
                return x;
            })
            .ToList();

        return result;
    }

    private static bool PassesNearLowerAnchorBrutality(HybridSelectionCandidate candidate)
    {
        ulong span = candidate.LowerDamageAnchor.SizeBytes > candidate.HigherDamageAnchor.SizeBytes
            ? candidate.LowerDamageAnchor.SizeBytes - candidate.HigherDamageAnchor.SizeBytes
            : 0;

        if (span == 0)
            return true;

        ulong distanceFromSmall = candidate.Prediction.PredictedSizeBytes > candidate.HigherDamageAnchor.SizeBytes
            ? candidate.Prediction.PredictedSizeBytes - candidate.HigherDamageAnchor.SizeBytes
            : 0;

        double fraction = distanceFromSmall / (double)span;
        if (fraction > Config.SelectionNearLowerAnchorBrutalZoneFractionOfPairSpan)
            return true;

        double requiredGain = Math.Max(
            Config.SelectionMinimumKldImprovementEpsilon,
            Math.Abs(candidate.HigherDamageAnchor.Kld - candidate.LowerDamageAnchor.Kld) *
            Config.SelectionNearAnchorRequiredKldGainFractionOfPairGap);

        return candidate.PredictedGainOverLine >= requiredGain;
    }

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

    private static bool Dominates(BenchmarkSnapshotRecord better, BenchmarkSnapshotRecord worse)
    {
        bool sameOrSmaller = better.SizeBytes <= worse.SizeBytes;
        bool strictlyLowerKld = better.Kld + Config.SelectionMinimumKldImprovementEpsilon < worse.Kld;
        return sameOrSmaller && strictlyLowerKld;
    }

    private sealed class AdjacentAnchorPair
    {
        public BenchmarkSnapshotRecord LowerDamageLarger { get; init; } = default!;
        public BenchmarkSnapshotRecord HigherDamageSmaller { get; init; } = default!;
    }
}
