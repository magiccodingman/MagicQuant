using System.Globalization;
using System.Numerics;
using System.Text.Json;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class AnomalyWorkflowService
{
    private const int DuckSmokeScanLimit = 0; // 0 means scan all predicted DuckDB rows; anomaly smoke must not be top-rank truncated.

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private readonly RemainingCombinationStore _store;
    private readonly HybridBenchmarkRepository _repository;
    private readonly QuantizationService _quantizationService;
    private readonly QuantFidelityComparerService _movement;
    private readonly AnomalyRuleRepository _rules;
    private readonly AnomalyAdjustedPredictionService _adjuster;

    public AnomalyWorkflowService(
        RemainingCombinationStore store,
        HybridBenchmarkRepository repository,
        QuantizationService quantizationService)
    {
        _store = store;
        _repository = repository;
        _quantizationService = quantizationService;
        _movement = new QuantFidelityComparerService();
        _rules = new AnomalyRuleRepository(_movement);
        _adjuster = new AnomalyAdjustedPredictionService(store);
    }

    public async Task<AnomalyRunResult> RunAsync(
        IReadOnlyCollection<BenchmarkSnapshotRecord> pureBaselineSnapshots,
        CancellationToken ct = default)
    {
        if (!Config.AnomalyDetection.Enabled || Config.AnomalyDetection.MaxAnomalyRefinementRounds <= 0)
        {
            AnsiConsole.MarkupLine("[grey]Anomaly detection disabled by config.[/]");
            return new AnomalyRunResult();
        }

        AnsiConsole.Write(new Rule("[yellow]Counterfactual Anomaly Smoke / Probe Pass[/]") { Justification = Justify.Left });

        var session = await _rules.StartSessionAsync("prediction-guided-selection", ct);
        try
        {
            var historical = await DetectHistoricalSmokeAsync(ct);
            var duck = await DetectDuckSmokeAsync(ct);
            var smoke = historical
                .Concat(duck)
                .GroupBy(x => TensorConfigIdentity.ToKey(x.CandidateConfig), StringComparer.Ordinal)
                .Select(g => g.OrderByDescending(x => x.IsConfirmedFromHistory).ThenByDescending(x => x.SmokeScore).First())
                .OrderByDescending(x => x.IsConfirmedFromHistory)
                .ThenByDescending(x => x.SmokeScore)
                .Take(Config.AnomalyDetection.MaxSmokeCandidatesPerReferenceZone * Math.Max(1, RuntimeSearchSpace.GetActiveCombinationBaselines().Count))
                .ToList();

            WriteSmokeConsoleSummary(historical.Count, duck.Count, smoke);
            await WriteJsonAsync("magicquant-anomaly-smoke-scan.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                historicalCount = historical.Count,
                duckPredictionSpaceCount = duck.Count,
                selectedSmokeCount = smoke.Count,
                smoke = smoke.Select(ToSmokeLog).ToList()
            }, ct);

            await WriteJsonAsync("magicquant-anomaly-seeds.json", smoke.Select(ToSmokeLog).ToList(), ct);

            var probes = await PlanProbesAsync(smoke, ct);
            await WriteJsonAsync("magicquant-anomaly-probes.json", probes.Select(ToProbeLog).ToList(), ct);

            var results = await ValidateProbesAsync(probes, ct);
            await _rules.PersistProbeResultsAsync(session.Id, results, ct);
            var upsertedRules = await _rules.UpsertRulesFromResultsAsync(results, ct);
            var applicableRules = await _rules.LoadApplicableRulesAsync(ct);
            var adjustment = await _adjuster.ApplyAsync(applicableRules, ct);

            await WriteJsonAsync("magicquant-anomaly-rules.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                upserted = upsertedRules.Select(ToRuleLog).ToList(),
                applicable = applicableRules.Select(ToRuleLog).ToList()
            }, ct);

            await WriteJsonAsync("magicquant-anomaly-adjusted-predictions-summary.json", adjustment, ct);
            await WriteFinalManifestAsync("magicquant.anomalies.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                smoke = smoke.Select(ToSmokeLog).ToList(),
                probes = probes.Select(ToProbeLog).ToList(),
                results = results.Select(ToResultLog).ToList(),
                rules = applicableRules.Select(ToRuleLog).ToList(),
                adjustment
            }, ct);
            await WriteFinalManifestAsync("magicquant.prediction-audit.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                note = "BaseRankSafeKld is normal PAVA gravity. FinalPredictedKld is BaseRankSafeKld plus scoped anomaly adjustments. Global PAVA is not rerun after anomaly exceptions.",
                adjustment
            }, ct);

            return new AnomalyRunResult
            {
                SmokeCandidates = smoke,
                ProbePlans = probes,
                ProbeResults = results,
                AdjustmentSummary = adjustment
            };
        }
        finally
        {
            await _rules.CompleteSessionAsync(session.Id, ct);
        }
    }


    private async Task<List<AnomalySmokeCandidate>> DetectHistoricalSmokeAsync(CancellationToken ct)
    {
        var snapshots = await LoadAllCurrentBenchmarkSnapshotsAsync(ct);
        var byKey = snapshots.ToDictionary(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal);
        var predictionLookup = await LoadPredictionLookupAsync(ct);
        var smoke = new List<AnomalySmokeCandidate>();
        int skippedIsolation = 0;
        int skippedSparse = 0;
        int skippedNonContextualTwin = 0;
        int skippedMixed = 0;
        int contextualScanned = 0;

        foreach (var candidate in snapshots.Where(x => !TensorConfigIdentity.IsPureBaseline(x.Config)))
        {
            if (ShouldSkipInvalidContextualAnomalyConfig(candidate.Config, "history", out var skipReason))
            {
                if (skipReason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase))
                {
                    skippedSparse++;
                    LogHistoricalSparseCandidateIgnored(candidate.Config, skipReason, skippedSparse);
                }
                else
                {
                    skippedIsolation++;
                    LogSkippedInvalidContextualAnomalyConfig("history", candidate.Config, skipReason, skippedIsolation);
                }
                continue;
            }

            contextualScanned++;
            var twinConfig = _movement.BuildBaseContextTwin(candidate.Config);
            if (TensorConfigIdentity.ToKey(twinConfig) == TensorConfigIdentity.ToKey(candidate.Config))
                continue;

            if (ShouldSkipInvalidContextualAnomalyConfig(twinConfig, "history-twin", out var twinSkipReason))
            {
                skippedNonContextualTwin++;
                LogSkippedInvalidContextualAnomalyConfig("history-twin", twinConfig, twinSkipReason, skippedNonContextualTwin);
                continue;
            }

            var movement = _movement.Analyze(twinConfig, candidate.Config);
            if (movement.Classification != AnomalyMovementClassification.MonotoneDowngrade)
            {
                if (movement.Classification == AnomalyMovementClassification.MixedTrade)
                {
                    skippedMixed++;
                    if (Config.AnomalyDetection.VerboseAnomalyLogging && skippedMixed <= 12)
                    {
                        AnsiConsole.MarkupLine("[grey]Ignored anomaly smoke:[/] classification=MixedTrade reason=normal protect/compress frontier behavior");
                    }
                }
                continue;
            }

            if (movement.DowngradeCount > Config.AnomalyDetection.MaxProbeGroupCount)
                continue;

            byKey.TryGetValue(TensorConfigIdentity.ToKey(twinConfig), out var twin);
            if (twin != null && ShouldSkipInvalidContextualAnomalyConfig(twin.Config, "history-existing-twin", out var existingTwinSkipReason))
            {
                skippedNonContextualTwin++;
                LogSkippedInvalidContextualAnomalyConfig("history-existing-twin", twin.Config, existingTwinSkipReason, skippedNonContextualTwin);
                continue;
            }

            predictionLookup.TryGetValue(TensorConfigIdentity.ToKey(candidate.Config), out var candidatePrediction);
            predictionLookup.TryGetValue(TensorConfigIdentity.ToKey(twinConfig), out var twinPrediction);

            bool confirmed = twin != null &&
                             candidate.SizeBytes <= twin.SizeBytes &&
                             twin.Kld - candidate.Kld >= Config.AnomalyDetection.MinActualGainVsTwinKld;

            if (!confirmed && twin != null && twin.Kld <= candidate.Kld)
                continue;

            smoke.Add(new AnomalySmokeCandidate
            {
                Source = "history",
                CandidateConfig = candidate.Config,
                TwinConfig = twinConfig,
                Movement = movement,
                CandidatePredictedKld = candidatePrediction?.BaseRankSafeKld ?? candidatePrediction?.FinalPredictedKld ?? candidate.Kld,
                TwinPredictedKld = twinPrediction?.BaseRankSafeKld ?? twinPrediction?.FinalPredictedKld ?? twin?.Kld ?? 0d,
                CandidatePredictedSizeBytes = candidatePrediction?.PredictedSizeBytes ?? candidate.SizeBytes,
                TwinPredictedSizeBytes = twinPrediction?.PredictedSizeBytes ?? twin?.SizeBytes ?? candidate.SizeBytes,
                SizeSavingsBytes = twin != null && twin.SizeBytes > candidate.SizeBytes ? twin.SizeBytes - candidate.SizeBytes : 0UL,
                PredictionSpaceGapVsTwin = (candidatePrediction?.BaseRankSafeKld ?? candidate.Kld) - (twinPrediction?.BaseRankSafeKld ?? twin?.Kld ?? candidate.Kld),
                CandidatePredictionRank = candidatePrediction?.PredictionRank,
                TwinPredictionRank = twinPrediction?.PredictionRank,
                SmokeScore = confirmed ? 1_000_000d : 100d,
                SmokeStrength = confirmed ? "ConfirmedHistory" : "HistoricalMissingTwin",
                HasActualTwin = twin != null,
                CandidateActualKld = candidate.Kld,
                TwinActualKld = twin?.Kld,
                CandidateActualSizeBytes = candidate.SizeBytes,
                TwinActualSizeBytes = twin?.SizeBytes,
                IsConfirmedFromHistory = confirmed,
                Message = confirmed
                    ? "Existing explicit contextual quantized benchmark history contains a monotone downgrade candidate that beats its higher-bit twin."
                    : "Existing explicit contextual quantized benchmark history has monotone downgrade smoke but the exact twin is missing."
            });
        }

        if (skippedIsolation > 0 || skippedNonContextualTwin > 0 || skippedSparse > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Historical contextual anomaly scan:[/] contextualScanned={contextualScanned:N0} skippedIsolation={skippedIsolation:N0} sparseIgnored={skippedSparse:N0} skippedTwins={skippedNonContextualTwin:N0}");
        }

        return smoke;
    }


    private async Task<List<AnomalySmokeCandidate>> DetectDuckSmokeAsync(CancellationToken ct)
    {
        var rows = await LoadPredictionRowsAsync(DuckSmokeScanLimit, ct);
        var explicitRows = new Dictionary<string, PredictionDuckRow>(StringComparer.Ordinal);
        var result = new List<AnomalySmokeCandidate>();
        int skippedIsolation = 0;
        int skippedSparse = 0;
        int normalizedSparse = 0;
        int skippedPure = 0;
        int skippedMixed = 0;
        int skippedMovement = 0;
        int skippedNoTwin = 0;
        int logicalTwinFallback = 0;
        int skippedSavings = 0;
        int skippedGap = 0;
        int contextualScanned = 0;

        foreach (var row in rows)
        {
            if (!_movement.TryNormalizeSparseDuckRowToActivatedContext(row.Config, out var activated, out var wasSparse, out var normalizeReason))
            {
                if (normalizeReason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase))
                    skippedSparse++;
                else
                    skippedIsolation++;

                LogSkippedInvalidContextualAnomalyConfig("duckdb", row.Config, normalizeReason, skippedIsolation + skippedSparse);
                continue;
            }

            if (wasSparse)
                normalizedSparse++;

            if (TensorConfigIdentity.ToKey(activated) == TensorConfigIdentity.ToKey(_movement.BuildBaseContextTwin(activated)))
            {
                skippedPure++;
                continue;
            }

            var normalizedRow = row with { Config = activated };
            string normalizedKey = TensorConfigIdentity.ToKey(activated);
            if (!explicitRows.ContainsKey(normalizedKey))
                explicitRows[normalizedKey] = normalizedRow;
        }

        foreach (var row in explicitRows.Values)
        {
            contextualScanned++;
            var twin = _movement.BuildBaseContextTwin(row.Config);

            if (ShouldSkipInvalidContextualAnomalyConfig(row.Config, "duckdb-normalized-candidate", out var candidateSkipReason))
            {
                skippedIsolation++;
                LogSkippedInvalidContextualAnomalyConfig("duckdb-normalized-candidate", row.Config, candidateSkipReason, skippedIsolation);
                continue;
            }

            if (ShouldSkipInvalidContextualAnomalyConfig(twin, "duckdb-twin", out var twinSkipReason))
            {
                skippedIsolation++;
                LogSkippedInvalidContextualAnomalyConfig("duckdb-twin", twin, twinSkipReason, skippedIsolation);
                continue;
            }

            var movement = _movement.Analyze(twin, row.Config);
            if (movement.Classification == AnomalyMovementClassification.MixedTrade)
            {
                skippedMixed++;
                if (Config.AnomalyDetection.VerboseAnomalyLogging && skippedMixed <= 12)
                {
                    AnsiConsole.MarkupLine("[grey]Ignored anomaly smoke:[/] classification=MixedTrade reason=normal protect/compress frontier behavior");
                }
                continue;
            }

            if (movement.Classification != AnomalyMovementClassification.MonotoneDowngrade)
            {
                skippedMovement++;
                continue;
            }

            if (movement.DowngradeCount <= 0 || movement.DowngradeCount > Config.AnomalyDetection.MaxProbeGroupCount)
            {
                skippedMovement++;
                continue;
            }

            var twinRow = await LoadContextualTwinPredictionRowAsync(twin, explicitRows, ct);
            if (twinRow == null)
            {
                skippedNoTwin++;
                continue;
            }

            if (TensorConfigIdentity.ToKey(twinRow.Config) != TensorConfigIdentity.ToKey(twin))
                logicalTwinFallback++;

            if (twinRow.PredictedSizeBytes <= row.PredictedSizeBytes)
            {
                skippedSavings++;
                continue;
            }

            ulong savingsBytes = twinRow.PredictedSizeBytes - row.PredictedSizeBytes;
            double savingsPercent = savingsBytes * 100d / Math.Max(1d, twinRow.PredictedSizeBytes);
            if (savingsPercent < Config.AnomalyDetection.MinPredictedSizeSavingsVsTwinPercent)
            {
                skippedSavings++;
                continue;
            }

            double gap = row.BaseRankSafeKld - twinRow.BaseRankSafeKld;
            if (gap > Config.AnomalyDetection.MaxPredictionSpaceGapVsTwinKld)
            {
                skippedGap++;
                continue;
            }

            double score = ComputeSmokeScore(gap, savingsPercent, movement.DowngradeCount, row.PredictionRank, twinRow.PredictionRank);

            result.Add(new AnomalySmokeCandidate
            {
                Source = "duckdb-prediction-space",
                CandidateConfig = row.Config,
                TwinConfig = twin,
                Movement = movement,
                CandidatePredictedKld = row.BaseRankSafeKld,
                TwinPredictedKld = twinRow.BaseRankSafeKld,
                CandidatePredictedSizeBytes = row.PredictedSizeBytes,
                TwinPredictedSizeBytes = twinRow.PredictedSizeBytes,
                SizeSavingsBytes = savingsBytes,
                PredictionSpaceGapVsTwin = gap,
                CandidatePredictionRank = row.PredictionRank,
                TwinPredictionRank = twinRow.PredictionRank,
                SmokeScore = score,
                SmokeStrength = gap <= 0d ? "Strong" : "Close",
                Message = "Prediction-space contextual monotone downgrade candidate is close enough to its higher-bit quantized twin to justify probes. Sparse DuckDB source rows, when present, were normalized into explicit active context before classification."
            });
        }

        AnsiConsole.MarkupLine("[yellow]DuckDB contextual smoke scan:[/]");
        AnsiConsole.MarkupLine($"[grey]  predicted rows scanned=[/] [cyan]{rows.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  sparse rows normalized to explicit context=[/] [cyan]{normalizedSparse:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  sparse rows skipped=[/] [cyan]{skippedSparse:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  BF16/exact rows skipped=[/] [cyan]{skippedIsolation:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  pure/logical reference rows skipped=[/] [cyan]{skippedPure:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  contextual quantized rows scanned=[/] [cyan]{contextualScanned:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  logical higher-bit twin fallback used=[/] [cyan]{logicalTwinFallback:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  no higher-bit twin found=[/] [cyan]{skippedNoTwin:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  movement not monotone downgrade=[/] [cyan]{skippedMovement:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  mixed trade ignored=[/] [cyan]{skippedMixed:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  size savings below threshold=[/] [cyan]{skippedSavings:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  prediction-space gap too large=[/] [cyan]{skippedGap:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  queued smoke candidates=[/] [cyan]{result.Count:N0}[/]");

        if (result.Count == 0 && rows.Count > 0)
        {
            AnsiConsole.MarkupLine(
                "[yellow]DuckDB contextual smoke scan produced zero candidates.[/] This is valid only if all predicted rows were filtered by explicit-context validity, monotone-downgrade movement, size-savings, or prediction-gap thresholds above.");
        }

        return result
            .GroupBy(x => x.TwinConfig.BaseQuant)
            .SelectMany(g => g.OrderByDescending(x => x.SmokeScore).Take(Config.AnomalyDetection.MaxSmokeCandidatesPerReferenceZone))
            .ToList();
    }


    private async Task<List<AnomalyProbePlan>> PlanProbesAsync(IReadOnlyList<AnomalySmokeCandidate> seeds, CancellationToken ct)
    {
        var plans = new List<AnomalyProbePlan>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var seed in seeds)
        {
            bool invalidSeedCandidate = ShouldSkipInvalidContextualAnomalyConfig(seed.CandidateConfig, "probe-seed-candidate", out var seedCandidateReason);
            bool invalidSeedTwin = ShouldSkipInvalidContextualAnomalyConfig(seed.TwinConfig, "probe-seed-twin", out var seedTwinReason);
            if (invalidSeedCandidate || invalidSeedTwin)
            {
                string reason = invalidSeedCandidate ? seedCandidateReason : seedTwinReason;
                AnsiConsole.MarkupLine($"[yellow]SkippedInvalidContextualAnomalyProbe:[/] source=probe-seed reason={Markup.Escape(reason)} candidate={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.CandidateQuant))} twin={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.TwinQuant))}");
                continue;
            }

            var reference = _movement.CreateActivatedContextBlanket(seed.TwinConfig.BaseQuant);
            _movement.EnsureAllActiveGroupsExplicit(reference, "probe-plan-reference");

            var changed = _movement.Analyze(reference, seed.CandidateConfig)
                .ChangedGroups
                .Where(x => x.Movement == QuantMovementKind.Downgrade)
                .OrderBy(x => x.Group.UniqueId)
                .Take(Config.AnomalyDetection.MaxProbeGroupCount)
                .ToList();

            if (changed.Count == 0)
                continue;

            var subsets = BuildProbeSubsets(changed);
            int perSeed = 0;
            foreach (var subset in subsets)
            {
                if (perSeed >= Config.AnomalyDetection.MaxProbesPerSeed || plans.Count >= Config.AnomalyDetection.MaxTotalProbesPerRun)
                    break;

                var probeConfig = reference;
                foreach (var g in subset)
                    probeConfig = _movement.WithStoredSlot(probeConfig, g.Group, g.CandidateStoredSlot);

                if (ShouldSkipInvalidContextualAnomalyConfig(probeConfig, "probe-plan", out var probeSkipReason))
                {
                    AnsiConsole.MarkupLine($"[yellow]SkippedInvalidContextualAnomalyProbe:[/] source=probe-plan reason={Markup.Escape(probeSkipReason)} probe={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)probeConfig))}");
                    continue;
                }

                _movement.EnsureAllActiveGroupsExplicit(probeConfig, "probe-plan-probe");

                if (await _rules.HasSuppressionOrRuleAsync(reference, subset, ct))
                    continue;

                string key = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probeConfig);
                if (!seen.Add(key))
                    continue;

                var plan = new AnomalyProbePlan
                {
                    Seed = seed,
                    ReferenceConfig = reference,
                    ProbeConfig = probeConfig,
                    ProbeGroups = subset,
                    ProbeType = ResolveProbeType(subset.Count, changed.Count),
                    HypothesisLabel = _movement.DescribeGroups(subset)
                };
                plans.Add(plan);
                perSeed++;
            }

            AnsiConsole.MarkupLine(
                $"[yellow]Potential anomaly smoke:[/] classification={seed.Movement.Classification} candidate={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.CandidateQuant))} " +
                $"higher-bit twin={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.TwinQuant))} changed groups={Markup.Escape(_movement.DescribeGroups(changed))} " +
                $"upgradeCount={seed.Movement.UpgradeCount} downgradeCount={seed.Movement.DowngradeCount} " +
                $"prediction-space gap={seed.PredictionSpaceGapVsTwin:0.000000} predicted size savings={seed.SizeSavingsBytes:N0} probes queued={perSeed:N0}");
        }

        return plans;
    }


    private async Task<List<AnomalyProbeResult>> ValidateProbesAsync(IReadOnlyList<AnomalyProbePlan> probes, CancellationToken ct)
    {
        if (probes.Count == 0)
            return new List<AnomalyProbeResult>();

        var results = new List<AnomalyProbeResult>();
        var safeProbes = new List<AnomalyProbePlan>();

        foreach (var plan in probes)
        {
            string referenceReason = string.Empty;
            string probeReason = string.Empty;
            bool invalidReference = ShouldSkipInvalidContextualAnomalyConfig(plan.ReferenceConfig, "validate-reference", out referenceReason);
            bool invalidProbe = ShouldSkipInvalidContextualAnomalyConfig(plan.ProbeConfig, "validate-probe", out probeReason);
            if (invalidReference || invalidProbe)
            {
                string reason = invalidReference ? referenceReason : probeReason;
                AnsiConsole.MarkupLine($"[yellow]SkippedInvalidContextualAnomalyProbe:[/] source=validate reason={Markup.Escape(reason)} reference={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)plan.ReferenceConfig))} probe={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)plan.ProbeConfig))}");
                results.Add(new AnomalyProbeResult
                {
                    Plan = plan,
                    Classification = AnomalyProbeClassification.SuppressionOnly,
                    RuleDirection = AnomalyRuleDirection.SuppressionOnly,
                    Accepted = false,
                    FailureCode = reason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase) ? "SKIPPED_SPARSE_CONTEXTUAL_PROBE" : "SKIPPED_ISOLATION_SAMPLE",
                    Message = reason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase)
                        ? "SkippedInvalidContextualAnomalyProbe: active groups must be explicitly stored for contextual anomaly probes."
                        : "SkippedIsolationSample: BF16/exact isolation rows are not contextual anomaly probes."
                });
                continue;
            }

            _movement.EnsureAllActiveGroupsExplicit(plan.ReferenceConfig, "pre-quantization-reference");
            _movement.EnsureAllActiveGroupsExplicit(plan.ProbeConfig, "pre-quantization-probe");
            safeProbes.Add(plan);
        }

        if (safeProbes.Count == 0)
            return results;

        foreach (var plan in safeProbes)
            WriteContextualProbeConsoleLog(plan);

        var quants = safeProbes
            .SelectMany(x => new[] { x.ReferenceConfig, x.ProbeConfig })
            .DistinctBy(TensorConfigIdentity.ToKey)
            .Select(x => (HybridQuant)x)
            .ToList();

        AnsiConsole.MarkupLine($"[grey]Validating anomaly probes:[/] unique quant builds/benchmarks=[cyan]{quants.Count:N0}[/] probe plans=[cyan]{safeProbes.Count:N0}[/]");
        await _quantizationService.ProcessHybridBatchAsync(quants, ct);

        foreach (var plan in safeProbes)
        {
            var reference = await _repository.LoadBenchmarkSnapshotAsync(plan.ReferenceConfig, ct);
            var probe = await _repository.LoadBenchmarkSnapshotAsync(plan.ProbeConfig, ct);
            var result = ClassifyProbe(plan, reference, probe);
            results.Add(result);

            WriteProbeOutcome(result);
        }

        return results;
    }

    private AnomalyProbeResult ClassifyProbe(
        AnomalyProbePlan plan,
        BenchmarkSnapshotRecord? reference,
        BenchmarkSnapshotRecord? probe)
    {
        if (reference == null)
        {
            return new AnomalyProbeResult
            {
                Plan = plan,
                ReferenceSnapshot = null,
                ProbeSnapshot = probe,
                Classification = AnomalyProbeClassification.MissingTwin,
                RuleDirection = AnomalyRuleDirection.SuppressionOnly,
                FailureCode = "REFERENCE_TWIN_MISSING",
                Message = "Reference/higher-bit twin benchmark was missing after anomaly probe validation."
            };
        }

        if (probe == null)
        {
            return new AnomalyProbeResult
            {
                Plan = plan,
                ReferenceSnapshot = reference,
                ProbeSnapshot = null,
                Classification = AnomalyProbeClassification.MissingProbeBenchmark,
                RuleDirection = AnomalyRuleDirection.SuppressionOnly,
                FailureCode = "PROBE_BENCHMARK_MISSING",
                Message = "Probe benchmark was missing after anomaly probe validation."
            };
        }

        double gain = reference.Kld - probe.Kld;
        bool sameOrSmaller = probe.SizeBytes <= reference.SizeBytes;
        if (sameOrSmaller && gain >= Config.AnomalyDetection.MinActualGainVsTwinKld)
        {
            return new AnomalyProbeResult
            {
                Plan = plan,
                ReferenceSnapshot = reference,
                ProbeSnapshot = probe,
                Classification = plan.ProbeType switch
                {
                    "single" => AnomalyProbeClassification.SingleGroupInversion,
                    "pair" => AnomalyProbeClassification.PairSynergy,
                    "full" when plan.ProbeGroups.Count >= 3 => AnomalyProbeClassification.HigherOrderSynergy,
                    _ => AnomalyProbeClassification.CounterfactualMdaViolation
                },
                RuleDirection = AnomalyRuleDirection.Beneficial,
                Accepted = true,
                ActualGainVsTwin = gain,
                Message = "Lower-fidelity monotone probe beat its higher-fidelity same-context twin."
            };
        }

        if (probe.Kld - reference.Kld >= Config.AnomalyDetection.MinActualGainVsTwinKld)
        {
            return new AnomalyProbeResult
            {
                Plan = plan,
                ReferenceSnapshot = reference,
                ProbeSnapshot = probe,
                Classification = AnomalyProbeClassification.HarmfulInteraction,
                RuleDirection = AnomalyRuleDirection.Harmful,
                Accepted = true,
                ActualGainVsTwin = gain,
                Message = "Probe was meaningfully worse than its higher-fidelity twin; persisted as harmful interaction."
            };
        }

        return new AnomalyProbeResult
        {
            Plan = plan,
            ReferenceSnapshot = reference,
            ProbeSnapshot = probe,
            Classification = AnomalyProbeClassification.NormalGravity,
            RuleDirection = AnomalyRuleDirection.SuppressionOnly,
            Accepted = false,
            ActualGainVsTwin = gain,
            FailureCode = "NORMAL_GRAVITY",
            Message = "Smoke rejected / normal MDA gravity confirmed."
        };
    }

    private async Task<List<BenchmarkSnapshotRecord>> LoadAllCurrentBenchmarkSnapshotsAsync(CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var modelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (modelHashId == null)
            return new List<BenchmarkSnapshotRecord>();

        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();
        int? imatrixId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, modelHashId.Value, createIfMissing: false, ct);

        var rows = await db.AiBenchmarks
            .AsNoTracking()
            .Include(x => x.TensorCombo)
            .Include(x => x.CategorBenchmarks)
            .Where(x => x.ArchitectureFamilyId == architectureFamilyId)
            .Where(x => x.TensorGroupProfileId == tensorGroupProfileId)
            .Where(x => x.AiModelHashId == modelHashId.Value)
            .Where(x => x.ImatrixDefinitionId == imatrixId)
            .ToListAsync(ct);

        var list = new List<BenchmarkSnapshotRecord>();
        foreach (var row in rows)
        {
            var general = row.CategorBenchmarks.FirstOrDefault(x => x.Category == (byte)BenchmarkCategory.General)
                          ?? row.CategorBenchmarks.OrderBy(x => x.Category).FirstOrDefault();
            if (general == null)
                continue;

            var config = new TensorConfig(row.TensorCombo.BaseQuant, row.TensorCombo.Embeddings, row.TensorCombo.LmHead, row.TensorCombo.AttnQ, row.TensorCombo.AttnKV, row.TensorCombo.AttnOutput, row.TensorCombo.FfnUpGate, row.TensorCombo.FfnDown, row.TensorCombo.MoeExperts, row.TensorCombo.MoeRouter);
            var quant = (HybridQuant)config;
            list.Add(new BenchmarkSnapshotRecord
            {
                Config = config,
                Quant = quant,
                DisplayName = HybridBenchmarkRepository.BuildDisplayName(quant),
                ProviderName = HybridBenchmarkRepository.ResolveProviderName(quant, exportNaming: false),
                BaselineFamily = HybridBenchmarkRepository.ResolveBaselineFamily(quant),
                IsHybrid = HybridBenchmarkRepository.IsTrueMagicQuantHybrid(quant),
                IsExternalRebuiltBaseline = HybridBenchmarkRepository.IsExternalRebuiltBaseline(quant),
                IsMaterializedTensorMapped = quant.Tensors.Count > 0,
                SizeBytes = row.SizeBytes,
                Kld = general.Kld,
                Ppl = general.Ppl
            });
        }

        return list;
    }

    private async Task<Dictionary<string, PredictionDuckRow>> LoadPredictionLookupAsync(CancellationToken ct)
    {
        var rows = await LoadPredictionRowsAsync(DuckSmokeScanLimit, ct);
        return rows.ToDictionary(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal);
    }

    private async Task<List<PredictionDuckRow>> LoadPredictionRowsAsync(int limit, CancellationToken ct)
    {
        using var c = new DuckDBConnection($"Data Source={_store.GetDatabaseFilePath()}");
        await c.OpenAsync(ct);
        await ConfigureDuckAsync(c, ct);

        using var cmd = c.CreateCommand();
        string limitSql = limit > 0 ? "\nLIMIT ?" : string.Empty;
        cmd.CommandText = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       COALESCE(BaseRankSafeKld, PredictedKld) AS BaseRankSafeKld,
       COALESCE(FinalPredictedKld, PredictedKld) AS FinalPredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank
FROM {CombinationDuckDbSchema.TableName}
WHERE COALESCE(BaseRankSafeKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
ORDER BY PredictionRank ASC{limitSql};";
        if (limit > 0)
            cmd.Parameters.Add(new DuckDBParameter { Value = limit });

        var rows = new List<PredictionDuckRow>();
        using var r = await cmd.ExecuteReaderAsync(ct);
        while (await r.ReadAsync(ct))
            rows.Add(ReadPredictionDuckRow(r));

        return rows;
    }

    private async Task<PredictionDuckRow?> LoadSinglePredictionRowAsync(TensorConfig config, CancellationToken ct)
    {
        using var c = new DuckDBConnection($"Data Source={_store.GetDatabaseFilePath()}");
        await c.OpenAsync(ct);
        await ConfigureDuckAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       COALESCE(BaseRankSafeKld, PredictedKld) AS BaseRankSafeKld,
       COALESCE(FinalPredictedKld, PredictedKld) AS FinalPredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank
FROM {CombinationDuckDbSchema.TableName}
WHERE BaseQuant = ?
  AND Embeddings = ?
  AND LmHead = ?
  AND AttnQ = ?
  AND AttnKV = ?
  AND AttnOutput = ?
  AND FfnUpGate = ?
  AND FfnDown = ?
  AND MoeExperts = ?
  AND MoeRouter = ?
LIMIT 1;";
        AddConfigParameters(cmd, config);

        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            return null;

        return ReadPredictionDuckRow(r);
    }


    private async Task<PredictionDuckRow?> LoadContextualTwinPredictionRowAsync(
        TensorConfig explicitTwin,
        IReadOnlyDictionary<string, PredictionDuckRow> explicitRows,
        CancellationToken ct)
    {
        string explicitKey = TensorConfigIdentity.ToKey(explicitTwin);
        if (explicitRows.TryGetValue(explicitKey, out var inMemoryExplicit))
            return inMemoryExplicit;

        var explicitRow = await LoadSinglePredictionRowAsync(explicitTwin, ct);
        if (explicitRow != null)
            return explicitRow with { Config = explicitTwin };

        // The normal generator may only contain the pure sparse carrier for an all-Q8/all-Q6
        // reference. For smoke scoring, that sparse carrier is allowed as a prediction source
        // only; the anomaly seed/probe/twin identity remains the explicit activated blanket.
        var sparsePure = new TensorConfig(
            explicitTwin.BaseQuant,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue,
            BaselineQuants.TensorConfigNullSlotValue);

        var sparseRow = await LoadSinglePredictionRowAsync(sparsePure, ct);
        return sparseRow == null ? null : sparseRow with { Config = sparsePure };
    }

    private static PredictionDuckRow ReadPredictionDuckRow(System.Data.Common.DbDataReader r)
    {
        return new PredictionDuckRow(
            new TensorConfig(
                ToByte(r.GetValue(0)),
                ToByte(r.GetValue(1)),
                ToByte(r.GetValue(2)),
                ToByte(r.GetValue(3)),
                ToByte(r.GetValue(4)),
                ToByte(r.GetValue(5)),
                ToByte(r.GetValue(6)),
                ToByte(r.GetValue(7)),
                ToByte(r.GetValue(8)),
                ToByte(r.GetValue(9))),
            ToDouble(r.GetValue(10)),
            ToDouble(r.GetValue(11)),
            ToUInt64(r.GetValue(12)),
            ToDouble(r.GetValue(13)),
            ToUInt64(r.GetValue(14)));
    }


    private bool ShouldSkipInvalidContextualAnomalyConfig(TensorConfig config, string source, out string reason)
    {
        if (_movement.TryValidateContextualAnomalyConfig(config, out var validationReason))
        {
            reason = string.Empty;
            return false;
        }

        reason = $"{source}: {validationReason}";
        return true;
    }

    private static void LogSkippedInvalidContextualAnomalyConfig(string source, TensorConfig config, string reason, int count)
    {
        if (!Config.AnomalyDetection.VerboseAnomalyLogging && count > 1)
            return;

        if (count > 12)
            return;

        string label = reason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase)
            ? "SkippedInvalidContextualAnomalyProbe"
            : "SkippedIsolationSample";

        string canonicalMessage = reason.Contains("SparseActiveGroup", StringComparison.OrdinalIgnoreCase)
            ? "active groups must be explicit for contextual anomaly probes"
            : "BF16/exact isolation rows are not contextual anomaly probes";

        AnsiConsole.MarkupLine(
            $"[yellow]{label}:[/] {Markup.Escape(canonicalMessage)}. source={Markup.Escape(source)} reason={Markup.Escape(reason)} config={Markup.Escape(TensorConfigIdentity.ToKey(config))} name={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)config))}");
    }

    private static void LogHistoricalSparseCandidateIgnored(TensorConfig config, string reason, int count)
    {
        if (!Config.AnomalyDetection.VerboseAnomalyLogging && count > 1)
            return;

        if (count > 12)
            return;

        AnsiConsole.MarkupLine(
            $"[yellow]HistoricalSparseCandidateIgnored:[/] reason=CannotTrustTensorComboIdentityForAnomalyRule detail={Markup.Escape(reason)} config={Markup.Escape(TensorConfigIdentity.ToKey(config))} name={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)config))}");
    }

    private void WriteContextualProbeConsoleLog(AnomalyProbePlan plan)
    {
        var movement = _movement.Analyze(plan.ReferenceConfig, plan.ProbeConfig);
        var referenceGroups = _movement.BuildEffectiveGroupVector(plan.ReferenceConfig);
        var probeGroups = _movement.BuildEffectiveGroupVector(plan.ProbeConfig);
        var inactive = _movement.BuildInactiveGroupList();

        AnsiConsole.MarkupLine("[yellow]Contextual anomaly probe:[/]");
        AnsiConsole.MarkupLine($"[grey]  kind=[/] [cyan]{Markup.Escape(plan.ProbeType)}[/]");
        AnsiConsole.MarkupLine($"[grey]  referenceQuant=[/] [cyan]{Markup.Escape(SafeName(plan.ReferenceConfig.BaseQuant))}[/]");
        AnsiConsole.MarkupLine($"[grey]  base=[/] [cyan]{Markup.Escape(SafeName(plan.ProbeConfig.BaseQuant))}[/]");
        AnsiConsole.MarkupLine("[grey]  effective groups:[/]");
        foreach (var item in probeGroups)
            AnsiConsole.MarkupLine($"[grey]    {Markup.Escape(item.Key)}=[/] [cyan]{Markup.Escape(item.Value)}[/]");

        if (inactive.Count > 0)
        {
            AnsiConsole.MarkupLine("[grey]  inactive groups:[/]");
            foreach (var group in inactive)
                AnsiConsole.MarkupLine($"[grey]    {Markup.Escape(group)}=NULL[/]");
        }

        AnsiConsole.MarkupLine($"[grey]  old BF16 isolation=[/] [cyan]false[/]");
        AnsiConsole.MarkupLine($"[grey]  movement=[/] [cyan]{movement.Classification}[/] [grey]up={movement.UpgradeCount} down={movement.DowngradeCount} same={movement.SameCount} unknown={movement.UnknownCount}[/]");
    }

    private static IReadOnlyList<IReadOnlyList<AnomalyChangedGroup>> BuildProbeSubsets(IReadOnlyList<AnomalyChangedGroup> changed)
    {
        var result = new List<IReadOnlyList<AnomalyChangedGroup>>();

        foreach (var item in changed)
            result.Add(new[] { item });

        for (int i = 0; i < changed.Count; i++)
        {
            for (int j = i + 1; j < changed.Count; j++)
                result.Add(new[] { changed[i], changed[j] });
        }

        result.Add(changed.ToList());

        if (changed.Count >= 3)
        {
            for (int i = 0; i < changed.Count; i++)
                result.Add(changed.Where((_, index) => index != i).ToList());
        }

        return result
            .GroupBy(x => string.Join(",", x.OrderBy(g => g.Group.UniqueId).Select(g => g.Group.UniqueId)), StringComparer.Ordinal)
            .Select(g => g.First())
            .ToList();
    }

    private static string ResolveProbeType(int subsetCount, int fullCount)
    {
        if (subsetCount == 1) return "single";
        if (subsetCount == 2) return "pair";
        if (subsetCount == fullCount) return "full";
        return "leave-one-out";
    }

    private static double ComputeSmokeScore(double gap, double savingsPercent, int changedGroupCount, ulong? candidateRank, ulong? twinRank)
    {
        double closeness = Math.Max(0d, Config.AnomalyDetection.MaxPredictionSpaceGapVsTwinKld - gap);
        double groupPenalty = Math.Max(1, changedGroupCount);
        double rankBonus = 0d;
        if (candidateRank.HasValue && twinRank.HasValue)
            rankBonus = Math.Clamp((double)twinRank.Value - candidateRank.Value, -10_000d, 10_000d) / 10_000d;

        return (closeness * 10_000d) + savingsPercent / groupPenalty + rankBonus;
    }

    private static void WriteSmokeConsoleSummary(int historicalCount, int duckCount, IReadOnlyList<AnomalySmokeCandidate> selected)
    {
        int monotone = selected.Count(x => x.Movement.Classification == AnomalyMovementClassification.MonotoneDowngrade);
        int existingTwins = selected.Count(x => x.HasActualTwin);
        int missingTwins = selected.Count - existingTwins;

        AnsiConsole.MarkupLine("[yellow]Anomaly smoke scan:[/]");
        AnsiConsole.MarkupLine($"[grey]  historical benchmarks scanned smoke=[/] [cyan]{historicalCount:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  DuckDB prediction-space smoke=[/] [cyan]{duckCount:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  monotone downgrade smoke candidates=[/] [cyan]{monotone:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  existing twins found=[/] [cyan]{existingTwins:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  missing twins queued=[/] [cyan]{missingTwins:N0}[/]");
    }

    private static void WriteProbeOutcome(AnomalyProbeResult result)
    {
        if (result.RuleDirection == AnomalyRuleDirection.Beneficial && result.ProbeSnapshot != null && result.ReferenceSnapshot != null)
        {
            AnsiConsole.MarkupLine(
                $"[green]Counterfactual MDA violation confirmed:[/] candidate={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(result.ProbeSnapshot.Quant))} " +
                $"twin={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(result.ReferenceSnapshot.Quant))} " +
                $"actual candidate KLD={result.ProbeSnapshot.Kld:0.000000} actual twin KLD={result.ReferenceSnapshot.Kld:0.000000} " +
                $"gain={result.ActualGainVsTwin:0.000000} classification={result.Classification}");
            return;
        }

        if (result.RuleDirection == AnomalyRuleDirection.SuppressionOnly && result.ProbeSnapshot != null && result.ReferenceSnapshot != null)
        {
            AnsiConsole.MarkupLine(
                $"[grey]Smoke rejected / normal gravity confirmed:[/] candidate={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(result.ProbeSnapshot.Quant))} " +
                $"twin={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(result.ReferenceSnapshot.Quant))} " +
                $"actual candidate KLD={result.ProbeSnapshot.Kld:0.000000} actual twin KLD={result.ReferenceSnapshot.Kld:0.000000} persisted suppression={Config.AnomalyDetection.PersistSuppressionResults}");
            return;
        }

        AnsiConsole.MarkupLine($"[yellow]Anomaly probe outcome:[/] {Markup.Escape(result.Classification.ToString())} {Markup.Escape(result.Message)}");
    }

    private static async Task ConfigureDuckAsync(DuckDBConnection c, CancellationToken ct)
    {
        await ExecuteDuckAsync(c, "SET preserve_insertion_order = false;", ct);
        await ExecuteDuckAsync(c, $"SET threads = {Math.Max(1, Environment.ProcessorCount)};", ct);
    }

    private static async Task ExecuteDuckAsync(DuckDBConnection c, string sql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        await cmd.ExecuteNonQueryAsync(ct);
    }

    private static void AddConfigParameters(DuckDBCommand cmd, TensorConfig c)
    {
        cmd.Parameters.Add(new DuckDBParameter { Value = c.BaseQuant });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.Embeddings });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.LmHead });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.AttnQ });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.AttnKV });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.AttnOutput });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.FfnUpGate });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.FfnDown });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.MoeExperts });
        cmd.Parameters.Add(new DuckDBParameter { Value = c.MoeRouter });
    }

    private static async Task WriteJsonAsync(string fileName, object payload, CancellationToken ct)
    {
        string dir = Cache.ModelMagicQuantDirectory ?? Cache.MagicQuantDirectory ?? Directory.GetCurrentDirectory();
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, fileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(payload, JsonOptions), ct);
    }

    private static async Task WriteFinalManifestAsync(string fileName, object payload, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(Cache.OutputDirectory))
            return;

        string manifestDir = Path.Combine(Cache.OutputDirectory!, "magicquant-manifest");
        Directory.CreateDirectory(manifestDir);
        string path = Path.Combine(manifestDir, fileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(payload, JsonOptions), ct);
    }


    private object ToSmokeLog(AnomalySmokeCandidate x)
    {
        return new
        {
            x.Source,
            isContextualAnomalySmoke = true,
            oldBf16Isolation = false,
            allActiveGroupsExplicit = _movement.HasAllActiveGroupsExplicit(x.CandidateConfig) && _movement.HasAllActiveGroupsExplicit(x.TwinConfig),
            candidate = TensorConfigIdentity.ToKey(x.CandidateConfig),
            twin = TensorConfigIdentity.ToKey(x.TwinConfig),
            candidateName = HybridBenchmarkRepository.BuildDisplayName(x.CandidateQuant),
            twinName = HybridBenchmarkRepository.BuildDisplayName(x.TwinQuant),
            referenceEffectiveGroups = _movement.BuildEffectiveGroupVector(x.TwinConfig),
            candidateEffectiveGroups = _movement.BuildEffectiveGroupVector(x.CandidateConfig),
            inactiveGroups = _movement.BuildInactiveGroupList(),
            movement = x.Movement.Classification.ToString(),
            x.Movement.UpgradeCount,
            x.Movement.DowngradeCount,
            x.Movement.SameCount,
            x.Movement.UnknownCount,
            changedGroups = x.Movement.ChangedGroups.Select(g => new
            {
                group = g.Group.Name,
                candidate = SafeName(g.CandidateQuantId),
                reference = SafeName(g.ReferenceQuantId),
                movement = g.Movement.ToString()
            }).ToList(),
            x.CandidatePredictedKld,
            x.TwinPredictedKld,
            x.PredictionSpaceGapVsTwin,
            x.CandidatePredictedSizeBytes,
            x.TwinPredictedSizeBytes,
            x.SizeSavingsBytes,
            x.CandidatePredictionRank,
            x.TwinPredictionRank,
            x.SmokeScore,
            x.SmokeStrength,
            x.HasActualTwin,
            x.CandidateActualKld,
            x.TwinActualKld,
            x.IsConfirmedFromHistory,
            x.Message
        };
    }

    private object ToProbeLog(AnomalyProbePlan x)
    {
        var movement = _movement.Analyze(x.ReferenceConfig, x.ProbeConfig);
        return new
        {
            isContextualAnomalyProbe = true,
            oldBf16Isolation = false,
            allActiveGroupsExplicit = _movement.HasAllActiveGroupsExplicit(x.ReferenceConfig) && _movement.HasAllActiveGroupsExplicit(x.ProbeConfig),
            reference = TensorConfigIdentity.ToKey(x.ReferenceConfig),
            probe = TensorConfigIdentity.ToKey(x.ProbeConfig),
            referenceName = HybridBenchmarkRepository.BuildDisplayName((HybridQuant)x.ReferenceConfig),
            probeName = HybridBenchmarkRepository.BuildDisplayName((HybridQuant)x.ProbeConfig),
            referenceEffectiveGroups = _movement.BuildEffectiveGroupVector(x.ReferenceConfig),
            candidateEffectiveGroups = _movement.BuildEffectiveGroupVector(x.ProbeConfig),
            inactiveGroups = _movement.BuildInactiveGroupList(),
            movementClassification = movement.Classification.ToString(),
            movement.UpgradeCount,
            movement.DowngradeCount,
            movement.SameCount,
            movement.UnknownCount,
            x.ProbeType,
            x.HypothesisLabel,
            groups = x.ProbeGroups.Select(g => new
            {
                group = g.Group.Name,
                candidate = SafeName(g.CandidateQuantId),
                reference = SafeName(g.ReferenceQuantId)
            }).ToList()
        };
    }

    private object ToResultLog(AnomalyProbeResult x)
    {
        return new
        {
            probe = ToProbeLog(x.Plan),
            classification = x.Classification.ToString(),
            direction = x.RuleDirection.ToString(),
            x.Accepted,
            x.ActualGainVsTwin,
            referenceKld = x.ReferenceSnapshot?.Kld,
            probeKld = x.ProbeSnapshot?.Kld,
            referenceSizeBytes = x.ReferenceSnapshot?.SizeBytes,
            probeSizeBytes = x.ProbeSnapshot?.SizeBytes,
            x.FailureCode,
            x.Message
        };
    }

    private static object ToRuleLog(AnomalyInteractionRule x)
    {
        return new
        {
            x.Id,
            x.RuleType,
            x.RuleDirection,
            x.RuleStatus,
            x.ReferenceQuantId,
            referenceQuant = SafeName(x.ReferenceQuantId),
            x.ReferenceContextKey,
            x.ReferenceEffectiveGroupsJson,
            x.CandidateEffectiveGroupsJson,
            x.InactiveGroupsJson,
            x.FullTensorConfigKey,
            x.GroupSetHash,
            x.GroupCount,
            x.MeanActualGainVsTwin,
            x.BestActualGainVsTwin,
            x.MeanPredictionSpaceGap,
            x.BestPredictionSpaceGap,
            x.AppliedPredictionSpaceAdjustmentKld,
            x.EvidenceCount,
            x.Confidence,
            groups = x.GroupStates.OrderBy(g => g.SortOrder).Select(g => new
            {
                g.TensorGroupId,
                candidate = SafeName(g.CandidateQuantId),
                reference = SafeName(g.ReferenceQuantId),
                g.Movement
            }).ToList()
        };
    }

    private static byte ToByte(object? value)
    {
        if (value is null || value is DBNull) return 0;
        if (value is BigInteger big) return (byte)big;
        return Convert.ToByte(value, CultureInfo.InvariantCulture);
    }

    private static double ToDouble(object? value)
    {
        if (value is null || value is DBNull) return 0d;
        if (value is BigInteger big) return (double)big;
        return Convert.ToDouble(value, CultureInfo.InvariantCulture);
    }

    private static ulong ToUInt64(object? value)
    {
        if (value is null || value is DBNull) return 0UL;
        if (value is BigInteger big) return (ulong)big;
        return Convert.ToUInt64(value, CultureInfo.InvariantCulture);
    }

    private static string SafeName(byte quantId)
    {
        try
        {
            return BaselineQuants.FromId(quantId).Names[0];
        }
        catch
        {
            return $"id:{quantId}";
        }
    }

    private sealed record PredictionDuckRow(
        TensorConfig Config,
        double BaseRankSafeKld,
        double FinalPredictedKld,
        ulong PredictedSizeBytes,
        double PredictionConfidence,
        ulong PredictionRank);
}
