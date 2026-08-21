using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using DuckDB.NET.Data;
using MagicQuant.Configuration;
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
    private AnomalySmokeScanDiagnostics? _lastDuckSmokeDiagnostics;

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
                duckDiagnostics = _lastDuckSmokeDiagnostics,
                smoke = smoke.Select(ToSmokeLog).ToList()
            }, ct);

            await WriteJsonAsync("magicquant-anomaly-seeds.json", smoke.Select(ToSmokeLog).ToList(), ct);
            await WriteJsonAsync("magicquant-synergy-smoke-scan.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                terminology = "CounterfactualSynergy smoke. Anomaly names are retained as backward-compatible aliases.",
                historicalCount = historical.Count,
                duckPredictionSpaceCount = duck.Count,
                selectedSmokeCount = smoke.Count,
                duckDiagnostics = _lastDuckSmokeDiagnostics,
                smoke = smoke.Select(ToSmokeLog).ToList()
            }, ct);

            var planningDiagnostics = new ProbePlanningDiagnostics();
            var probes = await PlanProbesAsync(smoke, planningDiagnostics, ct);

            var results = await ValidateProbesAsync(probes, ct);

            var transferProbes = await PlanSynergyTransferProbesAsync(results, planningDiagnostics, ct);
            if (transferProbes.Count > 0)
            {
                probes = probes.Concat(transferProbes).ToList();
                var transferResults = await ValidateProbesAsync(transferProbes, ct);
                results = results.Concat(transferResults).ToList();
            }

            var expansionProbes = await PlanConfirmedAnomalyExpansionProbesAsync(results, planningDiagnostics, ct);
            if (expansionProbes.Count > 0)
            {
                probes = probes.Concat(expansionProbes).ToList();
                var expansionResults = await ValidateProbesAsync(expansionProbes, ct);
                results = results.Concat(expansionResults).ToList();
            }

            var compositionDiagnostics = new List<SynergyCompositionProbeRecord>();
            var compositionProbes = await PlanSynergyCompositionProbesAsync(results, planningDiagnostics, compositionDiagnostics, ct);
            if (compositionProbes.Count > 0)
            {
                probes = probes.Concat(compositionProbes).ToList();
                var compositionResults = await ValidateProbesAsync(compositionProbes, ct);
                results = results.Concat(compositionResults).ToList();
                compositionDiagnostics = BuildCompositionDiagnostics(compositionProbes, compositionResults);
            }

            await WriteJsonAsync("magicquant-synergy-composition-probes.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                compositionProbes = compositionDiagnostics
            }, ct);

            await WriteJsonAsync("magicquant-anomaly-probes.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                planningDiagnostics,
                probes = probes.Select(ToProbeLog).ToList()
            }, ct);

            await _rules.PersistProbeResultsAsync(session.Id, results, ct);
            var upsertedRules = await _rules.UpsertRulesFromResultsAsync(results, ct);
            var applicableRules = await _rules.LoadApplicableRulesAsync(ct);
            var bestAnomaly = BuildBestConfirmedAnomalyReconciliation(results, Array.Empty<BenchmarkSnapshotRecord>());
            WriteBestAnomalyConsoleLog(bestAnomaly);
            var adjustment = await _adjuster.ApplyAsync(applicableRules, ct);

            await WriteJsonAsync("magicquant-anomaly-rules.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                upserted = upsertedRules.Select(ToRuleLog).ToList(),
                applicable = applicableRules.Select(ToRuleLog).ToList()
            }, ct);
            await WriteJsonAsync("magicquant-synergy-templates.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                templates = applicableRules.Select(ToSynergyTemplateLog).ToList()
            }, ct);
            await WriteJsonAsync("magicquant-synergy-probes.json", results.Select(ToResultLog).ToList(), ct);
            await WriteJsonAsync("magicquant-synergy-transfer-probes.json", results
                .Where(x => x.Plan.SeedClass == AnomalySeedClass.SynergyTransferProbe || x.Plan.SeedClass == AnomalySeedClass.ConfirmedAnomalyNeighborhoodProbe)
                .Select(ToResultLog).ToList(), ct);

            var wingSummary = BuildSynergyWingSummary(smoke, results, adjustment);
            WriteSynergyWingConsoleSummary(wingSummary);
            await WriteJsonAsync("magicquant-synergy-wing-summary.json", wingSummary, ct);

            await WriteJsonAsync("magicquant-anomaly-adjusted-predictions-summary.json", adjustment, ct);
            await WriteJsonAsync("magicquant-synergy-adjusted-predictions-summary.json", adjustment, ct);
            await WriteFinalManifestAsync("magicquant.anomalies.json", new
            {
                generatedAtUtc = DateTime.UtcNow,
                smoke = smoke.Select(ToSmokeLog).ToList(),
                probes = probes.Select(ToProbeLog).ToList(),
                results = results.Select(ToResultLog).ToList(),
                rules = applicableRules.Select(ToRuleLog).ToList(),
                bestConfirmedAnomaly = bestAnomaly,
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
                AdjustmentSummary = adjustment,
                BestAnomalyReconciliation = bestAnomaly
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
        await EmitQ8ContextReferenceDriftDiagnosticsAsync(byKey, ct);
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

            string explicitTwinKey = TensorConfigIdentity.ToKey(twinConfig);
            var sparseTwinConfig = BuildSparsePureContext(twinConfig.BaseQuant);
            string sparseTwinKey = TensorConfigIdentity.ToKey(sparseTwinConfig);
            bool searchedExplicit = true;
            bool searchedSparse = true;
            byKey.TryGetValue(explicitTwinKey, out var explicitTwin);
            byKey.TryGetValue(sparseTwinKey, out var sparseTwin);

            var twin = explicitTwin ?? sparseTwin;
            string twinLookupMode = explicitTwin != null
                ? (sparseTwin != null ? "explicit-context-preferred; sparse-pure-also-found" : "explicit-context-found")
                : sparseTwin != null
                    ? "sparse-pure-fallback-found"
                    : "missing; searched-explicit-context-and-sparse-pure";

            if (Config.AnomalyDetection.VerboseAnomalyLogging)
            {
                AnsiConsole.MarkupLine($"[grey]Historical twin lookup:[/] candidate={Markup.Escape(candidate.DisplayName)} searchedExplicitContext={searchedExplicit} searchedSparsePure={searchedSparse} mode={Markup.Escape(twinLookupMode)}");
            }

            if (twin != null && ShouldSkipInvalidContextualAnomalyConfig(twin.Config, "history-existing-twin", out var existingTwinSkipReason))
            {
                skippedNonContextualTwin++;
                LogSkippedInvalidContextualAnomalyConfig("history-existing-twin", twin.Config, existingTwinSkipReason, skippedNonContextualTwin);
                continue;
            }

            predictionLookup.TryGetValue(TensorConfigIdentity.ToKey(candidate.Config), out var candidatePrediction);
            predictionLookup.TryGetValue(explicitTwinKey, out var twinPrediction);
            if (twinPrediction == null)
                predictionLookup.TryGetValue(sparseTwinKey, out twinPrediction);

            bool confirmed = twin != null &&
                             candidate.SizeBytes <= twin.SizeBytes &&
                             twin.Kld - candidate.Kld >= Config.AnomalyDetection.MinActualGainVsTwinKld;

            if (!confirmed && twin != null && twin.Kld <= candidate.Kld)
                continue;

            ulong? predictedCandidateSize = candidatePrediction?.PredictedSizeBytes;
            ulong? predictedTwinSize = twinPrediction?.PredictedSizeBytes;
            ulong? predictedSavings = predictedCandidateSize.HasValue && predictedTwinSize.HasValue && predictedTwinSize.Value >= predictedCandidateSize.Value
                ? predictedTwinSize.Value - predictedCandidateSize.Value
                : null;
            ulong? actualSavings = twin != null && twin.SizeBytes >= candidate.SizeBytes ? twin.SizeBytes - candidate.SizeBytes : null;

            smoke.Add(new AnomalySmokeCandidate
            {
                Source = "history",
                CandidateConfig = candidate.Config,
                TwinConfig = twinConfig,
                Movement = movement,
                CandidatePredictedKld = candidatePrediction?.BaseRankSafeKld ?? candidatePrediction?.FinalPredictedKld ?? candidate.Kld,
                TwinPredictedKld = twinPrediction?.BaseRankSafeKld ?? twinPrediction?.FinalPredictedKld ?? twin?.Kld ?? candidate.Kld,
                CandidatePredictedSizeBytes = predictedCandidateSize,
                TwinPredictedSizeBytes = predictedTwinSize,
                PredictedSizeSavingsBytes = predictedSavings,
                ActualSizeSavingsBytes = actualSavings,
                PlannedProbeWillMeasureSize = twin == null || actualSavings == null,
                TwinLookupMode = twinLookupMode,
                TwinFoundInLookupDictionary = twinPrediction != null,
                PredictionSpaceGapVsTwin = (candidatePrediction?.BaseRankSafeKld ?? candidate.Kld) - (twinPrediction?.BaseRankSafeKld ?? twin?.Kld ?? candidate.Kld),
                CandidatePredictionRank = candidatePrediction?.PredictionRank,
                TwinPredictionRank = twinPrediction?.PredictionRank,
                SmokeScore = confirmed ? 1_000_000d : 100d,
                SmokeStrength = confirmed ? "ConfirmedHistory" : "HistoricalMissingTwin",
                SeedClass = confirmed ? AnomalySeedClass.ConfirmedHistoricalCounterfactual : AnomalySeedClass.HistoricalMissingTwin,
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
        var totalClock = Stopwatch.StartNew();
        var loadClock = Stopwatch.StartNew();
        var rows = await LoadPredictionRowsAsync(DuckSmokeScanLimit, ct);
        loadClock.Stop();

        var lookupClock = Stopwatch.StartNew();
        var lookupRows = new Dictionary<string, PredictionDuckRow>(StringComparer.Ordinal);
        var candidateRows = new Dictionary<string, PredictionDuckRow>(StringComparer.Ordinal);
        var result = new List<AnomalySmokeCandidate>();
        var rejected = new List<RejectedSmokePreview>();
        var closestGapFailures = new List<RejectedSmokePreview>();
        var existingKeys = await _rules.LoadExistingRuleSuppressionKeysAsync(ct);

        int skippedIsolation = 0;
        int skippedSparse = 0;
        int normalizedSparse = 0;
        int skippedPure = 0;
        int skippedMixed = 0;
        int skippedMovement = 0;
        int skippedNoTwin = 0;
        int skippedSavings = 0;
        int skippedGap = 0;
        int contextualScanned = 0;
        int twinLookupCount = 0;
        int dictionaryTwinHits = 0;
        int fallbackDbTwinLookups = 0;

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

            var normalizedRow = row with { Config = activated };
            AddOrPreferBetterPredictionRow(lookupRows, normalizedRow);

            // Keep the sparse pure carrier in the lookup as an optional prediction source,
            // but never let it become a contextual anomaly identity or probe/rule row.
            if (TensorConfigIdentity.IsPureBaseline(row.Config))
                AddOrPreferBetterPredictionRow(lookupRows, row);

            if (TensorConfigIdentity.ToKey(activated) == TensorConfigIdentity.ToKey(_movement.BuildBaseContextTwin(activated)))
            {
                skippedPure++;
                continue;
            }

            AddOrPreferBetterPredictionRow(candidateRows, normalizedRow);
        }
        lookupClock.Stop();

        var scanClock = Stopwatch.StartNew();
        foreach (var row in candidateRows.Values)
        {
            contextualScanned++;
            var twin = _movement.BuildBaseContextTwin(row.Config);
            string candidateKey = TensorConfigIdentity.ToKey(row.Config);
            string twinKey = TensorConfigIdentity.ToKey(twin);

            if (ShouldSkipInvalidContextualAnomalyConfig(row.Config, "duckdb-normalized-candidate", out var candidateSkipReason))
            {
                skippedIsolation++;
                AddRejectedPreview(rejected, row.Config, twin, null, null, null, null, null, "InvalidCandidate: " + candidateSkipReason, false, false);
                LogSkippedInvalidContextualAnomalyConfig("duckdb-normalized-candidate", row.Config, candidateSkipReason, skippedIsolation);
                continue;
            }

            if (ShouldSkipInvalidContextualAnomalyConfig(twin, "duckdb-twin", out var twinSkipReason))
            {
                skippedIsolation++;
                AddRejectedPreview(rejected, row.Config, twin, null, null, null, null, null, "InvalidTwin: " + twinSkipReason, false, false);
                LogSkippedInvalidContextualAnomalyConfig("duckdb-twin", twin, twinSkipReason, skippedIsolation);
                continue;
            }

            var movement = _movement.Analyze(twin, row.Config);
            bool matchedConfirmedPattern = existingKeys.Contains(_rules.BuildRuleSuppressionKey(twin, movement.ChangedGroups));

            if (movement.Classification == AnomalyMovementClassification.MixedTrade)
            {
                skippedMixed++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, null, null, null, "MixedTrade", matchedConfirmedPattern, false);
                if (Config.AnomalyDetection.VerboseAnomalyLogging && skippedMixed <= 12)
                {
                    AnsiConsole.MarkupLine("[grey]Ignored anomaly smoke:[/] classification=MixedTrade reason=normal protect/compress frontier behavior");
                }
                continue;
            }

            if (movement.Classification != AnomalyMovementClassification.MonotoneDowngrade)
            {
                skippedMovement++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, null, null, null, "MovementNotMonotoneDowngrade", matchedConfirmedPattern, false);
                continue;
            }

            if (movement.DowngradeCount <= 0 || movement.DowngradeCount > Config.AnomalyDetection.MaxProbeGroupCount)
            {
                skippedMovement++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, null, null, null, "ChangedGroupBudgetExceeded", matchedConfirmedPattern, false);
                continue;
            }

            twinLookupCount++;
            var twinRow = ResolveTwinFromLookupOnly(twin, lookupRows, out var twinLookupMode, out var twinFoundInDictionary);
            if (twinRow == null)
            {
                skippedNoTwin++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, null, null, null, "MissingTwinInPreloadedDictionary", matchedConfirmedPattern, false);
                continue;
            }

            if (twinFoundInDictionary)
                dictionaryTwinHits++;

            if (twinRow.PredictedSizeBytes <= row.PredictedSizeBytes)
            {
                skippedSavings++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, twinRow, null, null, "PredictedSizeSavingsNotPositive", matchedConfirmedPattern, true);
                continue;
            }

            ulong savingsBytes = twinRow.PredictedSizeBytes - row.PredictedSizeBytes;
            double savingsPercent = savingsBytes * 100d / Math.Max(1d, twinRow.PredictedSizeBytes);
            if (savingsPercent < Config.AnomalyDetection.MinPredictedSizeSavingsVsTwinPercent)
            {
                skippedSavings++;
                AddRejectedPreview(rejected, row.Config, twin, movement, row, twinRow, savingsBytes, null, "PredictedSizeSavingsBelowThreshold", matchedConfirmedPattern, true);
                continue;
            }

            double gap = row.BaseRankSafeKld - twinRow.BaseRankSafeKld;
            double score = ComputeSmokeScore(gap, savingsPercent, movement.DowngradeCount, row.PredictionRank, twinRow.PredictionRank);
            bool gapCatastrophic = gap > Math.Max(Config.AnomalyDetection.MaxPredictionSpaceGapVsTwinKld, Config.SynergyDetection.MaxSmokeGapKld);
            bool scoreTooLow = score < Config.SynergyDetection.MinSmokeScore && !matchedConfirmedPattern;
            if (gapCatastrophic || scoreTooLow)
            {
                skippedGap++;
                string reason = gapCatastrophic ? "PredictionSpaceGapTooLarge" : "BelowMinSmokeScore";
                var preview = AddRejectedPreview(rejected, row.Config, twin, movement, row, twinRow, savingsBytes, gap, reason, matchedConfirmedPattern, true);
                closestGapFailures.Add(preview);
                continue;
            }


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
                PredictedSizeSavingsBytes = savingsBytes,
                PlannedProbeWillMeasureSize = true,
                TwinLookupMode = twinLookupMode,
                TwinFoundInLookupDictionary = twinFoundInDictionary,
                PredictionSpaceGapVsTwin = gap,
                CandidatePredictionRank = row.PredictionRank,
                TwinPredictionRank = twinRow.PredictionRank,
                SmokeScore = score,
                SmokeStrength = gap <= 0d ? "Strong" : "Close",
                SeedClass = AnomalySeedClass.PredictionSpaceSmoke,
                MatchedConfirmedAnomalyPattern = matchedConfirmedPattern,
                Message = "Prediction-space contextual monotone downgrade candidate is close enough to its higher-bit quantized twin to justify probes. Twin lookup was dictionary-only from the preloaded DuckDB row set."
            });
        }
        scanClock.Stop();
        totalClock.Stop();

        var diagnostics = new AnomalySmokeScanDiagnostics
        {
            PredictedRowsScanned = rows.Count,
            SparseRowsNormalized = normalizedSparse,
            SparseRowsSkipped = skippedSparse,
            Bf16ExactRowsSkipped = skippedIsolation,
            PureReferenceRowsSkipped = skippedPure,
            ContextualRowsScanned = contextualScanned,
            TwinLookupCount = twinLookupCount,
            DictionaryTwinHits = dictionaryTwinHits,
            MissingTwins = skippedNoTwin,
            FallbackDbTwinLookups = fallbackDbTwinLookups,
            MovementNotMonotoneDowngrade = skippedMovement,
            MixedTradeIgnored = skippedMixed,
            SizeSavingsBelowThreshold = skippedSavings,
            PredictionSpaceGapTooLarge = skippedGap,
            QueuedSmokeCandidates = result.Count,
            LoadPredictedRowsMs = loadClock.ElapsedMilliseconds,
            BuildLookupDictionaryMs = lookupClock.ElapsedMilliseconds,
            ScanRowsMs = scanClock.ElapsedMilliseconds,
            RejectedPreview = rejected
                .OrderBy(x => x.SortOrder)
                .Take(25)
                .Select(x => x.ToLog())
                .ToList(),
            ClosestGapFailures = closestGapFailures
                .OrderBy(x => x.PredictionSpaceGap ?? double.MaxValue)
                .ThenByDescending(x => x.PredictedSizeSavingsBytes ?? 0UL)
                .Take(10)
                .Select(x => x.ToLog())
                .ToList()
        };

        _lastDuckSmokeDiagnostics = diagnostics;

        AnsiConsole.MarkupLine("[yellow]DuckDB contextual smoke scan:[/]");
        AnsiConsole.MarkupLine($"[grey]  predicted rows scanned=[/] [cyan]{rows.Count:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  load predicted rows ms=[/] [cyan]{diagnostics.LoadPredictedRowsMs:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  build lookup dictionary ms=[/] [cyan]{diagnostics.BuildLookupDictionaryMs:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  scan rows ms=[/] [cyan]{diagnostics.ScanRowsMs:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  sparse rows normalized to explicit context=[/] [cyan]{normalizedSparse:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  sparse rows skipped=[/] [cyan]{skippedSparse:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  BF16/exact rows skipped=[/] [cyan]{skippedIsolation:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  pure/logical reference rows kept for lookup but skipped as smoke=[/] [cyan]{skippedPure:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  contextual quantized rows scanned=[/] [cyan]{contextualScanned:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  twin lookup count=[/] [cyan]{twinLookupCount:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  dictionary twin hits=[/] [cyan]{dictionaryTwinHits:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  missing twins=[/] [cyan]{skippedNoTwin:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  fallback DB twin lookups=[/] [cyan]{fallbackDbTwinLookups:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  movement not monotone downgrade=[/] [cyan]{skippedMovement:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  mixed trade ignored=[/] [cyan]{skippedMixed:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  size savings below threshold=[/] [cyan]{skippedSavings:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  prediction-space gap too large=[/] [cyan]{skippedGap:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  queued smoke candidates=[/] [cyan]{result.Count:N0}[/]");
        AnsiConsole.MarkupLine("[yellow]DuckDB synergy smoke timings:[/]");
        AnsiConsole.MarkupLine($"[grey]  predicted row scan ms=[/] [cyan]{diagnostics.LoadPredictedRowsMs:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  virtual twin construction ms=[/] [cyan]0[/] [grey](computed inline while scanning)[/]");
        AnsiConsole.MarkupLine($"[grey]  twin lookup ms=[/] [cyan]{diagnostics.ScanRowsMs:N0}[/] [grey](dictionary-only in normal operation)[/]");
        AnsiConsole.MarkupLine($"[grey]  scoring ms=[/] [cyan]{diagnostics.ScanRowsMs:N0}[/]");
        AnsiConsole.MarkupLine($"[grey]  diagnostic formatting ms=[/] [cyan]deferred[/]");

        if (result.Count == 0 && rows.Count > 0)
        {
            AnsiConsole.MarkupLine("[yellow]DuckDB contextual smoke scan produced zero candidates.[/] Top rejected-smoke previews and closest gap failures were written to magicquant-anomaly-smoke-scan-duckdb-diagnostics.json.");
            foreach (var preview in closestGapFailures.OrderBy(x => x.PredictionSpaceGap ?? double.MaxValue).Take(10))
            {
                AnsiConsole.MarkupLine($"[grey]  rejected monotone gap:[/] candidate={Markup.Escape(preview.CandidateName)} twin={Markup.Escape(preview.TwinName)} gap={FmtNullable(preview.PredictionSpaceGap)} savings={FmtNullable(preview.PredictedSizeSavingsBytes)} reason={Markup.Escape(preview.RejectionReason)} matchedRule={preview.MatchedConfirmedAnomalyPattern}");
            }
        }

        await WriteJsonAsync("magicquant-anomaly-smoke-scan-duckdb-diagnostics.json", new
        {
            generatedAtUtc = DateTime.UtcNow,
            diagnostics,
            queued = result.Select(ToSmokeLog).ToList()
        }, ct);

        return result
            .GroupBy(x => x.TwinConfig.BaseQuant)
            .SelectMany(g => g.OrderByDescending(x => x.SmokeScore).Take(Config.AnomalyDetection.MaxSmokeCandidatesPerReferenceZone))
            .ToList();
    }


    private async Task<List<AnomalyProbePlan>> PlanProbesAsync(
        IReadOnlyList<AnomalySmokeCandidate> seeds,
        ProbePlanningDiagnostics diagnostics,
        CancellationToken ct)
    {
        var plans = new List<AnomalyProbePlan>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var existingRuleKeys = await _rules.LoadExistingRuleSuppressionKeysAsync(ct);
        diagnostics.ExistingRuleKeysLoaded = existingRuleKeys.Count;

        foreach (var seed in seeds)
        {
            bool invalidSeedCandidate = ShouldSkipInvalidContextualAnomalyConfig(seed.CandidateConfig, "probe-seed-candidate", out var seedCandidateReason);
            bool invalidSeedTwin = ShouldSkipInvalidContextualAnomalyConfig(seed.TwinConfig, "probe-seed-twin", out var seedTwinReason);
            if (invalidSeedCandidate || invalidSeedTwin)
            {
                diagnostics.SkippedInvalidMovement++;
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
            {
                diagnostics.SkippedInvalidMovement++;
                continue;
            }

            var subsets = BuildProbeSubsets(changed);
            int perSeed = 0;
            foreach (var subset in subsets)
            {
                if (perSeed >= Config.AnomalyDetection.MaxProbesPerSeed || plans.Count >= Config.AnomalyDetection.MaxTotalProbesPerRun)
                {
                    diagnostics.SkippedBudget++;
                    break;
                }

                var probeConfig = reference;
                foreach (var g in subset)
                    probeConfig = _movement.WithStoredSlot(probeConfig, g.Group, g.CandidateStoredSlot);

                if (ShouldSkipInvalidContextualAnomalyConfig(probeConfig, "probe-plan", out var probeSkipReason))
                {
                    diagnostics.SkippedInvalidMovement++;
                    AnsiConsole.MarkupLine($"[yellow]SkippedInvalidContextualAnomalyProbe:[/] source=probe-plan reason={Markup.Escape(probeSkipReason)} probe={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName((HybridQuant)probeConfig))}");
                    continue;
                }

                _movement.EnsureAllActiveGroupsExplicit(probeConfig, "probe-plan-probe");

                if (existingRuleKeys.Contains(_rules.BuildRuleSuppressionKey(reference, subset)))
                {
                    diagnostics.SkippedExistingRuleOrSuppression++;
                    continue;
                }

                string key = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probeConfig);
                if (!seen.Add(key))
                {
                    diagnostics.SkippedDuplicate++;
                    continue;
                }

                string probeType = ResolveProbeType(subset.Count, changed.Count);
                var plan = new AnomalyProbePlan
                {
                    Seed = seed,
                    ReferenceConfig = reference,
                    ProbeConfig = probeConfig,
                    ProbeGroups = subset,
                    ProbeType = probeType,
                    HypothesisLabel = _movement.DescribeGroups(subset),
                    SeedClass = seed.SeedClass,
                    ProbePriorityClass = probeType == "single"
                        ? AnomalySeedClass.ExploratorySingle
                        : probeType == "pair"
                            ? AnomalySeedClass.ExploratoryPair
                            : seed.SeedClass
                };
                plans.Add(plan);
                perSeed++;
                diagnostics.ProbesQueued++;
            }

            AnsiConsole.MarkupLine(
                $"[yellow]Potential anomaly smoke:[/] seedClass={seed.SeedClass} classification={seed.Movement.Classification} candidate={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.CandidateQuant))} " +
                $"higher-bit twin={Markup.Escape(HybridBenchmarkRepository.BuildDisplayName(seed.TwinQuant))} changed groups={Markup.Escape(_movement.DescribeGroups(changed))} " +
                $"upgradeCount={seed.Movement.UpgradeCount} downgradeCount={seed.Movement.DowngradeCount} " +
                $"prediction-space gap={seed.PredictionSpaceGapVsTwin:0.000000} predicted size savings={FmtNullable(seed.PredictedSizeSavingsBytes)} actual size savings={FmtNullable(seed.ActualSizeSavingsBytes)} plannedProbeWillMeasureSize={seed.PlannedProbeWillMeasureSize} probes queued={perSeed:N0}");
        }

        AnsiConsole.MarkupLine(
            $"[grey]Anomaly probe planning diagnostics:[/] existingRuleKeysLoaded={diagnostics.ExistingRuleKeysLoaded:N0} skippedExistingRuleOrSuppression={diagnostics.SkippedExistingRuleOrSuppression:N0} skippedDuplicate={diagnostics.SkippedDuplicate:N0} skippedInvalidMovement={diagnostics.SkippedInvalidMovement:N0} skippedBudget={diagnostics.SkippedBudget:N0} probesQueued={diagnostics.ProbesQueued:N0}");

        return plans;
    }


    private async Task<List<AnomalyProbePlan>> PlanSynergyTransferProbesAsync(
        IReadOnlyList<AnomalyProbeResult> currentResults,
        ProbePlanningDiagnostics diagnostics,
        CancellationToken ct)
    {
        var cfg = Config.SynergyDetection;
        if (!cfg.Enabled || !cfg.TransferProbeEnabled || cfg.MaxTotalTransferProbesPerRun <= 0)
            return new List<AnomalyProbePlan>();

        var contexts = ResolveTransferTargetContexts(cfg.TransferProbeContextStrata);
        if (contexts.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]Synergy transfer probes skipped:[/] no configured context-stratum quant names resolved to active baselines.");
            return new List<AnomalyProbePlan>();
        }

        var historicalRules = await _rules.LoadApplicableRulesAsync(ct);
        var templates = BuildTransferTemplates(historicalRules, currentResults)
            .Where(x => x.Confidence >= cfg.MinConfidenceToScheduleTransferProbe)
            .GroupBy(x => x.Key, StringComparer.Ordinal)
            .Select(g => g
                .OrderByDescending(x => x.Confidence)
                .ThenByDescending(x => x.ActualEffectMagnitude)
                .First())
            .ToList();

        if (templates.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]Synergy transfer probes:[/] no confirmed templates met the transfer confidence threshold.");
            return new List<AnomalyProbePlan>();
        }

        var existingRuleKeys = await _rules.LoadExistingRuleSuppressionKeysAsync(ct);
        var candidatesByContext = contexts.ToDictionary(
            x => x,
            x => BuildTransferCandidatesForContext(x, templates, existingRuleKeys, diagnostics));

        int globalLimit = Math.Max(0, cfg.MaxTotalTransferProbesPerRun);
        int perTemplateLimit = Math.Max(1, cfg.MaxTransferProbesPerTemplate);
        var cursors = contexts.ToDictionary(x => x, _ => 0);
        var perTemplateCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var selectedKeys = new HashSet<string>(StringComparer.Ordinal);
        var plans = new List<AnomalyProbePlan>();

        while (plans.Count < globalLimit)
        {
            bool addedInRound = false;
            foreach (var context in contexts)
            {
                var candidates = candidatesByContext[context];
                while (cursors[context] < candidates.Count)
                {
                    var candidate = candidates[cursors[context]++];
                    int used = perTemplateCounts.GetValueOrDefault(candidate.TemplateKey);
                    if (used >= perTemplateLimit || !selectedKeys.Add(candidate.IdentityKey))
                        continue;

                    plans.Add(candidate.Plan);
                    perTemplateCounts[candidate.TemplateKey] = used + 1;
                    diagnostics.ProbesQueued++;
                    diagnostics.TransferProbesQueued++;
                    addedInRound = true;
                    break;
                }

                if (plans.Count >= globalLimit)
                    break;
            }

            if (!addedInRound)
                break;
        }

        int candidateCount = candidatesByContext.Values.Sum(x => x.Count);
        diagnostics.SkippedBudget += Math.Max(0, candidateCount - plans.Count);
        AnsiConsole.MarkupLine(
            $"[yellow]Synergy context-transfer probes:[/] templates=[cyan]{templates.Count:N0}[/] " +
            $"strata=[cyan]{contexts.Count:N0}[/] candidates=[cyan]{candidateCount:N0}[/] " +
            $"queued=[cyan]{plans.Count:N0}[/] globalLimit=[cyan]{globalLimit:N0}[/] perTemplateLimit=[cyan]{perTemplateLimit:N0}[/]");

        foreach (var context in contexts)
        {
            int queued = plans.Count(x => x.ReferenceConfig.BaseQuant == context.QuantId);
            AnsiConsole.MarkupLine(
                $"[grey]  transfer stratum={Markup.Escape(context.Stratum)} reference={Markup.Escape(SafeName(context.QuantId))} " +
                $"candidates={candidatesByContext[context].Count:N0} queued={queued:N0}[/]");
        }

        return plans;
    }

    private List<SynergyTransferCandidate> BuildTransferCandidatesForContext(
        SynergyTransferContext context,
        IReadOnlyList<SynergyTransferTemplate> templates,
        IReadOnlySet<string> existingRuleKeys,
        ProbePlanningDiagnostics diagnostics)
    {
        var candidates = new List<SynergyTransferCandidate>();
        int targetTier = _movement.EffectiveTier(context.QuantId);

        foreach (var template in templates)
        {
            if (template.SourceReferenceQuantId == context.QuantId)
                continue;

            if (!TryBuildControlledTransferConfig(
                    context.QuantId,
                    template.Groups.Select(x => (x.Group.UniqueId, x.CandidateQuantId)).ToList(),
                    out var reference,
                    out var probe,
                    out var changed) ||
                changed.Count > Config.AnomalyDetection.MaxProbeGroupCount)
            {
                continue;
            }

            if (ShouldSkipInvalidContextualAnomalyConfig(probe, "synergy-context-transfer", out _))
            {
                diagnostics.SkippedInvalidMovement++;
                continue;
            }

            if (existingRuleKeys.Contains(_rules.BuildRuleSuppressionKey(reference, changed)))
            {
                diagnostics.SkippedExistingRuleOrSuppression++;
                continue;
            }

            string identityKey = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probe);
            var analyzedMovement = _movement.Analyze(reference, probe);
            var seed = new AnomalySmokeCandidate
            {
                Source = $"synergy-template-transfer:{template.Source}",
                CandidateConfig = probe,
                TwinConfig = reference,
                Movement = analyzedMovement,
                SmokeScore = 1_000_000d + template.Confidence + template.ActualEffectMagnitude,
                SmokeStrength = $"ControlledContextTransfer:{context.Stratum}",
                SeedClass = AnomalySeedClass.SynergyTransferProbe,
                MatchedConfirmedAnomalyPattern = true,
                PlannedProbeWillMeasureSize = true,
                Message = "Controlled blanket transfer probe: remeasures a confirmed tensor template under a different surrounding-fidelity context."
            };

            var plan = new AnomalyProbePlan
            {
                Seed = seed,
                ReferenceConfig = reference,
                ProbeConfig = probe,
                ProbeGroups = changed,
                ProbeType = "context-transfer",
                HypothesisLabel = $"{_movement.DescribeGroups(changed)} in {context.Stratum} {SafeName(context.QuantId)} blanket",
                SeedClass = AnomalySeedClass.SynergyTransferProbe,
                ProbePriorityClass = AnomalySeedClass.SynergyTransferProbe
            };

            double candidateTierDistance = changed
                .Select(x => Math.Abs(_movement.EffectiveTier(x.CandidateQuantId) - targetTier))
                .DefaultIfEmpty(int.MaxValue)
                .Average();
            candidates.Add(new SynergyTransferCandidate(
                template.Key,
                identityKey,
                plan,
                changed.Count,
                candidateTierDistance,
                template.Confidence,
                template.ActualEffectMagnitude));
        }

        return candidates
            .OrderBy(x => x.GroupCount)
            .ThenBy(x => x.CandidateTierDistance)
            .ThenByDescending(x => x.Confidence)
            .ThenByDescending(x => x.ActualEffectMagnitude)
            .ThenBy(x => x.IdentityKey, StringComparer.Ordinal)
            .ToList();
    }

    internal static bool TryBuildControlledTransferConfig(
        byte targetContextQuantId,
        IReadOnlyList<(byte TensorGroupId, byte CandidateQuantId)> templateGroups,
        out TensorConfig reference,
        out TensorConfig probe,
        out IReadOnlyList<AnomalyChangedGroup> changedGroups)
    {
        var movementService = new QuantFidelityComparerService();
        reference = movementService.CreateActivatedContextBlanket(targetContextQuantId);
        probe = reference;
        var changed = new List<AnomalyChangedGroup>();
        var activeGroupsById = movementService.ActiveGroups.ToDictionary(x => x.UniqueId);

        foreach (var state in templateGroups.OrderBy(x => x.TensorGroupId))
        {
            if (state.CandidateQuantId == targetContextQuantId ||
                !activeGroupsById.TryGetValue(state.TensorGroupId, out var group))
            {
                continue;
            }

            var movement = movementService.Compare(targetContextQuantId, state.CandidateQuantId);
            if (movement == QuantMovementKind.Unknown || BaselineQuants.IsNativeExactAlias(state.CandidateQuantId))
                continue;

            byte candidateStored = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(state.CandidateQuantId);
            changed.Add(new AnomalyChangedGroup
            {
                Group = group,
                ReferenceQuantId = targetContextQuantId,
                CandidateQuantId = state.CandidateQuantId,
                ReferenceStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(targetContextQuantId),
                CandidateStoredSlot = candidateStored,
                Movement = movement
            });
            probe = movementService.WithStoredSlot(probe, group, candidateStored);
        }

        changedGroups = changed;
        return changed.Count > 0;
    }

    private List<SynergyTransferTemplate> BuildTransferTemplates(
        IReadOnlyList<AnomalyInteractionRule> historicalRules,
        IReadOnlyList<AnomalyProbeResult> currentResults)
    {
        var activeGroupsById = _movement.ActiveGroups.ToDictionary(x => x.UniqueId);
        var knownQuantIds = BaselineQuants.All.Select(x => x.UniqueId).ToHashSet();
        var templates = new List<SynergyTransferTemplate>();

        foreach (var rule in historicalRules)
        {
            var groups = rule.GroupStates
                .Where(x => activeGroupsById.ContainsKey(x.TensorGroupId))
                .Where(x => knownQuantIds.Contains(x.CandidateQuantId) && !BaselineQuants.IsNativeExactAlias(x.CandidateQuantId))
                .OrderBy(x => x.TensorGroupId)
                .Select(x => new SynergyTransferTemplateGroup(activeGroupsById[x.TensorGroupId], x.CandidateQuantId))
                .ToList();
            if (groups.Count == 0)
                continue;

            templates.Add(new SynergyTransferTemplate(
                BuildTransferTemplateKey(groups),
                $"historical-rule:{rule.Id:N}:{rule.RuleDirection}",
                rule.ReferenceQuantId,
                rule.Confidence,
                Math.Abs(rule.MeanActualGainVsTwin),
                groups));
        }

        double effectScale = Math.Max(Config.AnomalyDetection.MinActualGainVsTwinKld, 1e-9d);
        foreach (var result in currentResults
                     .Where(x => x.Accepted && x.RuleDirection != AnomalyRuleDirection.SuppressionOnly)
                     .Where(x => x.ReferenceSnapshot != null && x.ProbeSnapshot != null))
        {
            var groups = result.Plan.ProbeGroups
                .Where(x => activeGroupsById.ContainsKey(x.Group.UniqueId))
                .Where(x => knownQuantIds.Contains(x.CandidateQuantId) && !BaselineQuants.IsNativeExactAlias(x.CandidateQuantId))
                .OrderBy(x => x.Group.UniqueId)
                .Select(x => new SynergyTransferTemplateGroup(activeGroupsById[x.Group.UniqueId], x.CandidateQuantId))
                .ToList();
            if (groups.Count == 0)
                continue;

            double confidence = Math.Clamp(0.70d + (0.15d * Math.Clamp(Math.Abs(result.ActualGainVsTwin) / effectScale, 0d, 2d)), 0d, 1d);
            templates.Add(new SynergyTransferTemplate(
                BuildTransferTemplateKey(groups),
                $"current-probe:{result.Plan.ProbeType}:{result.RuleDirection}",
                result.Plan.ReferenceConfig.BaseQuant,
                confidence,
                Math.Abs(result.ActualGainVsTwin),
                groups));
        }

        return templates;
    }

    private static string BuildTransferTemplateKey(IReadOnlyList<SynergyTransferTemplateGroup> groups)
        => string.Join("|", groups.OrderBy(x => x.Group.UniqueId).Select(x => $"{x.Group.UniqueId}:{x.CandidateQuantId}"));

    private static List<SynergyTransferContext> ResolveTransferTargetContexts(RuntimeSynergyTransferProbeContextStrataConfig strata)
    {
        var byName = BaselineQuants.All
            .SelectMany(x => x.Names.Select(name => (Name: name, Quant: x)))
            .GroupBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(x => x.Key, x => x.First().Quant, StringComparer.OrdinalIgnoreCase);
        var contexts = new List<SynergyTransferContext>();

        void add(IEnumerable<string> names, string stratum)
        {
            foreach (string name in names.Where(x => !string.IsNullOrWhiteSpace(x)))
            {
                if (!byName.TryGetValue(name.Trim(), out var quant) || BaselineQuants.IsNativeExactAlias(quant.UniqueId))
                    continue;

                if (contexts.All(x => x.QuantId != quant.UniqueId))
                    contexts.Add(new SynergyTransferContext(quant.UniqueId, stratum));
            }
        }

        add(strata.HighFidelityReferenceQuants, "high-fidelity");
        add(strata.MidFidelityReferenceQuants, "mid-fidelity");
        if (strata.LowFidelityEnabled)
            add(strata.LowFidelityReferenceQuants, "low-fidelity");

        return contexts;
    }



    private async Task<List<AnomalyProbePlan>> PlanConfirmedAnomalyExpansionProbesAsync(
        IReadOnlyList<AnomalyProbeResult> initialResults,
        ProbePlanningDiagnostics diagnostics,
        CancellationToken ct)
    {
        var cfg = Config.AnomalyDetection.ConfirmedAnomalyExpansion;
        if (!cfg.Enabled || cfg.MaxTotalExpansionProbes <= 0)
            return new List<AnomalyProbePlan>();

        var allowedReference = ResolveQuantNames(cfg.AllowedReferenceQuants).ToHashSet();
        var allowedCandidate = ResolveQuantNames(cfg.AllowedCandidateQuants).ToHashSet();
        var existingRuleKeys = await _rules.LoadExistingRuleSuppressionKeysAsync(ct);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var plans = new List<AnomalyProbePlan>();

        foreach (var result in initialResults
                     .Where(x => x.RuleDirection == AnomalyRuleDirection.Beneficial)
                     .Where(x => x.ReferenceSnapshot != null && x.ProbeSnapshot != null)
                     .OrderByDescending(x => x.ActualGainVsTwin))
        {
            if (plans.Count >= cfg.MaxTotalExpansionProbes)
                break;

            if (allowedReference.Count > 0 && !allowedReference.Contains(result.Plan.ReferenceConfig.BaseQuant))
                continue;

            var seedGroups = result.Plan.ProbeGroups.OrderBy(x => x.Group.UniqueId).ToList();
            if (seedGroups.Count == 0)
                continue;

            byte primaryCandidateQuant = seedGroups[0].CandidateQuantId;
            if (allowedCandidate.Count > 0 && !allowedCandidate.Contains(primaryCandidateQuant))
                continue;

            var reference = _movement.CreateActivatedContextBlanket(result.Plan.ReferenceConfig.BaseQuant);
            _movement.EnsureAllActiveGroupsExplicit(reference, "confirmed-anomaly-expansion-reference");

            int perRule = 0;
            foreach (var neighbor in _movement.ActiveGroups.Where(g => seedGroups.All(s => s.Group.UniqueId != g.UniqueId)).OrderBy(g => g.UniqueId))
            {
                if (perRule >= cfg.MaxNeighborsPerConfirmedRule || plans.Count >= cfg.MaxTotalExpansionProbes)
                    break;

                var groups = seedGroups
                    .Concat(new[]
                    {
                        new AnomalyChangedGroup
                        {
                            Group = neighbor,
                            ReferenceQuantId = reference.BaseQuant,
                            CandidateQuantId = primaryCandidateQuant,
                            ReferenceStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(reference.BaseQuant),
                            CandidateStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(primaryCandidateQuant),
                            Movement = QuantMovementKind.Downgrade
                        }
                    })
                    .OrderBy(x => x.Group.UniqueId)
                    .ToList();

                if (groups.Count > Config.AnomalyDetection.MaxProbeGroupCount)
                    continue;

                if (existingRuleKeys.Contains(_rules.BuildRuleSuppressionKey(reference, groups)))
                {
                    diagnostics.SkippedExistingRuleOrSuppression++;
                    continue;
                }

                var probe = reference;
                foreach (var group in groups)
                    probe = _movement.WithStoredSlot(probe, group.Group, group.CandidateStoredSlot);

                if (ShouldSkipInvalidContextualAnomalyConfig(probe, "confirmed-anomaly-expansion", out _))
                {
                    diagnostics.SkippedInvalidMovement++;
                    continue;
                }

                string key = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probe);
                if (!seen.Add(key))
                {
                    diagnostics.SkippedDuplicate++;
                    continue;
                }

                var seed = new AnomalySmokeCandidate
                {
                    Source = "confirmed-anomaly-neighborhood",
                    CandidateConfig = probe,
                    TwinConfig = reference,
                    Movement = _movement.Analyze(reference, probe),
                    CandidatePredictedKld = result.Plan.Seed.CandidatePredictedKld,
                    TwinPredictedKld = result.Plan.Seed.TwinPredictedKld,
                    PredictionSpaceGapVsTwin = result.Plan.Seed.PredictionSpaceGapVsTwin,
                    SmokeScore = 900_000d + Math.Max(0d, result.ActualGainVsTwin),
                    SmokeStrength = "ConfirmedAnomalyNeighborhood",
                    SeedClass = AnomalySeedClass.ConfirmedAnomalyNeighborhoodProbe,
                    MatchedConfirmedAnomalyPattern = true,
                    PlannedProbeWillMeasureSize = true,
                    Message = "Bounded neighborhood probe generated from a confirmed beneficial contextual anomaly."
                };

                plans.Add(new AnomalyProbePlan
                {
                    Seed = seed,
                    ReferenceConfig = reference,
                    ProbeConfig = probe,
                    ProbeGroups = groups,
                    ProbeType = "confirmed-neighborhood",
                    HypothesisLabel = _movement.DescribeGroups(groups),
                    SeedClass = AnomalySeedClass.ConfirmedAnomalyNeighborhoodProbe,
                    ProbePriorityClass = groups.Count == 1 ? AnomalySeedClass.ExploratorySingle : AnomalySeedClass.ExploratoryPair
                });

                perRule++;
                diagnostics.ProbesQueued++;
                diagnostics.ExpansionProbesQueued++;
            }
        }

        if (plans.Count > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]Confirmed anomaly neighborhood probes:[/] queued={plans.Count:N0} maxTotal={cfg.MaxTotalExpansionProbes:N0}");
        }

        return plans;
    }



    private async Task<List<AnomalyProbePlan>> PlanSynergyCompositionProbesAsync(
        IReadOnlyList<AnomalyProbeResult> results,
        ProbePlanningDiagnostics diagnostics,
        List<SynergyCompositionProbeRecord> previewRecords,
        CancellationToken ct)
    {
        var cfg = Config.SynergyDetection;
        var plans = new List<AnomalyProbePlan>();
        if (!cfg.Enabled || !cfg.CompositionProbeEnabled || cfg.MaxCompositionProbesPerRun <= 0)
            return plans;

        var templates = results
            .Where(x => x.RuleDirection == AnomalyRuleDirection.Beneficial)
            .Where(x => x.Accepted && x.ReferenceSnapshot != null && x.ProbeSnapshot != null)
            .Where(x => x.Plan.ProbeGroups.Count > 0)
            .Select(x => new
            {
                Result = x,
                Confidence = Math.Clamp(Math.Abs(x.ActualGainVsTwin) / Math.Max(Config.AnomalyDetection.MinActualGainVsTwinKld, 1e-9), 0d, 1d),
                Key = string.Join(",", x.Plan.ProbeGroups.OrderBy(g => g.Group.UniqueId).Select(g => $"{g.Group.UniqueId}:{g.CandidateQuantId}"))
            })
            .Where(x => x.Confidence >= cfg.MinTemplateConfidenceForComposition)
            .GroupBy(x => x.Key, StringComparer.Ordinal)
            .Select(g => g.OrderByDescending(x => x.Result.ActualGainVsTwin).First())
            .OrderByDescending(x => x.Result.ActualGainVsTwin)
            .Take(cfg.MaxTemplatesToCompose)
            .ToList();

        int considered = templates.Count;
        int candidates = 0;
        int reused = 0;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var existingRuleKeys = await _rules.LoadExistingRuleSuppressionKeysAsync(ct);

        for (int i = 0; i < templates.Count; i++)
        {
            for (int j = i + 1; j < templates.Count; j++)
            {
                if (plans.Count >= cfg.MaxCompositionProbesPerRun)
                    break;

                var a = templates[i].Result;
                var b = templates[j].Result;
                if (a.Plan.ReferenceConfig.BaseQuant != b.Plan.ReferenceConfig.BaseQuant)
                    continue;

                var merged = a.Plan.ProbeGroups
                    .Concat(b.Plan.ProbeGroups)
                    .GroupBy(g => g.Group.UniqueId)
                    .Select(g => g.OrderBy(x => x.CandidateQuantId).First())
                    .OrderBy(g => g.Group.UniqueId)
                    .ToList();

                if (merged.Count <= Math.Max(a.Plan.ProbeGroups.Count, b.Plan.ProbeGroups.Count))
                    continue;
                if (merged.Count > cfg.MaxTemplateCompositionGroupCount)
                    continue;

                candidates++;
                var reference = _movement.CreateActivatedContextBlanket(a.Plan.ReferenceConfig.BaseQuant);
                var probe = reference;
                foreach (var group in merged)
                    probe = _movement.WithStoredSlot(probe, group.Group, group.CandidateStoredSlot);

                if (ShouldSkipInvalidContextualAnomalyConfig(probe, "synergy-composition", out var skipReason))
                {
                    diagnostics.SkippedInvalidMovement++;
                    previewRecords.Add(BuildCompositionPreview(a, b, reference, probe, "SkippedInvalidContextualProbe", skipReason));
                    continue;
                }

                if (existingRuleKeys.Contains(_rules.BuildRuleSuppressionKey(reference, merged)))
                {
                    diagnostics.SkippedExistingRuleOrSuppression++;
                    reused++;
                    previewRecords.Add(BuildCompositionPreview(a, b, reference, probe, "ExistingRuleOrSuppression", "A matching composition rule/suppression already exists."));
                    continue;
                }

                string key = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probe);
                if (!seen.Add(key))
                {
                    diagnostics.SkippedDuplicate++;
                    reused++;
                    continue;
                }

                var movement = _movement.Analyze(reference, probe);
                var seed = new AnomalySmokeCandidate
                {
                    Source = "counterfactual-synergy-composition",
                    CandidateConfig = probe,
                    TwinConfig = reference,
                    Movement = movement,
                    SmokeScore = 950_000d + Math.Max(0d, a.ActualGainVsTwin) + Math.Max(0d, b.ActualGainVsTwin),
                    SmokeStrength = "CompositionProbe",
                    SeedClass = AnomalySeedClass.SynergyCompositionProbe,
                    MatchedConfirmedAnomalyPattern = true,
                    PlannedProbeWillMeasureSize = true,
                    Message = "Tiny composition probe: tests whether two confirmed synergy templates cooperate, add, overlap redundantly, or interfere."
                };

                plans.Add(new AnomalyProbePlan
                {
                    Seed = seed,
                    ReferenceConfig = reference,
                    ProbeConfig = probe,
                    ProbeGroups = merged,
                    ProbeType = "composition",
                    HypothesisLabel = _movement.DescribeGroups(merged),
                    SeedClass = AnomalySeedClass.SynergyCompositionProbe,
                    ProbePriorityClass = AnomalySeedClass.SynergyCompositionProbe
                });
                diagnostics.ProbesQueued++;
                diagnostics.CompositionProbesQueued++;
                previewRecords.Add(BuildCompositionPreview(a, b, reference, probe, "Queued", "Composition probe queued for real benchmark validation."));
            }
        }

        AnsiConsole.MarkupLine($"[yellow]Synergy composition probes:[/] source templates considered=[cyan]{considered:N0}[/] composition candidates=[cyan]{candidates:N0}[/] queued=[cyan]{plans.Count:N0}[/] reused/skipped existing=[cyan]{reused:N0}[/]");
        return plans;
    }

    private SynergyCompositionProbeRecord BuildCompositionPreview(
        AnomalyProbeResult a,
        AnomalyProbeResult b,
        TensorConfig reference,
        TensorConfig probe,
        string classification,
        string note)
    {
        var movement = _movement.Analyze(reference, probe);
        return new SynergyCompositionProbeRecord
        {
            CompositionId = TensorConfigIdentity.ToKey(reference) + "=>" + TensorConfigIdentity.ToKey(probe),
            SourceTemplateIds = new[]
            {
                TensorConfigIdentity.ToKey(a.Plan.ProbeConfig),
                TensorConfigIdentity.ToKey(b.Plan.ProbeConfig)
            },
            SourceTemplateLabels = new[] { a.Plan.HypothesisLabel, b.Plan.HypothesisLabel },
            CandidateEffectiveGroups = _movement.BuildEffectiveGroupVector(probe),
            TwinEffectiveGroups = _movement.BuildEffectiveGroupVector(reference),
            CombinedGroupCount = movement.DowngradeCount,
            Classification = classification,
            ActualCandidateKld = null,
            ActualTwinKld = null,
            ActualGainVsTwin = null,
            PredictedCandidateKld = null,
            PredictedTwinKld = null,
            PredictionSpaceGap = null,
            Notes = new[] { note }
        };
    }

    private List<SynergyCompositionProbeRecord> BuildCompositionDiagnostics(
        IReadOnlyList<AnomalyProbePlan> compositionPlans,
        IReadOnlyList<AnomalyProbeResult> compositionResults)
    {
        var records = new List<SynergyCompositionProbeRecord>();
        foreach (var result in compositionResults.Where(x => x.Plan.SeedClass == AnomalySeedClass.SynergyCompositionProbe))
        {
            string classification = result.Classification switch
            {
                AnomalyProbeClassification.HarmfulInteraction => "HarmfulInterference",
                AnomalyProbeClassification.ContaminatingPassenger => "HarmfulInterference",
                AnomalyProbeClassification.NormalGravity => "RedundantComposition",
                AnomalyProbeClassification.SuppressionOnly => "CompositionRejected",
                _ when result.RuleDirection == AnomalyRuleDirection.Beneficial && result.ActualGainVsTwin >= Config.AnomalyDetection.MinActualGainVsTwinKld * 2d => "SuperSynergy",
                _ when result.RuleDirection == AnomalyRuleDirection.Beneficial => "AdditiveComposition",
                _ => "CompositionRejected"
            };

            records.Add(new SynergyCompositionProbeRecord
            {
                CompositionId = TensorConfigIdentity.ToKey(result.Plan.ReferenceConfig) + "=>" + TensorConfigIdentity.ToKey(result.Plan.ProbeConfig),
                SourceTemplateIds = result.Plan.ProbeGroups.Select(g => $"{g.Group.UniqueId}:{g.ReferenceQuantId}->{g.CandidateQuantId}").ToList(),
                SourceTemplateLabels = new[] { result.Plan.HypothesisLabel },
                CandidateEffectiveGroups = _movement.BuildEffectiveGroupVector(result.Plan.ProbeConfig),
                TwinEffectiveGroups = _movement.BuildEffectiveGroupVector(result.Plan.ReferenceConfig),
                CombinedGroupCount = result.Plan.ProbeGroups.Count,
                Classification = classification,
                ActualCandidateKld = result.ProbeSnapshot?.Kld,
                ActualTwinKld = result.ReferenceSnapshot?.Kld,
                ActualGainVsTwin = result.ActualGainVsTwin,
                PredictedCandidateKld = result.Plan.Seed.CandidatePredictedKld,
                PredictedTwinKld = result.Plan.Seed.TwinPredictedKld,
                PredictionSpaceGap = result.Plan.Seed.PredictionSpaceGapVsTwin,
                Notes = new[] { result.Message }
            });
        }

        int super = records.Count(x => x.Classification == "SuperSynergy");
        int additive = records.Count(x => x.Classification == "AdditiveComposition");
        int harmful = records.Count(x => x.Classification == "HarmfulInterference");
        int rejected = records.Count(x => x.Classification == "CompositionRejected" || x.Classification == "RedundantComposition");
        AnsiConsole.MarkupLine($"[yellow]Synergy composition probes:[/] confirmed super-synergy=[cyan]{super:N0}[/] additive=[cyan]{additive:N0}[/] harmful/interference=[cyan]{harmful:N0}[/] rejected/redundant=[cyan]{rejected:N0}[/]");
        return records;
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
                    "composition" when gain >= Config.AnomalyDetection.MinActualGainVsTwinKld * 2d => AnomalyProbeClassification.SuperSynergy,
                    "composition" => AnomalyProbeClassification.AdditiveComposition,
                    "full" when plan.ProbeGroups.Count >= 3 => AnomalyProbeClassification.HigherOrderSynergy,
                    _ => AnomalyProbeClassification.CounterfactualMdaViolation
                },
                RuleDirection = AnomalyRuleDirection.Beneficial,
                Accepted = true,
                ActualGainVsTwin = gain,
                Message = plan.ProbeType == "composition"
                    ? "Composed counterfactual synergy template beat its higher-fidelity same-context twin."
                    : "Lower-fidelity monotone probe beat its higher-fidelity same-context twin."
            };
        }

        double harmfulMargin = Math.Max(Config.AnomalyDetection.MinActualGainVsTwinKld, Config.SynergyDetection.MinFailureMarginForContaminationKld);
        if (probe.Kld - reference.Kld >= harmfulMargin)
        {
            bool contaminatingPassenger = Config.SynergyDetection.ContaminatingPassengerDetectionEnabled &&
                                         (plan.SeedClass == AnomalySeedClass.ConfirmedAnomalyNeighborhoodProbe ||
                                          plan.SeedClass == AnomalySeedClass.SynergyTransferProbe ||
                                          plan.SeedClass == AnomalySeedClass.SynergyCompositionProbe) &&
                                         plan.ProbeGroups.Count > 1;

            return new AnomalyProbeResult
            {
                Plan = plan,
                ReferenceSnapshot = reference,
                ProbeSnapshot = probe,
                Classification = contaminatingPassenger
                    ? AnomalyProbeClassification.ContaminatingPassenger
                    : plan.ProbeType == "composition"
                        ? AnomalyProbeClassification.HarmfulInterference
                        : AnomalyProbeClassification.HarmfulInteraction,
                RuleDirection = AnomalyRuleDirection.Harmful,
                Accepted = true,
                ActualGainVsTwin = gain,
                Message = contaminatingPassenger
                    ? "Probe was meaningfully worse than its twin; persisted as scoped contaminating-passenger negative evidence."
                    : "Probe was meaningfully worse than its higher-fidelity twin; persisted as harmful interaction."
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


    private async Task EmitQ8ContextReferenceDriftDiagnosticsAsync(
        IReadOnlyDictionary<string, BenchmarkSnapshotRecord> byKey,
        CancellationToken ct)
    {
        byte q8 = BaselineQuants.Q8_0.UniqueId;
        var sparse = BuildSparsePureContext(q8);
        TensorConfig explicitContext;
        try
        {
            explicitContext = _movement.CreateActivatedContextBlanket(q8);
        }
        catch
        {
            return;
        }

        byKey.TryGetValue(TensorConfigIdentity.ToKey(sparse), out var sparseSnapshot);
        byKey.TryGetValue(TensorConfigIdentity.ToKey(explicitContext), out var explicitSnapshot);
        if (sparseSnapshot == null || explicitSnapshot == null)
            return;

        double kldDelta = explicitSnapshot.Kld - sparseSnapshot.Kld;
        long sizeDelta = unchecked((long)explicitSnapshot.SizeBytes - (long)sparseSnapshot.SizeBytes);
        bool material = Math.Abs(kldDelta) >= Config.AnomalyDetection.MinActualGainVsTwinKld || Math.Abs(sizeDelta) > 0;

        if (material)
            AnsiConsole.MarkupLine("[yellow]Q8_CONTEXT_REFERENCE_DRIFT[/]");
        else if (Config.AnomalyDetection.VerboseAnomalyLogging)
            AnsiConsole.MarkupLine("[grey]Q8 contextual reference drift check:[/]");

        if (material || Config.AnomalyDetection.VerboseAnomalyLogging)
        {
            AnsiConsole.MarkupLine($"[grey]  pureQ8Kld=[/] [cyan]{sparseSnapshot.Kld:0.000000}[/] [grey]explicitContextQ8Kld=[/] [cyan]{explicitSnapshot.Kld:0.000000}[/] [grey]kldDelta=[/] [cyan]{kldDelta:0.000000}[/]");
            AnsiConsole.MarkupLine($"[grey]  pureQ8SizeBytes=[/] [cyan]{sparseSnapshot.SizeBytes:N0}[/] [grey]explicitContextQ8SizeBytes=[/] [cyan]{explicitSnapshot.SizeBytes:N0}[/] [grey]sizeDeltaBytes=[/] [cyan]{sizeDelta:N0}[/]");
        }

        await WriteJsonAsync("magicquant-anomaly-q8-reference-drift.json", new
        {
            generatedAtUtc = DateTime.UtcNow,
            driftCode = material ? "Q8_CONTEXT_REFERENCE_DRIFT" : "none",
            pureQ8 = new { key = TensorConfigIdentity.ToKey(sparseSnapshot.Config), sparseSnapshot.DisplayName, sparseSnapshot.Kld, sparseSnapshot.SizeBytes },
            explicitContextQ8 = new { key = TensorConfigIdentity.ToKey(explicitSnapshot.Config), explicitSnapshot.DisplayName, explicitSnapshot.Kld, explicitSnapshot.SizeBytes },
            kldDelta,
            sizeDeltaBytes = sizeDelta,
            note = "Anomaly probes prefer the explicit all-active contextual Q8 twin. Sparse pure Q8 remains a useful baseline anchor but may not be identical if benchmark execution settings drifted. Compare NGL/benchmark run metadata in SQLite BenchmarkRuns if material drift appears."
        }, ct);
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


    private static void AddOrPreferBetterPredictionRow(IDictionary<string, PredictionDuckRow> rows, PredictionDuckRow row)
    {
        string key = TensorConfigIdentity.ToKey(row.Config);
        if (!rows.TryGetValue(key, out var existing) ||
            row.BaseRankSafeKld < existing.BaseRankSafeKld ||
            (Math.Abs(row.BaseRankSafeKld - existing.BaseRankSafeKld) < 1e-12 && row.PredictedSizeBytes < existing.PredictedSizeBytes))
        {
            rows[key] = row;
        }
    }

    private static TensorConfig BuildSparsePureContext(byte baseQuantId) => new(
        baseQuantId,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue,
        BaselineQuants.TensorConfigNullSlotValue);

    private PredictionDuckRow? ResolveTwinFromLookupOnly(
        TensorConfig explicitTwin,
        IReadOnlyDictionary<string, PredictionDuckRow> lookup,
        out string lookupMode,
        out bool foundInDictionary)
    {
        string explicitKey = TensorConfigIdentity.ToKey(explicitTwin);
        if (lookup.TryGetValue(explicitKey, out var explicitRow))
        {
            lookupMode = "explicit-context-found";
            foundInDictionary = true;
            return explicitRow with { Config = explicitTwin };
        }

        var sparsePure = BuildSparsePureContext(explicitTwin.BaseQuant);
        string sparseKey = TensorConfigIdentity.ToKey(sparsePure);
        if (lookup.TryGetValue(sparseKey, out var sparseRow))
        {
            lookupMode = "sparse-pure-prediction-fallback; explicit-context-identity-preserved";
            foundInDictionary = true;
            return sparseRow with { Config = sparsePure };
        }

        lookupMode = "missing; dictionary-only lookup searched explicit-context and sparse-pure";
        foundInDictionary = false;
        return null;
    }

    private RejectedSmokePreview AddRejectedPreview(
        List<RejectedSmokePreview> previews,
        TensorConfig candidate,
        TensorConfig twin,
        AnomalyMovementAnalysis? movement,
        PredictionDuckRow? candidateRow,
        PredictionDuckRow? twinRow,
        ulong? predictedSizeSavingsBytes,
        double? predictionSpaceGap,
        string rejectionReason,
        bool matchedConfirmedAnomalyPattern,
        bool twinFoundInLookup)
    {
        var preview = new RejectedSmokePreview(
            previews.Count,
            candidate,
            twin,
            movement,
            candidateRow,
            twinRow,
            predictedSizeSavingsBytes,
            predictionSpaceGap,
            rejectionReason,
            matchedConfirmedAnomalyPattern,
            twinFoundInLookup);

        if (previews.Count < 500 || rejectionReason.Contains("PredictionSpaceGap", StringComparison.OrdinalIgnoreCase) || matchedConfirmedAnomalyPattern)
            previews.Add(preview);

        return preview;
    }

    private static HashSet<byte> ResolveQuantNames(IEnumerable<string> names)
    {
        var map = BaselineQuants.GetAllRecognizedBaselines()
            .SelectMany(q => q.Names.Select(n => (Name: n, Quant: q)))
            .GroupBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.First().Quant.UniqueId, StringComparer.OrdinalIgnoreCase);

        var result = new HashSet<byte>();
        foreach (var name in names ?? Array.Empty<string>())
        {
            if (map.TryGetValue(name, out var id))
                result.Add(id);
        }

        return result;
    }

    private object? BuildBestConfirmedAnomalyReconciliation(
        IReadOnlyList<AnomalyProbeResult> results,
        IReadOnlyCollection<BenchmarkSnapshotRecord> selectedSurvivors)
    {
        var best = results
            .Where(x => x.RuleDirection == AnomalyRuleDirection.Beneficial)
            .Where(x => x.ReferenceSnapshot != null && x.ProbeSnapshot != null)
            .OrderByDescending(x => x.ActualGainVsTwin)
            .ThenBy(x => x.ProbeSnapshot!.Kld)
            .ThenBy(x => x.ProbeSnapshot!.SizeBytes)
            .FirstOrDefault();

        if (best == null || best.ReferenceSnapshot == null || best.ProbeSnapshot == null)
            return null;

        string key = TensorConfigIdentity.ToKey(best.ProbeSnapshot.Config);
        bool selected = selectedSurvivors.Any(x => TensorConfigIdentity.ToKey(x.Config) == key);
        string reasonNotSelected = selected
            ? string.Empty
            : selectedSurvivors.Count == 0
                ? "final selection has not run yet"
                : BuildReasonBestAnomalyNotSelected(best.ProbeSnapshot, selectedSurvivors);

        return new
        {
            candidate = TensorConfigIdentity.ToKey(best.ProbeSnapshot.Config),
            candidateName = best.ProbeSnapshot.DisplayName,
            twin = TensorConfigIdentity.ToKey(best.ReferenceSnapshot.Config),
            twinName = best.ReferenceSnapshot.DisplayName,
            actualCandidateKld = best.ProbeSnapshot.Kld,
            actualTwinKld = best.ReferenceSnapshot.Kld,
            actualGain = best.ActualGainVsTwin,
            actualCandidateSizeBytes = best.ProbeSnapshot.SizeBytes,
            actualTwinSizeBytes = best.ReferenceSnapshot.SizeBytes,
            actualSizeSavingsBytes = best.ReferenceSnapshot.SizeBytes >= best.ProbeSnapshot.SizeBytes ? best.ReferenceSnapshot.SizeBytes - best.ProbeSnapshot.SizeBytes : 0UL,
            classification = best.Classification.ToString(),
            selectedAsSurvivor = selected,
            reasonNotSelected
        };
    }

    private static string BuildReasonBestAnomalyNotSelected(
        BenchmarkSnapshotRecord anomaly,
        IReadOnlyCollection<BenchmarkSnapshotRecord> selectedSurvivors)
    {
        var dominator = selectedSurvivors.FirstOrDefault(x => x.SizeBytes <= anomaly.SizeBytes && x.Kld <= anomaly.Kld && (x.SizeBytes < anomaly.SizeBytes || x.Kld < anomaly.Kld));
        if (dominator != null)
            return $"dominated by selected survivor {dominator.DisplayName} (kld={dominator.Kld:0.000000}, size={dominator.SizeBytes})";

        var lowerKld = selectedSurvivors.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes).FirstOrDefault();
        if (lowerKld != null && lowerKld.Kld < anomaly.Kld)
            return $"selected frontier contains lower-KLD survivor {lowerKld.DisplayName}; anomaly was not a final dominance/spacing winner";

        return "not present in selected survivor set; no dominance reason was found in current reconciliation data";
    }

    private static void WriteBestAnomalyConsoleLog(object? reconciliation)
    {
        if (reconciliation == null)
        {
            AnsiConsole.MarkupLine("[grey]Best confirmed beneficial anomaly:[/] none");
            return;
        }

        string json = JsonSerializer.Serialize(reconciliation, JsonOptions);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        AnsiConsole.MarkupLine("[yellow]Best confirmed beneficial anomaly:[/]");
        AnsiConsole.MarkupLine($"[grey]  candidate=[/] [cyan]{Markup.Escape(root.GetProperty("candidateName").GetString() ?? "unknown")}[/]");
        AnsiConsole.MarkupLine($"[grey]  twin=[/] [cyan]{Markup.Escape(root.GetProperty("twinName").GetString() ?? "unknown")}[/]");
        AnsiConsole.MarkupLine($"[grey]  actualCandidateKld=[/] [cyan]{root.GetProperty("actualCandidateKld").GetDouble():0.000000}[/]");
        AnsiConsole.MarkupLine($"[grey]  actualTwinKld=[/] [cyan]{root.GetProperty("actualTwinKld").GetDouble():0.000000}[/]");
        AnsiConsole.MarkupLine($"[grey]  actualGain=[/] [cyan]{root.GetProperty("actualGain").GetDouble():0.000000}[/]");
        AnsiConsole.MarkupLine($"[grey]  selectedAsSurvivor=[/] [cyan]{root.GetProperty("selectedAsSurvivor").GetBoolean()}[/]");
        string reason = root.TryGetProperty("reasonNotSelected", out var r) ? r.GetString() ?? string.Empty : string.Empty;
        if (!string.IsNullOrWhiteSpace(reason))
            AnsiConsole.MarkupLine($"[grey]  reasonNotSelected=[/] [yellow]{Markup.Escape(reason)}[/]");
    }

    private static string FmtNullable(double? value) => value.HasValue ? value.Value.ToString("0.000000", CultureInfo.InvariantCulture) : "n/a";
    private static string FmtNullable(ulong? value) => value.HasValue ? value.Value.ToString("N0", CultureInfo.InvariantCulture) : "n/a";

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
        AnsiConsole.MarkupLine($"[grey]  seedClass=[/] [cyan]{Markup.Escape(plan.SeedClass.ToString())}[/] [grey]priority=[/] [cyan]{Markup.Escape(plan.ProbePriorityClass.ToString())}[/]");
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
        double maxGap = Math.Max(Config.AnomalyDetection.MaxPredictionSpaceGapVsTwinKld, Config.SynergyDetection.MaxSmokeGapKld);
        double closenessScore = maxGap <= 0d ? 0d : Math.Clamp((maxGap - Math.Max(0d, gap)) / maxGap, 0d, 1d);
        double savingsScore = Math.Clamp(savingsPercent / Math.Max(Config.AnomalyDetection.MinPredictedSizeSavingsVsTwinPercent, 0.01d), 0d, 2d) / 2d;
        double groupPenalty = 1d / Math.Max(1, changedGroupCount);
        double rankBonus = 0d;
        if (candidateRank.HasValue && twinRank.HasValue)
            rankBonus = Math.Clamp((double)twinRank.Value - candidateRank.Value, -10_000d, 10_000d) / 20_000d;

        return Math.Clamp((closenessScore * 0.55d) + (savingsScore * 0.30d) + (groupPenalty * 0.10d) + rankBonus, 0d, 1d);
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




    private IReadOnlyList<SynergyWingSummary> BuildSynergyWingSummary(
        IReadOnlyList<AnomalySmokeCandidate> smoke,
        IReadOnlyList<AnomalyProbeResult> results,
        AnomalyAdjustmentSummary adjustment)
    {
        var zones = new Dictionary<string, SynergyWingSummary>(StringComparer.OrdinalIgnoreCase)
        {
            ["Q8→Q6"] = new SynergyWingSummary { Zone = "Q8→Q6" },
            ["Q6→Q5/Q4"] = new SynergyWingSummary { Zone = "Q6→Q5/Q4" },
            ["Q5→IQ4/IQ3"] = new SynergyWingSummary { Zone = "Q5→IQ4/IQ3" },
            ["Other"] = new SynergyWingSummary { Zone = "Other" }
        };

        foreach (var item in smoke)
            zones[ResolveWingZone(item.TwinConfig.BaseQuant, item.Movement.ChangedGroups.Select(x => x.CandidateQuantId))].SmokeCount++;

        foreach (var result in results)
        {
            var zone = zones[ResolveWingZone(result.Plan.ReferenceConfig.BaseQuant, result.Plan.ProbeGroups.Select(x => x.CandidateQuantId))];
            if (result.RuleDirection == AnomalyRuleDirection.Beneficial)
            {
                zone.ConfirmedBeneficialTemplates++;
                zone.ValidationSuccessCount++;
            }
            else if (result.RuleDirection == AnomalyRuleDirection.Harmful)
            {
                zone.HarmfulTemplates++;
                zone.ValidationFailureCount++;
            }
            else
            {
                zone.SuppressionOnlyTemplates++;
                zone.ValidationFailureCount++;
            }
        }

        foreach (var match in adjustment.RuleMatches)
        {
            string text = JsonSerializer.Serialize(match);
            var zone = text.Contains("Q6_", StringComparison.OrdinalIgnoreCase) && text.Contains("Q8_0", StringComparison.OrdinalIgnoreCase)
                ? zones["Q8→Q6"]
                : text.Contains("Q5", StringComparison.OrdinalIgnoreCase) || text.Contains("Q4", StringComparison.OrdinalIgnoreCase)
                    ? zones["Q6→Q5/Q4"]
                    : zones["Other"];

            if (text.Contains("Harmful", StringComparison.OrdinalIgnoreCase) || text.Contains("Contaminating", StringComparison.OrdinalIgnoreCase))
                zone.CandidateRowsDemoted++;
            else
                zone.CandidateRowsAdjustedPositively++;
        }

        foreach (var zone in zones.Values)
        {
            if (zone.ConfirmedBeneficialTemplates == 0 && (zone.HarmfulTemplates > 0 || zone.SuppressionOnlyTemplates > 0))
                zone.Explanation = "No nonlinear wing survived because harmful/suppression evidence dominated or no adjusted candidate beat the frontier line.";
            else if (zone.ConfirmedBeneficialTemplates == 0 && zone.SmokeCount == 0)
                zone.Explanation = "No contextual synergy smoke was strong enough to probe in this fidelity zone.";
            else if (zone.ConfirmedBeneficialTemplates > 0)
                zone.Explanation = "Confirmed beneficial counterfactual synergy evidence exists in this fidelity zone.";
            else
                zone.Explanation = "Smoke existed but did not produce confirmed beneficial evidence.";
        }

        return zones.Values.ToList();
    }

    private static string ResolveWingZone(byte referenceQuantId, IEnumerable<byte> candidateQuantIds)
    {
        string reference = SafeName(referenceQuantId);
        var candidates = candidateQuantIds.Select(SafeName).ToList();
        if (reference.StartsWith("Q8", StringComparison.OrdinalIgnoreCase) && candidates.Any(x => x.Contains("Q6", StringComparison.OrdinalIgnoreCase)))
            return "Q8→Q6";
        if (reference.Contains("Q6", StringComparison.OrdinalIgnoreCase) && candidates.Any(x => x.Contains("Q5", StringComparison.OrdinalIgnoreCase) || x.Contains("Q4", StringComparison.OrdinalIgnoreCase) || x.Contains("IQ4", StringComparison.OrdinalIgnoreCase)))
            return "Q6→Q5/Q4";
        if ((reference.Contains("Q5", StringComparison.OrdinalIgnoreCase) || reference.Contains("Q4", StringComparison.OrdinalIgnoreCase)) && candidates.Any(x => x.Contains("IQ4", StringComparison.OrdinalIgnoreCase) || x.Contains("IQ3", StringComparison.OrdinalIgnoreCase) || x.Contains("Q3", StringComparison.OrdinalIgnoreCase)))
            return "Q5→IQ4/IQ3";
        return "Other";
    }

    private static void WriteSynergyWingConsoleSummary(IReadOnlyList<SynergyWingSummary> summaries)
    {
        AnsiConsole.MarkupLine("[yellow]Synergy wing summary:[/]");
        foreach (var zone in summaries)
        {
            AnsiConsole.MarkupLine($"[grey]  {Markup.Escape(zone.Zone)}:[/] beneficial=[cyan]{zone.ConfirmedBeneficialTemplates:N0}[/], harmful=[cyan]{zone.HarmfulTemplates:N0}[/], suppressed=[cyan]{zone.SuppressionOnlyTemplates:N0}[/], adjusted=[cyan]{zone.CandidateRowsAdjustedPositively:N0}[/], demoted=[cyan]{zone.CandidateRowsDemoted:N0}[/]");
        }
    }

    private object ToSynergyTemplateLog(AnomalyInteractionRule x)
    {
        Dictionary<string, string> selected = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string> raised = new(StringComparer.OrdinalIgnoreCase);
        foreach (var state in x.GroupStates.OrderBy(g => g.SortOrder))
        {
            string groupName = ResolveGroupName(state.TensorGroupId);
            selected[groupName] = SafeName(state.CandidateQuantId);
            raised[groupName] = SafeName(state.ReferenceQuantId);
        }

        return new
        {
            templateType = x.RuleType,
            referenceQuant = SafeName(x.ReferenceQuantId),
            selectedGroupStates = selected,
            raisedCounterfactualStates = raised,
            discoveryContext = TryDeserializeDictionary(x.ReferenceEffectiveGroupsJson),
            candidateContext = TryDeserializeDictionary(x.CandidateEffectiveGroupsJson),
            actualGainKld = x.BestActualGainVsTwin,
            sizeSavingsBytes = x.MetadataJson?.Contains("sizeSavingsBytes", StringComparison.OrdinalIgnoreCase) == true ? (object?)"see metadataJson" : null,
            x.Confidence,
            generalizationPolicy = "ExactStrong_TransferWeak",
            metadataJson = x.MetadataJson
        };
    }

    private static Dictionary<string, string> TryDeserializeDictionary(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, string>>(json) ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }
        catch
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }
    }

    private static string ResolveGroupName(byte groupId)
    {
        var group = TReg.All.FirstOrDefault(x => x.UniqueId == groupId);
        return group?.Name ?? $"group:{groupId}";
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
            x.PredictedSizeSavingsBytes,
            x.CandidateActualSizeBytes,
            x.TwinActualSizeBytes,
            x.ActualSizeSavingsBytes,
            x.PlannedProbeWillMeasureSize,
            x.TwinLookupMode,
            x.RejectionReason,
            x.MatchedConfirmedAnomalyPattern,
            x.TwinFoundInLookupDictionary,
            seedClass = x.SeedClass.ToString(),
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
            seedClass = x.SeedClass.ToString(),
            probePriorityClass = x.ProbePriorityClass.ToString(),
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


    private sealed record SynergyTransferContext(byte QuantId, string Stratum);

    private sealed record SynergyTransferTemplateGroup(TensorGroup Group, byte CandidateQuantId);

    private sealed record SynergyTransferTemplate(
        string Key,
        string Source,
        byte SourceReferenceQuantId,
        double Confidence,
        double ActualEffectMagnitude,
        IReadOnlyList<SynergyTransferTemplateGroup> Groups);

    private sealed record SynergyTransferCandidate(
        string TemplateKey,
        string IdentityKey,
        AnomalyProbePlan Plan,
        int GroupCount,
        double CandidateTierDistance,
        double Confidence,
        double ActualEffectMagnitude);


    private sealed class RejectedSmokePreview
    {
        public RejectedSmokePreview(
            int sortOrder,
            TensorConfig candidate,
            TensorConfig twin,
            AnomalyMovementAnalysis? movement,
            PredictionDuckRow? candidateRow,
            PredictionDuckRow? twinRow,
            ulong? predictedSizeSavingsBytes,
            double? predictionSpaceGap,
            string rejectionReason,
            bool matchedConfirmedAnomalyPattern,
            bool twinFoundInLookup)
        {
            SortOrder = sortOrder;
            Candidate = candidate;
            Twin = twin;
            Movement = movement;
            CandidateRow = candidateRow;
            TwinRow = twinRow;
            PredictedSizeSavingsBytes = predictedSizeSavingsBytes;
            PredictionSpaceGap = predictionSpaceGap;
            RejectionReason = rejectionReason;
            MatchedConfirmedAnomalyPattern = matchedConfirmedAnomalyPattern;
            TwinFoundInLookup = twinFoundInLookup;
        }

        public int SortOrder { get; }
        public TensorConfig Candidate { get; }
        public TensorConfig Twin { get; }
        public AnomalyMovementAnalysis? Movement { get; }
        public PredictionDuckRow? CandidateRow { get; }
        public PredictionDuckRow? TwinRow { get; }
        public ulong? PredictedSizeSavingsBytes { get; }
        public double? PredictionSpaceGap { get; }
        public string RejectionReason { get; }
        public bool MatchedConfirmedAnomalyPattern { get; }
        public bool TwinFoundInLookup { get; }
        public string CandidateName => HybridBenchmarkRepository.BuildDisplayName((HybridQuant)Candidate);
        public string TwinName => HybridBenchmarkRepository.BuildDisplayName((HybridQuant)Twin);

        public object ToLog() => new
        {
            candidate = TensorConfigIdentity.ToKey(Candidate),
            twin = TensorConfigIdentity.ToKey(Twin),
            candidateName = CandidateName,
            twinName = TwinName,
            movement = Movement?.Classification.ToString() ?? "Unknown",
            changedGroups = Movement?.ChangedGroups.Select(g => $"{g.Group.ShortCode}={AnomalyWorkflowService.SafeName(g.CandidateQuantId)}").ToList() ?? new List<string>(),
            smokeScore = Movement == null || CandidateRow == null || TwinRow == null || !PredictionSpaceGap.HasValue || !PredictedSizeSavingsBytes.HasValue
                ? (double?)null
                : AnomalyWorkflowService.ComputeSmokeScore(PredictionSpaceGap.Value, PredictedSizeSavingsBytes.Value * 100d / Math.Max(1d, TwinRow.PredictedSizeBytes), Movement.DowngradeCount, CandidateRow.PredictionRank, TwinRow.PredictionRank),
            predictedCandidateKld = CandidateRow?.BaseRankSafeKld,
            predictedTwinKld = TwinRow?.BaseRankSafeKld,
            predictionSpaceGap = PredictionSpaceGap,
            predictedCandidateSizeBytes = CandidateRow?.PredictedSizeBytes,
            predictedTwinSizeBytes = TwinRow?.PredictedSizeBytes,
            predictedSizeSavingsBytes = PredictedSizeSavingsBytes,
            rejectionReason = RejectionReason,
            matchedConfirmedAnomalyPattern = MatchedConfirmedAnomalyPattern,
            twinExistedInLookupDictionary = TwinFoundInLookup
        };
    }

    private sealed record PredictionDuckRow(
        TensorConfig Config,
        double BaseRankSafeKld,
        double FinalPredictedKld,
        ulong PredictedSizeBytes,
        double PredictionConfidence,
        ulong PredictionRank);
}
