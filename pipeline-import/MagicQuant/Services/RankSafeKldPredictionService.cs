using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Central prediction authority for MagicQuant's isolation-truth KLD estimator.
///
/// This intentionally replaces the old MDA / bucket-survival prediction path.
/// It combines:
/// - Q8-carrier, native-exact blanket, single-group isolation measurements
/// - additive isolation KLD
/// - bit-stress interaction correction
/// - rank-safe isotonic projection over the additive backbone
/// </summary>
public sealed class RankSafeKldPredictionService
{
    private readonly HybridBenchmarkRepository _repository;
    private readonly EffectiveCandidateStateResolverService _effectiveResolver;

    public RankSafeKldPredictionService(
        HybridBenchmarkRepository repository,
        EffectiveCandidateStateResolverService effectiveResolver)
    {
        _repository = repository;
        _effectiveResolver = effectiveResolver;
    }

    internal async Task<RankSafePredictionModel> BuildModelAsync(CancellationToken ct = default)
    {
        var context = await BuildContextAsync(ct);
        var fitRows = await LoadFitRowsAsync(context, Array.Empty<RankSafePredictionRow>(), ct);
        var fit = FitInteractionModel(fitRows, context);

        var notes = new List<string>(context.Notes)
        {
            $"Prediction fit rows: {fit.FitRowCount:N0}; alpha={fit.Alpha:G6}; beta={fit.Beta:G6}; bit-stress-threshold={fit.BitStressThreshold:G4}; fallback={fit.UsedFallback}."
        };

        context.Fit = fit;
        context.Notes = notes;
        return context;
    }

    public async Task<RankSafePredictionSet> PredictAsync(
        IReadOnlyCollection<TensorConfig> configs,
        CancellationToken ct = default)
    {
        if (configs == null)
            throw new ArgumentNullException(nameof(configs));

        var context = await BuildModelAsync(ct);
        var uniqueConfigs = configs
            .DistinctBy(TensorConfigIdentity.ToKey)
            .ToList();

        var rows = new List<RankSafePredictionRow>(uniqueConfigs.Count);

        foreach (var config in uniqueConfigs)
        {
            ct.ThrowIfCancellationRequested();
            var row = await PredictSingleAsync(config, context, ct);
            rows.Add(row);
        }

        foreach (var row in rows.Where(x => x.IsPredictable))
        {
            row.CrossTerm = ComputeCrossTerm(row.Config, context, context.Fit.BitStressThreshold);
            row.InteractionKld = Math.Max(0d, (context.Fit.Alpha * row.AdditiveKld) + (context.Fit.Beta * row.CrossTerm));
        }

        ApplyRankSafeProjection(rows);

        PrintPredictionDiagnostics(rows, context.Fit);
        return new RankSafePredictionSet
        {
            Rows = rows
                .OrderBy(x => x.PredictedKld)
                .ThenBy(x => x.IsSizePredictable ? 0 : 1)
                .ThenBy(x => x.PredictedSizeBytes)
                .ToList(),
            Fit = context.Fit,
            Notes = context.Notes
        };
    }

    private async Task<RankSafePredictionRow> PredictSingleAsync(
        TensorConfig config,
        RankSafePredictionModel context,
        CancellationToken ct)
    {
        var quant = (HybridQuant)config;
        var effective = await _effectiveResolver.ResolveAsync(config, ct);

        var row = new RankSafePredictionRow
        {
            Config = config,
            Quant = quant,
            IsPureBaseline = TensorConfigIdentity.IsPureBaseline(config),
            EffectiveStateKey = effective.EffectiveStateKey,
            HasUnknownMappings = effective.HasUnknownMappings,
            Notes = effective.Warnings.ToList()
        };

        if (row.IsPureBaseline)
        {
            if (context.PureSnapshotsByBaselineId.TryGetValue(config.BaseQuant, out var pureDirect))
            {
                row.PredictedSizeBytes = pureDirect.SizeBytes;
                row.AdditiveKld = pureDirect.Kld;
                row.InteractionKld = pureDirect.Kld;
                row.PredictedKld = pureDirect.Kld;
                row.PredictedPpl = pureDirect.Ppl;
                return row;
            }

            GuardAgainstDisabledPureBaselineSurrogateFallback(config.BaseQuant, context, row.Notes);
        }

        row.PredictedSizeBytes = PredictSize(config, context, row.Notes, out bool canPredictSize);
        row.IsSizePredictable = canPredictSize;
        row.AdditiveKld = PredictAdditiveKld(config, context, row.Notes, out bool canPredict);
        row.PredictedPpl = PredictPpl(config, context, row.Notes);

        if (!canPredict)
        {
            row.IsPredictable = false;
            row.InteractionKld = double.PositiveInfinity;
            row.PredictedKld = double.PositiveInfinity;
            return row;
        }

        row.InteractionKld = row.AdditiveKld;
        row.PredictedKld = row.AdditiveKld;
        return row;
    }

    private async Task<RankSafePredictionModel> BuildContextAsync(CancellationToken ct)
    {
        var activeGroups = TReg.All
            .Where(x => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        if (activeGroups.Count == 0)
            throw new InvalidOperationException("No active tensor groups were available for prediction.");

        var notes = new List<string>();
        var pureSnapshots = await _repository.LoadPureBaselineSnapshotsAsync(ct);
        var pureByBaselineId = pureSnapshots
            .GroupBy(x => x.Quant.BaseQuant.UniqueId)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(x => x.Kld).ThenBy(x => x.SizeBytes).First());

        if (!pureByBaselineId.TryGetValue(BaselineQuants.Q8_0.UniqueId, out var pureQ8))
            throw new InvalidOperationException("Rank-safe prediction requires a pure Q8_0 benchmark snapshot.");

        var nativeExactScheme = TensorWeightScheme.GetCurrentNativePrecisionScheme();
        var q8BaseOnlyQuant = HybridQuant.CreateExactBlanket(
            baseQuant: BaselineQuants.Q8_0,
            groups: activeGroups,
            exactScheme: nativeExactScheme);

        var q8BaseOnly = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)q8BaseOnlyQuant, ct);
        if (q8BaseOnly == null)
        {
            /*
             * Deprecated fallback, intentionally disabled:
             *
             * notes.Add("Q8 native-exact base-only anchor was missing. Size fallback will use pure Q8...");
             * q8BaseOnly = pureQ8;
             *
             * Base-only anchors define the additive size coordinate system. Falling back to a pure
             * Q8 model hides missing isolation truth and can flatten external/custom size geometry.
             */
            throw new InvalidOperationException(
                "Rank-safe prediction requires the Q8_0 native-exact base-only anchor. " +
                "The old pure-Q8 fallback is intentionally disabled; generate the missing base-only isolation sample instead.");
        }

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
            {
                baseOnlyByBaselineId[baseline.UniqueId] = directBaseOnlySnapshot;
                continue;
            }

            if (TryGetDisabledSurrogateBaselineId(baseline.UniqueId, out var disabledSurrogateId) &&
                baseOnlyByBaselineId.ContainsKey(disabledSurrogateId))
            {
                notes.Add(
                    $"Missing exact base-only anchor for external baseline {FormatBaselineForNote(baseline.UniqueId)} (id '{baseline.UniqueId}'). " +
                    $"A normalized surrogate {FormatBaselineForNote(disabledSurrogateId)} (id '{disabledSurrogateId}') exists, but surrogate base-size fallback is intentionally disabled.");
            }
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
                {
                    isolationByGroupAndBaseline[(group.UniqueId, baseline.UniqueId)] = snapshot;
                    continue;
                }

                if (TryGetDisabledSurrogateBaselineId(baseline.UniqueId, out var disabledSurrogateId) &&
                    isolationByGroupAndBaseline.ContainsKey((group.UniqueId, disabledSurrogateId)))
                {
                    notes.Add(
                        $"Missing exact isolation snapshot for group '{group.Name}' and external baseline {FormatBaselineForNote(baseline.UniqueId)} (id '{baseline.UniqueId}'). " +
                        $"A normalized surrogate {FormatBaselineForNote(disabledSurrogateId)} (id '{disabledSurrogateId}') exists, but surrogate isolation fallback is intentionally disabled.");
                }
            }
        }

        var missingQ8IsolationGroups = activeGroups
            .Where(group => !isolationByGroupAndBaseline.ContainsKey((group.UniqueId, BaselineQuants.Q8_0.UniqueId)))
            .Select(group => group.Name)
            .ToList();

        if (missingQ8IsolationGroups.Count > 0)
        {
            foreach (var groupName in missingQ8IsolationGroups)
            {
                notes.Add($"Missing KLD isolation snapshot for group '{groupName}' and baseline Q8_0. Q8_0 is a quantized state, not native truth; prediction will not silently fall back to zero for this group.");
            }
        }
        else
        {
            notes.Add($"Q8_0 isolation snapshots loaded for {activeGroups.Count:N0} active tensor groups. Q8_0 will contribute measured prediction-space KLD, not zero/native damage.");
        }

        var isolationDominanceBitTruthByGroupAndBaseline = BuildIsolationDominanceBitTruthOverrides(
            activeGroups,
            isolationByGroupAndBaseline,
            notes);

        AppendExternalCoverageDiagnostics(notes, activeGroups, pureByBaselineId, baseOnlyByBaselineId, isolationByGroupAndBaseline);

        return new RankSafePredictionModel(
            activeGroups: activeGroups,
            pureQ8: pureQ8,
            q8BaseOnly: q8BaseOnly,
            pureSnapshotsByBaselineId: pureByBaselineId,
            baseOnlySnapshotsByBaselineId: baseOnlyByBaselineId,
            isolationByGroupAndBaseline: isolationByGroupAndBaseline,
            isolationDominanceBitTruthByGroupAndBaseline: isolationDominanceBitTruthByGroupAndBaseline,
            notes: notes);
    }

    private async Task<List<FitObservation>> LoadFitRowsAsync(
        RankSafePredictionModel context,
        IReadOnlyList<RankSafePredictionRow> alreadyPredicted,
        CancellationToken ct)
    {
        var allBenchmarkRows = await _repository.LoadAllBenchmarkSnapshotsForCurrentContextAsync(
            category: (byte)BenchmarkCategory.General,
            strictImatrixContext: true,
            ct: ct);

        var alreadyByKey = alreadyPredicted.ToDictionary(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal);
        var fitRows = new List<FitObservation>();
        var skippedFitReasons = new HashSet<string>(StringComparer.Ordinal);

        foreach (var snapshot in allBenchmarkRows)
        {
            ct.ThrowIfCancellationRequested();

            /*
             * Keep the rank-safe predictor entirely in prediction space.
             *
             * Real pure baselines such as UD-Q6_K_XL can appear in the benchmark table
             * as BaseQuant=UD-Q6_K_XL with NULL group slots. That shape is a real artifact
             * identity, not a prediction-space base-only anchor. The prediction coordinate
             * system is still the Q8_0 carrier plus exact isolated group overrides, so pure
             * baselines are canonicalized to the virtual all-groups row before prediction:
             *
             *   real pure UD-Q6_K_XL       -> Q8_0 carrier with every active group = UD-Q6_K_XL
             *   real pure Q6_K             -> Q8_0 carrier with every active group = Q6_K
             *
             * The real snapshot.Kld remains the fit target. Only the config used to produce
             * the additive/cross-term prediction is canonicalized. This preserves the hard
             * separation between real benchmark truth and synthetic prediction geometry.
             */
            if (!TryCanonicalizeBenchmarkSnapshotConfigForPrediction(
                    snapshot.Config,
                    context,
                    out var predictionConfig,
                    out var skipReason))
            {
                /*
                 * This row is real benchmark truth, but it is not representable in the
                 * rank-safe prediction coordinate system. Do not throw here: old runs and
                 * helper paths can leave real/external-base synthetic artifacts in SQLite
                 * even though DuckDB prediction-space candidates always use the Q8_0
                 * carrier. Those rows are simply not fit observations for the synthetic
                 * model.
                 */
                if (!string.IsNullOrWhiteSpace(skipReason))
                    skippedFitReasons.Add(skipReason);

                continue;
            }

            var predictionKey = TensorConfigIdentity.ToKey(predictionConfig);

            RankSafePredictionRow predicted;
            if (!alreadyByKey.TryGetValue(predictionKey, out predicted!))
            {
                predicted = await PredictSingleAsync(predictionConfig, context, ct);
            }

            if (!predicted.IsPredictable || double.IsInfinity(predicted.AdditiveKld) || double.IsNaN(predicted.AdditiveKld))
                continue;

            fitRows.Add(new FitObservation
            {
                Config = predictionConfig,
                ActualKld = Math.Max(0d, snapshot.Kld),
                AdditiveKld = predicted.AdditiveKld
            });
        }

        if (skippedFitReasons.Count > 0)
            context.Notes = context.Notes.Concat(skippedFitReasons.OrderBy(x => x, StringComparer.Ordinal)).ToList();

        return fitRows;
    }

    private static bool TryCanonicalizeBenchmarkSnapshotConfigForPrediction(
        TensorConfig config,
        RankSafePredictionModel context,
        out TensorConfig predictionConfig,
        out string? skipReason)
    {
        predictionConfig = config;
        skipReason = null;

        if (config.BaseQuant == BaselineQuants.Q8_0.UniqueId)
            return true;

        var baseline = BaselineQuants.FromId(config.BaseQuant);
        if (BaselineQuants.IsNativeExactAlias(baseline.UniqueId))
            return true;

        if (TensorConfigIdentity.IsPureBaseline(config) &&
            !context.PureSnapshotsByBaselineId.ContainsKey(baseline.UniqueId))
        {
            throw new InvalidOperationException(
                $"Rank-safe prediction fit encountered pure baseline {baseline.Names[0]} (id '{baseline.UniqueId}'), but the pure benchmark snapshot was not loaded into context. " +
                "This is a critical truth-loading error, not a soft warning.");
        }

        byte inheritedBaseSlot = BaselineQuants.EncodeTensorConfigGroupSlot(baseline);

        predictionConfig = new TensorConfig(
            baseQuant: BaselineQuants.Q8_0.UniqueId,
            embeddings: CanonicalizePredictionSlot(TReg.Embeddings, config.Embeddings, inheritedBaseSlot, context),
            lmHead: CanonicalizePredictionSlot(TReg.LmHead, config.LmHead, inheritedBaseSlot, context),
            attnQ: CanonicalizePredictionSlot(TReg.AttnQ, config.AttnQ, inheritedBaseSlot, context),
            attnKV: CanonicalizePredictionSlot(TReg.AttnKV, config.AttnKV, inheritedBaseSlot, context),
            attnOutput: CanonicalizePredictionSlot(TReg.AttnOutput, config.AttnOutput, inheritedBaseSlot, context),
            ffnUpGate: CanonicalizePredictionSlot(TReg.FfnUpGate, config.FfnUpGate, inheritedBaseSlot, context),
            ffnDown: CanonicalizePredictionSlot(TReg.FfnDown, config.FfnDown, inheritedBaseSlot, context),
            moeExperts: CanonicalizePredictionSlot(TReg.MoeExperts, config.MoeExperts, inheritedBaseSlot, context),
            moeRouter: CanonicalizePredictionSlot(TReg.MoeRouter, config.MoeRouter, inheritedBaseSlot, context));

        return true;
    }

    private static byte CanonicalizePredictionSlot(
        TensorGroup group,
        byte storedValue,
        byte inheritedBaseSlot,
        RankSafePredictionModel context)
    {
        if (!context.ActiveGroups.Any(x => x.UniqueId == group.UniqueId))
            return BaselineQuants.TensorConfigNullSlotValue;

        return BaselineQuants.IsNullTensorConfigGroupSlot(storedValue)
            ? inheritedBaseSlot
            : storedValue;
    }

    private RankSafePredictionFit FitInteractionModel(
        IReadOnlyList<FitObservation> observations,
        RankSafePredictionModel context)
    {
        var usable = observations
            .Where(x => x.ActualKld >= 0d)
            .Where(x => !double.IsNaN(x.AdditiveKld) && !double.IsInfinity(x.AdditiveKld))
            .ToList();

        if (usable.Count < Math.Max(3, Config.PredictionMinimumFitRows))
        {
            return new RankSafePredictionFit
            {
                Alpha = 1.0d,
                Beta = 0.0d,
                BitStressThreshold = Config.PredictionDefaultBitStressThreshold,
                FitRowCount = usable.Count,
                UsedFallback = true
            };
        }

        RankSafePredictionFit? best = null;

        foreach (double threshold in Config.PredictionBitStressThresholdCandidates)
        {
            double s11 = 0d;
            double s12 = 0d;
            double s22 = 0d;
            double y1 = 0d;
            double y2 = 0d;

            var crossTerms = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (var row in usable)
            {
                double x1 = row.AdditiveKld;
                double x2 = ComputeCrossTerm(row.Config, context, threshold);
                double y = row.ActualKld;

                s11 += x1 * x1;
                s12 += x1 * x2;
                s22 += x2 * x2;
                y1 += x1 * y;
                y2 += x2 * y;
                crossTerms[TensorConfigIdentity.ToKey(row.Config)] = x2;
            }

            double det = (s11 * s22) - (s12 * s12);
            double alpha;
            double beta;

            if (Math.Abs(det) <= 1e-18d)
            {
                alpha = s11 <= 1e-18d ? 1.0d : y1 / s11;
                beta = 0.0d;
            }
            else
            {
                alpha = ((y1 * s22) - (y2 * s12)) / det;
                beta = ((s11 * y2) - (s12 * y1)) / det;
            }

            if (double.IsNaN(alpha) || double.IsInfinity(alpha))
                alpha = 1.0d;

            if (double.IsNaN(beta) || double.IsInfinity(beta))
                beta = 0.0d;

            // Keep the correction sane. The fit can be noisy when the only benchmarked
            // rows are pure baselines and isolation probes.
            alpha = Math.Clamp(alpha, 0.05d, 10.0d);
            beta = Math.Clamp(beta, -1_000_000d, 1_000_000d);

            double mae = usable
                .Select(x =>
                {
                    double cross = crossTerms[TensorConfigIdentity.ToKey(x.Config)];
                    double pred = Math.Max(0d, (alpha * x.AdditiveKld) + (beta * cross));
                    return Math.Abs(pred - x.ActualKld);
                })
                .Average();

            var candidate = new RankSafePredictionFit
            {
                Alpha = alpha,
                Beta = beta,
                BitStressThreshold = threshold,
                FitRowCount = usable.Count,
                FitMae = mae,
                UsedFallback = false
            };

            if (best == null || candidate.FitMae < best.FitMae)
                best = candidate;
        }

        return best ?? new RankSafePredictionFit
        {
            Alpha = 1.0d,
            Beta = 0.0d,
            BitStressThreshold = Config.PredictionDefaultBitStressThreshold,
            FitRowCount = usable.Count,
            UsedFallback = true
        };
    }

    private static void ApplyRankSafeProjection(IReadOnlyList<RankSafePredictionRow> rows)
    {
        var predictable = rows
            .Where(x => x.IsPredictable)
            .OrderBy(x => x.AdditiveKld)
            .ThenBy(x => x.InteractionKld)
            .ThenBy(x => x.PredictedSizeBytes)
            .ToList();

        if (predictable.Count == 0)
            return;

        double[] projected = Pava(predictable.Select(x => x.InteractionKld).ToArray());

        for (int i = 0; i < predictable.Count; i++)
            predictable[i].PredictedKld = Math.Max(0d, projected[i]);

        ulong rank = 1;
        foreach (var row in rows
                     .Where(x => x.IsPredictable)
                     .OrderBy(x => x.PredictedKld)
                     .ThenBy(x => x.PredictedSizeBytes))
        {
            row.PredictedRank = rank++;
        }
    }

    private static double[] Pava(double[] values)
    {
        var blocks = new List<PavaBlock>();

        foreach (double value in values)
        {
            blocks.Add(new PavaBlock { Sum = value, Weight = 1d, Count = 1 });

            while (blocks.Count >= 2)
            {
                var right = blocks[^1];
                var left = blocks[^2];

                if (left.Mean <= right.Mean)
                    break;

                left.Sum += right.Sum;
                left.Weight += right.Weight;
                left.Count += right.Count;
                blocks[^2] = left;
                blocks.RemoveAt(blocks.Count - 1);
            }
        }

        var result = new double[values.Length];
        int index = 0;
        foreach (var block in blocks)
        {
            double mean = block.Mean;
            for (int i = 0; i < block.Count; i++)
                result[index++] = mean;
        }

        return result;
    }

    private double PredictAdditiveKld(
        TensorConfig config,
        RankSafePredictionModel context,
        List<string> notes,
        out bool canPredict)
    {
        canPredict = true;
        double total = 0d;

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (IsZeroDamageAlias(effectiveBaselineId))
                continue;

            if (!TryResolveIsolationBaselineForPrediction(group, effectiveBaselineId, context, notes, out var resolved))
            {
                notes.Add(BuildMissingIsolationNote(group, effectiveBaselineId));
                canPredict = false;
                continue;
            }

            total += Math.Max(0d, resolved.Snapshot.Kld);
        }

        return Math.Max(0d, total);
    }

    private double PredictPpl(
        TensorConfig config,
        RankSafePredictionModel context,
        List<string> notes)
    {
        double total = 0d;

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (IsZeroDamageAlias(effectiveBaselineId))
                continue;

            if (TryResolveIsolationBaselineForPrediction(group, effectiveBaselineId, context, notes, out var resolved))
                total += resolved.Snapshot.Ppl;
            else
                notes.Add(BuildMissingIsolationNote(group, effectiveBaselineId));
        }

        return total;
    }

    private ulong PredictSize(
        TensorConfig config,
        RankSafePredictionModel context,
        List<string> notes,
        out bool canPredictSize)
    {
        canPredictSize = true;

        if (!TryResolveBaseOnlySnapshotForPrediction(config.BaseQuant, context, notes, out var baseOnlyAnchor))
        {
            notes.Add($"Missing base-only size anchor for base baseline {FormatBaselineForNote(config.BaseQuant)} (id '{config.BaseQuant}'). Size prediction is not safe for selection.");
            canPredictSize = false;
            return 0;
        }

        long total = (long)baseOnlyAnchor.SizeBytes;
        long q8ExactBlanketSize = (long)context.Q8BaseOnly.SizeBytes;

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            // Base-only anchors already hold every active group at native exact precision.
            // Exact aliases therefore contribute no size delta.
            if (BaselineQuants.IsNativeExactAlias(effectiveBaselineId))
                continue;

            if (!TryResolveIsolationBaselineForPrediction(group, effectiveBaselineId, context, notes, out var resolved))
            {
                notes.Add($"Missing group size-isolation snapshot for group '{group.Name}' and effective baseline {FormatBaselineForNote(effectiveBaselineId)} (id '{effectiveBaselineId}'). Size prediction is not safe for selection.");
                canPredictSize = false;
                continue;
            }

            total += (long)resolved.Snapshot.SizeBytes - q8ExactBlanketSize;
        }

        if (total <= 0)
        {
            notes.Add($"Predicted size collapsed to {total:N0} bytes. Size prediction is not safe for selection.");
            canPredictSize = false;
            return 0;
        }

        return (ulong)total;
    }

    private static Dictionary<(byte GroupId, byte BaselineId), double> BuildIsolationDominanceBitTruthOverrides(
        IReadOnlyList<TensorGroup> activeGroups,
        Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord> isolationByGroupAndBaseline,
        List<string> notes)
    {
        const double kldEpsilon = 1e-12;

        var result = new Dictionary<(byte GroupId, byte BaselineId), double>();
        var detailNotes = new List<string>();
        var baselinesById = BaselineQuants.GetAllRecognizedBaselines()
            .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
            .GroupBy(x => x.UniqueId)
            .ToDictionary(g => g.Key, g => g.First());

        foreach (var group in activeGroups.OrderBy(x => x.UniqueId))
        {
            var entries = isolationByGroupAndBaseline
                .Where(x => x.Key.GroupId == group.UniqueId && baselinesById.ContainsKey(x.Key.BaselineId))
                .Select(x => new IsolationBitTruthEntry(
                    Baseline: baselinesById[x.Key.BaselineId],
                    Snapshot: x.Value,
                    DeclaredBitRange: (double)baselinesById[x.Key.BaselineId].BitRange))
                .OrderByDescending(x => x.DeclaredBitRange)
                .ThenBy(x => x.Snapshot.Kld)
                .ThenBy(x => x.Snapshot.SizeBytes)
                .ToList();

            foreach (var candidate in entries)
            {
                double inheritedBitTruth = candidate.DeclaredBitRange;
                IsolationBitTruthEntry? strongestVictim = null;

                foreach (var victim in entries)
                {
                    if (victim.DeclaredBitRange <= candidate.DeclaredBitRange)
                        continue;

                    bool sameSizeOrSmaller = candidate.Snapshot.SizeBytes <= victim.Snapshot.SizeBytes;
                    bool lowerKld = candidate.Snapshot.Kld < victim.Snapshot.Kld - kldEpsilon;
                    if (!sameSizeOrSmaller || !lowerKld)
                        continue;

                    if (victim.DeclaredBitRange > inheritedBitTruth)
                    {
                        inheritedBitTruth = victim.DeclaredBitRange;
                        strongestVictim = victim;
                    }
                }

                if (inheritedBitTruth <= candidate.DeclaredBitRange)
                    continue;

                result[(group.UniqueId, candidate.Baseline.UniqueId)] = inheritedBitTruth;

                if (strongestVictim != null && detailNotes.Count < 32)
                {
                    detailNotes.Add(
                        $"Isolation bit-truth override: group '{group.Name}' treats {candidate.Baseline.Names[0]} as {inheritedBitTruth:G4}b stress truth instead of {candidate.DeclaredBitRange:G4}b because it isolated-dominated higher-fidelity {strongestVictim.Baseline.Names[0]} (candidate size={candidate.Snapshot.SizeBytes:N0}, kld={candidate.Snapshot.Kld:0.######}; victim size={strongestVictim.Snapshot.SizeBytes:N0}, kld={strongestVictim.Snapshot.Kld:0.######}).");
                }
            }
        }

        if (result.Count == 0)
        {
            notes.Add("Isolation bit-truth overrides: none. Declared quant bit ranges will drive bit-stress interaction correction.");
            return result;
        }

        notes.Add(
            $"Isolation bit-truth overrides active: {result.Count:N0} group/baseline state(s) inherit higher-fidelity stress truth because isolated sampling showed same-size-or-smaller lower-KLD dominance.");

        foreach (var detail in detailNotes)
            notes.Add(detail);

        if (result.Count > detailNotes.Count)
            notes.Add($"Isolation bit-truth overrides: {result.Count - detailNotes.Count:N0} additional override(s) omitted from diagnostics.");

        return result;
    }

    internal static double GetStressBitRangeForPrediction(
        TensorGroup group,
        byte baselineId,
        RankSafePredictionModel context)
    {
        if (IsZeroDamageAlias(baselineId))
            return 99d;

        double declared = BaselineQuants.FromId(baselineId).BitRange;
        return context.IsolationDominanceBitTruthByGroupAndBaseline.TryGetValue((group.UniqueId, baselineId), out var inherited)
            ? Math.Max(declared, inherited)
            : declared;
    }

    private double ComputeCrossTerm(TensorConfig config, RankSafePredictionModel context, double threshold)
    {
        var contributions = new List<(double Kld, double Bits)>();

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (IsZeroDamageAlias(effectiveBaselineId))
                continue;

            if (!TryResolveIsolationBaselineForPrediction(group, effectiveBaselineId, context, notes: null, out var resolved))
                continue;

            double stressBitRange = GetStressBitRangeForPrediction(group, resolved.BaselineId, context);
            contributions.Add((Math.Max(0d, resolved.Snapshot.Kld), stressBitRange));
        }

        double cross = 0d;
        for (int i = 0; i < contributions.Count; i++)
        {
            for (int j = i + 1; j < contributions.Count; j++)
            {
                double stressI = Math.Max(0d, threshold - contributions[i].Bits);
                double stressJ = Math.Max(0d, threshold - contributions[j].Bits);
                if (stressI <= 0d || stressJ <= 0d)
                    continue;

                cross += contributions[i].Kld * contributions[j].Kld * stressI * stressJ;
            }
        }

        return cross;
    }

    public static IReadOnlyList<(TensorGroup Group, byte EffectiveBaselineId)> EnumerateEffectiveBaselines(
        TensorConfig config,
        IReadOnlyList<TensorGroup>? activeGroups = null)
    {
        activeGroups ??= TReg.All
            .Where(x => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();

        var result = new List<(TensorGroup Group, byte EffectiveBaselineId)>(activeGroups.Count);

        foreach (var (group, storedValue) in TensorConfigIdentity.EnumerateGroupSlots(config))
        {
            if (!activeGroups.Any(x => x.UniqueId == group.UniqueId))
                continue;

            byte effective = BaselineQuants.IsNullTensorConfigGroupSlot(storedValue)
                ? config.BaseQuant
                : BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedValue);

            result.Add((group, effective));
        }

        return result;
    }

    public static byte NormalizeBaselineIdForIsolation(byte baselineId)
    {
        if (BaselineQuants.IsNativeExactAlias(baselineId))
            return baselineId;

        var baseline = BaselineQuants.FromId(baselineId);
        if (!baseline.IsExternalRepositoryBaseline)
            return baselineId;

        var builtIn = BaselineQuants.ResolveBuiltInStandardBaseline(baseline.QuantizeBaseArgumentName)
                      ?? BaselineQuants.ResolveBuiltInStandardBaseline(baseline.Names[0]);

        return builtIn?.UniqueId ?? baselineId;
    }

    internal static bool TryResolveIsolationBaselineForPrediction(
        TensorGroup group,
        byte effectiveBaselineId,
        RankSafePredictionModel context,
        List<string>? notes,
        out IsolationBaselineResolution resolution)
    {
        if (context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, effectiveBaselineId), out var exact))
        {
            resolution = new IsolationBaselineResolution(effectiveBaselineId, exact, false, null);
            return true;
        }

        if (TryGetDisabledSurrogateBaselineId(effectiveBaselineId, out var disabledSurrogateId) &&
            context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, disabledSurrogateId), out var disabledSurrogate))
        {
            /*
             * Deprecated surrogate fallback, intentionally disabled:
             *
             * resolution = new IsolationBaselineResolution(disabledSurrogateId, disabledSurrogate, true, disabledSurrogateId);
             * notes?.Add($"External baseline {FormatBaselineForNote(effectiveBaselineId)} used surrogate isolation {FormatBaselineForNote(disabledSurrogateId)} for group '{group.Name}'.");
             * return true;
             *
             * This used to collapse external/custom repositories such as Unsloth Dynamic into
             * their built-in llama.cpp family before prediction. MagicQuant should already have
             * exact isolated samples for every registered external candidate, so using this path
             * would hide a truth-coverage bug. Keep the old shape here only as a breadcrumb if a
             * future emergency compatibility mode is deliberately reintroduced.
             */
            _ = disabledSurrogate;
            ThrowExternalIsolationSurrogateFallbackDisabled(group, effectiveBaselineId, disabledSurrogateId);
        }

        if (IsExternalRepositoryBaseline(effectiveBaselineId))
            ThrowMissingExactExternalIsolation(group, effectiveBaselineId);

        resolution = default;
        return false;
    }

    internal static bool TryResolveBaseOnlySnapshotForPrediction(
        byte baselineId,
        RankSafePredictionModel context,
        List<string>? notes,
        out BenchmarkSnapshotRecord snapshot)
    {
        if (context.BaseOnlySnapshotsByBaselineId.TryGetValue(baselineId, out var exactSnapshot))
        {
            snapshot = exactSnapshot;
            return true;
        }

        if (TryGetDisabledSurrogateBaselineId(baselineId, out var disabledSurrogateId) &&
            context.BaseOnlySnapshotsByBaselineId.ContainsKey(disabledSurrogateId))
        {
            /*
             * Deprecated surrogate fallback, intentionally disabled:
             *
             * snapshot = context.BaseOnlySnapshotsByBaselineId[disabledSurrogateId];
             * notes?.Add($"External baseline {FormatBaselineForNote(baselineId)} used surrogate base-only size {FormatBaselineForNote(disabledSurrogateId)}.");
             * return true;
             *
             * Base-only anchors must preserve the exact runtime baseline id. Falling back
             * here makes UD-Q4_K_XL and Q4_K_M look byte-identical before selection even starts.
             */
            ThrowExternalBaseOnlySurrogateFallbackDisabled(baselineId, disabledSurrogateId);
        }

        if (IsExternalRepositoryBaseline(baselineId))
            throw new InvalidOperationException(
                $"Missing exact synthetic base-only anchor for external baseline {FormatBaselineForNote(baselineId)} (id '{baselineId}'). " +
                "The rank-safe predictor must not use real pure baseline snapshots as base-only prediction anchors. " +
                "Pure baselines are canonicalized to Q8_0-carrier virtual blankets before prediction; reaching size prediction with an external/custom BaseQuant means a non-canonical config escaped normalization.");

        snapshot = default!;
        return false;
    }

    internal static bool TryGetDisabledSurrogateBaselineId(byte baselineId, out byte surrogateBaselineId)
    {
        surrogateBaselineId = baselineId;

        if (BaselineQuants.IsNativeExactAlias(baselineId))
            return false;

        var baseline = BaselineQuants.FromId(baselineId);
        if (!baseline.IsExternalRepositoryBaseline)
            return false;

        var normalized = NormalizeBaselineIdForIsolation(baselineId);
        if (normalized == baselineId)
            return false;

        surrogateBaselineId = normalized;
        return true;
    }

    internal static bool IsExternalRepositoryBaseline(byte baselineId)
    {
        if (BaselineQuants.IsNativeExactAlias(baselineId))
            return false;

        return BaselineQuants.FromId(baselineId).IsExternalRepositoryBaseline;
    }

    private static void GuardAgainstDisabledPureBaselineSurrogateFallback(
        byte baselineId,
        RankSafePredictionModel context,
        List<string> notes)
    {
        if (!TryGetDisabledSurrogateBaselineId(baselineId, out var disabledSurrogateId) ||
            !context.PureSnapshotsByBaselineId.ContainsKey(disabledSurrogateId))
        {
            return;
        }

        /*
         * Deprecated surrogate fallback, intentionally disabled:
         *
         * var pureSurrogate = context.PureSnapshotsByBaselineId[disabledSurrogateId];
         * notes.Add($"Pure baseline {FormatBaselineForNote(baselineId)} used surrogate pure snapshot {FormatBaselineForNote(disabledSurrogateId)}.");
         *
         * Pure external baselines must not inherit standard-family prediction identity.
         */
        throw new InvalidOperationException(
            $"Missing exact pure snapshot for external baseline {FormatBaselineForNote(baselineId)} (id '{baselineId}'), " +
            $"but surrogate pure snapshot {FormatBaselineForNote(disabledSurrogateId)} (id '{disabledSurrogateId}') exists. " +
            "Surrogate pure-baseline fallback is disabled to prevent external/custom collapse.");
    }

    private static void ThrowExternalIsolationSurrogateFallbackDisabled(
        TensorGroup group,
        byte externalBaselineId,
        byte surrogateBaselineId)
    {
        throw new InvalidOperationException(
            $"Missing exact isolation snapshot for group '{group.Name}' and external baseline {FormatBaselineForNote(externalBaselineId)} (id '{externalBaselineId}'). " +
            $"Surrogate isolation {FormatBaselineForNote(surrogateBaselineId)} (id '{surrogateBaselineId}') exists, but fallback is disabled. " +
            "External/custom baselines must be scored from exact isolated prediction truth; regenerate/relearn the missing isolated sample instead of silently collapsing it.");
    }

    private static void ThrowExternalBaseOnlySurrogateFallbackDisabled(byte externalBaselineId, byte surrogateBaselineId)
    {
        throw new InvalidOperationException(
            $"Missing exact base-only anchor for external baseline {FormatBaselineForNote(externalBaselineId)} (id '{externalBaselineId}'). " +
            $"Surrogate base-only anchor {FormatBaselineForNote(surrogateBaselineId)} (id '{surrogateBaselineId}') exists, but fallback is disabled. " +
            "External/custom baselines must preserve exact runtime identity for size prediction.");
    }

    private static void ThrowMissingExactExternalIsolation(TensorGroup group, byte externalBaselineId)
    {
        throw new InvalidOperationException(
            $"Missing exact isolation snapshot for group '{group.Name}' and external baseline {FormatBaselineForNote(externalBaselineId)} (id '{externalBaselineId}'). " +
            "No surrogate fallback was used. MagicQuant expects external/custom isolated samples to exist before prediction materialization.");
    }

    private static void AppendExternalCoverageDiagnostics(
        List<string> notes,
        IReadOnlyList<TensorGroup> activeGroups,
        Dictionary<byte, BenchmarkSnapshotRecord> pureByBaselineId,
        Dictionary<byte, BenchmarkSnapshotRecord> baseOnlyByBaselineId,
        Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord> isolationByGroupAndBaseline)
    {
        var externalBaselines = BaselineQuants.GetAllRecognizedBaselines()
            .Where(x => x.IsExternalRepositoryBaseline)
            .OrderBy(x => x.UniqueId)
            .ToList();

        if (externalBaselines.Count == 0)
            return;

        notes.Add("External/custom surrogate fallback is disabled; missing exact external prediction truth will throw instead of collapsing to a standard family.");

        foreach (var baseline in externalBaselines)
        {
            int exactIsolation = activeGroups.Count(group => isolationByGroupAndBaseline.ContainsKey((group.UniqueId, baseline.UniqueId)));
            bool exactBaseOnly = baseOnlyByBaselineId.ContainsKey(baseline.UniqueId);
            bool exactPure = pureByBaselineId.ContainsKey(baseline.UniqueId);
            string baseAnchorText = exactBaseOnly
                ? "base-only=exact"
                : exactPure
                    ? "base-only=missing; pure-anchor=exact"
                    : "base-only=missing; pure-anchor=missing";
            string fallbackText = TryGetDisabledSurrogateBaselineId(baseline.UniqueId, out var fallbackId)
                ? $"; disabled fallback target would have been {FormatBaselineForNote(fallbackId)}:{fallbackId}"
                : string.Empty;

            notes.Add(
                $"External isolation exact coverage: {baseline.Names[0]}:{baseline.UniqueId} exact={exactIsolation}/{activeGroups.Count} groups; {baseAnchorText}{fallbackText}.");
        }
    }

    private sealed record IsolationBitTruthEntry(
        BaselineQuants Baseline,
        BenchmarkSnapshotRecord Snapshot,
        double DeclaredBitRange);

    internal readonly record struct IsolationBaselineResolution(
        byte BaselineId,
        BenchmarkSnapshotRecord Snapshot,
        bool IsSurrogate,
        byte? FallbackBaselineId);

    private static bool IsZeroDamageAlias(byte baselineId) => IsNativeExactZeroReferenceAlias(baselineId);

    private static bool IsNativeExactZeroReferenceAlias(byte baselineId)
    {
        // MagicQuant's zero-damage reference is native exact precision (BF16/F16/F32),
        // not Q8_0. Q8_0 is a real quantized state with measured per-group isolation
        // KLD and must flow through the same lookup path as Q6_K/Q5_K/Q4/etc.
        return BaselineQuants.IsNativeExactAlias(baselineId);
    }

    private static string BuildMissingIsolationNote(TensorGroup group, byte baselineId)
    {
        var baselineName = FormatBaselineForNote(baselineId);
        if (baselineId == BaselineQuants.Q8_0.UniqueId)
        {
            return $"Missing KLD isolation snapshot for group '{group.Name}' and baseline Q8_0. Q8_0 is quantized damage, not native truth; this row is marked incomplete instead of silently receiving zero KLD.";
        }

        return $"Missing KLD isolation snapshot for group '{group.Name}' and baseline {baselineName} (id '{baselineId}').";
    }

    private static string FormatBaselineForNote(byte baselineId)
    {
        try
        {
            return BaselineQuants.FromId(baselineId).Names[0];
        }
        catch
        {
            return $"id {baselineId}";
        }
    }

    private static void PrintPredictionDiagnostics(IReadOnlyCollection<RankSafePredictionRow> rows, RankSafePredictionFit fit)
    {
        int predictable = rows.Count(x => x.IsPredictable);
        int sizePredictable = rows.Count(x => x.IsPredictable && x.IsSizePredictable);
        int skipped = rows.Count - predictable;
        int unsafeSize = predictable - sizePredictable;

        AnsiConsole.MarkupLine($"[grey]Rank-safe prediction rows:[/] [cyan]{predictable:N0}[/] KLD-predictable / [cyan]{sizePredictable:N0}[/] size-safe / [yellow]{skipped:N0}[/] KLD-incomplete / [yellow]{unsafeSize:N0}[/] unsafe-size");
        AnsiConsole.MarkupLine($"[grey]Interaction fit:[/] alpha=[cyan]{fit.Alpha:G6}[/] beta=[cyan]{fit.Beta:G6}[/] bit-stress=[cyan]{fit.BitStressThreshold:G4}[/] fit-rows=[cyan]{fit.FitRowCount:N0}[/] fallback=[cyan]{fit.UsedFallback}[/]");

        if (predictable == 0)
            return;

        var sizeSafeRows = rows.Where(x => x.IsPredictable && x.IsSizePredictable).ToList();
        ulong minSize = sizeSafeRows.Count == 0 ? 0UL : sizeSafeRows.Min(x => x.PredictedSizeBytes);
        ulong maxSize = sizeSafeRows.Count == 0 ? 0UL : sizeSafeRows.Max(x => x.PredictedSizeBytes);
        double minKld = rows.Where(x => x.IsPredictable).Min(x => x.PredictedKld);
        double maxKld = rows.Where(x => x.IsPredictable).Max(x => x.PredictedKld);

        AnsiConsole.MarkupLine($"[grey]Predicted size spread, size-safe rows only:[/] [cyan]{ToGb(minSize):0.00}[/] GB .. [cyan]{ToGb(maxSize):0.00}[/] GB");
        AnsiConsole.MarkupLine($"[grey]Predicted KLD spread:[/] [cyan]{minKld:0.000000}[/] .. [cyan]{maxKld:0.000000}[/]");
    }

    private static double ToGb(ulong bytes) => bytes / 1024d / 1024d / 1024d;

    internal sealed class RankSafePredictionModel
    {
        public RankSafePredictionModel(
            IReadOnlyList<TensorGroup> activeGroups,
            BenchmarkSnapshotRecord pureQ8,
            BenchmarkSnapshotRecord q8BaseOnly,
            Dictionary<byte, BenchmarkSnapshotRecord> pureSnapshotsByBaselineId,
            Dictionary<byte, BenchmarkSnapshotRecord> baseOnlySnapshotsByBaselineId,
            Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord> isolationByGroupAndBaseline,
            Dictionary<(byte GroupId, byte BaselineId), double> isolationDominanceBitTruthByGroupAndBaseline,
            IReadOnlyList<string> notes)
        {
            ActiveGroups = activeGroups;
            PureQ8 = pureQ8;
            Q8BaseOnly = q8BaseOnly;
            PureSnapshotsByBaselineId = pureSnapshotsByBaselineId;
            BaseOnlySnapshotsByBaselineId = baseOnlySnapshotsByBaselineId;
            IsolationByGroupAndBaseline = isolationByGroupAndBaseline;
            IsolationDominanceBitTruthByGroupAndBaseline = isolationDominanceBitTruthByGroupAndBaseline;
            Notes = notes;
        }

        public IReadOnlyList<TensorGroup> ActiveGroups { get; }
        public BenchmarkSnapshotRecord PureQ8 { get; }
        public BenchmarkSnapshotRecord Q8BaseOnly { get; }
        public Dictionary<byte, BenchmarkSnapshotRecord> PureSnapshotsByBaselineId { get; }
        public Dictionary<byte, BenchmarkSnapshotRecord> BaseOnlySnapshotsByBaselineId { get; }
        public Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord> IsolationByGroupAndBaseline { get; }
        public Dictionary<(byte GroupId, byte BaselineId), double> IsolationDominanceBitTruthByGroupAndBaseline { get; }
        public IReadOnlyList<string> Notes { get; set; }
        public RankSafePredictionFit Fit { get; set; } = new();
    }

    private sealed class FitObservation
    {
        public TensorConfig Config { get; init; }
        public double ActualKld { get; init; }
        public double AdditiveKld { get; init; }
    }

    private struct PavaBlock
    {
        public double Sum;
        public double Weight;
        public int Count;
        public double Mean => Weight <= 0d ? 0d : Sum / Weight;
    }
}
