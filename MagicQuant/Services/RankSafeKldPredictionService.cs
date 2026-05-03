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

        byte normalizedBaseId = NormalizeBaselineIdForIsolation(config.BaseQuant);

        if (row.IsPureBaseline && context.PureSnapshotsByBaselineId.TryGetValue(config.BaseQuant, out var pureDirect))
        {
            row.PredictedSizeBytes = pureDirect.SizeBytes;
            row.AdditiveKld = pureDirect.Kld;
            row.InteractionKld = pureDirect.Kld;
            row.PredictedKld = pureDirect.Kld;
            row.PredictedPpl = pureDirect.Ppl;
            return row;
        }

        if (row.IsPureBaseline && context.PureSnapshotsByBaselineId.TryGetValue(normalizedBaseId, out var pureNormalized))
        {
            row.PredictedSizeBytes = pureNormalized.SizeBytes;
            row.AdditiveKld = pureNormalized.Kld;
            row.InteractionKld = pureNormalized.Kld;
            row.PredictedKld = pureNormalized.Kld;
            row.PredictedPpl = pureNormalized.Ppl;
            row.Notes.Add($"Pure baseline '{quant.BaseQuant.Names[0]}' was normalized to '{pureNormalized.Quant.BaseQuant.Names[0]}' for prediction.");
            return row;
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
            notes.Add("Q8 native-exact base-only anchor was missing. Size fallback will use pure Q8; KLD exact/Q8 contributions remain zero.");
            q8BaseOnly = pureQ8;
        }

        var baseOnlyByBaselineId = new Dictionary<byte, BenchmarkSnapshotRecord>
        {
            [BaselineQuants.Q8_0.UniqueId] = q8BaseOnly
        };

        foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines()
                     .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
                     .OrderBy(x => x.UniqueId))
        {
            byte normalizedBaselineId = NormalizeBaselineIdForIsolation(baseline.UniqueId);

            if (!baseOnlyByBaselineId.ContainsKey(baseline.UniqueId))
            {
                var directBaseOnlyQuant = HybridQuant.CreateExactBlanket(
                    baseQuant: baseline,
                    groups: activeGroups,
                    exactScheme: nativeExactScheme);

                var directBaseOnlySnapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)directBaseOnlyQuant, ct);
                if (directBaseOnlySnapshot != null)
                {
                    baseOnlyByBaselineId[baseline.UniqueId] = directBaseOnlySnapshot;
                    if (!baseOnlyByBaselineId.ContainsKey(normalizedBaselineId))
                        baseOnlyByBaselineId[normalizedBaselineId] = directBaseOnlySnapshot;
                    continue;
                }
            }

            if (baseOnlyByBaselineId.ContainsKey(normalizedBaselineId))
                continue;

            var normalizedBaseline = BaselineQuants.FromId(normalizedBaselineId);
            var baseOnlyQuant = HybridQuant.CreateExactBlanket(
                baseQuant: normalizedBaseline,
                groups: activeGroups,
                exactScheme: nativeExactScheme);

            var baseOnlySnapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)baseOnlyQuant, ct);
            if (baseOnlySnapshot != null)
                baseOnlyByBaselineId[normalizedBaselineId] = baseOnlySnapshot;
        }

        var isolationByGroupAndBaseline = new Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord>();

        foreach (var group in activeGroups)
        {
            foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines()
                         .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
                         .OrderBy(x => x.UniqueId))
            {
                var normalizedBaselineId = NormalizeBaselineIdForIsolation(baseline.UniqueId);

                if (isolationByGroupAndBaseline.ContainsKey((group.UniqueId, normalizedBaselineId)))
                    continue;

                var isolationQuant = HybridQuant.CreateExactBlanket(
                    baseQuant: BaselineQuants.Q8_0,
                    groups: activeGroups,
                    exactScheme: nativeExactScheme);

                isolationQuant.SetLearnedCandidateOverride(group, BaselineQuants.FromId(normalizedBaselineId));
                var snapshot = await _repository.LoadBenchmarkSnapshotAsync((TensorConfig)isolationQuant, ct);

                if (snapshot != null)
                    isolationByGroupAndBaseline[(group.UniqueId, normalizedBaselineId)] = snapshot;
            }
        }

        return new RankSafePredictionModel(
            activeGroups: activeGroups,
            pureQ8: pureQ8,
            q8BaseOnly: q8BaseOnly,
            pureSnapshotsByBaselineId: pureByBaselineId,
            baseOnlySnapshotsByBaselineId: baseOnlyByBaselineId,
            isolationByGroupAndBaseline: isolationByGroupAndBaseline,
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

        foreach (var snapshot in allBenchmarkRows)
        {
            ct.ThrowIfCancellationRequested();

            RankSafePredictionRow predicted;
            if (!alreadyByKey.TryGetValue(TensorConfigIdentity.ToKey(snapshot.Config), out predicted!))
            {
                predicted = await PredictSingleAsync(snapshot.Config, context, ct);
            }

            if (!predicted.IsPredictable || double.IsInfinity(predicted.AdditiveKld) || double.IsNaN(predicted.AdditiveKld))
                continue;

            fitRows.Add(new FitObservation
            {
                Config = snapshot.Config,
                ActualKld = Math.Max(0d, snapshot.Kld),
                AdditiveKld = predicted.AdditiveKld
            });
        }

        return fitRows;
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

            byte normalized = NormalizeBaselineIdForIsolation(effectiveBaselineId);
            if (IsZeroDamageAlias(normalized))
                continue;

            if (!context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, normalized), out var isolation))
            {
                notes.Add($"Missing KLD isolation snapshot for group '{group.Name}' and baseline id '{normalized}'.");
                canPredict = false;
                continue;
            }

            total += Math.Max(0d, isolation.Kld);
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

            byte normalized = NormalizeBaselineIdForIsolation(effectiveBaselineId);
            if (context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, normalized), out var isolation))
                total += isolation.Ppl;
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
        byte normalizedBaseId = NormalizeBaselineIdForIsolation(config.BaseQuant);

        if (!context.BaseOnlySnapshotsByBaselineId.TryGetValue(normalizedBaseId, out var baseOnlyAnchor))
        {
            notes.Add($"Missing base-only size anchor for base baseline id '{normalizedBaseId}'. Size prediction is not safe for selection.");
            canPredictSize = false;
            return 0;
        }

        long total = (long)baseOnlyAnchor.SizeBytes;
        long q8ExactBlanketSize = (long)context.Q8BaseOnly.SizeBytes;

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            byte normalizedTargetId = NormalizeBaselineIdForIsolation(effectiveBaselineId);

            // Base-only anchors already hold every active group at native exact precision.
            // Exact aliases therefore contribute no size delta.
            if (BaselineQuants.IsNativeExactAlias(normalizedTargetId))
                continue;

            if (!context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, normalizedTargetId), out var targetIsolation))
            {
                notes.Add($"Missing group size-isolation snapshot for group '{group.Name}' and effective baseline id '{normalizedTargetId}'. Size prediction is not safe for selection.");
                canPredictSize = false;
                continue;
            }

            total += (long)targetIsolation.SizeBytes - q8ExactBlanketSize;
        }

        if (total <= 0)
        {
            notes.Add($"Predicted size collapsed to {total:N0} bytes. Size prediction is not safe for selection.");
            canPredictSize = false;
            return 0;
        }

        return (ulong)total;
    }

    private double ComputeCrossTerm(TensorConfig config, RankSafePredictionModel context, double threshold)
    {
        var contributions = new List<(double Kld, double Bits)>();

        foreach (var (group, effectiveBaselineId) in EnumerateEffectiveBaselines(config, context.ActiveGroups))
        {
            if (IsZeroDamageAlias(effectiveBaselineId))
                continue;

            byte normalized = NormalizeBaselineIdForIsolation(effectiveBaselineId);
            if (IsZeroDamageAlias(normalized))
                continue;

            if (!context.IsolationByGroupAndBaseline.TryGetValue((group.UniqueId, normalized), out var isolation))
                continue;

            var baseline = BaselineQuants.FromId(normalized);
            contributions.Add((Math.Max(0d, isolation.Kld), baseline.BitRange));
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

    private static bool IsZeroDamageAlias(byte baselineId)
    {
        return baselineId == BaselineQuants.Q8_0.UniqueId ||
               BaselineQuants.IsNativeExactAlias(baselineId);
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
            IReadOnlyList<string> notes)
        {
            ActiveGroups = activeGroups;
            PureQ8 = pureQ8;
            Q8BaseOnly = q8BaseOnly;
            PureSnapshotsByBaselineId = pureSnapshotsByBaselineId;
            BaseOnlySnapshotsByBaselineId = baseOnlySnapshotsByBaselineId;
            IsolationByGroupAndBaseline = isolationByGroupAndBaseline;
            Notes = notes;
        }

        public IReadOnlyList<TensorGroup> ActiveGroups { get; }
        public BenchmarkSnapshotRecord PureQ8 { get; }
        public BenchmarkSnapshotRecord Q8BaseOnly { get; }
        public Dictionary<byte, BenchmarkSnapshotRecord> PureSnapshotsByBaselineId { get; }
        public Dictionary<byte, BenchmarkSnapshotRecord> BaseOnlySnapshotsByBaselineId { get; }
        public Dictionary<(byte GroupId, byte BaselineId), BenchmarkSnapshotRecord> IsolationByGroupAndBaseline { get; }
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