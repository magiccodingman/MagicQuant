using System.Globalization;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

/// <summary>
/// Materializes transient prediction/ranking metadata into DuckDB.
/// SQLite remains the long-term benchmark truth source; these columns only order
/// candidates before real quantization/benchmark validation.
/// </summary>
public sealed class DuckDbPredictionMaterializationService
{
    private readonly RemainingCombinationStore _store;
    private readonly RankSafeKldPredictionService _predictionService;

    private static readonly GroupSlot[] GroupSlots =
    [
        new(TReg.Embeddings, "Embeddings"),
        new(TReg.LmHead, "LmHead"),
        new(TReg.AttnQ, "AttnQ"),
        new(TReg.AttnKV, "AttnKV"),
        new(TReg.AttnOutput, "AttnOutput"),
        new(TReg.FfnUpGate, "FfnUpGate"),
        new(TReg.FfnDown, "FfnDown"),
        new(TReg.MoeExperts, "MoeExperts"),
        new(TReg.MoeRouter, "MoeRouter")
    ];

    public DuckDbPredictionMaterializationService(
        RemainingCombinationStore store,
        RankSafeKldPredictionService predictionService)
    {
        _store = store;
        _predictionService = predictionService;
    }

    public async Task<PredictionMaterializationStatus> MaterializeAsync(CancellationToken ct = default)
    {
        var model = await _predictionService.BuildModelAsync(ct);

        using var c = new DuckDBConnection($"Data Source={_store.GetDatabaseFilePath()}");
        await c.OpenAsync(ct);
        await ConfigureSessionAsync(c, ct);

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName}
SET PredictedKld = NULL,
    PredictedSizeBytes = NULL,
    PredictionConfidence = NULL,
    PredictionRank = NULL;", ct);

        await BuildLookupTablesAsync(c, model, ct);
        await BuildPredictionWorkTablesAsync(c, model, ct);
        await BuildPavaBlocksAsync(c, ct);
        await PersistProjectedPredictionsAsync(c, model, ct);

        var status = await _store.GetPredictionStatusAsync(ct);

        foreach (var note in model.Notes)
            AnsiConsole.MarkupLine($"[grey]Prediction materialization note:[/] {Markup.Escape(note)}");

        return status;
    }

    private static async Task BuildLookupTablesAsync(
        DuckDBConnection c,
        RankSafeKldPredictionService.RankSafePredictionModel model,
        CancellationToken ct)
    {
        await ExecuteAsync(c, @"
DROP TABLE IF EXISTS temp_effective_group_prediction;
DROP TABLE IF EXISTS temp_base_predicted_size;
DROP TABLE IF EXISTS temp_group_size_delta;

CREATE TEMP TABLE temp_effective_group_prediction (
    BaseQuant UTINYINT,
    GroupName VARCHAR,
    GroupId UTINYINT,
    StoredSlot UTINYINT,
    EffectiveBaselineId UTINYINT,
    NormalizedBaselineId UTINYINT,
    KldContribution DOUBLE,
    PplContribution DOUBLE,
    BitRange DOUBLE,
    IsZeroDamage BOOLEAN,
    IsKldPredictable BOOLEAN
);

CREATE TEMP TABLE temp_base_predicted_size (
    BaseQuant UTINYINT,
    BaseSizeBytes UBIGINT,
    IsSizePredictable BOOLEAN
);

CREATE TEMP TABLE temp_group_size_delta (
    BaseQuant UTINYINT,
    GroupName VARCHAR,
    GroupId UTINYINT,
    StoredSlot UTINYINT,
    DeltaBytes BIGINT,
    IsSizePredictable BOOLEAN
);", ct);

        var activeBaselines = RuntimeSearchSpace.GetActiveCombinationBaselines()
            .OrderBy(x => x.UniqueId)
            .ToList();

        var activeGroups = model.ActiveGroups
            .Select(g => GroupSlots.First(x => x.Group.UniqueId == g.UniqueId))
            .OrderBy(x => x.Group.UniqueId)
            .ToList();

        using var tx = c.BeginTransaction();

        foreach (var baseline in activeBaselines)
        {
            byte normalizedBase = RankSafeKldPredictionService.NormalizeBaselineIdForIsolation(baseline.UniqueId);
            bool hasBaseSize = model.BaseOnlySnapshotsByBaselineId.TryGetValue(normalizedBase, out var baseOnly);
            await ExecuteAsync(c,
                $"INSERT INTO temp_base_predicted_size VALUES ({baseline.UniqueId}, {SqlULong(hasBaseSize ? baseOnly!.SizeBytes : 0UL)}, {SqlBool(hasBaseSize)});",
                ct);

            var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseline);

            foreach (var slot in activeGroups)
            {
                var allowedForGroup = allowed[slot.Group.UniqueId];
                foreach (byte storedSlot in allowedForGroup)
                {
                    var effectiveBaselineId = GetEffectiveBaselineId(baseline.UniqueId, storedSlot);
                    var normalizedBaselineId = RankSafeKldPredictionService.NormalizeBaselineIdForIsolation(effectiveBaselineId);
                    bool zeroDamage = IsZeroDamageAlias(effectiveBaselineId) || IsZeroDamageAlias(normalizedBaselineId);

                    double kldContribution = 0d;
                    double pplContribution = 0d;
                    double bitRange = zeroDamage ? 99d : GetBitRange(normalizedBaselineId);
                    bool kldPredictable = true;

                    if (!zeroDamage)
                    {
                        if (model.IsolationByGroupAndBaseline.TryGetValue((slot.Group.UniqueId, normalizedBaselineId), out var isolation))
                        {
                            kldContribution = Math.Max(0d, isolation.Kld);
                            pplContribution = isolation.Ppl;
                        }
                        else
                        {
                            kldPredictable = false;
                        }
                    }

                    await ExecuteAsync(c, $@"
INSERT INTO temp_effective_group_prediction VALUES (
    {baseline.UniqueId},
    '{slot.ColumnName}',
    {slot.Group.UniqueId},
    {storedSlot},
    {effectiveBaselineId},
    {normalizedBaselineId},
    {SqlDouble(kldContribution)},
    {SqlDouble(pplContribution)},
    {SqlDouble(bitRange)},
    {SqlBool(zeroDamage)},
    {SqlBool(kldPredictable)}
);", ct);

                    long deltaBytes = 0L;
                    bool sizePredictable = true;

                    // This mirrors RankSafeKldPredictionService.PredictSize:
                    // base-only anchor starts with native-exact groups, then every active
                    // effective group contributes its measured isolation size delta.
                    if (!BaselineQuants.IsNativeExactAlias(normalizedBaselineId))
                    {
                        if (model.IsolationByGroupAndBaseline.TryGetValue((slot.Group.UniqueId, normalizedBaselineId), out var targetIsolation))
                        {
                            deltaBytes = (long)targetIsolation.SizeBytes - (long)model.Q8BaseOnly.SizeBytes;
                        }
                        else
                        {
                            sizePredictable = false;
                        }
                    }

                    await ExecuteAsync(c, $@"
INSERT INTO temp_group_size_delta VALUES (
    {baseline.UniqueId},
    '{slot.ColumnName}',
    {slot.Group.UniqueId},
    {storedSlot},
    {deltaBytes.ToString(CultureInfo.InvariantCulture)},
    {SqlBool(sizePredictable)}
);", ct);
                }
            }
        }

        tx.Commit();
    }

    private static async Task BuildPredictionWorkTablesAsync(
        DuckDBConnection c,
        RankSafeKldPredictionService.RankSafePredictionModel model,
        CancellationToken ct)
    {
        var active = model.ActiveGroups
            .Select(g => GroupSlots.First(x => x.Group.UniqueId == g.UniqueId))
            .OrderBy(x => x.Group.UniqueId)
            .ToList();

        string JoinEffective(GroupSlot slot, string alias) =>
            $"LEFT JOIN temp_effective_group_prediction {alias} ON {alias}.BaseQuant = t.BaseQuant AND {alias}.GroupName = '{slot.ColumnName}' AND {alias}.StoredSlot = t.{slot.ColumnName}";

        string JoinDelta(GroupSlot slot, string alias) =>
            $"LEFT JOIN temp_group_size_delta {alias} ON {alias}.BaseQuant = t.BaseQuant AND {alias}.GroupName = '{slot.ColumnName}' AND {alias}.StoredSlot = t.{slot.ColumnName}";

        string kldSum = active.Count == 0
            ? "0.0"
            : string.Join(" + ", active.Select((_, i) => $"COALESCE(e{i}.KldContribution, 0.0)"));

        string sizeSum = active.Count == 0
            ? "0"
            : string.Join(" + ", active.Select((_, i) => $"COALESCE(d{i}.DeltaBytes, 0)"));

        string kldPredictable = active.Count == 0
            ? "TRUE"
            : string.Join(" AND ", active.Select((_, i) => $"COALESCE(e{i}.IsKldPredictable, FALSE)"));

        string sizePredictable = active.Count == 0
            ? "b.IsSizePredictable"
            : "b.IsSizePredictable AND " + string.Join(" AND ", active.Select((_, i) => $"COALESCE(d{i}.IsSizePredictable, FALSE)"));

        string joins = string.Join(Environment.NewLine, active.Select((slot, i) => JoinEffective(slot, $"e{i}"))) +
                       Environment.NewLine +
                       string.Join(Environment.NewLine, active.Select((slot, i) => JoinDelta(slot, $"d{i}")));

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_prediction_work;
CREATE TEMP TABLE temp_prediction_work AS
SELECT
    CAST(ROW_NUMBER() OVER () AS UBIGINT) AS PredictionWorkId,
    t.{CombinationDuckDbSchema.SlotColumnList.Replace(", ", ", t.")},
    ({kldSum})::DOUBLE AS AdditiveKld,
    GREATEST(CAST(b.BaseSizeBytes AS BIGINT) + {sizeSum}, 0)::UBIGINT AS PredictedSizeBytesRaw,
    ({kldPredictable})::BOOLEAN AS IsKldPredictable,
    ({sizePredictable})::BOOLEAN AS IsSizePredictable
FROM {CombinationDuckDbSchema.TableName} t
JOIN temp_base_predicted_size b ON b.BaseQuant = t.BaseQuant
{joins};", ct);

        string contribUnions = string.Join(Environment.NewLine + "UNION ALL" + Environment.NewLine,
            active.Select((slot, i) => $@"
SELECT
    w.PredictionWorkId,
    CAST({i} AS UTINYINT) AS GroupOrder,
    e.KldContribution,
    e.BitRange,
    GREATEST(0.0, {SqlDouble(model.Fit.BitStressThreshold)} - e.BitRange) AS Stress
FROM temp_prediction_work w
JOIN temp_effective_group_prediction e
  ON e.BaseQuant = w.BaseQuant
 AND e.GroupName = '{slot.ColumnName}'
 AND e.StoredSlot = w.{slot.ColumnName}
WHERE w.IsKldPredictable"));

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_work_group_contrib;
CREATE TEMP TABLE temp_work_group_contrib AS
{contribUnions};", ct);

        await ExecuteAsync(c, @"
DROP TABLE IF EXISTS temp_work_cross_term;
CREATE TEMP TABLE temp_work_cross_term AS
SELECT
    a.PredictionWorkId,
    SUM(a.KldContribution * b.KldContribution * a.Stress * b.Stress) AS CrossTerm
FROM temp_work_group_contrib a
JOIN temp_work_group_contrib b
  ON a.PredictionWorkId = b.PredictionWorkId
 AND a.GroupOrder < b.GroupOrder
GROUP BY a.PredictionWorkId;", ct);

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_projection;
CREATE TEMP TABLE temp_projection AS
SELECT
    w.PredictionWorkId,
    w.AdditiveKld,
    GREATEST(0.0, ({SqlDouble(model.Fit.Alpha)} * w.AdditiveKld) + ({SqlDouble(model.Fit.Beta)} * COALESCE(x.CrossTerm, 0.0))) AS InteractionKld,
    w.PredictedSizeBytesRaw AS PredictedSizeBytes,
    COALESCE(x.CrossTerm, 0.0) AS CrossTerm
FROM temp_prediction_work w
LEFT JOIN temp_work_cross_term x ON x.PredictionWorkId = w.PredictionWorkId
WHERE w.IsKldPredictable
  AND w.IsSizePredictable
  AND w.PredictedSizeBytesRaw > 0;", ct);

        // PAVA is still the same rank-safe projection. It is just applied once
        // over the DuckDB-ordered work table instead of over a giant C# object list.
        await ExecuteAsync(c, @"
DROP TABLE IF EXISTS temp_prediction_order;
CREATE TEMP TABLE temp_prediction_order AS
SELECT
    CAST(ROW_NUMBER() OVER (
        ORDER BY AdditiveKld ASC,
                 InteractionKld ASC,
                 PredictedSizeBytes ASC
    ) AS UBIGINT) AS PredictionOrdinal,
    PredictionWorkId,
    AdditiveKld,
    InteractionKld,
    PredictedSizeBytes
FROM temp_projection;", ct);
    }

    private static async Task BuildPavaBlocksAsync(DuckDBConnection c, CancellationToken ct)
    {
        await ExecuteAsync(c, @"
DROP TABLE IF EXISTS temp_pava_blocks;
CREATE TEMP TABLE temp_pava_blocks (
    StartOrdinal UBIGINT,
    EndOrdinal UBIGINT,
    ProjectedKld DOUBLE,
    BlockCount UBIGINT,
    MeanAdjustment DOUBLE
);", ct);

        var blocks = new List<PavaBlock>();

        using (var cmd = c.CreateCommand())
        {
            cmd.CommandText = @"
SELECT PredictionOrdinal, InteractionKld
FROM temp_prediction_order
ORDER BY PredictionOrdinal ASC;";

            using var r = await cmd.ExecuteReaderAsync(ct);
            while (await r.ReadAsync(ct))
            {
                ct.ThrowIfCancellationRequested();

                ulong ordinal = Convert.ToUInt64(r.GetValue(0));
                double value = Math.Max(0d, Convert.ToDouble(r.GetValue(1), CultureInfo.InvariantCulture));

                blocks.Add(new PavaBlock
                {
                    StartOrdinal = ordinal,
                    EndOrdinal = ordinal,
                    Sum = value,
                    Count = 1
                });

                while (blocks.Count >= 2 && blocks[^2].Mean > blocks[^1].Mean)
                {
                    var right = blocks[^1];
                    var left = blocks[^2];

                    left.EndOrdinal = right.EndOrdinal;
                    left.Sum += right.Sum;
                    left.Count += right.Count;

                    blocks[^2] = left;
                    blocks.RemoveAt(blocks.Count - 1);
                }
            }
        }

        if (blocks.Count == 0)
            return;

        using var tx = c.BeginTransaction();
        using var insert = c.CreateCommand();
        insert.CommandText = "INSERT INTO temp_pava_blocks VALUES (?, ?, ?, ?, ?);";

        foreach (var block in blocks)
        {
            insert.Parameters.Clear();
            double projected = Math.Max(0d, block.Mean);
            insert.Parameters.Add(new DuckDBParameter { Value = block.StartOrdinal });
            insert.Parameters.Add(new DuckDBParameter { Value = block.EndOrdinal });
            insert.Parameters.Add(new DuckDBParameter { Value = projected });
            insert.Parameters.Add(new DuckDBParameter { Value = block.Count });
            insert.Parameters.Add(new DuckDBParameter { Value = Math.Abs(projected - block.Mean) });
            await insert.ExecuteNonQueryAsync(ct);
        }

        tx.Commit();
    }

    private static async Task PersistProjectedPredictionsAsync(
        DuckDBConnection c,
        RankSafeKldPredictionService.RankSafePredictionModel model,
        CancellationToken ct)
    {
        double baseConfidence = ComputeBaseConfidence(model.Fit);

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_projected_prediction;
CREATE TEMP TABLE temp_projected_prediction AS
SELECT
    o.PredictionWorkId,
    b.ProjectedKld AS PredictedKld,
    o.InteractionKld,
    o.PredictedSizeBytes,
    b.BlockCount,
    ABS(b.ProjectedKld - o.InteractionKld) / GREATEST(b.ProjectedKld, 1e-9) AS AdjustmentRatio,
    GREATEST(0.25, 1.0 / SQRT(GREATEST(CAST(b.BlockCount AS DOUBLE), 1.0))) AS PlateauPenalty
FROM temp_prediction_order o
JOIN temp_pava_blocks b
  ON o.PredictionOrdinal BETWEEN b.StartOrdinal AND b.EndOrdinal;", ct);

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_ranked_prediction;
CREATE TEMP TABLE temp_ranked_prediction AS
SELECT
    w.{CombinationDuckDbSchema.SlotColumnList.Replace(", ", ", w.")},
    p.PredictedKld,
    p.PredictedSizeBytes,
    LEAST(1.0, GREATEST(0.0,
        CASE WHEN NOT {CombinationDuckDbSchema.QualifyHybridPredicate("w")}
             THEN 1.0
             ELSE {SqlDouble(baseConfidence)} * (1.0 / (1.0 + p.AdjustmentRatio)) * p.PlateauPenalty
        END
    )) AS PredictionConfidence
FROM temp_prediction_work w
JOIN temp_projected_prediction p ON p.PredictionWorkId = w.PredictionWorkId;", ct);

        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_ranked_prediction_with_rank;
CREATE TEMP TABLE temp_ranked_prediction_with_rank AS
SELECT
    *,
    CAST(ROW_NUMBER() OVER (
        ORDER BY PredictedKld ASC,
                 PredictedSizeBytes ASC,
                 PredictionConfidence DESC,
                 BaseQuant ASC,
                 Embeddings ASC,
                 LmHead ASC,
                 AttnQ ASC,
                 AttnKV ASC,
                 AttnOutput ASC,
                 FfnUpGate ASC,
                 FfnDown ASC,
                 MoeExperts ASC,
                 MoeRouter ASC
    ) AS UBIGINT) AS PredictionRank
FROM temp_ranked_prediction;", ct);

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName} t
SET PredictedKld = r.PredictedKld,
    PredictedSizeBytes = r.PredictedSizeBytes,
    PredictionConfidence = r.PredictionConfidence,
    PredictionRank = r.PredictionRank
FROM temp_ranked_prediction_with_rank r
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "r")};", ct);
    }

    private static double ComputeBaseConfidence(RankSafePredictionFit fit)
    {
        if (fit.UsedFallback)
            return 0.65d;

        double denom = Math.Max(Config.PredictionMinimumFitRows * 4.0d, 1.0d);
        return Math.Clamp(fit.FitRowCount / denom, 0.35d, 1.0d);
    }

    private static byte GetEffectiveBaselineId(byte baseQuant, byte storedSlot)
    {
        return BaselineQuants.IsNullTensorConfigGroupSlot(storedSlot)
            ? baseQuant
            : BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedSlot);
    }

    private static bool IsZeroDamageAlias(byte baselineId)
    {
        return baselineId == BaselineQuants.Q8_0.UniqueId ||
               BaselineQuants.IsNativeExactAlias(baselineId);
    }

    private static double GetBitRange(byte baselineId)
    {
        if (IsZeroDamageAlias(baselineId))
            return 99d;

        return BaselineQuants.FromId(baselineId).BitRange;
    }

    private static async Task ConfigureSessionAsync(DuckDBConnection connection, CancellationToken ct)
    {
        await ExecuteAsync(connection, "SET preserve_insertion_order = false;", ct);
        await ExecuteAsync(connection, $"SET threads = {Math.Max(1, Environment.ProcessorCount)};", ct);
    }

    private static async Task ExecuteAsync(DuckDBConnection c, string sql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        await cmd.ExecuteNonQueryAsync(ct);
    }

    private static string SqlDouble(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            return "0.0";

        return value.ToString("R", CultureInfo.InvariantCulture);
    }

    private static string SqlULong(ulong value) => value.ToString(CultureInfo.InvariantCulture);
    private static string SqlBool(bool value) => value ? "TRUE" : "FALSE";

    private readonly record struct GroupSlot(TensorGroup Group, string ColumnName);

    private struct PavaBlock
    {
        public ulong StartOrdinal;
        public ulong EndOrdinal;
        public double Sum;
        public ulong Count;
        public double Mean => Count == 0 ? 0d : Sum / Count;
    }
}

public sealed class PredictionMaterializationStatus
{
    public long TotalRows { get; init; }
    public long PredictedRows { get; init; }
    public long MissingPredictionRows { get; init; }
    public long RankedRows { get; init; }
    public double? MinPredictedKld { get; init; }
    public double? MaxPredictedKld { get; init; }
    public ulong? MinPredictedSizeBytes { get; init; }
    public ulong? MaxPredictedSizeBytes { get; init; }
}
