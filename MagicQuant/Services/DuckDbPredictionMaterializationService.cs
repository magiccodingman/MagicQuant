using System.Globalization;
using System.Numerics;
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

        PrintModelCoverageDiagnostics(model);

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName}
SET PredictedKld = NULL,
    PredictedSizeBytes = NULL,
    PredictionConfidence = NULL,
    PredictionRank = NULL,
    BaseRankSafeKld = NULL,
    AnomalyAdjustmentKld = 0.0,
    FinalPredictedKld = NULL;", ct);

        await BuildLookupTablesAsync(c, model, ct);
        await PrintLookupDiagnosticsAsync(c, model, ct);
        await BuildPredictionWorkTablesAsync(c, model, ct);
        await PrintPredictionWorkDiagnosticsAsync(c, ct);
        await BuildPavaBlocksAsync(c, ct);
        await PersistProjectedPredictionsAsync(c, model, ct);

        var status = await _store.GetPredictionStatusAsync(ct);
        await PrintFinalMaterializationDiagnosticsAsync(c, status, ct);

        foreach (var note in model.Notes)
            AnsiConsole.MarkupLine($"[grey]Prediction materialization note:[/] {Markup.Escape(note)}");

        if (status.TotalRows > 0 && status.PredictedRows == 0)
        {
            throw new InvalidOperationException(
                "Prediction materialization produced zero predicted rows. This is not a valid no-hybrid result. " +
                "The diagnostics above should identify whether DuckDB BaseQuant IDs, base-only anchors, " +
                "or group isolation/profile-scoped truth rows are missing.");
        }

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

        var duckDbBaseQuantIds = await LoadDistinctBaseQuantIdsAsync(c, ct);
        var runtimeBaseQuantIds = RuntimeSearchSpace.GetActiveCombinationBaselines()
            .Select(x => x.UniqueId)
            .OrderBy(x => x)
            .ToList();

        if (!duckDbBaseQuantIds.SequenceEqual(runtimeBaseQuantIds))
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Prediction carrier mismatch:[/] DuckDB BaseQuant IDs=[cyan]{Markup.Escape(FormatBaselineIds(duckDbBaseQuantIds))}[/], " +
                $"Runtime active IDs=[cyan]{Markup.Escape(FormatBaselineIds(runtimeBaseQuantIds))}[/]. " +
                "Using DuckDB BaseQuant IDs as the scoring source of truth.");
        }

        var activeBaselines = duckDbBaseQuantIds
            .Select(BaselineQuants.FromId)
            .OrderBy(x => x.UniqueId)
            .ToList();

        var activeGroups = model.ActiveGroups
            .Select(g => GroupSlots.First(x => x.Group.UniqueId == g.UniqueId))
            .OrderBy(x => x.Group.UniqueId)
            .ToList();

        var warnedMissingQ8IsolationGroups = new HashSet<byte>();

        using var tx = c.BeginTransaction();

        foreach (var baseline in activeBaselines)
        {
            byte normalizedBase = RankSafeKldPredictionService.NormalizeBaselineIdForIsolation(baseline.UniqueId);
            bool hasBaseSize = model.BaseOnlySnapshotsByBaselineId.TryGetValue(baseline.UniqueId, out var baseOnly) ||
                               model.BaseOnlySnapshotsByBaselineId.TryGetValue(normalizedBase, out baseOnly);
            await ExecuteAsync(c,
                $"INSERT INTO temp_base_predicted_size VALUES ({baseline.UniqueId}, {SqlULong(hasBaseSize ? baseOnly!.SizeBytes : 0UL)}, {SqlBool(hasBaseSize)});",
                ct);

            foreach (var slot in activeGroups)
            {
                var storedSlotsForGroup = await LoadDistinctStoredSlotsAsync(c, baseline.UniqueId, slot.ColumnName, ct);
                foreach (byte storedSlot in storedSlotsForGroup)
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
                            if (normalizedBaselineId == BaselineQuants.Q8_0.UniqueId && warnedMissingQ8IsolationGroups.Add(slot.Group.UniqueId))
                            {
                                AnsiConsole.MarkupLine($"[yellow]Missing KLD isolation snapshot for group '{Markup.Escape(slot.Group.Name)}' and baseline Q8_0 while building DuckDB prediction lookup. Q8_0 is quantized damage, not native truth; matching rows will stay unpredicted instead of receiving zero KLD.[/]");
                            }
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

                ulong ordinal = ToUInt64(r.GetValue(0));
                double value = Math.Max(0d, ToDouble(r.GetValue(1)));

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
    PredictionRank = r.PredictionRank,
    BaseRankSafeKld = r.PredictedKld,
    AnomalyAdjustmentKld = 0.0,
    FinalPredictedKld = r.PredictedKld
FROM temp_ranked_prediction_with_rank r
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "r")};", ct);
    }

    private static async Task<IReadOnlyList<byte>> LoadDistinctBaseQuantIdsAsync(DuckDBConnection c, CancellationToken ct)
    {
        var result = new List<byte>();
        using var cmd = c.CreateCommand();
        cmd.CommandText = $@"
SELECT DISTINCT BaseQuant
FROM {CombinationDuckDbSchema.TableName}
ORDER BY BaseQuant;";

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            result.Add(ToByte(reader.GetValue(0)));

        return result;
    }

    private static async Task<IReadOnlyList<byte>> LoadDistinctStoredSlotsAsync(
        DuckDBConnection c,
        byte baseQuant,
        string columnName,
        CancellationToken ct)
    {
        var result = new List<byte>();
        using var cmd = c.CreateCommand();
        cmd.CommandText = $@"
SELECT DISTINCT {columnName}
FROM {CombinationDuckDbSchema.TableName}
WHERE BaseQuant = ?
ORDER BY {columnName};";
        cmd.Parameters.Add(new DuckDBParameter { Value = baseQuant });

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            result.Add(ToByte(reader.GetValue(0)));

        return result;
    }

    private static void PrintModelCoverageDiagnostics(RankSafeKldPredictionService.RankSafePredictionModel model)
    {
        string activeGroups = string.Join(", ", model.ActiveGroups.Select(x => $"{x.Name}:{x.UniqueId}"));
        string baseOnly = FormatBaselineIds(model.BaseOnlySnapshotsByBaselineId.Keys.OrderBy(x => x).ToList());

        AnsiConsole.MarkupLine($"[grey]Prediction model active groups:[/] {Markup.Escape(activeGroups)}");
        AnsiConsole.MarkupLine($"[grey]Prediction model base-only anchors:[/] [cyan]{model.BaseOnlySnapshotsByBaselineId.Count:N0}[/] ({Markup.Escape(baseOnly)})");
        AnsiConsole.MarkupLine($"[grey]Prediction model isolation anchors:[/] [cyan]{model.IsolationByGroupAndBaseline.Count:N0}[/]");

        foreach (var group in model.ActiveGroups.OrderBy(x => x.UniqueId))
        {
            var ids = model.IsolationByGroupAndBaseline.Keys
                .Where(x => x.GroupId == group.UniqueId)
                .Select(x => x.BaselineId)
                .Distinct()
                .OrderBy(x => x)
                .ToList();

            AnsiConsole.MarkupLine($"[grey]  - isolation coverage {Markup.Escape(group.Name)}:[/] [cyan]{ids.Count:N0}[/] ({Markup.Escape(FormatBaselineIds(ids))})");
        }
    }

    private static async Task PrintLookupDiagnosticsAsync(
        DuckDBConnection c,
        RankSafeKldPredictionService.RankSafePredictionModel model,
        CancellationToken ct)
    {
        long totalRows = await ScalarLongAsync(c, $"SELECT COUNT(*) FROM {CombinationDuckDbSchema.TableName};", ct);
        long baseLookupRows = await ScalarLongAsync(c, "SELECT COUNT(*) FROM temp_base_predicted_size;", ct);
        long missingBaseJoin = await ScalarLongAsync(c, $@"
SELECT COUNT(*)
FROM {CombinationDuckDbSchema.TableName} t
LEFT JOIN temp_base_predicted_size b ON b.BaseQuant = t.BaseQuant
WHERE b.BaseQuant IS NULL;", ct);

        AnsiConsole.MarkupLine($"[grey]DuckDB prediction lookup rows:[/] total=[cyan]{totalRows:N0}[/] base-lookups=[cyan]{baseLookupRows:N0}[/] missing-base-join=[cyan]{missingBaseJoin:N0}[/]");

        await PrintBaseLookupRowsAsync(c, ct);
        await PrintMissingGroupLookupRowsAsync(c, model, ct);
    }

    private static async Task PrintBaseLookupRowsAsync(DuckDBConnection c, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = @"
SELECT BaseQuant, BaseSizeBytes, IsSizePredictable
FROM temp_base_predicted_size
ORDER BY BaseQuant;";

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            byte baseId = ToByte(reader.GetValue(0));
            ulong bytes = ToUInt64(reader.GetValue(1));
            bool predictable = ToBool(reader.GetValue(2));
            AnsiConsole.MarkupLine($"[grey]  - base lookup {Markup.Escape(FormatBaselineId(baseId))}:[/] size={bytes:N0} predictable={predictable}");
        }
    }

    private static async Task PrintMissingGroupLookupRowsAsync(
        DuckDBConnection c,
        RankSafeKldPredictionService.RankSafePredictionModel model,
        CancellationToken ct)
    {
        var active = model.ActiveGroups
            .Select(g => GroupSlots.First(x => x.Group.UniqueId == g.UniqueId))
            .OrderBy(x => x.Group.UniqueId)
            .ToList();

        foreach (var slot in active)
        {
            using var cmd = c.CreateCommand();
            cmd.CommandText = $@"
SELECT t.BaseQuant, t.{slot.ColumnName}, COUNT(*) AS MissingRows
FROM {CombinationDuckDbSchema.TableName} t
LEFT JOIN temp_effective_group_prediction e
  ON e.BaseQuant = t.BaseQuant
 AND e.GroupName = '{slot.ColumnName}'
 AND e.StoredSlot = t.{slot.ColumnName}
WHERE e.BaseQuant IS NULL
GROUP BY t.BaseQuant, t.{slot.ColumnName}
ORDER BY MissingRows DESC
LIMIT 5;";

            using var reader = await cmd.ExecuteReaderAsync(ct);
            bool wroteHeader = false;
            while (await reader.ReadAsync(ct))
            {
                if (!wroteHeader)
                {
                    AnsiConsole.MarkupLine($"[yellow]Missing effective lookup rows for group {Markup.Escape(slot.ColumnName)}:[/]");
                    wroteHeader = true;
                }

                byte baseId = ToByte(reader.GetValue(0));
                byte storedSlot = ToByte(reader.GetValue(1));
                long count = ToInt64(reader.GetValue(2));
                AnsiConsole.MarkupLine($"[yellow]  - base={Markup.Escape(FormatBaselineId(baseId))} stored={Markup.Escape(FormatStoredSlot(storedSlot))} rows={count:N0}[/]");
            }
        }
    }

    private static async Task PrintPredictionWorkDiagnosticsAsync(DuckDBConnection c, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = @"
SELECT
    COUNT(*) AS WorkRows,
    COALESCE(SUM(CASE WHEN IsKldPredictable THEN 1 ELSE 0 END), 0) AS KldPredictableRows,
    COALESCE(SUM(CASE WHEN IsSizePredictable THEN 1 ELSE 0 END), 0) AS SizePredictableRows,
    COALESCE(SUM(CASE WHEN IsKldPredictable AND IsSizePredictable THEN 1 ELSE 0 END), 0) AS ProjectableRows
FROM temp_prediction_work;";

        using (var reader = await cmd.ExecuteReaderAsync(ct))
        {
            await reader.ReadAsync(ct);
            long workRows = ToInt64(reader.GetValue(0));
            long kldRows = ToInt64(reader.GetValue(1));
            long sizeRows = ToInt64(reader.GetValue(2));
            long projectableRows = ToInt64(reader.GetValue(3));
            AnsiConsole.MarkupLine($"[grey]DuckDB prediction work rows:[/] work=[cyan]{workRows:N0}[/] kld-ok=[cyan]{kldRows:N0}[/] size-ok=[cyan]{sizeRows:N0}[/] projectable=[cyan]{projectableRows:N0}[/]");
        }

        await PrintPredictionFailureBreakdownAsync(c, ct);

        long projected = await ScalarLongAsync(c, "SELECT COUNT(*) FROM temp_projection;", ct);
        long ordered = await ScalarLongAsync(c, "SELECT COUNT(*) FROM temp_prediction_order;", ct);
        AnsiConsole.MarkupLine($"[grey]DuckDB projection rows:[/] projection=[cyan]{projected:N0}[/] ordered=[cyan]{ordered:N0}[/]");
    }

    private static async Task PrintPredictionFailureBreakdownAsync(DuckDBConnection c, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = @"
SELECT
    BaseQuant,
    COUNT(*) AS Rows,
    COALESCE(SUM(CASE WHEN NOT IsKldPredictable THEN 1 ELSE 0 END), 0) AS KldMissing,
    COALESCE(SUM(CASE WHEN NOT IsSizePredictable THEN 1 ELSE 0 END), 0) AS SizeMissing
FROM temp_prediction_work
GROUP BY BaseQuant
ORDER BY BaseQuant;";

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            byte baseId = ToByte(reader.GetValue(0));
            long rows = ToInt64(reader.GetValue(1));
            long kldMissing = ToInt64(reader.GetValue(2));
            long sizeMissing = ToInt64(reader.GetValue(3));
            AnsiConsole.MarkupLine($"[grey]  - work {Markup.Escape(FormatBaselineId(baseId))}:[/] rows={rows:N0} missing-kld={kldMissing:N0} missing-size={sizeMissing:N0}");
        }
    }

    private static async Task PrintFinalMaterializationDiagnosticsAsync(
        DuckDBConnection c,
        PredictionMaterializationStatus status,
        CancellationToken ct)
    {
        long rankedRows = await ScalarLongAsync(c, "SELECT COUNT(*) FROM temp_ranked_prediction_with_rank;", ct);
        AnsiConsole.MarkupLine($"[grey]DuckDB final materialized prediction rows:[/] predicted=[cyan]{status.PredictedRows:N0}[/] / {status.TotalRows:N0}, ranked-temp=[cyan]{rankedRows:N0}[/]");
    }

    private static async Task<long> ScalarLongAsync(DuckDBConnection c, string sql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        return ToInt64(await cmd.ExecuteScalarAsync(ct) ?? 0L);
    }

    private static string FormatBaselineIds(IReadOnlyCollection<byte> ids)
    {
        if (ids.Count == 0)
            return "none";

        return string.Join(", ", ids.Select(FormatBaselineId));
    }

    private static string FormatBaselineId(byte id)
    {
        try
        {
            var baseline = BaselineQuants.FromId(id);
            return $"{baseline.Names[0]}:{id}";
        }
        catch
        {
            return $"unknown:{id}";
        }
    }

    private static string FormatStoredSlot(byte storedSlot)
    {
        if (BaselineQuants.IsNullTensorConfigGroupSlot(storedSlot))
            return $"base/null:{storedSlot}";

        byte decoded = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedSlot);
        return $"{FormatBaselineId(decoded)} stored:{storedSlot}";
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
        // Only native exact aliases are zero-reference states. Q8_0 is intentionally
        // excluded: it has measured isolation KLD and must be scored like every other
        // quant baseline in prediction space.
        return BaselineQuants.IsNativeExactAlias(baselineId);
    }

    private static double GetBitRange(byte baselineId)
    {
        if (IsZeroDamageAlias(baselineId))
            return 99d;

        return BaselineQuants.FromId(baselineId).BitRange;
    }

    private static long ToInt64(object? value)
    {
        if (value is null or DBNull)
            return 0L;

        return value switch
        {
            long x => x,
            int x => x,
            short x => x,
            sbyte x => x,
            byte x => x,
            uint x => checked((long)x),
            ulong x => checked((long)x),
            BigInteger x => checked((long)x),
            decimal x => checked((long)x),
            double x => checked((long)x),
            float x => checked((long)x),
            IConvertible x => x.ToInt64(CultureInfo.InvariantCulture),
            _ => long.Parse(value.ToString() ?? "0", CultureInfo.InvariantCulture)
        };
    }

    private static ulong ToUInt64(object? value)
    {
        if (value is null or DBNull)
            return 0UL;

        return value switch
        {
            ulong x => x,
            long x => checked((ulong)x),
            int x => checked((ulong)x),
            short x => checked((ulong)x),
            sbyte x => checked((ulong)x),
            byte x => x,
            uint x => x,
            BigInteger x => checked((ulong)x),
            decimal x => checked((ulong)x),
            double x => checked((ulong)x),
            float x => checked((ulong)x),
            IConvertible x => x.ToUInt64(CultureInfo.InvariantCulture),
            _ => ulong.Parse(value.ToString() ?? "0", CultureInfo.InvariantCulture)
        };
    }

    private static byte ToByte(object? value)
    {
        if (value is null or DBNull)
            return 0;

        return value switch
        {
            byte x => x,
            sbyte x => checked((byte)x),
            short x => checked((byte)x),
            int x => checked((byte)x),
            long x => checked((byte)x),
            ushort x => checked((byte)x),
            uint x => checked((byte)x),
            ulong x => checked((byte)x),
            BigInteger x => checked((byte)x),
            IConvertible x => x.ToByte(CultureInfo.InvariantCulture),
            _ => byte.Parse(value.ToString() ?? "0", CultureInfo.InvariantCulture)
        };
    }

    private static double ToDouble(object? value)
    {
        if (value is null or DBNull)
            return 0d;

        return value switch
        {
            double x => x,
            float x => x,
            decimal x => (double)x,
            BigInteger x => (double)x,
            IConvertible x => x.ToDouble(CultureInfo.InvariantCulture),
            _ => double.Parse(value.ToString() ?? "0", CultureInfo.InvariantCulture)
        };
    }

    private static bool ToBool(object? value)
    {
        if (value is null or DBNull)
            return false;

        return value switch
        {
            bool x => x,
            byte x => x != 0,
            sbyte x => x != 0,
            short x => x != 0,
            int x => x != 0,
            long x => x != 0,
            ushort x => x != 0,
            uint x => x != 0,
            ulong x => x != 0,
            BigInteger x => x != BigInteger.Zero,
            IConvertible x => x.ToBoolean(CultureInfo.InvariantCulture),
            _ => bool.Parse(value.ToString() ?? "false")
        };
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