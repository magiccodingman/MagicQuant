using System.Collections.Immutable;
using System.Diagnostics;
using System.Numerics;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public class QuantDatabaseService
{
    private const string DbFileNamePrefix = "MagicQuant_Combinations";
    private const string TableName = CombinationDuckDbSchema.TableName;

    private static readonly string[] ExpectedColumnTypes = CombinationDuckDbSchema.ExpectedColumnTypes;

    private static string CreateTableSql => CombinationDuckDbSchema.CreateTableSql;

    private static async Task ConfigureFastLoadSessionAsync(DuckDBConnection connection, CancellationToken ct)
    {
        // These are safe session-level tweaks for this write-heavy workload.
        // We do not care about insertion order for tensor combo staging.
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = "SET preserve_insertion_order = false;";
            await cmd.ExecuteNonQueryAsync(ct);
        }

        // Let DuckDB use the available machine parallelism.
        int threadCount = Math.Max(1, Environment.ProcessorCount);
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = $"SET threads = {threadCount};";
            await cmd.ExecuteNonQueryAsync(ct);
        }
    }

    private static async Task RecreateTableAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var createCmd = connection.CreateCommand();
        createCmd.CommandText = CreateTableSql;
        await createCmd.ExecuteNonQueryAsync(ct);
    }

    public async Task<long> GetRemainingCombinationCountAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {TableName};";

        return ToInt64(await cmd.ExecuteScalarAsync(ct));
    }

    public async Task<List<TensorConfig>> GetRemainingTensorConfigsAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);
        long count = await GetRowCountAsync(connection, ct);
        if (count > Config.MaxInMemoryCombinationLoadRows)
            throw new InvalidOperationException($"Refusing to load {count:N0} DuckDB tensor configs into memory. Use SQL-native filtering/streaming instead.");

        var results = new List<TensorConfig>();

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $@"
        SELECT
            BaseQuant,
            Embeddings,
            LmHead,
            AttnQ,
            AttnKV,
            AttnOutput,
            FfnUpGate,
            FfnDown,
            MoeExperts,
            MoeRouter
        FROM {TableName}
        ORDER BY
            BaseQuant,
            Embeddings,
            LmHead,
            AttnQ,
            AttnKV,
            AttnOutput,
            FfnUpGate,
            FfnDown,
            MoeExperts,
            MoeRouter;";

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            results.Add(new TensorConfig(
                baseQuant: Convert.ToByte(reader.GetValue(0)),
                embeddings: Convert.ToByte(reader.GetValue(1)),
                lmHead: Convert.ToByte(reader.GetValue(2)),
                attnQ: Convert.ToByte(reader.GetValue(3)),
                attnKV: Convert.ToByte(reader.GetValue(4)),
                attnOutput: Convert.ToByte(reader.GetValue(5)),
                ffnUpGate: Convert.ToByte(reader.GetValue(6)),
                ffnDown: Convert.ToByte(reader.GetValue(7)),
                moeExperts: Convert.ToByte(reader.GetValue(8)),
                moeRouter: Convert.ToByte(reader.GetValue(9))
            ));
        }

        return results;
    }

    private static string GetDuckDbDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Cache.ModelMagicQuantDirectory!;

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Cache.MagicQuantDirectory!;

        throw new InvalidOperationException(
            "Neither Cache.ModelMagicQuantDirectory nor Cache.MagicQuantDirectory is set.");
    }

    private static string BuildContextAwareDuckDbFileName()
    {
        string model = string.IsNullOrWhiteSpace(Cache.CurrentModelId) ? "unknown-model" : Cache.CurrentModelId;
        string imatrix = Cache.IsImatrixAvailable ? (Cache.ActiveImatrixIdentityHash ?? "imatrix-unknown") : "no-imatrix";
        string hp = RuntimeSearchSpace.AllowHighPrecisionHybrids ? "hp-on" : "hp-off";
        return $"{DbFileNamePrefix}_{model}_{imatrix}_{hp}.duckdb";
    }

    private string ConnectionString => $"Data Source={Path.Combine(GetDuckDbDirectory(), BuildContextAwareDuckDbFileName())}";

    public async Task InitializeAsync(bool forceRebuild = false, CancellationToken ct = default)
    {
        var duckDbDirectory = GetDuckDbDirectory();
        Directory.CreateDirectory(duckDbDirectory);

        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        BigInteger expectedTotal = ComboCounter.CountAll();
        bool tableShapeOk = await HasExpectedTableShapeAsync(connection, ct);
        long currentDbCount = tableShapeOk
            ? await GetRowCountAsync(connection, ct)
            : -1;
        long currentNormalDbCount = tableShapeOk
            ? await GetNormalRowCountAsync(connection, ct)
            : -1;

        AnsiConsole.MarkupLine(
            $"[bold]DuckDB Check:[/] Current Rows: [cyan]{currentDbCount:N0}[/] | Normal Candidate Rows: [cyan]{currentNormalDbCount:N0}[/] | Expected Normal Rows: [yellow]{expectedTotal:N0}[/]");

        if (!tableShapeOk)
            AnsiConsole.MarkupLine("[yellow]DuckDB table shape is missing or stale. Rebuild required.[/]");

        if (forceRebuild || !tableShapeOk || new BigInteger(currentNormalDbCount) != expectedTotal)
        {
            AnsiConsole.MarkupLine("[bold red]DuckDB empty, mismatch, forced, or stale.[/] Initializing/Rebuilding...");
            await RebuildDatabaseAsync(connection, expectedTotal, ct);
        }
        else
        {
            var virtualAnchorStats = await AppendVirtualPredictionAnchorRowsAsync(connection, ct);
            AnsiConsole.MarkupLine(
                $"[bold green]DuckDB is synchronized and ready.[/] [grey]Virtual anchors inserted={virtualAnchorStats.InsertedRows:N0}, marked={virtualAnchorStats.MarkedExistingRows:N0}[/]");
        }
    }

    public async Task RebuildAsync(CancellationToken ct = default)
    {
        await InitializeAsync(forceRebuild: true, ct: ct);
    }

    public async Task<long> PrunePredictedLargerThanQ8Async(
        RequiredSampleGenerationResult fullPlan,
        CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        var predictionContext = await BuildPredictionContextAsync(fullPlan, ct);

        if (predictionContext == null)
        {
            AnsiConsole.MarkupLine("[yellow]Predicted-size pruning skipped: prediction context was incomplete.[/]");
            return 0;
        }

        if (Config.ManualMaxPredictedSizeBytes <= 0 && predictionContext.ShouldSkipPureQ8CeilingPruning)
        {
            if (!string.IsNullOrWhiteSpace(predictionContext.SkipPureQ8CeilingReason))
            {
                AnsiConsole.MarkupLine($"[green]Predicted-size pruning removed 0 combinations.[/] [grey]{Markup.Escape(predictionContext.SkipPureQ8CeilingReason!)}[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[green]Predicted-size pruning removed 0 combinations.[/]");
            }

            return 0;
        }

        long beforeCount = await GetRowCountAsync(connection, ct);

        ulong sizeCeilingBytes = Config.ManualMaxPredictedSizeBytes > 0
            ? Config.ManualMaxPredictedSizeBytes
            : predictionContext.PureQ8BaseSize;
        await BuildPredictedSizeLookupTablesAsync(connection, predictionContext, ct);

        using (var pruneCmd = connection.CreateCommand())
        {
            pruneCmd.CommandText = $@"
DROP TABLE IF EXISTS tensor_configs_pruned;
CREATE TABLE tensor_configs_pruned AS
SELECT t.*
FROM {TableName} t
JOIN temp_base_predicted_size b ON b.BaseQuant = t.BaseQuant
LEFT JOIN temp_group_size_delta de   ON de.BaseQuant = t.BaseQuant AND de.GroupName = 'Embeddings' AND de.StoredSlot = t.Embeddings
LEFT JOIN temp_group_size_delta dl   ON dl.BaseQuant = t.BaseQuant AND dl.GroupName = 'LmHead' AND dl.StoredSlot = t.LmHead
LEFT JOIN temp_group_size_delta daq  ON daq.BaseQuant = t.BaseQuant AND daq.GroupName = 'AttnQ' AND daq.StoredSlot = t.AttnQ
LEFT JOIN temp_group_size_delta dakv ON dakv.BaseQuant = t.BaseQuant AND dakv.GroupName = 'AttnKV' AND dakv.StoredSlot = t.AttnKV
LEFT JOIN temp_group_size_delta dao  ON dao.BaseQuant = t.BaseQuant AND dao.GroupName = 'AttnOutput' AND dao.StoredSlot = t.AttnOutput
LEFT JOIN temp_group_size_delta dfu  ON dfu.BaseQuant = t.BaseQuant AND dfu.GroupName = 'FfnUpGate' AND dfu.StoredSlot = t.FfnUpGate
LEFT JOIN temp_group_size_delta dfd  ON dfd.BaseQuant = t.BaseQuant AND dfd.GroupName = 'FfnDown' AND dfd.StoredSlot = t.FfnDown
LEFT JOIN temp_group_size_delta dme  ON dme.BaseQuant = t.BaseQuant AND dme.GroupName = 'MoeExperts' AND dme.StoredSlot = t.MoeExperts
LEFT JOIN temp_group_size_delta dmr  ON dmr.BaseQuant = t.BaseQuant AND dmr.GroupName = 'MoeRouter' AND dmr.StoredSlot = t.MoeRouter
WHERE CAST(b.BaseSizeBytes AS BIGINT)
    + COALESCE(de.DeltaBytes, 0) + COALESCE(dl.DeltaBytes, 0) + COALESCE(daq.DeltaBytes, 0)
    + COALESCE(dakv.DeltaBytes, 0) + COALESCE(dao.DeltaBytes, 0) + COALESCE(dfu.DeltaBytes, 0)
    + COALESCE(dfd.DeltaBytes, 0) + COALESCE(dme.DeltaBytes, 0) + COALESCE(dmr.DeltaBytes, 0)
    <= CAST({sizeCeilingBytes} AS BIGINT);
DROP TABLE {TableName};
ALTER TABLE tensor_configs_pruned RENAME TO {TableName};";
            await pruneCmd.ExecuteNonQueryAsync(ct);
        }

        long afterCount = await GetRowCountAsync(connection, ct);
        long removed = beforeCount - afterCount;

        string ceilingLabel = Config.ManualMaxPredictedSizeBytes > 0
            ? $"manual ceiling {Config.ManualMaxPredictedSizeBytes:N0} bytes"
            : "pure Q8";

        AnsiConsole.MarkupLine($"[yellow]Predicted-size pruning removed:[/] [red]{removed:N0}[/] combo(s) larger than {ceilingLabel}.");
        return removed;
    }

    public async Task<long> PruneHighPrecisionHybridCandidatesAsync(CancellationToken ct = default)
    {
        if (RuntimeSearchSpace.AllowHighPrecisionHybrids)
            return 0;

        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        long beforeCount = await GetRowCountAsync(connection, ct);
        byte bf16Stored = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(BaselineQuants.BF16_Hybrid.UniqueId);
        byte f16Stored = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(BaselineQuants.F16_Hybrid.UniqueId);
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = $@"
DROP TABLE IF EXISTS tensor_configs_pruned;
CREATE TABLE tensor_configs_pruned AS
SELECT *
FROM {TableName}
WHERE Embeddings NOT IN ({bf16Stored}, {f16Stored})
  AND LmHead NOT IN ({bf16Stored}, {f16Stored})
  AND AttnQ NOT IN ({bf16Stored}, {f16Stored})
  AND AttnKV NOT IN ({bf16Stored}, {f16Stored})
  AND AttnOutput NOT IN ({bf16Stored}, {f16Stored})
  AND FfnUpGate NOT IN ({bf16Stored}, {f16Stored})
  AND FfnDown NOT IN ({bf16Stored}, {f16Stored})
  AND MoeExperts NOT IN ({bf16Stored}, {f16Stored})
  AND MoeRouter NOT IN ({bf16Stored}, {f16Stored});
DROP TABLE {TableName};
ALTER TABLE tensor_configs_pruned RENAME TO {TableName};";
            await cmd.ExecuteNonQueryAsync(ct);
        }
        long afterCount = await GetRowCountAsync(connection, ct);
        return beforeCount - afterCount;
    }

    private async Task<bool> HasExpectedTableShapeAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var existsCmd = connection.CreateCommand();
        existsCmd.CommandText = $"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{TableName}'";
        long exists = (long)(await existsCmd.ExecuteScalarAsync(ct) ?? 0L);

        if (exists == 0)
            return false;

        var actual = new List<string>();

        using var shapeCmd = connection.CreateCommand();
        shapeCmd.CommandText = $@"
            SELECT lower(data_type)
            FROM information_schema.columns
            WHERE table_name = '{TableName}'
            ORDER BY ordinal_position;";

        using var reader = await shapeCmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            actual.Add(Convert.ToString(reader.GetValue(0)) ?? string.Empty);
        }

        if (actual.Count != ExpectedColumnTypes.Length)
            return false;

        for (int i = 0; i < ExpectedColumnTypes.Length; i++)
        {
            if (!string.Equals(actual[i], ExpectedColumnTypes[i], StringComparison.Ordinal))
                return false;
        }

        return true;
    }

    private async Task<long> GetRowCountAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var countCmd = connection.CreateCommand();
        countCmd.CommandText = $"SELECT COUNT(*) FROM {TableName}";
        return ToInt64(await countCmd.ExecuteScalarAsync(ct));
    }

    private async Task<long> GetNormalRowCountAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var countCmd = connection.CreateCommand();
        countCmd.CommandText = $"SELECT COUNT(*) FROM {TableName} WHERE COALESCE(IsVirtualPredictionAnchor, FALSE) = FALSE";
        return ToInt64(await countCmd.ExecuteScalarAsync(ct));
    }

    private async Task RebuildDatabaseAsync(
        DuckDBConnection connection,
        BigInteger expectedTotal,
        CancellationToken ct)
    {
        AnsiConsole.MarkupLine($"[yellow]Starting SQL-native tensor combination generation for {expectedTotal:N0} rows...[/]");

        await RecreateTableAsync(connection, ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        BigInteger insertedGrandTotal = BigInteger.Zero;
        var overallSw = Stopwatch.StartNew();

        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            ct.ThrowIfCancellationRequested();

            var baseSw = Stopwatch.StartNew();
            string baseName = baseline.Names.FirstOrDefault() ?? baseline.UniqueId.ToString();
            long before = await GetRowCountAsync(connection, ct);
            BigInteger expectedForBase = await InsertBaselineCombinationsSqlAsync(connection, baseline, ct);
            long after = await GetRowCountAsync(connection, ct);
            BigInteger delta = new(after - before);
            if (delta != expectedForBase)
                throw new InvalidOperationException($"Baseline {baseName} inserted {delta} rows, expected {expectedForBase}.");
            insertedGrandTotal += delta;

            baseSw.Stop();

            double rowsPerSec = baseSw.Elapsed.TotalSeconds <= 0
                ? 0 : (double)(long)expectedForBase / baseSw.Elapsed.TotalSeconds;

            AnsiConsole.MarkupLine(
                $"[bold green]Base complete:[/] {Markup.Escape(baseName)}  " +
                $"[grey]| Inserted:[/] {expectedForBase:N0} rows  " +
                $"[grey]| Time:[/] {baseSw.Elapsed.TotalMinutes:N2} min  " +
                $"[grey]| Rate:[/] {rowsPerSec:N0} rows/sec");
        }

        var virtualAnchorStats = await AppendVirtualPredictionAnchorRowsAsync(connection, ct);

        overallSw.Stop();

        long finalCount = await GetRowCountAsync(connection, ct);
        BigInteger expectedIncludingVirtualRows = expectedTotal + new BigInteger(virtualAnchorStats.InsertedRows);

        double finalRate = overallSw.Elapsed.TotalSeconds <= 0
            ? 0
            : (double)(long)insertedGrandTotal / overallSw.Elapsed.TotalSeconds;

        AnsiConsole.MarkupLine(
            $"[bold green]DuckDB rebuild complete.[/] " +
            $"[grey]| Inserted tracked:[/] {insertedGrandTotal:N0}  " +
            $"[grey]| Virtual anchors inserted:[/] {virtualAnchorStats.InsertedRows:N0}  " +
            $"[grey]| Virtual anchors marked:[/] {virtualAnchorStats.MarkedExistingRows:N0}  " +
            $"[grey]| Final row count:[/] {finalCount:N0}  " +
            $"[grey]| Time:[/] {overallSw.Elapsed.TotalMinutes:N2} min  " +
            $"[grey]| Avg rate:[/] {finalRate:N0} rows/sec");
        if (new BigInteger(finalCount) != expectedIncludingVirtualRows)
            throw new InvalidOperationException($"Final tensor_configs row count mismatch. actual={finalCount:N0}, expected={expectedIncludingVirtualRows:N0} (normal={expectedTotal:N0}, virtual-inserted={virtualAnchorStats.InsertedRows:N0}).");
    }

    private static async Task<VirtualAnchorInsertStats> AppendVirtualPredictionAnchorRowsAsync(
        DuckDBConnection connection,
        CancellationToken ct)
    {
        var activeGroups = GetVirtualPredictionAnchorActiveGroups();
        if (activeGroups.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]Virtual prediction anchors skipped:[/] no active tensor groups were available.");
            return new VirtualAnchorInsertStats();
        }

        var carrier = ChooseVirtualPredictionAnchorCarrier();
        var anchorBaselines = GetVirtualPredictionAnchorBaselines()
            .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
            .OrderByDescending(x => x.BitRange)
            .ThenBy(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .ToList();

        if (anchorBaselines.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]Virtual prediction anchors skipped:[/] no active baseline identities were available.");
            return new VirtualAnchorInsertStats();
        }

        int inserted = 0;
        int markedExisting = 0;
        var preview = new List<string>();

        foreach (var baseline in anchorBaselines)
        {
            ct.ThrowIfCancellationRequested();

            var quant = HybridQuant.CreateLearnedCandidateBlanket(
                baseQuant: carrier,
                groups: activeGroups,
                candidateBaseline: baseline);

            var config = (TensorConfig)quant;
            bool rowAlreadyExisted = await VirtualAnchorRowExistsAsync(connection, config, ct);
            await UpsertVirtualAnchorRowAsync(connection, config, baseline, ct);

            if (rowAlreadyExisted)
                markedExisting++;
            else
                inserted++;

            if (preview.Count < 12)
                preview.Add($"{baseline.Names[0]} -> {TensorConfigIdentity.ToKey(config)}");
        }

        string groupList = string.Join(", ", activeGroups.Select(x => x.Name));
        AnsiConsole.MarkupLine(
            $"[green]Virtual prediction anchors staged:[/] inserted=[cyan]{inserted:N0}[/], marked-existing=[cyan]{markedExisting:N0}[/], " +
            $"carrier=[cyan]{Markup.Escape(carrier.Names[0])}[/], groups=[cyan]{Markup.Escape(groupList)}[/]");

        foreach (var item in preview)
            AnsiConsole.MarkupLine($"  [grey]- {Markup.Escape(item)}[/]");

        if (anchorBaselines.Count > preview.Count)
            AnsiConsole.MarkupLine($"  [grey]- ... {anchorBaselines.Count - preview.Count:N0} more virtual anchors[/]");

        return new VirtualAnchorInsertStats
        {
            InsertedRows = inserted,
            MarkedExistingRows = markedExisting
        };
    }

    private static IReadOnlyList<TensorGroup> GetVirtualPredictionAnchorActiveGroups()
    {
        return TReg.All
            .Where(x => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == x.UniqueId))
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    private static IReadOnlyList<BaselineQuants> GetVirtualPredictionAnchorBaselines()
    {
        bool hasUsableImatrix = RuntimeSearchSpace.HasUsableImatrix();

        return BaselineQuants.GetLearningBaselines(hasUsableImatrix)
            .Concat(BaselineQuants.GetGroupCombinationCandidates(hasUsableImatrix, RuntimeSearchSpace.AllowHighPrecisionHybrids))
            .Concat(BaselineQuants.GetCombinationCarrierBaselines(hasUsableImatrix))
            .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
            .GroupBy(x => NormalizeAnchorKey(x.CanonicalKey), StringComparer.Ordinal)
            .Select(g => g.OrderBy(x => x.UniqueId).First())
            .OrderBy(x => x.UniqueId)
            .ToList();
    }

    private static BaselineQuants ChooseVirtualPredictionAnchorCarrier()
    {
        var activeCarriers = RuntimeSearchSpace.GetActiveCombinationBaselines()
            .OrderBy(x => x.UniqueId)
            .ToList();

        if (activeCarriers.Count == 0)
            return BaselineQuants.Q8_0;

        if (activeCarriers.Count == 1)
            return activeCarriers[0];

        var q8 = activeCarriers.FirstOrDefault(x => x.UniqueId == BaselineQuants.Q8_0.UniqueId);
        if (q8 != null)
            return q8;

        return activeCarriers
            .OrderByDescending(x => x.BitRange)
            .ThenByDescending(x => x.ExplicitCandidateSortOrder)
            .ThenBy(x => x.UniqueId)
            .First();
    }

    private static async Task<bool> VirtualAnchorRowExistsAsync(
        DuckDBConnection connection,
        TensorConfig config,
        CancellationToken ct)
    {
        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {TableName} WHERE {BuildSlotPredicateSql(config)};";
        var value = await cmd.ExecuteScalarAsync(ct);
        return ToInt64(value) > 0;
    }

    private static async Task UpsertVirtualAnchorRowAsync(
        DuckDBConnection connection,
        TensorConfig config,
        BaselineQuants baseline,
        CancellationToken ct)
    {
        if (await VirtualAnchorRowExistsAsync(connection, config, ct))
        {
            using var update = connection.CreateCommand();
            update.CommandText = $@"
UPDATE {TableName}
SET IsProtectedAnchor = TRUE,
    IsVirtualPredictionAnchor = TRUE,
    AnchorBaselineRuntimeId = {baseline.UniqueId},
    AnchorBaselineCanonicalKey = {SqlString(baseline.CanonicalKey)},
    AnchorDisplayName = {SqlString(baseline.Names.FirstOrDefault() ?? baseline.CanonicalKey)}
WHERE {BuildSlotPredicateSql(config)};";
            await update.ExecuteNonQueryAsync(ct);
            return;
        }

        using var insert = connection.CreateCommand();
        insert.CommandText = $@"
INSERT INTO {TableName}
({CombinationDuckDbSchema.SlotColumnList}, IsProtectedAnchor, IsVirtualPredictionAnchor, AnchorBaselineRuntimeId, AnchorBaselineCanonicalKey, AnchorDisplayName)
VALUES ({config.BaseQuant}, {config.Embeddings}, {config.LmHead}, {config.AttnQ}, {config.AttnKV}, {config.AttnOutput}, {config.FfnUpGate}, {config.FfnDown}, {config.MoeExperts}, {config.MoeRouter}, TRUE, TRUE, {baseline.UniqueId}, {SqlString(baseline.CanonicalKey)}, {SqlString(baseline.Names.FirstOrDefault() ?? baseline.CanonicalKey)});";
        await insert.ExecuteNonQueryAsync(ct);
    }

    private static string BuildSlotPredicateSql(TensorConfig config)
    {
        return $"BaseQuant = {config.BaseQuant} AND Embeddings = {config.Embeddings} AND LmHead = {config.LmHead} AND AttnQ = {config.AttnQ} AND AttnKV = {config.AttnKV} AND AttnOutput = {config.AttnOutput} AND FfnUpGate = {config.FfnUpGate} AND FfnDown = {config.FfnDown} AND MoeExperts = {config.MoeExperts} AND MoeRouter = {config.MoeRouter}";
    }

    private static string SqlString(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "NULL";

        return $"'{value.Replace("'", "''")}'";
    }

    private static long ToInt64(object? value)
    {
        if (value is null || value is DBNull)
            return 0L;

        if (value is BigInteger big)
            return (long)big;

        return Convert.ToInt64(value);
    }

    private static string NormalizeAnchorKey(string? value) => (value ?? string.Empty).Trim().ToLowerInvariant();

    private sealed class VirtualAnchorInsertStats
    {
        public int InsertedRows { get; init; }
        public int MarkedExistingRows { get; init; }
    }

    private async Task BulkAppendAsync(
        DuckDBConnection connection,
        IReadOnlyCollection<TensorConfig> rows,
        string label,
        CancellationToken ct)
    {
        // Emergency/small debug use only. Do NOT use for full search-space generation or trillion-scale pruning.
        // Insert only the ten tensor slot columns; DuckDB prediction columns intentionally remain NULL
        // until DuckDbPredictionMaterializationService scores/ranks the transient search space.
        if (rows.Count == 0)
            return;

        await ConfigureFastLoadSessionAsync(connection, ct);

        using var tx = connection.BeginTransaction();
        using var insert = connection.CreateCommand();
        insert.CommandText = $@"
INSERT INTO {TableName} ({CombinationDuckDbSchema.SlotColumnList})
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

        foreach (var row in rows)
        {
            ct.ThrowIfCancellationRequested();

            insert.Parameters.Clear();
            insert.Parameters.Add(new DuckDBParameter { Value = row.BaseQuant });
            insert.Parameters.Add(new DuckDBParameter { Value = row.Embeddings });
            insert.Parameters.Add(new DuckDBParameter { Value = row.LmHead });
            insert.Parameters.Add(new DuckDBParameter { Value = row.AttnQ });
            insert.Parameters.Add(new DuckDBParameter { Value = row.AttnKV });
            insert.Parameters.Add(new DuckDBParameter { Value = row.AttnOutput });
            insert.Parameters.Add(new DuckDBParameter { Value = row.FfnUpGate });
            insert.Parameters.Add(new DuckDBParameter { Value = row.FfnDown });
            insert.Parameters.Add(new DuckDBParameter { Value = row.MoeExperts });
            insert.Parameters.Add(new DuckDBParameter { Value = row.MoeRouter });

            await insert.ExecuteNonQueryAsync(ct);
        }

        tx.Commit();
    }

    private async Task<PredictionContext?> BuildPredictionContextAsync(
        RequiredSampleGenerationResult fullPlan,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var model = await db.AiModelHashes
            .FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);

        if (model == null)
            return null;

        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, model.Id, createIfMissing: false, ct);

        var pureQ8 = await LoadSnapshotByQuantAsync(
            db,
            model.Id,
            imatrixDefinitionId,
            HybridQuant.CreatePureBaseline(BaselineQuants.Q8_0),
            ct);

        if (pureQ8 == null)
            return null;

        var activeCombinationBaselines = RuntimeSearchSpace.GetActiveCombinationBaselines().ToList();

        var carrierBaseOnlyPlans = fullPlan.Plans
            .Where(x => x.Kind == RequiredSampleKind.BaseOnlyIsolation)
            .Where(x => x.TestedBaselineId.HasValue)
            .Where(x => x.Key.StartsWith("carrier-baseonly:", StringComparison.Ordinal) ||
                        x.Key.StartsWith("baseonly:", StringComparison.Ordinal))
            .Where(x =>
            {
                var baseline = BaselineQuants.FromId(x.TestedBaselineId!.Value);
                return baseline.IsCombinationCarrierCandidate;
            })
            .GroupBy(x => x.TestedBaselineId!.Value)
            .Select(g => g.First())
            .ToList();

        var loadedCarrierSnapshots = new List<(byte BaselineId, BenchmarkRow Snapshot)>();
        foreach (var plan in carrierBaseOnlyPlans)
        {
            var snap = await LoadSnapshotByQuantAsync(db, model.Id, imatrixDefinitionId, plan.Quant, ct);
            if (snap != null)
                loadedCarrierSnapshots.Add((plan.TestedBaselineId!.Value, snap));
        }

        ulong representativeCarrierBaseOnlySize = loadedCarrierSnapshots.Count > 0
            ? loadedCarrierSnapshots
                .OrderByDescending(x => BaselineQuants.FromId(x.BaselineId).BitRange)
                .ThenByDescending(x => BaselineQuants.FromId(x.BaselineId).ExplicitCandidateSortOrder)
                .Select(x => x.Snapshot.SizeBytes)
                .First()
            : pureQ8.SizeBytes;

        bool carrierBaseOnlyTruthCollapsed = loadedCarrierSnapshots.Count > 1 &&
                                             loadedCarrierSnapshots
                                                 .Select(x => x.Snapshot.SizeBytes)
                                                 .Distinct()
                                                 .Count() == 1;

        string? skipPureQ8PruneReason = null;
        if (carrierBaseOnlyTruthCollapsed && activeCombinationBaselines.Count == 1 && Config.ManualMaxPredictedSizeBytes <= 0)
        {
            var safeCarrier = activeCombinationBaselines[0];
            skipPureQ8PruneReason =
                $"Skipped pure-Q8 size pruning because base-carrier isolation truth collapsed across carriers and the search already resolved to the single deterministic safe carrier '{safeCarrier.Names[0]}'.";
        }

        var pureBaselineSizes = new Dictionary<byte, ulong>();
        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            var snap = await LoadSnapshotByQuantAsync(db, model.Id, imatrixDefinitionId, HybridQuant.CreatePureBaseline(baseline), ct);
            if (snap != null)
                pureBaselineSizes[baseline.UniqueId] = snap.SizeBytes;
        }

        pureBaselineSizes[BaselineQuants.Q8_0.UniqueId] = pureQ8.SizeBytes;

        var sizeByGroupAndCandidate = new Dictionary<(byte GroupId, byte CandidateId), ulong>();

        var groupPlans = fullPlan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolationProbe || x.Kind == RequiredSampleKind.GroupIsolationContinuation)
            .Where(x => x.TestedBaselineId == BaselineQuants.Q8_0.UniqueId)
            .ToList();

        foreach (var plan in groupPlans)
        {
            if (!plan.TargetGroupId.HasValue || !plan.TestedCandidateId.HasValue)
                continue;

            var snap = await LoadSnapshotByQuantAsync(db, model.Id, imatrixDefinitionId, plan.Quant, ct);
            if (snap == null)
                continue;

            sizeByGroupAndCandidate[(plan.TargetGroupId!.Value, plan.TestedCandidateId!.Value)] = snap.SizeBytes;
        }

        return new PredictionContext(
            pureQ8BaseSize: pureQ8.SizeBytes,
            pureBaselineSizes: pureBaselineSizes,
            carrierBaseOnlySize: representativeCarrierBaseOnlySize,
            sizesByGroupAndCandidate: sizeByGroupAndCandidate,
            shouldSkipPureQ8CeilingPruning: !string.IsNullOrWhiteSpace(skipPureQ8PruneReason),
            skipPureQ8CeilingReason: skipPureQ8PruneReason);
    }

    private static async Task<BenchmarkRow?> LoadSnapshotByQuantAsync(
        MagicQuantContext db,
        uint modelId,
        int? imatrixDefinitionId,
        HybridQuant quant,
        CancellationToken ct)
    {
        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();
        var lookup = (TensorConfig)quant;

        var row = await db.AiBenchmarks
            .Join(db.TensorCombos,
                b => b.TensorComboId,
                c => c.Id,
                (b, c) => new { b, c })
            .FirstOrDefaultAsync(x =>
                    x.b.ArchitectureFamilyId == architectureFamilyId &&
                    x.b.TensorGroupProfileId == tensorGroupProfileId &&
                    x.b.AiModelHashId == modelId &&
                    x.b.ImatrixDefinitionId == imatrixDefinitionId &&
                    x.c.BaseQuant == lookup.BaseQuant &&
                    x.c.Embeddings == lookup.Embeddings &&
                    x.c.LmHead == lookup.LmHead &&
                    x.c.AttnQ == lookup.AttnQ &&
                    x.c.AttnKV == lookup.AttnKV &&
                    x.c.AttnOutput == lookup.AttnOutput &&
                    x.c.FfnUpGate == lookup.FfnUpGate &&
                    x.c.FfnDown == lookup.FfnDown &&
                    x.c.MoeExperts == lookup.MoeExperts &&
                    x.c.MoeRouter == lookup.MoeRouter,
                ct);

        if (row == null)
            return null;

        return new BenchmarkRow { SizeBytes = row.b.SizeBytes };
    }

    private static async Task<BigInteger> InsertBaselineCombinationsSqlAsync(DuckDBConnection connection, BaselineQuants baseline, CancellationToken ct)
    {
        var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseline);
        if (allowed.Length != 9 || allowed.Any(x => x == null || x.Length == 0))
            throw new InvalidOperationException($"Invalid allowed candidate dimensions for baseline {baseline.Names[0]}.");
        await CreateTempDimensionTableAsync(connection, "temp_dim_embeddings", allowed[0], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_lm_head", allowed[1], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_attn_q", allowed[2], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_attn_kv", allowed[3], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_attn_output", allowed[4], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_ffn_up_gate", allowed[5], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_ffn_down", allowed[6], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_moe_experts", allowed[7], ct);
        await CreateTempDimensionTableAsync(connection, "temp_dim_moe_router", allowed[8], ct);
        using var cmd = connection.CreateCommand();
        cmd.CommandText = $@"
INSERT INTO {TableName} (BaseQuant,Embeddings,LmHead,AttnQ,AttnKV,AttnOutput,FfnUpGate,FfnDown,MoeExperts,MoeRouter)
SELECT CAST({baseline.UniqueId} AS UTINYINT), e.v, lh.v, aq.v, akv.v, ao.v, fu.v, fd.v, me.v, mr.v
FROM temp_dim_embeddings e
CROSS JOIN temp_dim_lm_head lh
CROSS JOIN temp_dim_attn_q aq
CROSS JOIN temp_dim_attn_kv akv
CROSS JOIN temp_dim_attn_output ao
CROSS JOIN temp_dim_ffn_up_gate fu
CROSS JOIN temp_dim_ffn_down fd
CROSS JOIN temp_dim_moe_experts me
CROSS JOIN temp_dim_moe_router mr;";
        await cmd.ExecuteNonQueryAsync(ct);
        return ProductOfDimensionLengths(allowed);
    }

    private static async Task BuildPredictedSizeLookupTablesAsync(DuckDBConnection connection, PredictionContext predictionContext, CancellationToken ct)
    {
        using (var create = connection.CreateCommand())
        {
            create.CommandText = @"DROP TABLE IF EXISTS temp_base_predicted_size;
DROP TABLE IF EXISTS temp_group_size_delta;
CREATE TEMP TABLE temp_base_predicted_size (BaseQuant UTINYINT, BaseSizeBytes UBIGINT);
CREATE TEMP TABLE temp_group_size_delta (BaseQuant UTINYINT, GroupName VARCHAR, StoredSlot UTINYINT, DeltaBytes BIGINT);";
            await create.ExecuteNonQueryAsync(ct);
        }
        string[] groupNames = ["Embeddings","LmHead","AttnQ","AttnKV","AttnOutput","FfnUpGate","FfnDown","MoeExperts","MoeRouter"];
        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            using (var b = connection.CreateCommand())
            {
                b.CommandText = $"INSERT INTO temp_base_predicted_size VALUES ({baseline.UniqueId}, {predictionContext.GetBaseSizeForSql(baseline.UniqueId)});";
                await b.ExecuteNonQueryAsync(ct);
            }
            var allowed = ComboLogic.GetAllowedCandidateIdsPerGroup(baseline);
            for (int i = 0; i < groupNames.Length; i++)
            foreach (byte slot in allowed[i])
            {
                long delta = predictionContext.GetRelativeSizeDeltaForSql((byte)(i + 1), baseline.UniqueId, slot);
                using var d = connection.CreateCommand();
                d.CommandText = $"INSERT INTO temp_group_size_delta VALUES ({baseline.UniqueId}, '{groupNames[i]}', {slot}, {delta});";
                await d.ExecuteNonQueryAsync(ct);
            }
        }
    }

    private sealed class BenchmarkRow
    {
        public ulong SizeBytes { get; set; }
    }

    private static string BuildValuesSql(IReadOnlyList<byte> values) =>
        $"(VALUES {string.Join(", ", values.Select(v => $"({v})"))}) AS t(v)";

    private static async Task CreateTempDimensionTableAsync(DuckDBConnection connection, string tableName, IReadOnlyList<byte> values, CancellationToken ct)
    {
        if (values.Count == 0)
            throw new InvalidOperationException($"Dimension {tableName} had zero candidates.");
        using var cmd = connection.CreateCommand();
        cmd.CommandText = $@"DROP TABLE IF EXISTS {tableName};
CREATE TEMP TABLE {tableName} AS
SELECT CAST(v AS UTINYINT) AS v
FROM {BuildValuesSql(values)};";
        await cmd.ExecuteNonQueryAsync(ct);
    }

    private static BigInteger ProductOfDimensionLengths(ImmutableArray<byte[]> allowed)
    {
        BigInteger product = BigInteger.One;
        foreach (var dim in allowed)
            product *= dim.Length;
        return product;
    }

    private sealed class PredictionContext
    {
        private readonly Dictionary<byte, ulong> _pureBaselineSizes;
        private readonly Dictionary<(byte GroupId, byte CandidateId), ulong> _sizesByGroupAndCandidate;

        public ulong PureQ8BaseSize { get; }
        public ulong CarrierBaseOnlySize { get; }
        public bool ShouldSkipPureQ8CeilingPruning { get; }
        public string? SkipPureQ8CeilingReason { get; }

        public PredictionContext(
            ulong pureQ8BaseSize,
            Dictionary<byte, ulong> pureBaselineSizes,
            ulong carrierBaseOnlySize,
            Dictionary<(byte GroupId, byte CandidateId), ulong> sizesByGroupAndCandidate,
            bool shouldSkipPureQ8CeilingPruning,
            string? skipPureQ8CeilingReason)
        {
            PureQ8BaseSize = pureQ8BaseSize;
            CarrierBaseOnlySize = carrierBaseOnlySize;
            ShouldSkipPureQ8CeilingPruning = shouldSkipPureQ8CeilingPruning;
            SkipPureQ8CeilingReason = skipPureQ8CeilingReason;
            _pureBaselineSizes = pureBaselineSizes;
            _sizesByGroupAndCandidate = sizesByGroupAndCandidate;
        }

        public ulong Predict(TensorConfig config)
        {
            long total = (long)GetBaseSizeForSql(config.BaseQuant);
            total += GetRelativeSizeDeltaForSql(TReg.Embeddings.UniqueId, config.BaseQuant, config.Embeddings);
            total += GetRelativeSizeDeltaForSql(TReg.LmHead.UniqueId, config.BaseQuant, config.LmHead);
            total += GetRelativeSizeDeltaForSql(TReg.AttnQ.UniqueId, config.BaseQuant, config.AttnQ);
            total += GetRelativeSizeDeltaForSql(TReg.AttnKV.UniqueId, config.BaseQuant, config.AttnKV);
            total += GetRelativeSizeDeltaForSql(TReg.AttnOutput.UniqueId, config.BaseQuant, config.AttnOutput);
            total += GetRelativeSizeDeltaForSql(TReg.FfnUpGate.UniqueId, config.BaseQuant, config.FfnUpGate);
            total += GetRelativeSizeDeltaForSql(TReg.FfnDown.UniqueId, config.BaseQuant, config.FfnDown);
            total += GetRelativeSizeDeltaForSql(TReg.MoeExperts.UniqueId, config.BaseQuant, config.MoeExperts);
            total += GetRelativeSizeDeltaForSql(TReg.MoeRouter.UniqueId, config.BaseQuant, config.MoeRouter);

            if (total < 0)
                total = 0;

            return (ulong)total;
        }

        public ulong GetBaseSizeForSql(byte baseQuant)
        {
            if (_pureBaselineSizes.TryGetValue(baseQuant, out var directBase))
                return directBase;

            if (TryGetDisabledSurrogateBaselineId(baseQuant, out var disabledSurrogateId) &&
                _pureBaselineSizes.ContainsKey(disabledSurrogateId))
            {
                /*
                 * Deprecated surrogate fallback, intentionally disabled:
                 *
                 * return _pureBaselineSizes[disabledSurrogateId];
                 *
                 * This made external/custom carriers inherit standard-family base size in the
                 * SQL pre-pruning path. The RankSafe materializer now requires exact external
                 * base-only truth, and this older helper should fail the same way.
                 */
                throw new InvalidOperationException(
                    $"Missing exact pure/base size for external baseline {FormatBaselineForSql(baseQuant)} (id '{baseQuant}'), " +
                    $"but disabled surrogate {FormatBaselineForSql(disabledSurrogateId)} (id '{disabledSurrogateId}') exists. " +
                    "SQL size prediction fallback is disabled to prevent external/custom collapse.");
            }

            throw new InvalidOperationException(
                $"Missing pure/base size for baseline {FormatBaselineForSql(baseQuant)} (id '{baseQuant}'). " +
                "SQL size prediction no longer falls back to Q8_0 because missing size truth should stop the run.");
        }

        public long GetRelativeSizeDeltaForSql(byte groupId, byte baseQuant, byte storedSlot)
        {
            if (storedSlot == 0)
                return 0;

            byte decoded = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(storedSlot);
            if (BaselineQuants.IsNativeExactAlias(decoded))
                return 0;

            if (decoded == baseQuant)
                return 0;

            ulong candidateSize = GetExactGroupSizeOrThrow(groupId, decoded, "candidate");
            ulong baseSize = GetExactGroupSizeOrThrow(groupId, baseQuant, "base");
            return (long)candidateSize - (long)baseSize;
        }

        private ulong GetExactGroupSizeOrThrow(byte groupId, byte baselineId, string role)
        {
            if (_sizesByGroupAndCandidate.TryGetValue((groupId, baselineId), out var exactSize))
                return exactSize;

            if (TryGetDisabledSurrogateBaselineId(baselineId, out var disabledSurrogateId) &&
                _sizesByGroupAndCandidate.ContainsKey((groupId, disabledSurrogateId)))
            {
                /*
                 * Deprecated surrogate fallback, intentionally disabled:
                 *
                 * return _sizesByGroupAndCandidate[(groupId, disabledSurrogateId)];
                 *
                 * Group-size deltas must be based on the exact runtime baseline id. Re-enabling
                 * this would collapse external/custom group assignments into their standard
                 * family before DuckDB ranking ever sees them.
                 */
                throw new InvalidOperationException(
                    $"Missing exact group-size isolation for {role} baseline {FormatBaselineForSql(baselineId)} (id '{baselineId}') " +
                    $"in tensor group id '{groupId}', but disabled surrogate {FormatBaselineForSql(disabledSurrogateId)} (id '{disabledSurrogateId}') exists. " +
                    "Regenerate the exact isolated sample instead of using SQL size fallback.");
            }

            throw new InvalidOperationException(
                $"Missing group-size isolation for {role} baseline {FormatBaselineForSql(baselineId)} (id '{baselineId}') in tensor group id '{groupId}'. " +
                "SQL size prediction no longer returns zero for missing isolation truth.");
        }

        private static bool TryGetDisabledSurrogateBaselineId(byte baselineId, out byte surrogateBaselineId)
        {
            surrogateBaselineId = baselineId;

            if (BaselineQuants.IsNativeExactAlias(baselineId))
                return false;

            var baseline = BaselineQuants.FromId(baselineId);
            if (!baseline.IsExternalRepositoryBaseline)
                return false;

            var builtIn = BaselineQuants.ResolveBuiltInStandardBaseline(baseline.QuantizeBaseArgumentName)
                          ?? BaselineQuants.ResolveBuiltInStandardBaseline(baseline.Names[0]);

            if (builtIn == null || builtIn.UniqueId == baselineId)
                return false;

            surrogateBaselineId = builtIn.UniqueId;
            return true;
        }

        private static string FormatBaselineForSql(byte baselineId)
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
    }
}