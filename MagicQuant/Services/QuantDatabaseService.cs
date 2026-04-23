using System.Diagnostics;
using System.Numerics;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public class QuantDatabaseService
{
    private const string DbFileNamePrefix = "MagicQuant_Combinations";
    private const string TableName = "tensor_configs";

    // Keep this moderate so generation still yields often enough for progress.
    private const int GeneratorBatchSize = 250_000;

    // Appender heartbeat. Lower = chattier.
    private const long InsertProgressLogEveryRows = 50_000;

    private static readonly string[] ExpectedColumnTypes =
    [
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint",
        "utinyint"
    ];

    private static string CreateTableSql => $@"
        DROP TABLE IF EXISTS {TableName};
        CREATE TABLE {TableName} (
            BaseQuant UTINYINT,
            Embeddings UTINYINT,
            LmHead UTINYINT,
            AttnQ UTINYINT,
            AttnKV UTINYINT,
            AttnOutput UTINYINT,
            FfnUpGate UTINYINT,
            FfnDown UTINYINT,
            MoeExperts UTINYINT,
            MoeRouter UTINYINT
        );";

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

        return (long)(await cmd.ExecuteScalarAsync(ct) ?? 0L);
    }

    public async Task<List<TensorConfig>> GetRemainingTensorConfigsAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

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

        AnsiConsole.MarkupLine(
            $"[bold]DuckDB Check:[/] Current Rows: [cyan]{currentDbCount:N0}[/] | Expected: [yellow]{expectedTotal:N0}[/]");

        if (!tableShapeOk)
            AnsiConsole.MarkupLine("[yellow]DuckDB table shape is missing or stale. Rebuild required.[/]");

        if (forceRebuild || !tableShapeOk || currentDbCount != expectedTotal)
        {
            AnsiConsole.MarkupLine("[bold red]DuckDB empty, mismatch, forced, or stale.[/] Initializing/Rebuilding...");
            await RebuildDatabaseAsync(connection, expectedTotal, ct);
        }
        else
        {
            AnsiConsole.MarkupLine("[bold green]DuckDB is synchronized and ready.[/]");
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

        var rows = new List<TensorConfig>();

        using (var select = connection.CreateCommand())
        {
            select.CommandText = $@"
                SELECT BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter
                FROM {TableName};";

            using var reader = await select.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct))
            {
                rows.Add(new TensorConfig(
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
        }

        var kept = new List<TensorConfig>(rows.Count);
        var predictedByBase = new Dictionary<byte, List<ulong>>();

        ulong sizeCeilingBytes = Config.ManualMaxPredictedSizeBytes > 0
            ? Config.ManualMaxPredictedSizeBytes
            : predictionContext.PureQ8BaseSize;

        foreach (var row in rows)
        {
            ulong predicted = predictionContext.Predict(row);

            if (!predictedByBase.TryGetValue(row.BaseQuant, out var bucket))
            {
                bucket = new List<ulong>();
                predictedByBase[row.BaseQuant] = bucket;
            }

            bucket.Add(predicted);

            if (predicted <= sizeCeilingBytes)
                kept.Add(row);
        }

        if (predictedByBase.Count > 0)
        {
            ulong globalMin = predictedByBase.Values.SelectMany(x => x).Min();
            ulong globalMax = predictedByBase.Values.SelectMany(x => x).Max();
            AnsiConsole.MarkupLine($"[grey]Stage-1 predicted size spread:[/] [cyan]{globalMin / 1024d / 1024d / 1024d:F2}[/] [grey]GB ..[/] [cyan]{globalMax / 1024d / 1024d / 1024d:F2}[/] [grey]GB[/]");

            foreach (var kv in predictedByBase.OrderBy(x => BaselineQuants.FromId(x.Key).BitRange).ThenBy(x => x.Key))
            {
                var baseline = BaselineQuants.FromId(kv.Key);
                ulong min = kv.Value.Min();
                ulong max = kv.Value.Max();
                AnsiConsole.MarkupLine(
                    $"[grey]Stage-1 base {Markup.Escape(baseline.Names[0])} (BitRange {baseline.BitRange}) ->[/] [cyan]{kv.Value.Count:N0}[/] [grey]candidate(s),[/] [cyan]{min / 1024d / 1024d / 1024d:F2}[/] [grey]GB ..[/] [cyan]{max / 1024d / 1024d / 1024d:F2}[/] [grey]GB[/]");
            }
        }

        long removed = rows.Count - kept.Count;

        if (removed <= 0)
        {
            AnsiConsole.MarkupLine("[green]Predicted-size pruning removed 0 combinations.[/]");
            return 0;
        }

        await RecreateTableAsync(connection, ct);
        await BulkAppendAsync(connection, kept, "predicted-size-prune", ct);

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

        var rows = await GetRemainingTensorConfigsAsync(ct);
        var kept = rows.Where(x =>
            x.Embeddings != BaselineQuants.BF16_Hybrid.UniqueId && x.Embeddings != BaselineQuants.F16_Hybrid.UniqueId &&
            x.LmHead != BaselineQuants.BF16_Hybrid.UniqueId && x.LmHead != BaselineQuants.F16_Hybrid.UniqueId &&
            x.AttnQ != BaselineQuants.BF16_Hybrid.UniqueId && x.AttnQ != BaselineQuants.F16_Hybrid.UniqueId &&
            x.AttnKV != BaselineQuants.BF16_Hybrid.UniqueId && x.AttnKV != BaselineQuants.F16_Hybrid.UniqueId &&
            x.AttnOutput != BaselineQuants.BF16_Hybrid.UniqueId && x.AttnOutput != BaselineQuants.F16_Hybrid.UniqueId &&
            x.FfnUpGate != BaselineQuants.BF16_Hybrid.UniqueId && x.FfnUpGate != BaselineQuants.F16_Hybrid.UniqueId &&
            x.FfnDown != BaselineQuants.BF16_Hybrid.UniqueId && x.FfnDown != BaselineQuants.F16_Hybrid.UniqueId &&
            x.MoeExperts != BaselineQuants.BF16_Hybrid.UniqueId && x.MoeExperts != BaselineQuants.F16_Hybrid.UniqueId &&
            x.MoeRouter != BaselineQuants.BF16_Hybrid.UniqueId && x.MoeRouter != BaselineQuants.F16_Hybrid.UniqueId).ToList();

        long removed = rows.Count - kept.Count;
        if (removed <= 0)
            return 0;

        await RecreateTableAsync(connection, ct);
        await BulkAppendAsync(connection, kept, "high-precision-prune", ct);

        return removed;
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
        return (long)(await countCmd.ExecuteScalarAsync(ct) ?? 0L);
    }

    private async Task RebuildDatabaseAsync(
        DuckDBConnection connection,
        BigInteger expectedTotal,
        CancellationToken ct)
    {
        long totalTarget = (long)expectedTotal;

        AnsiConsole.MarkupLine($"[yellow]Starting bulk insert of {totalTarget:N0} rows...[/]");
        AnsiConsole.MarkupLine(
            $"[grey]Generator batch size:[/] {GeneratorBatchSize:N0}  [grey]| Appender heartbeat:[/] every {InsertProgressLogEveryRows:N0} rows");

        await RecreateTableAsync(connection, ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        long insertedGrandTotal = 0;
        var overallSw = Stopwatch.StartNew();

        using DuckDBAppender appender = connection.CreateAppender(TableName);

        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines())
        {
            ct.ThrowIfCancellationRequested();

            long baseInserted = 0;
            int baseBatchNumber = 0;
            var baseSw = Stopwatch.StartNew();
            string baseName = baseline.Names.FirstOrDefault() ?? baseline.UniqueId.ToString();

            AnsiConsole.MarkupLine($"[cyan]Generating + inserting base:[/] [bold]{Markup.Escape(baseName)}[/]");

            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(baseline, batchSize: GeneratorBatchSize))
            {
                ct.ThrowIfCancellationRequested();

                baseBatchNumber++;
                int batchCount = batch.Count;
                var batchSw = Stopwatch.StartNew();

                AnsiConsole.MarkupLine(
                    $"  [grey]Base batch #{baseBatchNumber} generated:[/] {batchCount:N0} rows  [grey]| Base inserted before batch:[/] {baseInserted:N0}");

                var progress = new InsertProgress
                {
                    InsertedTotal = insertedGrandTotal,
                    LastLoggedTotal = insertedGrandTotal,
                    ProgressLogEveryRows = InsertProgressLogEveryRows,
                    TotalTarget = totalTarget,
                    BaseName = baseName,
                    BatchNumber = baseBatchNumber
                };

                AppendRows(appender, batch, progress, overallSw, ct);

                insertedGrandTotal = progress.InsertedTotal;
                baseInserted += batchCount;

                batchSw.Stop();

                double grandPct = totalTarget == 0 ? 100d : insertedGrandTotal * 100d / totalTarget;

                AnsiConsole.MarkupLine(
                    $"  [green]Base batch #{baseBatchNumber} done:[/] {batchCount:N0} rows in {batchSw.Elapsed.TotalSeconds:N1}s  " +
                    $"[grey]| Base running:[/] {baseInserted:N0}  [grey]| Grand total:[/] {insertedGrandTotal:N0}/{totalTarget:N0} ({grandPct:N2}%)");

                batch.Clear();
            }

            baseSw.Stop();

            double rowsPerSec = baseSw.Elapsed.TotalSeconds <= 0
                ? 0
                : baseInserted / baseSw.Elapsed.TotalSeconds;

            AnsiConsole.MarkupLine(
                $"[bold green]Base complete:[/] {Markup.Escape(baseName)}  " +
                $"[grey]| Inserted:[/] {baseInserted:N0} rows  " +
                $"[grey]| Time:[/] {baseSw.Elapsed.TotalMinutes:N2} min  " +
                $"[grey]| Rate:[/] {rowsPerSec:N0} rows/sec");
        }

        appender.Close();
        overallSw.Stop();

        long finalCount = await GetRowCountAsync(connection, ct);

        double finalRate = overallSw.Elapsed.TotalSeconds <= 0
            ? 0
            : insertedGrandTotal / overallSw.Elapsed.TotalSeconds;

        AnsiConsole.MarkupLine(
            $"[bold green]DuckDB rebuild complete.[/] " +
            $"[grey]| Inserted tracked:[/] {insertedGrandTotal:N0}  " +
            $"[grey]| Final row count:[/] {finalCount:N0}  " +
            $"[grey]| Time:[/] {overallSw.Elapsed.TotalMinutes:N2} min  " +
            $"[grey]| Avg rate:[/] {finalRate:N0} rows/sec");
    }

    private async Task BulkAppendAsync(
        DuckDBConnection connection,
        IReadOnlyCollection<TensorConfig> rows,
        string label,
        CancellationToken ct)
    {
        if (rows.Count == 0)
            return;

        await ConfigureFastLoadSessionAsync(connection, ct);

        using DuckDBAppender appender = connection.CreateAppender(TableName);

        var progress = new InsertProgress
        {
            InsertedTotal = 0,
            LastLoggedTotal = 0,
            ProgressLogEveryRows = InsertProgressLogEveryRows,
            TotalTarget = rows.Count,
            BaseName = label,
            BatchNumber = 1
        };

        AppendRows(appender, rows, progress, Stopwatch.StartNew(), ct);
        appender.Close();
    }

    private static void AppendRows(
        DuckDBAppender appender,
        IReadOnlyCollection<TensorConfig> rows,
        InsertProgress progress,
        Stopwatch overallSw,
        CancellationToken ct)
    {
        foreach (var row in rows)
        {
            ct.ThrowIfCancellationRequested();

            appender.CreateRow()
                .AppendValue(row.BaseQuant)
                .AppendValue(row.Embeddings)
                .AppendValue(row.LmHead)
                .AppendValue(row.AttnQ)
                .AppendValue(row.AttnKV)
                .AppendValue(row.AttnOutput)
                .AppendValue(row.FfnUpGate)
                .AppendValue(row.FfnDown)
                .AppendValue(row.MoeExperts)
                .AppendValue(row.MoeRouter)
                .EndRow();

            progress.InsertedTotal++;

            if (progress.InsertedTotal - progress.LastLoggedTotal >= progress.ProgressLogEveryRows)
            {
                double elapsedSeconds = Math.Max(0.001, overallSw.Elapsed.TotalSeconds);
                double rowsPerSecond = progress.InsertedTotal / elapsedSeconds;
                double pct = progress.TotalTarget <= 0 ? 100d : progress.InsertedTotal * 100d / progress.TotalTarget;

                long remaining = Math.Max(0, progress.TotalTarget - progress.InsertedTotal);
                double etaSeconds = rowsPerSecond <= 0 ? 0 : remaining / rowsPerSecond;
                var eta = TimeSpan.FromSeconds(etaSeconds);

                AnsiConsole.MarkupLine(
                    $"    [grey]Progress[/] [green]{progress.InsertedTotal:N0}[/]/[yellow]{progress.TotalTarget:N0}[/] " +
                    $"({pct:N2}%)  [grey]| Rate:[/] {rowsPerSecond:N0}/sec  " +
                    $"[grey]| ETA:[/] {eta:hh\\:mm\\:ss}  " +
                    $"[grey]| Label:[/] {Markup.Escape(progress.BaseName)}  " +
                    $"[grey]| Batch:[/] {progress.BatchNumber}");

                progress.LastLoggedTotal = progress.InsertedTotal;
            }
        }
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
        var lookup = (TensorConfig)quant;

        var row = await db.AiBenchmarks
            .Join(db.TensorCombos,
                b => b.TensorComboId,
                c => c.Id,
                (b, c) => new { b, c })
            .FirstOrDefaultAsync(x =>
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

    private sealed class BenchmarkRow
    {
        public ulong SizeBytes { get; set; }
    }

    private sealed class InsertProgress
    {
        public long InsertedTotal { get; set; }
        public long LastLoggedTotal { get; set; }
        public long ProgressLogEveryRows { get; set; }
        public long TotalTarget { get; set; }
        public string BaseName { get; set; } = string.Empty;
        public int BatchNumber { get; set; }
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
            byte normalizedBaseId = NormalizeBaselineIdForIsolation(config.BaseQuant);
            long total = (long)(_pureBaselineSizes.TryGetValue(config.BaseQuant, out var directBase)
                ? directBase
                : _pureBaselineSizes.TryGetValue(normalizedBaseId, out var normalizedBase)
                    ? normalizedBase
                    : PureQ8BaseSize);

            ApplyRelativeDelta(TReg.Embeddings.UniqueId, normalizedBaseId, config.Embeddings, ref total);
            ApplyRelativeDelta(TReg.LmHead.UniqueId, normalizedBaseId, config.LmHead, ref total);
            ApplyRelativeDelta(TReg.AttnQ.UniqueId, normalizedBaseId, config.AttnQ, ref total);
            ApplyRelativeDelta(TReg.AttnKV.UniqueId, normalizedBaseId, config.AttnKV, ref total);
            ApplyRelativeDelta(TReg.AttnOutput.UniqueId, normalizedBaseId, config.AttnOutput, ref total);
            ApplyRelativeDelta(TReg.FfnUpGate.UniqueId, normalizedBaseId, config.FfnUpGate, ref total);
            ApplyRelativeDelta(TReg.FfnDown.UniqueId, normalizedBaseId, config.FfnDown, ref total);
            ApplyRelativeDelta(TReg.MoeExperts.UniqueId, normalizedBaseId, config.MoeExperts, ref total);
            ApplyRelativeDelta(TReg.MoeRouter.UniqueId, normalizedBaseId, config.MoeRouter, ref total);

            if (total < 0)
                total = 0;

            return (ulong)total;
        }

        private void ApplyRelativeDelta(byte groupId, byte baseCandidateId, byte candidateId, ref long total)
        {
            if (BaselineQuants.IsNullTensorConfigGroupSlot(candidateId) ||
                candidateId == BaselineQuants.BF16_Hybrid.UniqueId ||
                candidateId == BaselineQuants.F16_Hybrid.UniqueId)
                return;

            byte normalizedCandidateId = NormalizeBaselineIdForIsolation(candidateId);
            if (normalizedCandidateId == baseCandidateId)
                return;

            if (!_sizesByGroupAndCandidate.TryGetValue((groupId, normalizedCandidateId), out var candidateSize))
                return;

            if (!_sizesByGroupAndCandidate.TryGetValue((groupId, baseCandidateId), out var baseSize))
                return;

            total += (long)candidateSize - (long)baseSize;
        }

        private static byte NormalizeBaselineIdForIsolation(byte baselineId)
        {
            var baseline = BaselineQuants.FromId(baselineId);
            if (!baseline.IsExternalRepositoryBaseline)
                return baselineId;

            var builtIn = BaselineQuants.ResolveBuiltInStandardBaseline(baseline.QuantizeBaseArgumentName)
                          ?? BaselineQuants.ResolveBuiltInStandardBaseline(baseline.Names[0]);

            return builtIn?.UniqueId ?? baselineId;
        }
    }
}