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

    public async Task<long> GetRemainingCombinationCountAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);

        var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {TableName};";

        return (long)(await cmd.ExecuteScalarAsync(ct) ?? 0L);
    }

    public async Task<List<TensorConfig>> GetRemainingTensorConfigsAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);

        var results = new List<TensorConfig>();

        var cmd = connection.CreateCommand();
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
                baseQuant:  Convert.ToByte(reader.GetValue(0)),
                embeddings: Convert.ToByte(reader.GetValue(1)),
                lmHead:     Convert.ToByte(reader.GetValue(2)),
                attnQ:      Convert.ToByte(reader.GetValue(3)),
                attnKV:     Convert.ToByte(reader.GetValue(4)),
                attnOutput: Convert.ToByte(reader.GetValue(5)),
                ffnUpGate:  Convert.ToByte(reader.GetValue(6)),
                ffnDown:    Convert.ToByte(reader.GetValue(7)),
                moeExperts: Convert.ToByte(reader.GetValue(8)),
                moeRouter:  Convert.ToByte(reader.GetValue(9))
            ));
        }

        return results;
    }
    
    private static string GetDuckDbDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Cache.ModelMagicQuantDirectory;

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Cache.MagicQuantDirectory;

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

        BigInteger expectedTotal = ComboCounter.CountAll();
        long currentDbCount = await GetRowCountAsync(connection, ct);

        AnsiConsole.MarkupLine(
            $"[bold]DuckDB Check:[/] Current Rows: [cyan]{currentDbCount:N0}[/] | Expected: [yellow]{expectedTotal:N0}[/]");

        if (forceRebuild || currentDbCount != expectedTotal)
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

        var predictionContext = await BuildPredictionContextAsync(fullPlan, ct);

        if (predictionContext == null)
        {
            AnsiConsole.MarkupLine("[yellow]Predicted-size pruning skipped: prediction context was incomplete.[/]");
            return 0;
        }

        var rows = new List<TensorConfig>();

        var select = connection.CreateCommand();
        select.CommandText = $@"
            SELECT BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter
            FROM {TableName};";

        using (var reader = await select.ExecuteReaderAsync(ct))
        {
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

        foreach (var row in rows)
        {
            ulong predicted = predictionContext.Predict(row);
            if (predicted <= predictionContext.PureQ8BaseSize)
                kept.Add(row);
        }

        long removed = rows.Count - kept.Count;

        if (removed <= 0)
        {
            AnsiConsole.MarkupLine("[green]Predicted-size pruning removed 0 combinations.[/]");
            return 0;
        }

        var createCmd = connection.CreateCommand();
        createCmd.CommandText = $@"
            DROP TABLE IF EXISTS {TableName};
            CREATE TABLE {TableName} (
                BaseQuant TINYINT,
                Embeddings TINYINT,
                LmHead TINYINT,
                AttnQ TINYINT,
                AttnKV TINYINT,
                AttnOutput TINYINT,
                FfnUpGate TINYINT,
                FfnDown TINYINT,
                MoeExperts TINYINT,
                MoeRouter TINYINT
            );";
        await createCmd.ExecuteNonQueryAsync(ct);

        await BulkInsertAsync(connection, kept, ct);

        AnsiConsole.MarkupLine($"[yellow]Predicted-size pruning removed:[/] [red]{removed:N0}[/] combo(s) larger than pure Q8.");
        return removed;
    }


    public async Task<long> PruneHighPrecisionHybridCandidatesAsync(CancellationToken ct = default)
    {
        if (RuntimeSearchSpace.AllowHighPrecisionHybrids)
            return 0;

        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);

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

        var createCmd = connection.CreateCommand();
        createCmd.CommandText = $@"
            DROP TABLE IF EXISTS {TableName};
            CREATE TABLE {TableName} (
                BaseQuant TINYINT, Embeddings TINYINT, LmHead TINYINT, AttnQ TINYINT, AttnKV TINYINT,
                AttnOutput TINYINT, FfnUpGate TINYINT, FfnDown TINYINT, MoeExperts TINYINT, MoeRouter TINYINT
            );";
        await createCmd.ExecuteNonQueryAsync(ct);
        await BulkInsertAsync(connection, kept, ct);
        return removed;
    }

    private async Task<long> GetRowCountAsync(DuckDBConnection connection, CancellationToken ct)
    {
        var checkCmd = connection.CreateCommand();
        checkCmd.CommandText = $"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{TableName}'";
        var exists = (long)(await checkCmd.ExecuteScalarAsync(ct) ?? 0);

        if (exists == 0)
            return -1;

        var countCmd = connection.CreateCommand();
        countCmd.CommandText = $"SELECT COUNT(*) FROM {TableName}";
        return (long)(await countCmd.ExecuteScalarAsync(ct) ?? 0);
    }

    private async Task RebuildDatabaseAsync(
        DuckDBConnection connection,
        BigInteger expectedTotal,
        CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();

        var createCmd = connection.CreateCommand();
        createCmd.CommandText = $@"
            DROP TABLE IF EXISTS {TableName};
            CREATE TABLE {TableName} (
                BaseQuant TINYINT,
                Embeddings TINYINT,
                LmHead TINYINT,
                AttnQ TINYINT,
                AttnKV TINYINT,
                AttnOutput TINYINT,
                FfnUpGate TINYINT,
                FfnDown TINYINT,
                MoeExperts TINYINT,
                MoeRouter TINYINT
            );";
        await createCmd.ExecuteNonQueryAsync(ct);

        long insertedTotal = 0;
        var bases = RuntimeSearchSpace.GetActiveCombinationBaselines();

        AnsiConsole.MarkupLine($"Starting bulk insert of {expectedTotal:N0} rows...");

        foreach (var baseline in bases)
        {
            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(baseline, ct: ct))
            {
                await BulkInsertAsync(connection, batch, ct);
                insertedTotal += batch.Count;
                AnsiConsole.MarkupLine($"  Inserted batch... Total so far: {insertedTotal:N0}");
            }
        }

        sw.Stop();
        AnsiConsole.MarkupLine($"DuckDB rebuild complete! in {sw.Elapsed.TotalSeconds:F2}s");
    }

    private async Task BulkInsertAsync(
        DuckDBConnection connection,
        IReadOnlyCollection<TensorConfig> rows,
        CancellationToken ct)
    {
        if (rows.Count == 0)
            return;

        using var tx = connection.BeginTransaction();

        foreach (var row in rows)
        {
            var cmd = connection.CreateCommand();
            cmd.Transaction = tx;
            cmd.CommandText = $@"
                INSERT INTO {TableName}
                (BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

            cmd.Parameters.Add(new DuckDBParameter { Value = row.BaseQuant });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.Embeddings });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.LmHead });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.AttnQ });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.AttnKV });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.AttnOutput });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.FfnUpGate });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.FfnDown });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.MoeExperts });
            cmd.Parameters.Add(new DuckDBParameter { Value = row.MoeRouter });

            await cmd.ExecuteNonQueryAsync(ct);
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

        var carrierBaseOnlyPlan = fullPlan.Plans.FirstOrDefault(x =>
            x.Kind == RequiredSampleKind.BaseOnlyIsolation &&
            x.TestedBaselineId == BaselineQuants.Q8_0.UniqueId &&
            x.Key.StartsWith("carrier-baseonly:", StringComparison.Ordinal));

        if (pureQ8 == null || carrierBaseOnlyPlan == null)
            return null;

        var carrier = await LoadSnapshotByQuantAsync(db, model.Id, imatrixDefinitionId, carrierBaseOnlyPlan.Quant, ct);
        if (carrier == null)
            return null;

        var deltaByGroupAndCandidate = new Dictionary<(byte GroupId, byte CandidateId), long>();

        var groupPlans = fullPlan.Plans
            .Where(x => x.Kind == RequiredSampleKind.GroupIsolationProbe || x.Kind == RequiredSampleKind.GroupIsolationContinuation)
            .Where(x => x.TestedBaselineId == BaselineQuants.Q8_0.UniqueId)
            .ToList();

        foreach (var plan in groupPlans)
        {
            if (!plan.TargetGroupId.HasValue || !plan.TestedSchemeId.HasValue)
                continue;

            var snap = await LoadSnapshotByQuantAsync(db, model.Id, imatrixDefinitionId, plan.Quant, ct);
            if (snap == null)
                continue;

            long delta = (long)snap.SizeBytes - (long)carrier.SizeBytes;
            deltaByGroupAndCandidate[(plan.TargetGroupId.Value, plan.TestedSchemeId.Value)] = delta;
        }

        return new PredictionContext(
            pureQ8BaseSize: pureQ8.SizeBytes,
            carrierBaseOnlySize: carrier.SizeBytes,
            deltas: deltaByGroupAndCandidate);
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

    private sealed class PredictionContext
    {
        private readonly Dictionary<(byte GroupId, byte CandidateId), long> _deltas;

        public ulong PureQ8BaseSize { get; }
        public ulong CarrierBaseOnlySize { get; }

        public PredictionContext(
            ulong pureQ8BaseSize,
            ulong carrierBaseOnlySize,
            Dictionary<(byte GroupId, byte CandidateId), long> deltas)
        {
            PureQ8BaseSize = pureQ8BaseSize;
            CarrierBaseOnlySize = carrierBaseOnlySize;
            _deltas = deltas;
        }

        public ulong Predict(TensorConfig config)
        {
            long total = (long)CarrierBaseOnlySize;

            AddDelta(TReg.Embeddings.UniqueId, config.Embeddings, ref total);
            AddDelta(TReg.LmHead.UniqueId, config.LmHead, ref total);
            AddDelta(TReg.AttnQ.UniqueId, config.AttnQ, ref total);
            AddDelta(TReg.AttnKV.UniqueId, config.AttnKV, ref total);
            AddDelta(TReg.AttnOutput.UniqueId, config.AttnOutput, ref total);
            AddDelta(TReg.FfnUpGate.UniqueId, config.FfnUpGate, ref total);
            AddDelta(TReg.FfnDown.UniqueId, config.FfnDown, ref total);
            AddDelta(TReg.MoeExperts.UniqueId, config.MoeExperts, ref total);
            AddDelta(TReg.MoeRouter.UniqueId, config.MoeRouter, ref total);

            if (total < 0)
                total = 0;

            return (ulong)total;
        }

        private void AddDelta(byte groupId, byte candidateId, ref long total)
        {
            if (candidateId == BaselineQuants.BF16_Hybrid.UniqueId || candidateId == BaselineQuants.F16_Hybrid.UniqueId)
                return;

            if (_deltas.TryGetValue((groupId, candidateId), out long delta))
                total += delta;
        }
    }
}