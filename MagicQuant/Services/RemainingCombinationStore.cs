using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using System.Runtime.CompilerServices;

namespace MagicQuant.Services;

public sealed class RemainingCombinationStore
{
    private const string DbFileNamePrefix = "MagicQuant_Combinations";
    private const string TableName = CombinationDuckDbSchema.TableName;

    private static string ConnectionString => $"Data Source={Path.Combine(GetDuckDbDirectory(), BuildContextAwareDuckDbFileName())}";

    public string GetDatabaseFilePath() => Path.Combine(GetDuckDbDirectory(), BuildContextAwareDuckDbFileName());

    public async Task<long> CountAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {TableName};";
        return Convert.ToInt64(await cmd.ExecuteScalarAsync(ct) ?? 0L);
    }

    public async Task<List<TensorConfig>> LoadAllAsync(CancellationToken ct = default)
    {
        long count = await CountAsync(ct);
        if (count > Config.MaxInMemoryCombinationLoadRows)
            throw new InvalidOperationException($"Refusing to load {count:N0} DuckDB tensor configs into memory. Use SQL-native filtering/streaming instead.");

        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        var results = new List<TensorConfig>();

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList}
FROM {TableName}
ORDER BY {CombinationDuckDbSchema.SlotColumnList};";

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            results.Add(ReadTensorConfig(reader));

        return results;
    }

    public async IAsyncEnumerable<TensorConfig> StreamAsync(
        string? whereSql = null,
        string? orderBySql = null,
        long? limit = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        string sql = $@"SELECT {CombinationDuckDbSchema.SlotColumnList} FROM {TableName}";
        if (!string.IsNullOrWhiteSpace(whereSql))
            sql += $" WHERE {whereSql}";
        if (!string.IsNullOrWhiteSpace(orderBySql))
            sql += $" ORDER BY {orderBySql}";
        if (limit.HasValue)
            sql += $" LIMIT {limit.Value}";

        using var cmd = connection.CreateCommand();
        cmd.CommandText = sql;

        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
            yield return ReadTensorConfig(reader);
    }

    public async Task ReplaceAllAsync(
        IReadOnlyCollection<TensorConfig> configs,
        string reason,
        CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);
        await RecreateTableAsync(connection, ct);

        using var tx = connection.BeginTransaction();
        using var insert = connection.CreateCommand();
        insert.CommandText = $@"
INSERT INTO {TableName}
({CombinationDuckDbSchema.SlotColumnList})
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

        foreach (var config in configs)
        {
            insert.Parameters.Clear();
            AddSlotParameters(insert, config);
            await insert.ExecuteNonQueryAsync(ct);
        }

        tx.Commit();
    }

    public async Task<PredictionMaterializationStatus> GetPredictionStatusAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $@"
SELECT
    COUNT(*) AS TotalRows,
    COUNT(PredictedKld) AS PredictedRows,
    COUNT(PredictionRank) AS RankedRows,
    MIN(PredictedKld) AS MinPredictedKld,
    MAX(PredictedKld) AS MaxPredictedKld,
    MIN(PredictedSizeBytes) AS MinPredictedSizeBytes,
    MAX(PredictedSizeBytes) AS MaxPredictedSizeBytes
FROM {TableName};";

        using var r = await cmd.ExecuteReaderAsync(ct);
        await r.ReadAsync(ct);

        long total = Convert.ToInt64(r.GetValue(0));
        long predicted = Convert.ToInt64(r.GetValue(1));
        long ranked = Convert.ToInt64(r.GetValue(2));

        return new PredictionMaterializationStatus
        {
            TotalRows = total,
            PredictedRows = predicted,
            MissingPredictionRows = Math.Max(0, total - predicted),
            RankedRows = ranked,
            MinPredictedKld = r.IsDBNull(3) ? null : Convert.ToDouble(r.GetValue(3)),
            MaxPredictedKld = r.IsDBNull(4) ? null : Convert.ToDouble(r.GetValue(4)),
            MinPredictedSizeBytes = r.IsDBNull(5) ? null : Convert.ToUInt64(r.GetValue(5)),
            MaxPredictedSizeBytes = r.IsDBNull(6) ? null : Convert.ToUInt64(r.GetValue(6))
        };
    }

    public async Task<IReadOnlyList<RankSafePredictionRow>> QueryStrictDominanceCandidatesAsync(
        BenchmarkSnapshotRecord anchor,
        int limit,
        CancellationToken ct = default)
    {
        string sql = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       PredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank
FROM {TableName}
WHERE PredictedKld IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {CombinationDuckDbSchema.HybridPredicateSql}
  AND PredictedSizeBytes <= ?
  AND PredictedKld + ? < ?
ORDER BY PredictedSizeBytes ASC,
         PredictedKld ASC,
         PredictionRank ASC,
         PredictionConfidence DESC
LIMIT ?;";

        return await QueryPredictedRowsAsync(
            sql,
            new object[] { anchor.SizeBytes, Config.SelectionMinimumKldImprovementEpsilon, anchor.Kld, limit },
            ct);
    }

    public async Task<IReadOnlyList<HybridSelectionCandidate>> QueryBetterThanLinearCandidatesAsync(
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        ulong minSize,
        ulong maxSize,
        HybridSelectionReason reason,
        string windowLabel,
        int limit,
        CancellationToken ct = default)
    {
        string sql = $@"
WITH scored AS (
    SELECT {CombinationDuckDbSchema.SlotColumnList},
           PredictedKld,
           PredictedSizeBytes,
           PredictionConfidence,
           PredictionRank,
           (CAST(? AS DOUBLE)
             + ((CAST(PredictedSizeBytes AS DOUBLE) - CAST(? AS DOUBLE)) / GREATEST(CAST(? AS DOUBLE), 1.0))
             * (CAST(? AS DOUBLE) - CAST(? AS DOUBLE))) AS LinearExpectedKld
    FROM {TableName}
    WHERE PredictedKld IS NOT NULL
      AND PredictedSizeBytes IS NOT NULL
      AND PredictionRank IS NOT NULL
      AND {CombinationDuckDbSchema.HybridPredicateSql}
      AND PredictedSizeBytes BETWEEN ? AND ?
),
ranked AS (
    SELECT *,
           LinearExpectedKld - PredictedKld AS Gain
    FROM scored
)
SELECT {CombinationDuckDbSchema.SlotColumnList},
       PredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank,
       LinearExpectedKld,
       Gain
FROM ranked
WHERE Gain > ?
ORDER BY Gain DESC,
         PredictionConfidence DESC,
         PredictedSizeBytes ASC,
         PredictedKld ASC,
         PredictionRank ASC
LIMIT ?;";

        double denominator = Math.Max(
            (double)lowerDamageLarger.SizeBytes - higherDamageSmaller.SizeBytes,
            1d);

        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;

        foreach (var value in new object[]
                 {
                     higherDamageSmaller.Kld,
                     (double)higherDamageSmaller.SizeBytes,
                     denominator,
                     lowerDamageLarger.Kld,
                     higherDamageSmaller.Kld,
                     minSize,
                     maxSize,
                     Config.SelectionMinimumKldImprovementEpsilon,
                     limit
                 })
        {
            cmd.Parameters.Add(new DuckDBParameter { Value = value });
        }

        var list = new List<HybridSelectionCandidate>();
        using var r = await cmd.ExecuteReaderAsync(ct);
        int attempt = 0;
        while (await r.ReadAsync(ct))
        {
            var prediction = MapPredictedRow(r);
            double line = Convert.ToDouble(r.GetValue(14));
            double gain = Convert.ToDouble(r.GetValue(15));

            list.Add(new HybridSelectionCandidate
            {
                Prediction = prediction,
                Reason = reason,
                HigherDamageAnchor = higherDamageSmaller,
                LowerDamageAnchor = lowerDamageLarger,
                WindowMinSizeBytes = minSize,
                WindowMaxSizeBytes = maxSize,
                LinearExpectedKld = line,
                PredictedGainOverLine = gain,
                WindowLabel = windowLabel,
                AttemptOrder = ++attempt
            });
        }

        return list;
    }

    private async Task<IReadOnlyList<RankSafePredictionRow>> QueryPredictedRowsAsync(
        string sql,
        object[] args,
        CancellationToken ct)
    {
        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        foreach (var arg in args)
            cmd.Parameters.Add(new DuckDBParameter { Value = arg });

        using var r = await cmd.ExecuteReaderAsync(ct);
        var list = new List<RankSafePredictionRow>();
        while (await r.ReadAsync(ct))
            list.Add(MapPredictedRow(r));

        return list;
    }

    private static RankSafePredictionRow MapPredictedRow(System.Data.Common.DbDataReader r)
    {
        var config = ReadTensorConfig(r);

        return new RankSafePredictionRow
        {
            Config = config,
            Quant = (HybridQuant)config,
            PredictedKld = Convert.ToDouble(r.GetValue(10)),
            PredictedSizeBytes = Convert.ToUInt64(r.GetValue(11)),
            PredictionConfidence = Convert.ToDouble(r.GetValue(12)),
            PredictedRank = Convert.ToUInt64(r.GetValue(13)),
            IsPredictable = true,
            IsSizePredictable = true
        };
    }

    private static TensorConfig ReadTensorConfig(System.Data.Common.DbDataReader r)
    {
        return new TensorConfig(
            Convert.ToByte(r.GetValue(0)),
            Convert.ToByte(r.GetValue(1)),
            Convert.ToByte(r.GetValue(2)),
            Convert.ToByte(r.GetValue(3)),
            Convert.ToByte(r.GetValue(4)),
            Convert.ToByte(r.GetValue(5)),
            Convert.ToByte(r.GetValue(6)),
            Convert.ToByte(r.GetValue(7)),
            Convert.ToByte(r.GetValue(8)),
            Convert.ToByte(r.GetValue(9)));
    }

    private static void AddSlotParameters(DuckDBCommand command, TensorConfig config)
    {
        command.Parameters.Add(new DuckDBParameter { Value = config.BaseQuant });
        command.Parameters.Add(new DuckDBParameter { Value = config.Embeddings });
        command.Parameters.Add(new DuckDBParameter { Value = config.LmHead });
        command.Parameters.Add(new DuckDBParameter { Value = config.AttnQ });
        command.Parameters.Add(new DuckDBParameter { Value = config.AttnKV });
        command.Parameters.Add(new DuckDBParameter { Value = config.AttnOutput });
        command.Parameters.Add(new DuckDBParameter { Value = config.FfnUpGate });
        command.Parameters.Add(new DuckDBParameter { Value = config.FfnDown });
        command.Parameters.Add(new DuckDBParameter { Value = config.MoeExperts });
        command.Parameters.Add(new DuckDBParameter { Value = config.MoeRouter });
    }

    private static async Task ConfigureFastLoadSessionAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = "SET preserve_insertion_order = false;";
            await cmd.ExecuteNonQueryAsync(ct);
        }

        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = $"SET threads = {Math.Max(1, Environment.ProcessorCount)};";
            await cmd.ExecuteNonQueryAsync(ct);
        }
    }

    private static async Task RecreateTableAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var createCmd = connection.CreateCommand();
        createCmd.CommandText = CombinationDuckDbSchema.CreateTableSql;
        await createCmd.ExecuteNonQueryAsync(ct);
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
}
