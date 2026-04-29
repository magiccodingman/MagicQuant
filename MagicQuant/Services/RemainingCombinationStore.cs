using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Models;
using System.Runtime.CompilerServices;

namespace MagicQuant.Services;

public sealed class RemainingCombinationStore
{
    private const string DbFileNamePrefix = "MagicQuant_Combinations";
    private const string TableName = "tensor_configs";

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
SELECT BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter
FROM {TableName}
ORDER BY BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter;";

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
                moeRouter: Convert.ToByte(reader.GetValue(9))));
        }

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
        string sql = $@"SELECT BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter FROM {TableName}";
        if (!string.IsNullOrWhiteSpace(whereSql)) sql += $" WHERE {whereSql}";
        if (!string.IsNullOrWhiteSpace(orderBySql)) sql += $" ORDER BY {orderBySql}";
        if (limit.HasValue) sql += $" LIMIT {limit.Value}";
        using var cmd = connection.CreateCommand();
        cmd.CommandText = sql;
        using var reader = await cmd.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            yield return new TensorConfig(
                Convert.ToByte(reader.GetValue(0)), Convert.ToByte(reader.GetValue(1)), Convert.ToByte(reader.GetValue(2)),
                Convert.ToByte(reader.GetValue(3)), Convert.ToByte(reader.GetValue(4)), Convert.ToByte(reader.GetValue(5)),
                Convert.ToByte(reader.GetValue(6)), Convert.ToByte(reader.GetValue(7)), Convert.ToByte(reader.GetValue(8)),
                Convert.ToByte(reader.GetValue(9)));
        }
    }

    public async Task ReplaceAllAsync(IReadOnlyCollection<TensorConfig> configs, string reason, CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);
        await RecreateTableAsync(connection, ct);

        using var tx = connection.BeginTransaction();
        using var insert = connection.CreateCommand();
        insert.CommandText = $@"
INSERT INTO {TableName}
(BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";

        foreach (var config in configs)
        {
            insert.Parameters.Clear();
            insert.Parameters.Add(new DuckDBParameter { Value = config.BaseQuant });
            insert.Parameters.Add(new DuckDBParameter { Value = config.Embeddings });
            insert.Parameters.Add(new DuckDBParameter { Value = config.LmHead });
            insert.Parameters.Add(new DuckDBParameter { Value = config.AttnQ });
            insert.Parameters.Add(new DuckDBParameter { Value = config.AttnKV });
            insert.Parameters.Add(new DuckDBParameter { Value = config.AttnOutput });
            insert.Parameters.Add(new DuckDBParameter { Value = config.FfnUpGate });
            insert.Parameters.Add(new DuckDBParameter { Value = config.FfnDown });
            insert.Parameters.Add(new DuckDBParameter { Value = config.MoeExperts });
            insert.Parameters.Add(new DuckDBParameter { Value = config.MoeRouter });
            await insert.ExecuteNonQueryAsync(ct);
        }

        tx.Commit();
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
        createCmd.CommandText = $@"
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
