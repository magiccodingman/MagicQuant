using System.Diagnostics;
using System.Numerics;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public class QuantDatabaseService
{
    private const string DbFileName = "MagicQuant_Combinations.duckdb";
    private const string TableName = "tensor_configs";

    private static string GetDuckDbDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Cache.ModelMagicQuantDirectory;

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Cache.MagicQuantDirectory;

        throw new InvalidOperationException(
            "Neither Cache.ModelMagicQuantDirectory nor Cache.MagicQuantDirectory is set.");
    }

    private string ConnectionString => $"Data Source={Path.Combine(GetDuckDbDirectory(), DbFileName)}";

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

        AnsiConsole.MarkupLine($"[grey]Starting bulk insert of {expectedTotal:N0} rows...[/]");

        foreach (var baseline in bases)
        {
            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(
                         baseline,
                         batchSize: 1_000_000,
                         ct: ct))
            {
                using (var appender = connection.CreateAppender(TableName))
                {
                    foreach (var config in batch)
                    {
                        var row = appender.CreateRow();

                        row.AppendValue(config.BaseQuant);
                        row.AppendValue(config.Embeddings);
                        row.AppendValue(config.LmHead);
                        row.AppendValue(config.AttnQ);
                        row.AppendValue(config.AttnKV);
                        row.AppendValue(config.AttnOutput);
                        row.AppendValue(config.FfnUpGate);
                        row.AppendValue(config.FfnDown);
                        row.AppendValue(config.MoeExperts);
                        row.AppendValue(config.MoeRouter);

                        row.EndRow();
                    }
                }

                insertedTotal += batch.Count;
                AnsiConsole.MarkupLine($"  [grey]Inserted batch... Total so far:[/] {insertedTotal:N0}");
                batch.Clear();
            }
        }

        sw.Stop();
        AnsiConsole.MarkupLine($"[bold green]DuckDB rebuild complete![/] in {sw.Elapsed.TotalSeconds:F2}s");
    }
}