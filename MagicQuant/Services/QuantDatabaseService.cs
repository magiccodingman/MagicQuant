using System.Diagnostics;
using System.Numerics;
using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public class QuantDatabaseService
{
    private const string DbFileName = "MagicQuant.duckdb";
    private const string TableName = "tensor_configs";
    
    // Connection string points to the file in your cache directory
    private string ConnectionString => $"Data Source={Path.Combine(Cache.MagicQuantDirectory, DbFileName)}";

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // 1. Ensure directory exists
        Directory.CreateDirectory(Cache.MagicQuantDirectory);

        // 2. Open connection to check state
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);

        BigInteger expectedTotal = ComboCounter.CountAll();
        long currentDbCount = await GetRowCountAsync(connection, ct);

        AnsiConsole.MarkupLine($"[bold]DB Check:[/] Current Rows: [cyan]{currentDbCount:N0}[/] | Expected: [yellow]{expectedTotal:N0}[/]");

        // 3. Validation Logic: If counts mismatch or table missing, rebuild.
        if (currentDbCount != expectedTotal)
        {
            
            AnsiConsole.MarkupLine("[bold red]Database empty, mismatch, or new.[/] Initializing/Rebuilding...");

            await RebuildDatabaseAsync(connection, expectedTotal, ct);
        }
        else
        {
            AnsiConsole.MarkupLine("[bold green]Database is synchronized and ready.[/]");
        }
    }

    private async Task<long> GetRowCountAsync(DuckDBConnection connection, CancellationToken ct)
    {
        // Check if table exists first
        var checkCmd = connection.CreateCommand();
        checkCmd.CommandText = $"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{TableName}'";
        var exists = (long)(await checkCmd.ExecuteScalarAsync(ct) ?? 0);

        if (exists == 0) return -1; // Marker for "Table doesn't exist"

        // Get count
        var countCmd = connection.CreateCommand();
        countCmd.CommandText = $"SELECT COUNT(*) FROM {TableName}";
        return (long)(await countCmd.ExecuteScalarAsync(ct) ?? 0);
    }

    private async Task RebuildDatabaseAsync(DuckDBConnection connection, BigInteger expectedTotal, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();

        // 1. Drop and Recreate Table
        // We map sbyte (C#) to TINYINT (DuckDB)
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

        // 2. Generate and Bulk Insert
        // We use the Appender for high-performance bulk writing
        
        long insertedTotal = 0;

        // Iterate through your existing generator logic
        var bases = BaselineQuants.All
            .Where(b => b.BaseConversionBase != null)
            .ToList();

        AnsiConsole.MarkupLine($"[grey]Starting bulk insert of {expectedTotal:N0} rows...[/]");

        foreach (var b in bases)
        {
            // We reuse the generator you already wrote
            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(b, batchSize: 1_000_000, ct: ct))
            {
                // OPEN APPENDER for this batch
                // Note: DuckDB Appender is synchronous by design for max speed
                using (var appender = connection.CreateAppender(TableName))
                {
                    foreach (var config in batch)
                    {
                        var row = appender.CreateRow();
                        
                        // Precise mapping of struct fields
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
                } // Appender.Dispose() commits the batch automatically

                insertedTotal += batch.Count;
                AnsiConsole.MarkupLine($"  [grey]Inserted batch... Total so far:[/] {insertedTotal:N0}");
                
                // Clear memory in the batch list as per your previous logic
                batch.Clear();
            }
        }

        sw.Stop();
        AnsiConsole.MarkupLine($"[bold green]Rebuild Complete![/] in {sw.Elapsed.TotalSeconds:F2}s");
    }
}