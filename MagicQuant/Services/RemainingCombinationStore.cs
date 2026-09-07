using DuckDB.NET.Data;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace MagicQuant.Services;

public sealed class RemainingCombinationStore
{
    private const string TableName = CombinationDuckDbSchema.TableName;

    private static string ConnectionString => $"Data Source={CombinationDatabasePathService.GetPath()}";

    public string GetDatabaseFilePath() => CombinationDatabasePathService.GetPath();

    public async Task<long> CountAsync(CancellationToken ct = default)
    {
        using var connection = new DuckDBConnection(ConnectionString);
        await connection.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(connection, ct);
        await EnsureTensorConfigsTableExistsAsync(connection, ct);

        using var cmd = connection.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {TableName};";
        return ToInt64(await cmd.ExecuteScalarAsync(ct));
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
        await EnsureTensorConfigsTableExistsAsync(connection, ct);

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
        await EnsureTensorConfigsTableExistsAsync(connection, ct);

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

        long total = ToInt64(r.GetValue(0));
        long predicted = ToInt64(r.GetValue(1));
        long ranked = ToInt64(r.GetValue(2));

        return new PredictionMaterializationStatus
        {
            TotalRows = total,
            PredictedRows = predicted,
            MissingPredictionRows = Math.Max(0, total - predicted),
            RankedRows = ranked,
            MinPredictedKld = r.IsDBNull(3) ? null : ToDouble(r.GetValue(3)),
            MaxPredictedKld = r.IsDBNull(4) ? null : ToDouble(r.GetValue(4)),
            MinPredictedSizeBytes = r.IsDBNull(5) ? null : ToUInt64(r.GetValue(5)),
            MaxPredictedSizeBytes = r.IsDBNull(6) ? null : ToUInt64(r.GetValue(6))
        };
    }

    public async Task<IReadOnlyList<PredictedAnchorRow>> GetPredictedAnchorRowsAsync(CancellationToken ct = default)
    {
        string sql = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       COALESCE(AnchorDisplayName, AnchorBaselineCanonicalKey, '') AS AnchorDisplayName,
       COALESCE(AnchorBaselineCanonicalKey, '') AS AnchorBaselineCanonicalKey,
       COALESCE(AnchorBaselineRuntimeId, 0) AS AnchorBaselineRuntimeId,
       {CombinationDuckDbSchema.EffectivePredictedKldSql} AS PredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank,
       COALESCE(IsVirtualPredictionAnchor, FALSE) AS IsVirtualPredictionAnchor
FROM {TableName}
WHERE {CombinationDuckDbSchema.VirtualPredictionAnchorPredicateSql}
  AND COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
ORDER BY PredictedKld ASC,
         PredictedSizeBytes ASC,
         PredictionRank ASC;";

        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);
        await EnsureTensorConfigsTableExistsAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;

        var list = new List<PredictedAnchorRow>();
        using var r = await cmd.ExecuteReaderAsync(ct);
        while (await r.ReadAsync(ct))
            list.Add(MapPredictedAnchorRow(r));

        return list;
    }

    public async Task<PredictedAnchorRow?> FindPredictedAnchorForRealAnchorAsync(
        BenchmarkSnapshotRecord realAnchor,
        IReadOnlyList<PredictedAnchorRow> predictedAnchors,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();

        var sourceBaseline = HybridBenchmarkRepository.ResolveSourceBaselineForProvider(realAnchor.Quant);
        var predictionSpaceConfig = CanonicalizeConfigForPredictionSpace(realAnchor.Config);
        string predictionSpaceKey = TensorConfigIdentity.ToKey(predictionSpaceConfig);

        var byConfig = predictedAnchors.FirstOrDefault(x =>
            string.Equals(x.ConfigKey, predictionSpaceKey, StringComparison.Ordinal));
        if (byConfig != null)
            return byConfig;

        if (HybridBenchmarkRepository.IsTrueMagicQuantHybrid(realAnchor.Quant))
            return await QueryPredictedAnchorForConfigAsync(realAnchor, sourceBaseline, ct);

        string canonicalKey = NormalizeAnchorKey(sourceBaseline.CanonicalKey);

        var byCanonical = predictedAnchors
            .Where(x => !string.IsNullOrWhiteSpace(x.BaselineCanonicalKey))
            .FirstOrDefault(x => string.Equals(NormalizeAnchorKey(x.BaselineCanonicalKey), canonicalKey, StringComparison.Ordinal));

        if (byCanonical != null)
            return byCanonical;

        var byRuntimeId = predictedAnchors.FirstOrDefault(x => x.RuntimeBaselineId == sourceBaseline.UniqueId);
        if (byRuntimeId != null)
            return byRuntimeId;

        var displayNames = sourceBaseline.Names
            .Concat([realAnchor.DisplayName, realAnchor.BaselineFamily])
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(NormalizeAnchorKey)
            .ToHashSet(StringComparer.Ordinal);

        var byDisplay = predictedAnchors.FirstOrDefault(x => displayNames.Contains(NormalizeAnchorKey(x.DisplayName)));
        if (byDisplay != null)
            return byDisplay;

        // Accepted MagicQuant hybrids can become real anchors in later phases. They will
        // not have a virtual baseline-anchor row, but their own tensor config should still
        // be present and scored in DuckDB. Use that predicted row as the phase-local
        // prediction-space anchor instead of falling back to real KLD/size.
        return await QueryPredictedAnchorForConfigAsync(realAnchor, sourceBaseline, ct);
    }

    private async Task<PredictedAnchorRow?> QueryPredictedAnchorForConfigAsync(
        BenchmarkSnapshotRecord realAnchor,
        BaselineQuants sourceBaseline,
        CancellationToken ct)
    {
        var predictionSpaceConfig = CanonicalizeConfigForPredictionSpace(realAnchor.Config);

        string sql = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       {CombinationDuckDbSchema.EffectivePredictedKldSql} AS PredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank,
       COALESCE(IsVirtualPredictionAnchor, FALSE) AS IsVirtualPredictionAnchor
FROM {TableName}
WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {BuildSlotPredicateSql(predictionSpaceConfig)}
LIMIT 1;";

        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);
        await EnsureTensorConfigsTableExistsAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;

        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            return null;

        var config = ReadTensorConfig(r);
        return new PredictedAnchorRow
        {
            Config = config,
            ConfigKey = TensorConfigIdentity.ToKey(config),
            DisplayName = realAnchor.DisplayName,
            BaselineCanonicalKey = sourceBaseline.CanonicalKey,
            RuntimeBaselineId = sourceBaseline.UniqueId,
            PredictedKld = ToDouble(r.GetValue(10)),
            PredictedSizeBytes = ToUInt64(r.GetValue(11)),
            PredictionConfidence = ToDouble(r.GetValue(12)),
            PredictionRank = ToUInt64(r.GetValue(13)),
            IsVirtualPredictionAnchor = ToBoolean(r.GetValue(14))
        };
    }

    public Task<long> CountStrictDominanceCandidatesAsync(
        PredictedAnchorRow anchor,
        CancellationToken ct = default)
    {
        return CountStrictDominanceCandidatesAsync(anchor, anchor.PredictedSizeBytes, ct);
    }

    public async Task<long> CountStrictDominanceCandidatesAsync(
        PredictedAnchorRow anchor,
        ulong maxSizeBytes,
        CancellationToken ct = default)
    {
        string sql = $@"
SELECT COUNT(*)
FROM {TableName}
WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
  AND {CombinationDuckDbSchema.HybridPredicateSql}
  AND PredictedSizeBytes <= ?
  AND {CombinationDuckDbSchema.EffectivePredictedKldSql} + ? < ?;";

        return await ExecuteCountAsync(
            sql,
            new object[] { maxSizeBytes, Config.SelectionMinimumKldImprovementEpsilon, anchor.PredictedKld },
            ct);
    }

    public async Task<long> CountPredictedHybridCandidatesInSizeWindowAsync(
        ulong minSize,
        ulong maxSize,
        CancellationToken ct = default)
    {
        string sql = $@"
SELECT COUNT(*)
FROM {TableName}
WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
  AND {CombinationDuckDbSchema.HybridPredicateSql}
  AND PredictedSizeBytes BETWEEN ? AND ?;";

        return await ExecuteCountAsync(sql, new object[] { minSize, maxSize }, ct);
    }

    public Task<long> CountBetterThanLinearCandidatesAsync(
        PredictedAnchorRow higherDamageSmaller,
        PredictedAnchorRow lowerDamageLarger,
        ulong minSize,
        ulong maxSize,
        CancellationToken ct = default)
    {
        return CountBetterThanLinearCandidatesAsync(
            higherDamageSmaller,
            lowerDamageLarger,
            predictionWindowMinSize: minSize,
            predictionWindowMaxSize: maxSize,
            deterministicWindowMinSize: minSize,
            deterministicWindowMaxSize: maxSize,
            ct: ct);
    }

    public async Task<long> CountBetterThanLinearCandidatesAsync(
        PredictedAnchorRow higherDamageSmaller,
        PredictedAnchorRow lowerDamageLarger,
        ulong predictionWindowMinSize,
        ulong predictionWindowMaxSize,
        ulong deterministicWindowMinSize,
        ulong deterministicWindowMaxSize,
        CancellationToken ct = default)
    {
        var effectiveWindow = IntersectSizeWindows(
            predictionWindowMinSize,
            predictionWindowMaxSize,
            deterministicWindowMinSize,
            deterministicWindowMaxSize);

        if (effectiveWindow == null)
            return 0;

        string sql = $@"
WITH scored AS (
    SELECT {CombinationDuckDbSchema.EffectivePredictedKldSql} AS PredictedKld,
           PredictedSizeBytes,
           (CAST(? AS DOUBLE)
             + ((CAST(PredictedSizeBytes AS DOUBLE) - CAST(? AS DOUBLE)) / GREATEST(CAST(? AS DOUBLE), 1.0))
             * (CAST(? AS DOUBLE) - CAST(? AS DOUBLE))) AS LinearExpectedKld
    FROM {TableName}
    WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
      AND PredictedSizeBytes IS NOT NULL
      AND PredictionRank IS NOT NULL
      AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
      AND {CombinationDuckDbSchema.HybridPredicateSql}
      AND PredictedSizeBytes BETWEEN ? AND ?
)
SELECT COUNT(*)
FROM scored
WHERE LinearExpectedKld - PredictedKld > ?;";

        double denominator = Math.Max(
            (double)lowerDamageLarger.PredictedSizeBytes - higherDamageSmaller.PredictedSizeBytes,
            1d);

        return await ExecuteCountAsync(
            sql,
            new object[]
            {
                higherDamageSmaller.PredictedKld,
                (double)higherDamageSmaller.PredictedSizeBytes,
                denominator,
                lowerDamageLarger.PredictedKld,
                higherDamageSmaller.PredictedKld,
                effectiveWindow.Value.Min,
                effectiveWindow.Value.Max,
                Config.SelectionMinimumKldImprovementEpsilon
            },
            ct);
    }

    public Task<IReadOnlyList<RankSafePredictionRow>> QueryStrictDominanceCandidatesAsync(
        PredictedAnchorRow anchor,
        int limit,
        CancellationToken ct = default)
    {
        return QueryStrictDominanceCandidatesAsync(anchor, anchor.PredictedSizeBytes, limit, ct);
    }

    public async Task<IReadOnlyList<RankSafePredictionRow>> QueryStrictDominanceCandidatesAsync(
        PredictedAnchorRow anchor,
        ulong maxSizeBytes,
        int limit,
        CancellationToken ct = default)
    {
        string sql = $@"
SELECT {CombinationDuckDbSchema.SlotColumnList},
       {CombinationDuckDbSchema.EffectivePredictedKldSql} AS PredictedKld,
       PredictedSizeBytes,
       PredictionConfidence,
       PredictionRank,
       COALESCE(AnomalyAdjustmentKld, 0.0) AS AnomalyAdjustmentKld
FROM {TableName}
WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionRank IS NOT NULL
  AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
  AND {CombinationDuckDbSchema.HybridPredicateSql}
  AND PredictedSizeBytes <= ?
  AND {CombinationDuckDbSchema.EffectivePredictedKldSql} + ? < ?
ORDER BY PredictedSizeBytes ASC,
         PredictedKld ASC,
         PredictionRank ASC,
         PredictionConfidence DESC
LIMIT ?;";

        return await QueryPredictedRowsAsync(
            sql,
            new object[] { maxSizeBytes, Config.SelectionMinimumKldImprovementEpsilon, anchor.PredictedKld, limit },
            ct);
    }

    public async Task<IReadOnlyList<HybridSelectionCandidate>> QueryBetterThanLinearCandidatesAsync(
        BenchmarkSnapshotRecord higherDamageSmaller,
        BenchmarkSnapshotRecord lowerDamageLarger,
        PredictedAnchorRow higherDamagePredictionAnchor,
        PredictedAnchorRow lowerDamagePredictionAnchor,
        ulong predictionWindowMinSize,
        ulong predictionWindowMaxSize,
        ulong realValidationWindowMinSize,
        ulong realValidationWindowMaxSize,
        HybridSelectionReason reason,
        string windowLabel,
        int limit,
        CancellationToken ct = default)
    {
        var effectiveWindow = IntersectSizeWindows(
            predictionWindowMinSize,
            predictionWindowMaxSize,
            realValidationWindowMinSize,
            realValidationWindowMaxSize);

        if (effectiveWindow == null)
            return Array.Empty<HybridSelectionCandidate>();

        string sql = $@"
WITH scored AS (
    SELECT {CombinationDuckDbSchema.SlotColumnList},
           {CombinationDuckDbSchema.EffectivePredictedKldSql} AS PredictedKld,
           PredictedSizeBytes,
           PredictionConfidence,
           PredictionRank,
           (CAST(? AS DOUBLE)
             + ((CAST(PredictedSizeBytes AS DOUBLE) - CAST(? AS DOUBLE)) / GREATEST(CAST(? AS DOUBLE), 1.0))
             * (CAST(? AS DOUBLE) - CAST(? AS DOUBLE))) AS LinearExpectedKld
    FROM {TableName}
    WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
      AND PredictedSizeBytes IS NOT NULL
      AND PredictionRank IS NOT NULL
      AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql}
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
            (double)lowerDamagePredictionAnchor.PredictedSizeBytes - higherDamagePredictionAnchor.PredictedSizeBytes,
            1d);

        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);
        await EnsureTensorConfigsTableExistsAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;

        foreach (var value in new object[]
                 {
                     higherDamagePredictionAnchor.PredictedKld,
                     (double)higherDamagePredictionAnchor.PredictedSizeBytes,
                     denominator,
                     lowerDamagePredictionAnchor.PredictedKld,
                     higherDamagePredictionAnchor.PredictedKld,
                     effectiveWindow.Value.Min,
                     effectiveWindow.Value.Max,
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
            double line = ToDouble(r.GetValue(14));
            double gain = ToDouble(r.GetValue(15));

            list.Add(new HybridSelectionCandidate
            {
                Prediction = prediction,
                Reason = reason,
                HigherDamageAnchor = higherDamageSmaller,
                LowerDamageAnchor = lowerDamageLarger,
                HigherDamagePredictionAnchor = higherDamagePredictionAnchor,
                LowerDamagePredictionAnchor = lowerDamagePredictionAnchor,
                PredictionWindowMinSizeBytes = predictionWindowMinSize,
                PredictionWindowMaxSizeBytes = predictionWindowMaxSize,
                WindowMinSizeBytes = realValidationWindowMinSize,
                WindowMaxSizeBytes = realValidationWindowMaxSize,
                LinearExpectedKld = line,
                PredictedGainOverLine = gain,
                WindowLabel = windowLabel,
                AttemptOrder = ++attempt
            });
        }

        return list;
    }

    private static (ulong Min, ulong Max)? IntersectSizeWindows(
        ulong firstMin,
        ulong firstMax,
        ulong secondMin,
        ulong secondMax)
    {
        ulong min = Math.Max(firstMin, secondMin);
        ulong max = Math.Min(firstMax, secondMax);
        return max < min ? null : (min, max);
    }

    private async Task<IReadOnlyList<RankSafePredictionRow>> QueryPredictedRowsAsync(
        string sql,
        object[] args,
        CancellationToken ct)
    {
        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);
        await EnsureTensorConfigsTableExistsAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        foreach (var arg in args)
            cmd.Parameters.Add(new DuckDBParameter { Value = arg });

        using var r = await cmd.ExecuteReaderAsync(ct);
        var list = new List<RankSafePredictionRow>();
        while (await r.ReadAsync(ct))
            list.Add(MapPredictedRow(r, anomalyAdjustmentColumnIndex: r.FieldCount > 14 ? 14 : null));

        return list;
    }

    private static PredictedAnchorRow MapPredictedAnchorRow(System.Data.Common.DbDataReader r)
    {
        var config = ReadTensorConfig(r);
        string canonicalKey = Convert.ToString(r.GetValue(11)) ?? string.Empty;
        byte runtimeBaselineId = ToByte(r.GetValue(12));
        string displayName = Convert.ToString(r.GetValue(10)) ?? canonicalKey;

        return new PredictedAnchorRow
        {
            Config = config,
            ConfigKey = TensorConfigIdentity.ToKey(config),
            DisplayName = string.IsNullOrWhiteSpace(displayName) ? canonicalKey : displayName,
            BaselineCanonicalKey = canonicalKey,
            RuntimeBaselineId = runtimeBaselineId,
            PredictedKld = ToDouble(r.GetValue(13)),
            PredictedSizeBytes = ToUInt64(r.GetValue(14)),
            PredictionConfidence = ToDouble(r.GetValue(15)),
            PredictionRank = ToUInt64(r.GetValue(16)),
            IsVirtualPredictionAnchor = ToBoolean(r.GetValue(17))
        };
    }

    private static RankSafePredictionRow MapPredictedRow(System.Data.Common.DbDataReader r, int? anomalyAdjustmentColumnIndex = null)
    {
        var config = ReadTensorConfig(r);

        return new RankSafePredictionRow
        {
            Config = config,
            Quant = (HybridQuant)config,
            PredictedKld = ToDouble(r.GetValue(10)),
            PredictedSizeBytes = ToUInt64(r.GetValue(11)),
            PredictionConfidence = ToDouble(r.GetValue(12)),
            PredictedRank = ToUInt64(r.GetValue(13)),
            AnomalyAdjustmentKld = anomalyAdjustmentColumnIndex.HasValue ? ToDouble(r.GetValue(anomalyAdjustmentColumnIndex.Value)) : 0d,
            IsPredictable = true,
            IsSizePredictable = true
        };
    }

    private static TensorConfig ReadTensorConfig(System.Data.Common.DbDataReader r)
    {
        return new TensorConfig(
            ToByte(r.GetValue(0)),
            ToByte(r.GetValue(1)),
            ToByte(r.GetValue(2)),
            ToByte(r.GetValue(3)),
            ToByte(r.GetValue(4)),
            ToByte(r.GetValue(5)),
            ToByte(r.GetValue(6)),
            ToByte(r.GetValue(7)),
            ToByte(r.GetValue(8)),
            ToByte(r.GetValue(9)));
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

    private async Task<long> ExecuteCountAsync(string sql, object[] args, CancellationToken ct)
    {
        using var c = new DuckDBConnection(ConnectionString);
        await c.OpenAsync(ct);
        await ConfigureFastLoadSessionAsync(c, ct);
        await EnsureTensorConfigsTableExistsAsync(c, ct);

        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        foreach (var arg in args)
            cmd.Parameters.Add(new DuckDBParameter { Value = arg });

        return ToInt64(await cmd.ExecuteScalarAsync(ct));
    }

    private static TensorConfig CanonicalizeConfigForPredictionSpace(TensorConfig config)
    {
        if (config.BaseQuant == BaselineQuants.Q8_0.UniqueId)
            return config;

        var baseBaseline = BaselineQuants.FromId(config.BaseQuant);
        byte inheritedBaseSlot = BaselineQuants.EncodeTensorConfigGroupSlot(baseBaseline);

        return new TensorConfig(
            baseQuant: BaselineQuants.Q8_0.UniqueId,
            embeddings: CanonicalizePredictionSlot(TReg.Embeddings, config.Embeddings, inheritedBaseSlot),
            lmHead: CanonicalizePredictionSlot(TReg.LmHead, config.LmHead, inheritedBaseSlot),
            attnQ: CanonicalizePredictionSlot(TReg.AttnQ, config.AttnQ, inheritedBaseSlot),
            attnKV: CanonicalizePredictionSlot(TReg.AttnKV, config.AttnKV, inheritedBaseSlot),
            attnOutput: CanonicalizePredictionSlot(TReg.AttnOutput, config.AttnOutput, inheritedBaseSlot),
            ffnUpGate: CanonicalizePredictionSlot(TReg.FfnUpGate, config.FfnUpGate, inheritedBaseSlot),
            ffnDown: CanonicalizePredictionSlot(TReg.FfnDown, config.FfnDown, inheritedBaseSlot),
            moeExperts: CanonicalizePredictionSlot(TReg.MoeExperts, config.MoeExperts, inheritedBaseSlot),
            moeRouter: CanonicalizePredictionSlot(TReg.MoeRouter, config.MoeRouter, inheritedBaseSlot));
    }

    private static byte CanonicalizePredictionSlot(TensorGroup group, byte storedValue, byte inheritedBaseSlot)
    {
        if (Cache.UnusedTensorGroups.Any(x => x.UniqueId == group.UniqueId))
            return BaselineQuants.TensorConfigNullSlotValue;

        return BaselineQuants.IsNullTensorConfigGroupSlot(storedValue)
            ? inheritedBaseSlot
            : storedValue;
    }

    private static string BuildSlotPredicateSql(TensorConfig config)
    {
        return $"BaseQuant = {config.BaseQuant} AND Embeddings = {config.Embeddings} AND LmHead = {config.LmHead} AND AttnQ = {config.AttnQ} AND AttnKV = {config.AttnKV} AND AttnOutput = {config.AttnOutput} AND FfnUpGate = {config.FfnUpGate} AND FfnDown = {config.FfnDown} AND MoeExperts = {config.MoeExperts} AND MoeRouter = {config.MoeRouter}";
    }

    private static string NormalizeAnchorKey(string? value) => (value ?? string.Empty).Trim().ToLowerInvariant();

    private static long ToInt64(object? value)
    {
        if (value is null || value is DBNull)
            return 0L;

        if (value is BigInteger big)
            return (long)big;

        return Convert.ToInt64(value);
    }

    private static ulong ToUInt64(object? value)
    {
        if (value is null || value is DBNull)
            return 0UL;

        if (value is BigInteger big)
            return (ulong)big;

        return Convert.ToUInt64(value);
    }

    private static byte ToByte(object? value)
    {
        if (value is null || value is DBNull)
            return 0;

        if (value is BigInteger big)
            return (byte)big;

        return Convert.ToByte(value);
    }

    private static double ToDouble(object? value)
    {
        if (value is null || value is DBNull)
            return 0d;

        if (value is BigInteger big)
            return (double)big;

        return Convert.ToDouble(value);
    }

    private static bool ToBoolean(object? value)
    {
        if (value is null || value is DBNull)
            return false;

        if (value is bool b)
            return b;

        if (value is BigInteger big)
            return big != BigInteger.Zero;

        return Convert.ToBoolean(value);
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

    private static async Task EnsureTensorConfigsTableExistsAsync(DuckDBConnection connection, CancellationToken ct)
    {
        using var cmd = connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?;";
        cmd.Parameters.Add(new DuckDBParameter { Value = TableName });

        long matches = ToInt64(await cmd.ExecuteScalarAsync(ct));
        if (matches > 0)
            return;

        throw new InvalidOperationException(
            $"DuckDB search-space table '{TableName}' does not exist in '{CombinationDatabasePathService.GetPath()}'. " +
            "This almost always means the generator and prediction reader are using different DuckDB filenames, " +
            "or prediction started before QuantDatabaseService initialized/rebuilt the search-space table.");
    }
}