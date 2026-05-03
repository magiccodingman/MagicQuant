using DuckDB.NET.Data;
using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using System.Numerics;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class AnomalyAdjustedPredictionService
{
    private readonly RemainingCombinationStore _store;

    public AnomalyAdjustedPredictionService(RemainingCombinationStore store)
    {
        _store = store;
    }

    public async Task<AnomalyAdjustmentSummary> ApplyAsync(
        IReadOnlyCollection<AnomalyInteractionRule> rules,
        CancellationToken ct)
    {
        if (rules.Count == 0)
            return new AnomalyAdjustmentSummary { DuckDbPath = _store.GetDatabaseFilePath() };

        using var c = new DuckDBConnection($"Data Source={_store.GetDatabaseFilePath()}");
        await c.OpenAsync(ct);
        await ConfigureSessionAsync(c, ct);

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName}
SET AnomalyAdjustmentKld = 0.0,
    FinalPredictedKld = BaseRankSafeKld,
    PredictedKld = BaseRankSafeKld
WHERE BaseRankSafeKld IS NOT NULL;", ct);

        long totalMatched = 0;
        var matchLogs = new List<object>();

        foreach (var rule in rules.OrderByDescending(x => x.Confidence).ThenBy(x => x.Id))
        {
            string where = BuildRuleWhere(rule);
            if (string.IsNullOrWhiteSpace(where))
                continue;

            double adjustment = rule.AppliedPredictionSpaceAdjustmentKld;
            if (Math.Abs(adjustment) <= 0d)
                continue;

            long before = await CountMatchesAsync(c, where, ct);
            if (before == 0)
                continue;

            var beforeStats = await LoadPredictionStatsAsync(c, where, ct);

            string expression = adjustment < 0d
                ? $"GREATEST(COALESCE(AnomalyAdjustmentKld, 0.0) + ({SqlDouble(adjustment)}), -LEAST({SqlDouble(Config.AnomalyDetection.MaxNegativeAdjustmentKld)}, COALESCE(BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))"
                : $"LEAST(COALESCE(AnomalyAdjustmentKld, 0.0) + ({SqlDouble(adjustment)}), LEAST({SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)}, COALESCE(BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))";

            await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName}
SET AnomalyAdjustmentKld = {expression},
    FinalPredictedKld = GREATEST(0.0, COALESCE(BaseRankSafeKld, PredictedKld, 0.0) + {expression}),
    PredictedKld = GREATEST(0.0, COALESCE(BaseRankSafeKld, PredictedKld, 0.0) + {expression})
WHERE {where};", ct);

            var afterStats = await LoadPredictionStatsAsync(c, where, ct);

            totalMatched += before;
            var actual = ExtractActualEffect(rule);
            var log = new
            {
                ruleId = rule.Id,
                direction = rule.RuleDirection,
                ruleType = rule.RuleType,
                referenceQuant = SafeName(rule.ReferenceQuantId),
                groupSetHash = rule.GroupSetHash,
                basePredictedKld = beforeStats.AverageBasePredictedKld,
                adjustment,
                adjustedPredictedKld = afterStats.AverageFinalPredictedKld,
                actualCandidateKld = actual.CandidateKld,
                actualTwinKld = actual.TwinKld,
                actualGainOrHarm = actual.GainOrHarm,
                adjustmentReason = actual.HasActualEffect ? "measured-actual-counterfactual-effect" : "prediction-space-gap-fallback",
                matchedRows = before,
                confidence = rule.Confidence,
                groups = rule.GroupStates
                    .OrderBy(x => x.SortOrder)
                    .Select(x => new
                    {
                        x.TensorGroupId,
                        candidate = SafeName(x.CandidateQuantId),
                        reference = SafeName(x.ReferenceQuantId),
                        x.Movement
                    })
                    .ToList()
            };
            matchLogs.Add(log);

            AnsiConsole.MarkupLine(
                $"[green]Applying anomaly rule:[/] rule=[cyan]{Markup.Escape(DescribeRule(rule))}[/] direction=[cyan]{Markup.Escape(rule.RuleDirection)}[/] " +
                $"basePredictedKld=[cyan]{beforeStats.AverageBasePredictedKld:0.000000}[/] adjustment=[cyan]{adjustment:0.000000}[/] " +
                $"adjustedPredictedKld=[cyan]{afterStats.AverageFinalPredictedKld:0.000000}[/] " +
                $"actualCandidateKld=[cyan]{FmtNullable(actual.CandidateKld)}[/] actualTwinKld=[cyan]{FmtNullable(actual.TwinKld)}[/] " +
                $"actualGainOrHarm=[cyan]{FmtNullable(actual.GainOrHarm)}[/] reason=[cyan]{Markup.Escape(actual.HasActualEffect ? "measured-actual-counterfactual-effect" : "prediction-space-gap-fallback")}[/] " +
                $"matched DuckDB rows=[cyan]{before:N0}[/]");
        }

        await ReRankAsync(c, ct);

        return new AnomalyAdjustmentSummary
        {
            AppliedRuleCount = rules.Count,
            MatchedRowCount = totalMatched,
            DuckDbPath = _store.GetDatabaseFilePath(),
            RuleMatches = matchLogs
        };
    }


    private static string BuildRuleWhere(AnomalyInteractionRule rule)
    {
        if (rule.GroupStates.Count == 0)
            return string.Empty;

        if (BaselineQuants.IsNativeExactAlias(rule.ReferenceQuantId) ||
            rule.GroupStates.Any(x => BaselineQuants.IsNativeExactAlias(x.CandidateQuantId) || BaselineQuants.IsNativeExactAlias(x.ReferenceQuantId)))
        {
            return string.Empty;
        }

        var states = rule.GroupStates.ToDictionary(x => x.TensorGroupId, x => x.CandidateQuantId);
        var predicates = new List<string>
        {
            CombinationDuckDbSchema.ActiveCandidatePredicateSql,
            "BaseRankSafeKld IS NOT NULL",
            $"BaseQuant = {rule.ReferenceQuantId}"
        };

        // Match against the full normalized effective active vector. Sparse DuckDB rows
        // may still exist from the normal search space, so matching normalizes NULL slot
        // value 0 to BaseQuant, while explicit contextual probe/rule persistence remains
        // strict and never stores sparse anomaly identities.
        foreach (var group in ActiveGroups())
        {
            string? column = ColumnNameForGroupId(group.UniqueId);
            if (column == null)
                return string.Empty;

            byte expectedQuantId = states.TryGetValue(group.UniqueId, out var candidateQuantId)
                ? candidateQuantId
                : rule.ReferenceQuantId;

            predicates.Add($"(CASE WHEN {column} = 0 THEN BaseQuant ELSE CAST({column} AS INTEGER) - 1 END) = {expectedQuantId}");
        }

        return string.Join(" AND ", predicates);
    }

    private static IReadOnlyList<TensorGroup> ActiveGroups()
    {
        TensorGroup[] ordered =
        [
            TReg.Embeddings,
            TReg.LmHead,
            TReg.AttnQ,
            TReg.AttnKV,
            TReg.AttnOutput,
            TReg.FfnUpGate,
            TReg.FfnDown,
            TReg.MoeExperts,
            TReg.MoeRouter
        ];

        return ordered
            .Where(g => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == g.UniqueId))
            .OrderBy(g => g.UniqueId)
            .ToList();
    }

    private static string? ColumnNameForGroupId(byte groupId)
    {
        if (groupId == TReg.Embeddings.UniqueId) return "Embeddings";
        if (groupId == TReg.LmHead.UniqueId) return "LmHead";
        if (groupId == TReg.AttnQ.UniqueId) return "AttnQ";
        if (groupId == TReg.AttnKV.UniqueId) return "AttnKV";
        if (groupId == TReg.AttnOutput.UniqueId) return "AttnOutput";
        if (groupId == TReg.FfnUpGate.UniqueId) return "FfnUpGate";
        if (groupId == TReg.FfnDown.UniqueId) return "FfnDown";
        if (groupId == TReg.MoeExperts.UniqueId) return "MoeExperts";
        if (groupId == TReg.MoeRouter.UniqueId) return "MoeRouter";
        return null;
    }

    private static async Task ReRankAsync(DuckDBConnection c, CancellationToken ct)
    {
        await ExecuteAsync(c, $@"
DROP TABLE IF EXISTS temp_anomaly_rerank;
CREATE TEMP TABLE temp_anomaly_rerank AS
SELECT {CombinationDuckDbSchema.SlotColumnList},
       CAST(ROW_NUMBER() OVER (
           ORDER BY COALESCE(FinalPredictedKld, PredictedKld) ASC,
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
       ) AS UBIGINT) AS NewPredictionRank
FROM {CombinationDuckDbSchema.TableName}
WHERE COALESCE(FinalPredictedKld, PredictedKld) IS NOT NULL
  AND PredictedSizeBytes IS NOT NULL
  AND PredictionConfidence IS NOT NULL
  AND {CombinationDuckDbSchema.ActiveCandidatePredicateSql};

UPDATE {CombinationDuckDbSchema.TableName} t
SET PredictionRank = r.NewPredictionRank
FROM temp_anomaly_rerank r
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "r")};", ct);
    }

    private static async Task<long> CountMatchesAsync(DuckDBConnection c, string where, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {CombinationDuckDbSchema.TableName} WHERE {where};";
        return ToInt64(await cmd.ExecuteScalarAsync(ct));
    }


    private static async Task<PredictionMatchStats> LoadPredictionStatsAsync(DuckDBConnection c, string where, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = $@"
SELECT AVG(COALESCE(BaseRankSafeKld, PredictedKld)),
       AVG(COALESCE(FinalPredictedKld, PredictedKld))
FROM {CombinationDuckDbSchema.TableName}
WHERE {where};";

        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            return new PredictionMatchStats(0d, 0d);

        return new PredictionMatchStats(ToDouble(r.GetValue(0)), ToDouble(r.GetValue(1)));
    }

    private static ActualRuleEffect ExtractActualEffect(AnomalyInteractionRule rule)
    {
        if (string.IsNullOrWhiteSpace(rule.MetadataJson))
            return new ActualRuleEffect(null, null, null, false);

        try
        {
            using var doc = JsonDocument.Parse(rule.MetadataJson);
            var root = doc.RootElement;
            double? candidate = TryGetDouble(root, "actualCandidateKld");
            double? twin = TryGetDouble(root, "actualTwinKld");
            double? gain = TryGetDouble(root, "actualGainOrHarm");
            return new ActualRuleEffect(candidate, twin, gain, candidate.HasValue && twin.HasValue && gain.HasValue);
        }
        catch
        {
            return new ActualRuleEffect(null, null, null, false);
        }
    }

    private static double? TryGetDouble(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var d)
            ? d
            : null;
    }

    private static string FmtNullable(double? value) => value.HasValue ? value.Value.ToString("0.000000") : "n/a";

    private static double ToDouble(object? value)
    {
        if (value is null || value is DBNull)
            return 0d;

        if (value is BigInteger big)
            return (double)big;

        return Convert.ToDouble(value);
    }

    private static long ToInt64(object? value)
    {
        if (value is null || value is DBNull)
            return 0L;

        if (value is BigInteger big)
            return (long)big;

        return Convert.ToInt64(value);
    }

    private static async Task ExecuteAsync(DuckDBConnection c, string sql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        await cmd.ExecuteNonQueryAsync(ct);
    }

    private static async Task ConfigureSessionAsync(DuckDBConnection c, CancellationToken ct)
    {
        await ExecuteAsync(c, "SET preserve_insertion_order = false;", ct);
        await ExecuteAsync(c, $"SET threads = {Math.Max(1, Environment.ProcessorCount)};", ct);
    }

    private static string SqlDouble(double value) => value.ToString(System.Globalization.CultureInfo.InvariantCulture);

    private static string DescribeRule(AnomalyInteractionRule rule)
    {
        return string.Join(" + ", rule.GroupStates
            .OrderBy(x => x.SortOrder)
            .Select(x => $"{ColumnNameForGroupId(x.TensorGroupId)}={SafeName(x.CandidateQuantId)}")) +
               $" in {SafeName(rule.ReferenceQuantId)} context";
    }

    private readonly record struct PredictionMatchStats(double AverageBasePredictedKld, double AverageFinalPredictedKld);
    private readonly record struct ActualRuleEffect(double? CandidateKld, double? TwinKld, double? GainOrHarm, bool HasActualEffect);

    private static string SafeName(byte quantId)
    {
        try
        {
            return BaselineQuants.FromId(quantId).Names[0];
        }
        catch
        {
            return $"id:{quantId}";
        }
    }
}