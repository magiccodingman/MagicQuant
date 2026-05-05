using DuckDB.NET.Data;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;
using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace MagicQuant.Services;

public sealed class AnomalyAdjustedPredictionService
{
    private const double UpdateEpsilon = 1e-15d;

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
            if (IsDirection(rule, AnomalyRuleDirection.SuppressionOnly))
            {
                var log = LogSuppressionOnly(rule);
                matchLogs.Add(log);
                AnsiConsole.MarkupLine(
                    $"[grey]Anomaly rule suppression-only:[/] rule=[cyan]{Markup.Escape(DescribeRule(rule))}[/] no prediction score mutation.");
                continue;
            }

            if (IsDirection(rule, AnomalyRuleDirection.Beneficial))
            {
                AnsiConsole.MarkupLine("[grey]Broad beneficial same-selected-group adjustment disabled; applying pairwise twin ordering only.[/]");
                var result = await ApplyBeneficialPairwiseOrderingRuleAsync(c, rule, ct);
                if (result.HasValue)
                {
                    var pairwise = result.Value;
                    totalMatched += pairwise.MatchedCandidateRows;
                    matchLogs.Add(pairwise.LogObject);
                }

                continue;
            }

            if (IsDirection(rule, AnomalyRuleDirection.Harmful))
            {
                var result = await ApplyBroadHarmfulDemotionRuleAsync(c, rule, ct);
                if (result.HasValue)
                {
                    var harmful = result.Value;
                    totalMatched += harmful.MatchedRows;
                    matchLogs.Add(harmful.LogObject);
                }

                continue;
            }
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

    private static async Task<BeneficialPairwiseResult?> ApplyBeneficialPairwiseOrderingRuleAsync(
        DuckDBConnection c,
        AnomalyInteractionRule rule,
        CancellationToken ct)
    {
        string candidateWhere = BuildRuleCandidateWhere(rule, "c");
        if (string.IsNullOrWhiteSpace(candidateWhere))
            return null;

        await ExecuteAsync(c, "DROP TABLE IF EXISTS temp_anomaly_rule_candidates;", ct);
        await ExecuteAsync(c, "DROP TABLE IF EXISTS temp_anomaly_rule_twins;", ct);
        await ExecuteAsync(c, "DROP TABLE IF EXISTS temp_anomaly_rule_updates;", ct);

        await ExecuteAsync(c, $@"
CREATE TEMP TABLE temp_anomaly_rule_candidates AS
SELECT {CombinationDuckDbSchema.QualifySlotColumnList("c")},
       COALESCE(c.BaseRankSafeKld, c.PredictedKld) AS CandidateBaseRankSafeKld,
       COALESCE(c.FinalPredictedKld, c.PredictedKld, c.BaseRankSafeKld) AS CandidateCurrentFinalKld
FROM {CombinationDuckDbSchema.TableName} c
WHERE {candidateWhere};", ct);

        long matchedCandidateRows = await CountTempRowsAsync(c, "temp_anomaly_rule_candidates", ct);
        if (matchedCandidateRows == 0)
            return null;

        string twinJoin = BuildPairwiseTwinJoinPredicate(rule, "c", "t");
        if (string.IsNullOrWhiteSpace(twinJoin))
            return null;

        await ExecuteAsync(c, $@"
CREATE TEMP TABLE temp_anomaly_rule_twins AS
SELECT {CombinationDuckDbSchema.QualifySlotColumnList("c")},
       MIN(COALESCE(t.FinalPredictedKld, t.PredictedKld, t.BaseRankSafeKld)) AS TwinEffectiveKld
FROM temp_anomaly_rule_candidates c
JOIN {CombinationDuckDbSchema.TableName} t
  ON {twinJoin}
WHERE COALESCE(t.FinalPredictedKld, t.PredictedKld, t.BaseRankSafeKld) IS NOT NULL
GROUP BY {CombinationDuckDbSchema.QualifySlotColumnList("c")};", ct);

        long twinRowsFound = await CountTempRowsAsync(c, "temp_anomaly_rule_twins", ct);
        long missingTwinRows = Math.Max(0, matchedCandidateRows - twinRowsFound);

        double margin = Math.Max(0d, Config.AnomalyDetection.PredictionSpaceViolationMargin);
        await ExecuteAsync(c, $@"
CREATE TEMP TABLE temp_anomaly_rule_updates AS
SELECT {CombinationDuckDbSchema.QualifySlotColumnList("c")},
       c.CandidateBaseRankSafeKld,
       c.CandidateCurrentFinalKld,
       tw.TwinEffectiveKld,
       GREATEST(0.0, LEAST(c.CandidateCurrentFinalKld, tw.TwinEffectiveKld - {SqlDouble(margin)})) AS NewFinalKld,
       c.CandidateCurrentFinalKld - GREATEST(0.0, LEAST(c.CandidateCurrentFinalKld, tw.TwinEffectiveKld - {SqlDouble(margin)})) AS OrderingAdjustmentApplied
FROM temp_anomaly_rule_candidates c
JOIN temp_anomaly_rule_twins tw
  ON {CombinationDuckDbSchema.BuildSlotEqualityPredicate("c", "tw")}
WHERE GREATEST(0.0, LEAST(c.CandidateCurrentFinalKld, tw.TwinEffectiveKld - {SqlDouble(margin)})) < c.CandidateCurrentFinalKld - {SqlDouble(UpdateEpsilon)};", ct);

        var stats = await LoadPairwiseUpdateStatsAsync(c, ct);

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName} t
SET FinalPredictedKld = u.NewFinalKld,
    PredictedKld = u.NewFinalKld,
    AnomalyAdjustmentKld = u.NewFinalKld - COALESCE(t.BaseRankSafeKld, t.PredictedKld, 0.0)
FROM temp_anomaly_rule_updates u
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "u")};", ct);

        double advisoryCap = Math.Max(0d, Config.AnomalyDetection.MaxConfirmedPairwiseOrderingAdjustmentKld);
        bool exceededAdvisoryCap = advisoryCap > 0d && stats.MaxOrderingAdjustmentApplied > advisoryCap;
        var actual = ExtractActualEffect(rule);

        var log = new
        {
            ruleId = rule.Id,
            direction = rule.RuleDirection,
            ruleType = rule.RuleType,
            applicationMode = "pairwise-twin-ordering",
            broadBeneficialSameSelectedGroupAdjustment = "disabled",
            referenceQuant = SafeName(rule.ReferenceQuantId),
            groupSetHash = rule.GroupSetHash,
            matchedCandidateRows,
            twinRowsFound,
            missingTwinRows,
            rowsReordered = stats.RowsReordered,
            maxOrderingAdjustmentApplied = stats.MaxOrderingAdjustmentApplied,
            meanOrderingAdjustmentApplied = stats.MeanOrderingAdjustmentApplied,
            minCandidateBefore = stats.MinCandidateBefore,
            meanTwinEffectiveKld = stats.MeanTwinEffectiveKld,
            meanCandidateAfter = stats.MeanCandidateAfter,
            predictionSpaceViolationMargin = margin,
            advisoryMaxConfirmedPairwiseOrderingAdjustmentKld = advisoryCap,
            exceededAdvisoryCap,
            actualCandidateKld = actual.CandidateKld,
            actualTwinKld = actual.TwinKld,
            actualGainOrHarm = actual.GainOrHarm,
            confidence = rule.Confidence,
            groups = BuildGroupLog(rule)
        };

        AnsiConsole.MarkupLine(
            $"[green]Applying beneficial anomaly rule as pairwise ordering:[/] rule=[cyan]{Markup.Escape(DescribeRule(rule))}[/] " +
            $"matchedCandidateRows=[cyan]{matchedCandidateRows:N0}[/] twinRowsFound=[cyan]{twinRowsFound:N0}[/] missingTwinRows=[cyan]{missingTwinRows:N0}[/] " +
            $"rowsReordered=[cyan]{stats.RowsReordered:N0}[/] maxOrderingAdjustmentApplied=[cyan]{stats.MaxOrderingAdjustmentApplied:0.000000}[/] " +
            $"meanTwinEffectiveKld=[cyan]{stats.MeanTwinEffectiveKld:0.000000}[/] meanCandidateAfter=[cyan]{stats.MeanCandidateAfter:0.000000}[/]");

        if (missingTwinRows > 0)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Beneficial anomaly twin lookup miss:[/] rule=[cyan]{Markup.Escape(DescribeRule(rule))}[/] missingTwinRows=[cyan]{missingTwinRows:N0}[/]. No broad fallback boost was applied.");
        }

        if (exceededAdvisoryCap)
        {
            AnsiConsole.MarkupLine(
                $"[yellow]Pairwise ordering adjustment exceeded advisory cap:[/] maxApplied=[cyan]{stats.MaxOrderingAdjustmentApplied:0.000000}[/], advisoryCap=[cyan]{advisoryCap:0.000000}[/]. Confirmed pairwise ordering was preserved anyway.");
        }

        return new BeneficialPairwiseResult(matchedCandidateRows, twinRowsFound, missingTwinRows, stats.RowsReordered, log);
    }

    private static async Task<BroadRuleResult?> ApplyBroadHarmfulDemotionRuleAsync(
        DuckDBConnection c,
        AnomalyInteractionRule rule,
        CancellationToken ct)
    {
        string where = BuildRuleCandidateWhere(rule, null);
        if (string.IsNullOrWhiteSpace(where))
            return null;

        double adjustment = rule.AppliedPredictionSpaceAdjustmentKld;
        if (adjustment <= 0d)
            adjustment = Math.Max(Config.AnomalyDetection.PredictionSpaceViolationMargin, Math.Abs(adjustment));

        if (adjustment <= 0d)
            return null;

        long before = await CountMatchesAsync(c, where, ct);
        if (before == 0)
            return null;

        var beforeStats = await LoadPredictionStatsAsync(c, where, ct);

        string expression = $"LEAST(COALESCE(AnomalyAdjustmentKld, 0.0) + ({SqlDouble(adjustment)}), LEAST({SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)}, COALESCE(BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))";

        await ExecuteAsync(c, $@"
UPDATE {CombinationDuckDbSchema.TableName}
SET AnomalyAdjustmentKld = {expression},
    FinalPredictedKld = GREATEST(0.0, COALESCE(BaseRankSafeKld, PredictedKld, 0.0) + {expression}),
    PredictedKld = GREATEST(0.0, COALESCE(BaseRankSafeKld, PredictedKld, 0.0) + {expression})
WHERE {where};", ct);

        var afterStats = await LoadPredictionStatsAsync(c, where, ct);
        var actual = ExtractActualEffect(rule);
        var log = new
        {
            ruleId = rule.Id,
            direction = rule.RuleDirection,
            ruleType = rule.RuleType,
            applicationMode = "broad-harmful-demotion",
            referenceQuant = SafeName(rule.ReferenceQuantId),
            groupSetHash = rule.GroupSetHash,
            basePredictedKld = beforeStats.AverageBasePredictedKld,
            adjustment,
            adjustedPredictedKld = afterStats.AverageFinalPredictedKld,
            actualCandidateKld = actual.CandidateKld,
            actualTwinKld = actual.TwinKld,
            actualGainOrHarm = actual.GainOrHarm,
            broadHarmfulDemotionMatches = before,
            totalAdjustedRows = before,
            matchedRows = before,
            confidence = rule.Confidence,
            groups = BuildGroupLog(rule)
        };

        AnsiConsole.MarkupLine(
            $"[yellow]Applying broad harmful demotion:[/] rule=[cyan]{Markup.Escape(DescribeRule(rule))}[/] " +
            $"matchedRows=[cyan]{before:N0}[/] adjustment=[cyan]{adjustment:0.000000}[/] " +
            $"beforeAvg=[cyan]{beforeStats.AverageFinalPredictedKld:0.000000}[/] afterAvg=[cyan]{afterStats.AverageFinalPredictedKld:0.000000}[/]");

        return new BroadRuleResult(before, log);
    }

    private static string BuildRuleCandidateWhere(AnomalyInteractionRule rule, string? alias)
    {
        if (rule.GroupStates.Count == 0)
            return string.Empty;

        if (BaselineQuants.IsNativeExactAlias(rule.ReferenceQuantId) ||
            rule.GroupStates.Any(x => BaselineQuants.IsNativeExactAlias(x.CandidateQuantId) || BaselineQuants.IsNativeExactAlias(x.ReferenceQuantId)))
        {
            return string.Empty;
        }

        string q(string column) => string.IsNullOrWhiteSpace(alias) ? column : $"{alias}.{column}";

        var predicates = new List<string>
        {
            $"COALESCE({q("IsProtectedAnchor")}, FALSE) = FALSE",
            $"{q("BaseRankSafeKld")} IS NOT NULL",
            $"{q("BaseQuant")} = {rule.ReferenceQuantId}"
        };

        foreach (var state in rule.GroupStates.OrderBy(x => x.SortOrder))
        {
            string? column = ColumnNameForGroupId(state.TensorGroupId);
            if (column == null)
                return string.Empty;

            predicates.Add($"{EffectiveQuantSql(alias, column)} = {state.CandidateQuantId}");
        }

        return string.Join(" AND ", predicates);
    }

    private static string BuildPairwiseTwinJoinPredicate(AnomalyInteractionRule rule, string candidateAlias, string twinAlias)
    {
        if (rule.GroupStates.Count == 0)
            return string.Empty;

        var byColumn = new Dictionary<string, AnomalyInteractionRuleGroupState>(StringComparer.Ordinal);
        foreach (var state in rule.GroupStates.OrderBy(x => x.SortOrder))
        {
            string? column = ColumnNameForGroupId(state.TensorGroupId);
            if (column == null)
                return string.Empty;

            byColumn[column] = state;
        }

        var predicates = new List<string>
        {
            $"{twinAlias}.BaseQuant = {candidateAlias}.BaseQuant"
        };

        foreach (string column in CombinationDuckDbSchema.SlotColumns.Skip(1))
        {
            if (!byColumn.TryGetValue(column, out var state))
            {
                predicates.Add($"{twinAlias}.{column} = {candidateAlias}.{column}");
                continue;
            }

            byte referenceStoredSlot = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(state.ReferenceQuantId);
            predicates.Add($"({twinAlias}.{column} = {referenceStoredSlot} OR ({state.ReferenceQuantId} = {twinAlias}.BaseQuant AND {twinAlias}.{column} = 0))");
        }

        return string.Join(" AND ", predicates);
    }

    private static string EffectiveQuantSql(string? alias, string column)
    {
        string prefix = string.IsNullOrWhiteSpace(alias) ? string.Empty : alias + ".";
        return $"(CASE WHEN {prefix}{column} = 0 THEN {prefix}BaseQuant ELSE CAST({prefix}{column} AS INTEGER) - 1 END)";
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

    private static async Task<long> CountTempRowsAsync(DuckDBConnection c, string tableName, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM {tableName};";
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

    private static async Task<PairwiseUpdateStats> LoadPairwiseUpdateStatsAsync(DuckDBConnection c, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = @"
SELECT COUNT(*),
       COALESCE(MAX(OrderingAdjustmentApplied), 0.0),
       COALESCE(AVG(OrderingAdjustmentApplied), 0.0),
       COALESCE(MIN(CandidateCurrentFinalKld), 0.0),
       COALESCE(AVG(TwinEffectiveKld), 0.0),
       COALESCE(AVG(NewFinalKld), 0.0)
FROM temp_anomaly_rule_updates;";

        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            return new PairwiseUpdateStats(0, 0d, 0d, 0d, 0d, 0d);

        return new PairwiseUpdateStats(
            ToInt64(r.GetValue(0)),
            ToDouble(r.GetValue(1)),
            ToDouble(r.GetValue(2)),
            ToDouble(r.GetValue(3)),
            ToDouble(r.GetValue(4)),
            ToDouble(r.GetValue(5)));
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

    private static object LogSuppressionOnly(AnomalyInteractionRule rule)
    {
        return new
        {
            ruleId = rule.Id,
            direction = rule.RuleDirection,
            ruleType = rule.RuleType,
            applicationMode = "suppression-only",
            scoreMutation = false,
            referenceQuant = SafeName(rule.ReferenceQuantId),
            groupSetHash = rule.GroupSetHash,
            matchedRows = 0,
            totalAdjustedRows = 0,
            confidence = rule.Confidence,
            groups = BuildGroupLog(rule)
        };
    }

    private static IReadOnlyList<object> BuildGroupLog(AnomalyInteractionRule rule)
    {
        return rule.GroupStates
            .OrderBy(x => x.SortOrder)
            .Select(x => (object)new
            {
                x.TensorGroupId,
                group = ColumnNameForGroupId(x.TensorGroupId),
                candidate = SafeName(x.CandidateQuantId),
                reference = SafeName(x.ReferenceQuantId),
                x.Movement
            })
            .ToList();
    }

    private static bool IsDirection(AnomalyInteractionRule rule, AnomalyRuleDirection direction) =>
        string.Equals(rule.RuleDirection, direction.ToString(), StringComparison.OrdinalIgnoreCase);

    private static string FmtNullable(double? value) => value.HasValue ? value.Value.ToString("0.000000", CultureInfo.InvariantCulture) : "n/a";

    private static double ToDouble(object? value)
    {
        if (value is null || value is DBNull)
            return 0d;

        if (value is BigInteger big)
            return (double)big;

        return Convert.ToDouble(value, CultureInfo.InvariantCulture);
    }

    private static long ToInt64(object? value)
    {
        if (value is null || value is DBNull)
            return 0L;

        if (value is BigInteger big)
            return (long)big;

        return Convert.ToInt64(value, CultureInfo.InvariantCulture);
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

    private static string SqlDouble(double value) => value.ToString(CultureInfo.InvariantCulture);

    private static string DescribeRule(AnomalyInteractionRule rule)
    {
        return string.Join(" + ", rule.GroupStates
            .OrderBy(x => x.SortOrder)
            .Select(x => $"{ColumnNameForGroupId(x.TensorGroupId)}={SafeName(x.CandidateQuantId)}>{SafeName(x.ReferenceQuantId)}")) +
               $" in {SafeName(rule.ReferenceQuantId)} context";
    }

    private readonly record struct PredictionMatchStats(double AverageBasePredictedKld, double AverageFinalPredictedKld);
    private readonly record struct ActualRuleEffect(double? CandidateKld, double? TwinKld, double? GainOrHarm, bool HasActualEffect);
    private readonly record struct BeneficialPairwiseResult(long MatchedCandidateRows, long TwinRowsFound, long MissingTwinRows, long RowsReordered, object LogObject);
    private readonly record struct BroadRuleResult(long MatchedRows, object LogObject);
    private readonly record struct PairwiseUpdateStats(
        long RowsReordered,
        double MaxOrderingAdjustmentApplied,
        double MeanOrderingAdjustmentApplied,
        double MinCandidateBefore,
        double MeanTwinEffectiveKld,
        double MeanCandidateAfter);

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
