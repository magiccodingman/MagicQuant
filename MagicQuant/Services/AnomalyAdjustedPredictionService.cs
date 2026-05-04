using DuckDB.NET.Data;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
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
        long exactMatched = 0;
        long sameSelectedMatched = 0;
        long equivalentMatched = 0;
        long harmfulMatched = 0;
        long suppressedMatched = 0;
        var matchLogs = new List<object>();

        foreach (var rule in rules.OrderByDescending(x => x.Confidence).ThenBy(x => x.Id))
        {
            if (!IsRuleUsable(rule))
                continue;

            var tierResults = new List<RuleTierApplyResult>();
            tierResults.Add(await ApplyRuleTierAsync(
                c,
                rule,
                SynergyTemplateMatchTier.ExactContext,
                Config.SynergyDetection.ExactContextConfidenceMultiplier,
                ct));

            if (Config.SynergyDetection.Enabled)
            {
                tierResults.Add(await ApplyRuleTierAsync(
                    c,
                    rule,
                    SynergyTemplateMatchTier.SameSelectedGroups,
                    Config.SynergyDetection.SameSelectedGroupsConfidenceMultiplier,
                    ct));

                tierResults.Add(await ApplyRuleTierAsync(
                    c,
                    rule,
                    SynergyTemplateMatchTier.EquivalentQuantFamily,
                    Config.SynergyDetection.EquivalentQuantFamilyConfidenceMultiplier,
                    ct));
            }

            foreach (var tier in tierResults.Where(x => x.MatchedRows > 0))
            {
                totalMatched += tier.MatchedRows;
                if (tier.Tier == SynergyTemplateMatchTier.ExactContext) exactMatched += tier.MatchedRows;
                if (tier.Tier == SynergyTemplateMatchTier.SameSelectedGroups) sameSelectedMatched += tier.MatchedRows;
                if (tier.Tier == SynergyTemplateMatchTier.EquivalentQuantFamily) equivalentMatched += tier.MatchedRows;
                if (rule.RuleDirection == AnomalyRuleDirection.Harmful.ToString()) harmfulMatched += tier.MatchedRows;
                if (rule.RuleDirection == AnomalyRuleDirection.SuppressionOnly.ToString()) suppressedMatched += tier.MatchedRows;

                matchLogs.Add(new
                {
                    ruleId = rule.Id,
                    templateType = ResolveTemplateType(rule),
                    direction = rule.RuleDirection,
                    tier = tier.Tier.ToString(),
                    tierMultiplier = tier.Multiplier,
                    referenceQuant = SafeName(rule.ReferenceQuantId),
                    groupSetHash = rule.GroupSetHash,
                    exactContextMatches = tier.Tier == SynergyTemplateMatchTier.ExactContext ? tier.MatchedRows : 0,
                    sameSelectedGroupMatches = tier.Tier == SynergyTemplateMatchTier.SameSelectedGroups ? tier.MatchedRows : 0,
                    equivalentQuantFamilyMatches = tier.Tier == SynergyTemplateMatchTier.EquivalentQuantFamily ? tier.MatchedRows : 0,
                    totalAdjustedRows = tier.MatchedRows,
                    basePredictedKld = tier.Before.AverageBasePredictedKld,
                    adjustedPredictedKld = tier.After.AverageFinalPredictedKld,
                    averageAdjustment = tier.After.AverageAnomalyAdjustmentKld,
                    actualCandidateKld = ExtractActualEffect(rule).CandidateKld,
                    actualTwinKld = ExtractActualEffect(rule).TwinKld,
                    actualGainOrHarm = ExtractActualEffect(rule).GainOrHarm,
                    adjustmentReason = "prediction-space-virtual-twin-rank-movement",
                    confidence = rule.Confidence,
                    effectiveConfidence = tier.EffectiveConfidence,
                    candidatePredicateReason = tier.CandidatePredicateReason,
                    virtualTwinPredicateReason = tier.VirtualTwinPredicateReason,
                    groups = rule.GroupStates
                        .OrderBy(x => x.SortOrder)
                        .Select(x => new
                        {
                            x.TensorGroupId,
                            group = ColumnNameForGroupId(x.TensorGroupId),
                            candidate = SafeName(x.CandidateQuantId),
                            reference = SafeName(x.ReferenceQuantId),
                            x.Movement
                        })
                        .ToList()
                });

                AnsiConsole.MarkupLine(
                    $"[green]Applying synergy template:[/] template=[cyan]{Markup.Escape(DescribeRule(rule))}[/] tier=[cyan]{tier.Tier}[/] direction=[cyan]{Markup.Escape(rule.RuleDirection)}[/] " +
                    $"basePredictedKld=[cyan]{tier.Before.AverageBasePredictedKld:0.000000}[/] adjustedPredictedKld=[cyan]{tier.After.AverageFinalPredictedKld:0.000000}[/] " +
                    $"avgAdjustment=[cyan]{tier.After.AverageAnomalyAdjustmentKld:0.000000}[/] reason=[cyan]prediction-space-virtual-twin-rank-movement[/] matched DuckDB rows=[cyan]{tier.MatchedRows:N0}[/]");
            }

            if (tierResults.All(x => x.MatchedRows == 0))
            {
                matchLogs.Add(new
                {
                    ruleId = rule.Id,
                    templateType = ResolveTemplateType(rule),
                    direction = rule.RuleDirection,
                    referenceQuant = SafeName(rule.ReferenceQuantId),
                    groupSetHash = rule.GroupSetHash,
                    exactContextMatches = 0,
                    sameSelectedGroupMatches = 0,
                    equivalentQuantFamilyMatches = 0,
                    totalAdjustedRows = 0,
                    reason = "No DuckDB rows matched this transferable synergy template. Either the search space does not contain the selected group states, virtual raised twins were not generated/predictable, active groups were sparse/native-exact, or confidence/generalization thresholds blocked the tier."
                });
            }
        }

        await ReRankAsync(c, ct);

        return new AnomalyAdjustmentSummary
        {
            AppliedRuleCount = rules.Count,
            MatchedRowCount = totalMatched,
            ExactContextMatches = exactMatched,
            SameSelectedGroupMatches = sameSelectedMatched,
            EquivalentQuantFamilyMatches = equivalentMatched,
            SuppressedMatches = suppressedMatched,
            HarmfulMatches = harmfulMatched,
            DuckDbPath = _store.GetDatabaseFilePath(),
            RuleMatches = matchLogs
        };
    }

    private static async Task<RuleTierApplyResult> ApplyRuleTierAsync(
        DuckDBConnection c,
        AnomalyInteractionRule rule,
        SynergyTemplateMatchTier tier,
        double tierMultiplier,
        CancellationToken ct)
    {
        var empty = new RuleTierApplyResult(
            tier,
            0,
            tierMultiplier,
            0d,
            new PredictionMatchStats(0d, 0d, 0d),
            new PredictionMatchStats(0d, 0d, 0d),
            string.Empty,
            string.Empty);

        if (tierMultiplier <= 0d)
            return empty;

        double effectiveConfidence = rule.Confidence * tierMultiplier;
        if (effectiveConfidence < Config.SynergyDetection.MinConfidenceToApplyAdjustment)
            return empty with { CandidatePredicateReason = "BlockedByMinSynergyConfidence" };

        var matchSql = BuildRuleMatchSubquery(rule, tier, out var candidateReason, out var twinReason);
        if (string.IsNullOrWhiteSpace(matchSql))
            return empty with { CandidatePredicateReason = candidateReason, VirtualTwinPredicateReason = twinReason };

        long before = await CountMatchesAsync(c, matchSql, ct);
        if (before == 0)
            return empty with { CandidatePredicateReason = candidateReason, VirtualTwinPredicateReason = twinReason };

        var beforeStats = await LoadPredictionStatsAsync(c, matchSql, ct);
        string signedMagnitude = BuildAdjustmentMagnitudeSql(rule, effectiveConfidence);

        await ExecuteAsync(c, $@"
WITH matches AS (
{matchSql}
), calculated AS (
    SELECT {CombinationDuckDbSchema.SlotColumnList},
           {signedMagnitude} AS Delta
    FROM matches
)
UPDATE {CombinationDuckDbSchema.TableName} t
SET AnomalyAdjustmentKld = CASE
        WHEN calculated.Delta < 0 THEN GREATEST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, -LEAST({SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentFractionOfBaseKld)}))
        ELSE LEAST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, LEAST({SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))
    END,
    FinalPredictedKld = GREATEST(0.0, COALESCE(t.BaseRankSafeKld, t.PredictedKld, 0.0) + CASE
        WHEN calculated.Delta < 0 THEN GREATEST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, -LEAST({SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentFractionOfBaseKld)}))
        ELSE LEAST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, LEAST({SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))
    END),
    PredictedKld = GREATEST(0.0, COALESCE(t.BaseRankSafeKld, t.PredictedKld, 0.0) + CASE
        WHEN calculated.Delta < 0 THEN GREATEST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, -LEAST({SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentFractionOfBaseKld)}))
        ELSE LEAST(COALESCE(t.AnomalyAdjustmentKld, 0.0) + calculated.Delta, LEAST({SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)}, COALESCE(t.BaseRankSafeKld, 0.0) * {SqlDouble(Config.AnomalyDetection.MaxAdjustmentFractionOfBaseKld)}))
    END)
FROM calculated
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "calculated")};", ct);

        var afterStats = await LoadPredictionStatsAsync(c, matchSql, ct);
        return new RuleTierApplyResult(tier, before, tierMultiplier, effectiveConfidence, beforeStats, afterStats, candidateReason, twinReason);
    }

    private static string BuildAdjustmentMagnitudeSql(AnomalyInteractionRule rule, double effectiveConfidence)
    {
        string baseGap = "(COALESCE(CandidateBaseRankSafeKld, CandidatePredictedKld, 0.0) - COALESCE(VirtualTwinBaseRankSafeKld, VirtualTwinPredictedKld, COALESCE(CandidateBaseRankSafeKld, CandidatePredictedKld, 0.0)))";
        string magnitude = $"(GREATEST({baseGap}, 0.0) + {SqlDouble(Config.AnomalyDetection.PredictionSpaceViolationMargin)}) * {SqlDouble(Config.AnomalyDetection.AnomalyAdjustmentShrinkFactor)} * {SqlDouble(effectiveConfidence)}";

        if (rule.RuleDirection == AnomalyRuleDirection.Harmful.ToString())
            return $"LEAST({magnitude}, {SqlDouble(Config.AnomalyDetection.MaxPositiveAdjustmentKld)})";

        return $"-LEAST({magnitude}, {SqlDouble(Config.SynergyDetection.MaxNegativeAdjustmentKld)})";
    }

    private static string BuildRuleMatchSubquery(
        AnomalyInteractionRule rule,
        SynergyTemplateMatchTier tier,
        out string candidateReason,
        out string virtualTwinReason)
    {
        candidateReason = tier switch
        {
            SynergyTemplateMatchTier.ExactContext => "Full explicit discovery context must match the confirmed dome probe.",
            SynergyTemplateMatchTier.SameSelectedGroups => "Candidate must contain the same selected group states as the confirmed synergy template.",
            SynergyTemplateMatchTier.EquivalentQuantFamily => "Candidate selected groups must use a related quant family/tier.",
            _ => "Unsupported tier."
        };
        virtualTwinReason = "Virtual twin is constructed by raising only selected groups to their counterfactual reference quant while preserving all other effective group states.";

        if (rule.GroupStates.Count == 0)
            return string.Empty;

        if (BaselineQuants.IsNativeExactAlias(rule.ReferenceQuantId) ||
            rule.GroupStates.Any(x => BaselineQuants.IsNativeExactAlias(x.CandidateQuantId) || BaselineQuants.IsNativeExactAlias(x.ReferenceQuantId)))
        {
            candidateReason = "BF16/native/exact template states are not valid contextual synergy templates.";
            return string.Empty;
        }

        var selected = rule.GroupStates.OrderBy(x => x.SortOrder).ToList();
        var predicates = new List<string>
        {
            "t.BaseRankSafeKld IS NOT NULL",
            "t.PredictedSizeBytes IS NOT NULL",
            "COALESCE(t.IsProtectedAnchor, FALSE) = FALSE"
        };

        // Keep transfer controlled for now: a Q8-dome template applies inside rows with the same base/reference quant.
        predicates.Add($"t.BaseQuant = {rule.ReferenceQuantId}");

        foreach (var state in selected)
        {
            string? column = ColumnNameForGroupId(state.TensorGroupId);
            if (column == null)
                return string.Empty;

            if (tier == SynergyTemplateMatchTier.EquivalentQuantFamily)
            {
                var equivalentIds = EquivalentQuantIds(state.CandidateQuantId);
                if (equivalentIds.Count == 0)
                    return string.Empty;

                predicates.Add($"{EffectiveSql("t", column)} IN ({string.Join(",", equivalentIds.Select(x => x.ToString(CultureInfo.InvariantCulture)))})");
            }
            else
            {
                predicates.Add($"{EffectiveSql("t", column)} = {state.CandidateQuantId}");
            }
        }

        if (tier == SynergyTemplateMatchTier.ExactContext)
        {
            var states = selected.ToDictionary(x => x.TensorGroupId, x => x.CandidateQuantId);
            foreach (var group in ActiveGroups())
            {
                string? column = ColumnNameForGroupId(group.UniqueId);
                if (column == null)
                    return string.Empty;

                byte expected = states.TryGetValue(group.UniqueId, out var q) ? q : rule.ReferenceQuantId;
                predicates.Add($"{EffectiveSql("t", column)} = {expected}");
            }
        }
        else
        {
            // Transfer tiers are deliberately weaker than exact context. Keep the
            // original discovery row out of transfer-tier matching so it does not
            // receive duplicate exact + generalized adjustments.
            var selectedGroupIds = selected.Select(x => x.TensorGroupId).ToHashSet();
            var surroundingDifferencePredicates = new List<string>();

            foreach (var group in ActiveGroups())
            {
                if (selectedGroupIds.Contains(group.UniqueId))
                    continue;

                string? column = ColumnNameForGroupId(group.UniqueId);
                if (column == null)
                    return string.Empty;

                surroundingDifferencePredicates.Add($"{EffectiveSql("t", column)} <> {rule.ReferenceQuantId}");
            }

            if (surroundingDifferencePredicates.Count > 0)
                predicates.Add("(" + string.Join(" OR ", surroundingDifferencePredicates) + ")");
        }

        if (tier == SynergyTemplateMatchTier.EquivalentQuantFamily)
        {
            // Equivalent-family matching must be meaningfully broader than exact
            // same-selected-group matching; otherwise the same rows receive both
            // transfer tiers. Require at least one selected group to use a related
            // non-identical quant family member.
            var selectedQuantDifferencePredicates = new List<string>();
            foreach (var state in selected)
            {
                string? column = ColumnNameForGroupId(state.TensorGroupId);
                if (column == null)
                    return string.Empty;

                selectedQuantDifferencePredicates.Add($"{EffectiveSql("t", column)} <> {state.CandidateQuantId}");
            }

            if (selectedQuantDifferencePredicates.Count > 0)
                predicates.Add("(" + string.Join(" OR ", selectedQuantDifferencePredicates) + ")");
        }

        string virtualTwinJoin = BuildVirtualTwinJoinPredicate(selected);
        if (string.IsNullOrWhiteSpace(virtualTwinJoin))
            return string.Empty;

        return $@"    SELECT t.{CombinationDuckDbSchema.SlotColumnList.Replace(", ", ", t.")},
           COALESCE(t.BaseRankSafeKld, t.PredictedKld) AS CandidateBaseRankSafeKld,
           COALESCE(t.PredictedKld, t.BaseRankSafeKld) AS CandidatePredictedKld,
           COALESCE(vt.BaseRankSafeKld, vt.PredictedKld) AS VirtualTwinBaseRankSafeKld,
           COALESCE(vt.PredictedKld, vt.BaseRankSafeKld) AS VirtualTwinPredictedKld
    FROM {CombinationDuckDbSchema.TableName} t
    JOIN {CombinationDuckDbSchema.TableName} vt ON {virtualTwinJoin}
    WHERE {string.Join(" AND ", predicates)}";
    }

    private static string BuildVirtualTwinJoinPredicate(IReadOnlyList<AnomalyInteractionRuleGroupState> selected)
    {
        var selectedMap = selected.ToDictionary(x => x.TensorGroupId, x => x.ReferenceQuantId);
        var predicates = new List<string>
        {
            "vt.BaseQuant = t.BaseQuant",
            "vt.BaseRankSafeKld IS NOT NULL"
        };

        foreach (var group in ActiveGroups())
        {
            string? column = ColumnNameForGroupId(group.UniqueId);
            if (column == null)
                return string.Empty;

            if (selectedMap.TryGetValue(group.UniqueId, out var referenceQuantId))
                predicates.Add($"{EffectiveSql("vt", column)} = {referenceQuantId}");
            else
                predicates.Add($"{EffectiveSql("vt", column)} = {EffectiveSql("t", column)}");
        }

        return string.Join(" AND ", predicates);
    }

    private static string EffectiveSql(string alias, string column) =>
        $"(CASE WHEN {alias}.{column} = 0 THEN {alias}.BaseQuant ELSE CAST({alias}.{column} AS INTEGER) - 1 END)";

    private static IReadOnlyList<byte> EquivalentQuantIds(byte quantId)
    {
        int tier = EffectiveTier(quantId);
        if (tier < 0)
            return Array.Empty<byte>();

        return BaselineQuants.GetAllRecognizedBaselines()
            .Where(x => !BaselineQuants.IsNativeExactAlias(x.UniqueId))
            .Where(x => EffectiveTier(x.UniqueId) == tier)
            .Select(x => x.UniqueId)
            .Distinct()
            .OrderBy(x => x)
            .ToList();
    }

    private static bool IsRuleUsable(AnomalyInteractionRule rule)
    {
        if (rule.GroupStates.Count == 0)
            return false;

        if (rule.RuleStatus != AnomalyRuleStatus.Confirmed.ToString())
            return false;

        if (rule.RuleDirection != AnomalyRuleDirection.Beneficial.ToString() &&
            rule.RuleDirection != AnomalyRuleDirection.Harmful.ToString())
        {
            return false;
        }

        if (BaselineQuants.IsNativeExactAlias(rule.ReferenceQuantId))
            return false;

        return rule.GroupStates.All(x =>
            !BaselineQuants.IsNativeExactAlias(x.CandidateQuantId) &&
            !BaselineQuants.IsNativeExactAlias(x.ReferenceQuantId));
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
DROP TABLE IF EXISTS temp_synergy_rerank;
CREATE TEMP TABLE temp_synergy_rerank AS
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
FROM temp_synergy_rerank r
WHERE {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "r")};", ct);
    }

    private static async Task<long> CountMatchesAsync(DuckDBConnection c, string matchSql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = $"SELECT COUNT(*) FROM ({matchSql}) q;";
        return ToInt64(await cmd.ExecuteScalarAsync(ct));
    }

    private static async Task<PredictionMatchStats> LoadPredictionStatsAsync(DuckDBConnection c, string matchSql, CancellationToken ct)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = $@"
SELECT AVG(COALESCE(t.BaseRankSafeKld, t.PredictedKld)),
       AVG(COALESCE(t.FinalPredictedKld, t.PredictedKld)),
       AVG(COALESCE(t.AnomalyAdjustmentKld, 0.0))
FROM {CombinationDuckDbSchema.TableName} t
JOIN ({matchSql}) m ON {CombinationDuckDbSchema.BuildSlotEqualityPredicate("t", "m")};";

        using var r = await cmd.ExecuteReaderAsync(ct);
        if (!await r.ReadAsync(ct))
            return new PredictionMatchStats(0d, 0d, 0d);

        return new PredictionMatchStats(ToDouble(r.GetValue(0)), ToDouble(r.GetValue(1)), ToDouble(r.GetValue(2)));
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

    private static string ResolveTemplateType(AnomalyInteractionRule rule)
    {
        return rule.RuleType switch
        {
            "SingleGroupInversion" => "CounterfactualSynergy.SingleGroupInversion",
            "PairSynergy" => "CounterfactualSynergy.PairSynergy",
            "HigherOrderSynergy" => "CounterfactualSynergy.HigherOrderSynergy",
            _ => $"CounterfactualSynergy.{rule.RuleType}"
        };
    }

    private static double? TryGetDouble(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var d)
            ? d
            : null;
    }

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
            .Select(x => $"{ColumnNameForGroupId(x.TensorGroupId)}={SafeName(x.CandidateQuantId)}")) +
               $" transferred from {SafeName(rule.ReferenceQuantId)} dome";
    }

    private static int EffectiveTier(byte quantId)
    {
        if (BaselineQuants.IsNativeExactAlias(quantId))
            return 160;

        var baseline = BaselineQuants.FromId(quantId);
        string name = baseline.Names[0].ToUpperInvariant();

        if (name.Contains("Q8") || baseline.BitRange >= 8) return 80;
        if (name.Contains("Q6") || baseline.BitRange == 6) return 60;
        if (name.Contains("Q5") || baseline.BitRange == 5) return 50;
        if (name.Contains("Q4") || name.Contains("IQ4") || baseline.BitRange == 4) return 40;
        if (name.Contains("Q3") || name.Contains("IQ3") || baseline.BitRange == 3) return 30;
        if (name.Contains("Q2") || name.Contains("IQ2") || baseline.BitRange == 2) return 20;

        return baseline.BitRange > 0 ? baseline.BitRange * 10 : -1;
    }

    private readonly record struct PredictionMatchStats(double AverageBasePredictedKld, double AverageFinalPredictedKld, double AverageAnomalyAdjustmentKld);
    private readonly record struct ActualRuleEffect(double? CandidateKld, double? TwinKld, double? GainOrHarm, bool HasActualEffect);
    private readonly record struct RuleTierApplyResult(
        SynergyTemplateMatchTier Tier,
        long MatchedRows,
        double Multiplier,
        double EffectiveConfidence,
        PredictionMatchStats Before,
        PredictionMatchStats After,
        string CandidatePredicateReason,
        string VirtualTwinPredicateReason);

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
