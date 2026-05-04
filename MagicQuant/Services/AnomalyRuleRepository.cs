using System.Text.Json;
using MagicQuant.Models;
using Microsoft.EntityFrameworkCore;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;

namespace MagicQuant.Services;

public sealed class AnomalyRuleRepository
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false
    };

    private readonly QuantFidelityComparerService _movement;

    public AnomalyRuleRepository(QuantFidelityComparerService movement)
    {
        _movement = movement;
    }

    public async Task<AnomalyProbeSession> StartSessionAsync(string sourceRunLabel, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var scope = await ResolveScopeAsync(db, ct);
        var session = new AnomalyProbeSession
        {
            ArchitectureFamilyId = scope.ArchitectureFamilyId,
            TensorGroupProfileId = scope.TensorGroupProfileId,
            AiModelHashId = scope.AiModelHashId,
            ImatrixDefinitionId = scope.ImatrixDefinitionId,
            BenchmarkCategory = (byte)BenchmarkCategory.General,
            StartedUtc = DateTime.UtcNow,
            SourceRunLabel = sourceRunLabel,
            ConfigJson = JsonSerializer.Serialize(Config.AnomalyDetection, JsonOptions)
        };

        db.AnomalyProbeSessions.Add(session);
        await db.SaveChangesAsync(ct);
        return session;
    }

    public async Task CompleteSessionAsync(Guid sessionId, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var session = await db.AnomalyProbeSessions.FirstOrDefaultAsync(x => x.Id == sessionId, ct);
        if (session == null)
            return;

        session.CompletedUtc = DateTime.UtcNow;
        await db.SaveChangesAsync(ct);
    }

    public async Task<IReadOnlyList<AnomalyProbeObservation>> PersistProbeResultsAsync(
        Guid sessionId,
        IReadOnlyCollection<AnomalyProbeResult> results,
        CancellationToken ct)
    {
        if (results.Count == 0)
            return Array.Empty<AnomalyProbeObservation>();

        await using var db = new MagicQuantContext();
        var scope = await ResolveScopeAsync(db, ct);
        var observations = new List<AnomalyProbeObservation>();

        foreach (var result in results)
        {
            if (!_movement.IsContextualQuantizedConfig(result.Plan.ReferenceConfig) ||
                !_movement.IsContextualQuantizedConfig(result.Plan.ProbeConfig))
            {
                // Invalid/sparse/BF16-exact anomaly attempts are logged by the workflow and
                // intentionally not persisted as contextual anomaly observations.
                continue;
            }

            _movement.EnsureAllActiveGroupsExplicit(result.Plan.ReferenceConfig, "persist-observation-reference");
            _movement.EnsureAllActiveGroupsExplicit(result.Plan.ProbeConfig, "persist-observation-probe");

            var referenceCombo = await EnsureTensorComboAsync(db, result.Plan.ReferenceConfig, ct);
            var probeCombo = await EnsureTensorComboAsync(db, result.Plan.ProbeConfig, ct);
            var groups = result.Plan.ProbeGroups.OrderBy(x => x.Group.UniqueId).ToList();
            var movement = result.Plan.Seed.Movement;

            var observation = new AnomalyProbeObservation
            {
                SessionId = sessionId,
                ArchitectureFamilyId = scope.ArchitectureFamilyId,
                TensorGroupProfileId = scope.TensorGroupProfileId,
                AiModelHashId = scope.AiModelHashId,
                ImatrixDefinitionId = scope.ImatrixDefinitionId,
                BenchmarkCategory = (byte)BenchmarkCategory.General,
                ReferenceTensorComboId = referenceCombo.Id,
                ProbeTensorComboId = probeCombo.Id,
                ProbeType = result.Plan.ProbeType,
                Classification = result.Classification.ToString(),
                HypothesisLabel = result.Plan.HypothesisLabel,
                MovementClassification = movement.Classification.ToString(),
                ChangedGroupSetHash = _movement.BuildChangedGroupHash(groups),
                ChangedGroupsJson = JsonSerializer.Serialize(groups.Select(ToGroupLog), JsonOptions),
                CandidateQuantsJson = JsonSerializer.Serialize(groups.ToDictionary(x => x.Group.Name, x => BaselineQuants.FromId(x.CandidateQuantId).Names[0]), JsonOptions),
                ReferenceEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(result.Plan.ReferenceConfig), JsonOptions),
                CandidateEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(result.Plan.ProbeConfig), JsonOptions),
                InactiveGroupsJson = JsonSerializer.Serialize(_movement.BuildInactiveGroupList(), JsonOptions),
                ReferenceTensorConfigKey = TensorConfigIdentity.ToKey(result.Plan.ReferenceConfig),
                ProbeTensorConfigKey = TensorConfigIdentity.ToKey(result.Plan.ProbeConfig),
                IsContextualAnomalyProbe = true,
                OldBf16Isolation = false,
                AllActiveGroupsExplicit = _movement.HasAllActiveGroupsExplicit(result.Plan.ReferenceConfig) && _movement.HasAllActiveGroupsExplicit(result.Plan.ProbeConfig),
                ReferenceQuantId = result.Plan.ReferenceConfig.BaseQuant,
                ActualKld = result.ProbeSnapshot?.Kld ?? 0d,
                PredictedKld = result.Plan.Seed.CandidatePredictedKld,
                ReferenceActualKld = result.ReferenceSnapshot?.Kld ?? 0d,
                ReferencePredictedKld = result.Plan.Seed.TwinPredictedKld,
                ActualGainVsTwin = result.ActualGainVsTwin,
                PredictionSpaceGapVsTwin = result.Plan.Seed.PredictionSpaceGapVsTwin,
                SizeSavingsBytes = ComputeSizeSavings(result.ReferenceSnapshot, result.ProbeSnapshot),
                UpgradeCount = movement.UpgradeCount,
                DowngradeCount = movement.DowngradeCount,
                SameCount = movement.SameCount,
                UnknownCount = movement.UnknownCount,
                NetBitDelta = movement.NetBitDelta,
                RuleDirection = result.RuleDirection.ToString(),
                Accepted = result.Accepted,
                FailureCode = result.FailureCode,
                Message = result.Message,
                CreatedUtc = DateTime.UtcNow
            };

            db.AnomalyProbeObservations.Add(observation);
            observations.Add(observation);
        }

        await db.SaveChangesAsync(ct);
        return observations;
    }

    public async Task<IReadOnlyList<AnomalyInteractionRule>> UpsertRulesFromResultsAsync(
        IReadOnlyCollection<AnomalyProbeResult> results,
        CancellationToken ct)
    {
        var eligible = results
            .Where(x => x.RuleDirection != AnomalyRuleDirection.SuppressionOnly || Config.AnomalyDetection.PersistSuppressionResults)
            .Where(x => x.ReferenceSnapshot != null && x.ProbeSnapshot != null)
            .Where(x => _movement.IsContextualQuantizedConfig(x.Plan.ReferenceConfig))
            .Where(x => _movement.IsContextualQuantizedConfig(x.Plan.ProbeConfig))
            .ToList();

        if (eligible.Count == 0)
            return Array.Empty<AnomalyInteractionRule>();

        await using var db = new MagicQuantContext();
        var scope = await ResolveScopeAsync(db, ct);
        var upserted = new List<AnomalyInteractionRule>();

        foreach (var group in eligible.GroupBy(BuildRuleKey, StringComparer.Ordinal))
        {
            var first = group.First();
            _movement.EnsureAllActiveGroupsExplicit(first.Plan.ReferenceConfig, "upsert-rule-reference");
            _movement.EnsureAllActiveGroupsExplicit(first.Plan.ProbeConfig, "upsert-rule-probe");
            var probeGroups = first.Plan.ProbeGroups.OrderBy(x => x.Group.UniqueId).ToList();
            string groupSetHash = _movement.BuildChangedGroupHash(probeGroups);
            string direction = first.RuleDirection.ToString();
            byte referenceQuantId = first.Plan.ReferenceConfig.BaseQuant;

            var rule = await db.AnomalyInteractionRules
                .Include(x => x.GroupStates)
                .FirstOrDefaultAsync(x =>
                    x.ArchitectureFamilyId == scope.ArchitectureFamilyId &&
                    x.TensorGroupProfileId == scope.TensorGroupProfileId &&
                    x.AiModelHashId == scope.AiModelHashId &&
                    x.ImatrixDefinitionId == scope.ImatrixDefinitionId &&
                    x.BenchmarkCategory == (byte)BenchmarkCategory.General &&
                    x.ReferenceQuantId == referenceQuantId &&
                    x.GroupSetHash == groupSetHash &&
                    x.RuleDirection == direction,
                    ct);

            bool isNew = rule == null;
            if (rule == null)
            {
                rule = new AnomalyInteractionRule
                {
                    ArchitectureFamilyId = scope.ArchitectureFamilyId,
                    TensorGroupProfileId = scope.TensorGroupProfileId,
                    AiModelHashId = scope.AiModelHashId,
                    ImatrixDefinitionId = scope.ImatrixDefinitionId,
                    BenchmarkCategory = (byte)BenchmarkCategory.General,
                    ReferenceQuantId = referenceQuantId,
                    ReferenceContextKey = _movement.ReferenceContextKey(first.Plan.ReferenceConfig),
                    ReferenceEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(first.Plan.ReferenceConfig), JsonOptions),
                    CandidateEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(first.Plan.ProbeConfig), JsonOptions),
                    InactiveGroupsJson = JsonSerializer.Serialize(_movement.BuildInactiveGroupList(), JsonOptions),
                    FullTensorConfigKey = TensorConfigIdentity.ToKey(first.Plan.ProbeConfig),
                    RuleDirection = direction,
                    GroupSetHash = groupSetHash,
                    CreatedUtc = DateTime.UtcNow
                };
                db.AnomalyInteractionRules.Add(rule);
            }

            var rows = group.ToList();
            rule.RuleType = ResolveRuleType(rows);
            rule.RuleStatus = first.RuleDirection == AnomalyRuleDirection.Beneficial || first.RuleDirection == AnomalyRuleDirection.Harmful
                ? AnomalyRuleStatus.Confirmed.ToString()
                : AnomalyRuleStatus.Suppressed.ToString();
            rule.Status = rule.RuleStatus;
            rule.MovementClassification = first.Plan.Seed.Movement.Classification.ToString();
            rule.GroupCount = probeGroups.Count;
            rule.EvidenceCount = Math.Max(rule.EvidenceCount, 0) + rows.Count;
            rule.MeanActualGainVsTwin = rows.Average(x => x.ActualGainVsTwin);
            rule.BestActualGainVsTwin = rows.Max(x => x.ActualGainVsTwin);
            rule.MeanPredictionSpaceGap = rows.Average(x => x.Plan.Seed.PredictionSpaceGapVsTwin);
            rule.BestPredictionSpaceGap = rows.Min(x => x.Plan.Seed.PredictionSpaceGapVsTwin);
            rule.ShrinkFactor = Config.AnomalyDetection.AnomalyAdjustmentShrinkFactor;
            rule.Confidence = ComputeConfidence(rows);
            rule.AppliedPredictionSpaceAdjustmentKld = ComputePredictionAdjustment(first, rule.Confidence);
            rule.ReferenceContextKey = _movement.ReferenceContextKey(first.Plan.ReferenceConfig);
            rule.ReferenceEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(first.Plan.ReferenceConfig), JsonOptions);
            rule.CandidateEffectiveGroupsJson = JsonSerializer.Serialize(_movement.BuildEffectiveGroupVector(first.Plan.ProbeConfig), JsonOptions);
            rule.InactiveGroupsJson = JsonSerializer.Serialize(_movement.BuildInactiveGroupList(), JsonOptions);
            rule.FullTensorConfigKey = TensorConfigIdentity.ToKey(first.Plan.ProbeConfig);
            rule.UpdatedUtc = DateTime.UtcNow;
            rule.MetadataJson = JsonSerializer.Serialize(new
            {
                source = "counterfactual-twin-probe",
                isContextualAnomalyProbe = true,
                oldBf16Isolation = false,
                allActiveGroupsExplicit = true,
                referenceEffectiveGroups = _movement.BuildEffectiveGroupVector(first.Plan.ReferenceConfig),
                candidateEffectiveGroups = _movement.BuildEffectiveGroupVector(first.Plan.ProbeConfig),
                inactiveGroups = _movement.BuildInactiveGroupList(),
                first.Plan.ProbeType,
                first.Plan.HypothesisLabel,
                actualCandidateKld = first.ProbeSnapshot?.Kld,
                actualTwinKld = first.ReferenceSnapshot?.Kld,
                actualGainOrHarm = first.ActualGainVsTwin,
                adjustmentReason = first.ReferenceSnapshot != null && first.ProbeSnapshot != null
                    ? "measured-actual-counterfactual-effect"
                    : "prediction-space-gap-fallback",
                groups = probeGroups.Select(ToGroupLog).ToList()
            }, JsonOptions);

            if (!isNew)
                db.AnomalyInteractionRuleGroupStates.RemoveRange(rule.GroupStates);

            rule.GroupStates = probeGroups.Select((x, i) => new AnomalyInteractionRuleGroupState
            {
                RuleId = rule.Id,
                TensorGroupId = x.Group.UniqueId,
                CandidateQuantId = x.CandidateQuantId,
                ReferenceQuantId = x.ReferenceQuantId,
                Movement = x.Movement.ToString(),
                SortOrder = i
            }).ToList();

            upserted.Add(rule);
        }

        await db.SaveChangesAsync(ct);
        return upserted;
    }

    public async Task<IReadOnlyList<AnomalyInteractionRule>> LoadApplicableRulesAsync(CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var scope = await ResolveScopeAsync(db, ct);
        var minConfidence = Config.AnomalyDetection.MinRuleConfidenceToApply;

        var rules = await db.AnomalyInteractionRules
            .AsNoTracking()
            .Include(x => x.GroupStates)
            .Where(x => x.ArchitectureFamilyId == scope.ArchitectureFamilyId)
            .Where(x => x.TensorGroupProfileId == scope.TensorGroupProfileId)
            .Where(x => x.AiModelHashId == scope.AiModelHashId)
            .Where(x => x.ImatrixDefinitionId == scope.ImatrixDefinitionId)
            .Where(x => x.BenchmarkCategory == (byte)BenchmarkCategory.General)
            .Where(x => x.RuleStatus == AnomalyRuleStatus.Confirmed.ToString())
            .Where(x => x.Confidence >= minConfidence)
            .Where(x => x.RuleDirection == AnomalyRuleDirection.Beneficial.ToString() || x.RuleDirection == AnomalyRuleDirection.Harmful.ToString())
            .ToListAsync(ct);

        return rules
            .Where(_movement.IsContextualQuantizedRule)
            .ToList();
    }

    public async Task<HashSet<string>> LoadExistingRuleSuppressionKeysAsync(CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var scope = await ResolveScopeAsync(db, ct);

        var rows = await db.AnomalyInteractionRules
            .AsNoTracking()
            .Where(x => x.ArchitectureFamilyId == scope.ArchitectureFamilyId)
            .Where(x => x.TensorGroupProfileId == scope.TensorGroupProfileId)
            .Where(x => x.AiModelHashId == scope.AiModelHashId)
            .Where(x => x.ImatrixDefinitionId == scope.ImatrixDefinitionId)
            .Where(x => x.BenchmarkCategory == (byte)BenchmarkCategory.General)
            .Where(x => x.RuleStatus != AnomalyRuleStatus.Retired.ToString())
            .Select(x => new { x.ReferenceQuantId, x.GroupSetHash })
            .ToListAsync(ct);

        return rows
            .Select(x => BuildRuleSuppressionKey(x.ReferenceQuantId, x.GroupSetHash))
            .ToHashSet(StringComparer.Ordinal);
    }

    public string BuildRuleSuppressionKey(TensorConfig reference, IReadOnlyList<AnomalyChangedGroup> groups)
        => BuildRuleSuppressionKey(reference.BaseQuant, _movement.BuildChangedGroupHash(groups));

    public async Task<bool> HasSuppressionOrRuleAsync(
        TensorConfig reference,
        IReadOnlyList<AnomalyChangedGroup> groups,
        CancellationToken ct)
    {
        var keys = await LoadExistingRuleSuppressionKeysAsync(ct);
        return keys.Contains(BuildRuleSuppressionKey(reference, groups));
    }

    private static string BuildRuleSuppressionKey(byte referenceQuantId, string groupSetHash)
        => $"ref={referenceQuantId}|groups={groupSetHash}";

    private async Task<AnomalyScope> ResolveScopeAsync(MagicQuantContext db, CancellationToken ct)
    {
        uint aiModelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdAsync(db, ct);
        int? imatrixId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, aiModelHashId, createIfMissing: false, ct);
        return new AnomalyScope(
            TensorGroupProfileService.RequireCurrentArchitectureFamilyId(),
            TensorGroupProfileService.RequireCurrentProfileId(),
            aiModelHashId,
            imatrixId);
    }

    private static async Task<TensorCombo> EnsureTensorComboAsync(MagicQuantContext db, TensorConfig config, CancellationToken ct)
    {
        var combo = await db.TensorCombos.FirstOrDefaultAsync(x =>
            x.BaseQuant == config.BaseQuant &&
            x.Embeddings == config.Embeddings &&
            x.LmHead == config.LmHead &&
            x.AttnQ == config.AttnQ &&
            x.AttnKV == config.AttnKV &&
            x.AttnOutput == config.AttnOutput &&
            x.FfnUpGate == config.FfnUpGate &&
            x.FfnDown == config.FfnDown &&
            x.MoeExperts == config.MoeExperts &&
            x.MoeRouter == config.MoeRouter,
            ct);

        if (combo != null)
            return combo;

        combo = new TensorCombo(config);
        db.TensorCombos.Add(combo);
        await db.SaveChangesAsync(ct);
        return combo;
    }

    private static object ToGroupLog(AnomalyChangedGroup x)
    {
        return new
        {
            groupId = x.Group.UniqueId,
            group = x.Group.Name,
            shortCode = x.Group.ShortCode,
            candidateQuantId = x.CandidateQuantId,
            candidateQuant = BaselineQuants.FromId(x.CandidateQuantId).Names[0],
            referenceQuantId = x.ReferenceQuantId,
            referenceQuant = BaselineQuants.FromId(x.ReferenceQuantId).Names[0],
            movement = x.Movement.ToString()
        };
    }

    private static ulong ComputeSizeSavings(BenchmarkSnapshotRecord? reference, BenchmarkSnapshotRecord? probe)
    {
        if (reference == null || probe == null || reference.SizeBytes <= probe.SizeBytes)
            return 0UL;

        return reference.SizeBytes - probe.SizeBytes;
    }

    private static string BuildRuleKey(AnomalyProbeResult result)
    {
        var groups = result.Plan.ProbeGroups
            .OrderBy(x => x.Group.UniqueId)
            .Select(x => $"{x.Group.UniqueId}:{x.ReferenceQuantId}->{x.CandidateQuantId}");

        return $"{result.RuleDirection}|ref={TensorConfigIdentity.ToKey(result.Plan.ReferenceConfig)}|probe={TensorConfigIdentity.ToKey(result.Plan.ProbeConfig)}|{string.Join("|", groups)}";
    }

    private static string ResolveRuleType(IReadOnlyList<AnomalyProbeResult> rows)
    {
        var first = rows[0];
        if (first.RuleDirection == AnomalyRuleDirection.SuppressionOnly)
            return "SuppressionOnly";

        return first.Plan.ProbeType switch
        {
            "single" => "SingleGroupInversion",
            "pair" => "PairSynergy",
            "composition" => rows.Any(x => x.RuleDirection == AnomalyRuleDirection.Harmful) ? "HarmfulInterferenceComposition" : "CounterfactualSynergyComposition",
            "confirmed-neighborhood" => rows.Any(x => x.Classification == AnomalyProbeClassification.ContaminatingPassenger) ? "ContaminatingPassenger" : "ConfirmedAnomalyNeighborhood",
            "full" => rows.Any(x => x.Plan.ProbeGroups.Count >= 3) ? "HigherOrderSynergy" : "PairSynergy",
            "leave-one-out" => "HigherOrderSynergy",
            _ => rows.Any(x => x.Classification == AnomalyProbeClassification.ContaminatingPassenger) ? "ContaminatingPassenger" : "ContextOnly"
        };
    }

    private static double ComputeConfidence(IReadOnlyList<AnomalyProbeResult> rows)
    {
        if (rows.Count == 0)
            return 0d;

        double accepted = rows.Count(x => x.Accepted) / (double)rows.Count;
        double gain = Math.Clamp(rows.Max(x => Math.Abs(x.ActualGainVsTwin)) / Math.Max(Config.AnomalyDetection.MinActualGainVsTwinKld, 1e-9), 0d, 2d) / 2d;
        return Math.Clamp((accepted * 0.70d) + (gain * 0.30d), 0d, 1d);
    }

    private static double ComputePredictionAdjustment(AnomalyProbeResult result, double confidence)
    {
        var cfg = Config.AnomalyDetection;
        var synergy = Config.SynergyDetection;

        // Prediction KLD is rank-relative. Even when real probes confirm a beneficial
        // counterfactual effect, the adjustment is sized by how far the candidate must
        // move in prediction space to sit below its virtual/same-context twin. Actual
        // KLD affects confidence/classification, not raw numeric subtraction.
        double baseGap = result.Plan.Seed.PredictionSpaceGapVsTwin;
        double required = result.RuleDirection switch
        {
            AnomalyRuleDirection.Beneficial => -(Math.Max(0d, baseGap) + cfg.PredictionSpaceViolationMargin),
            AnomalyRuleDirection.Harmful => Math.Max(cfg.PredictionSpaceViolationMargin, Math.Abs(baseGap) + cfg.PredictionSpaceViolationMargin),
            _ => 0d
        };

        double multiplier = result.Plan.SeedClass switch
        {
            AnomalySeedClass.SynergyCompositionProbe => synergy.SameSelectedGroupsConfidenceMultiplier,
            AnomalySeedClass.SynergyTransferProbe => synergy.SameSelectedGroupsConfidenceMultiplier,
            AnomalySeedClass.ConfirmedAnomalyNeighborhoodProbe => synergy.SameSelectedGroupsConfidenceMultiplier,
            _ => synergy.ExactContextConfidenceMultiplier
        };

        if (result.Classification == AnomalyProbeClassification.ContaminatingPassenger)
            multiplier *= synergy.ContaminationPenaltyConfidenceMultiplier;

        double adjusted = required * confidence * cfg.AnomalyAdjustmentShrinkFactor * multiplier;
        if (adjusted < 0d)
            return Math.Max(adjusted, -Math.Min(cfg.MaxNegativeAdjustmentKld, synergy.MaxNegativeAdjustmentKld));

        return Math.Min(adjusted, cfg.MaxPositiveAdjustmentKld);
    }

    private readonly record struct AnomalyScope(int ArchitectureFamilyId, int TensorGroupProfileId, uint AiModelHashId, int? ImatrixDefinitionId);
}