using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;

namespace MagicQuant.Services;

public sealed class QuantFidelityComparerService
{
    private static readonly TensorGroup[] OrderedGroups =
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

    public IReadOnlyList<TensorGroup> ActiveGroups => OrderedGroups
        .Where(g => !Cache.UnusedTensorGroups.Any(u => u.UniqueId == g.UniqueId))
        .OrderBy(g => g.UniqueId)
        .ToList();

    public IReadOnlyList<TensorGroup> InactiveGroups => OrderedGroups
        .Where(g => Cache.UnusedTensorGroups.Any(u => u.UniqueId == g.UniqueId))
        .OrderBy(g => g.UniqueId)
        .ToList();

    public AnomalyMovementAnalysis Analyze(TensorConfig reference, TensorConfig candidate)
    {
        var changed = new List<AnomalyChangedGroup>();
        int upgrades = 0;
        int downgrades = 0;
        int same = 0;
        int unknown = 0;
        int lateral = 0;
        int net = 0;

        foreach (var group in ActiveGroups)
        {
            byte referenceStored = GetStoredSlot(reference, group);
            byte candidateStored = GetStoredSlot(candidate, group);
            byte referenceQuant = EffectiveQuantId(reference, group);
            byte candidateQuant = EffectiveQuantId(candidate, group);
            var movement = Compare(referenceQuant, candidateQuant);

            switch (movement)
            {
                case QuantMovementKind.Upgrade:
                    upgrades++;
                    break;
                case QuantMovementKind.Downgrade:
                    downgrades++;
                    break;
                case QuantMovementKind.Same:
                    same++;
                    break;
                case QuantMovementKind.LateralOrEquivalent:
                    lateral++;
                    break;
                case QuantMovementKind.Unknown:
                    unknown++;
                    break;
            }

            net += EffectiveTier(candidateQuant) - EffectiveTier(referenceQuant);

            if (movement != QuantMovementKind.Same || referenceStored != candidateStored)
            {
                changed.Add(new AnomalyChangedGroup
                {
                    Group = group,
                    ReferenceQuantId = referenceQuant,
                    CandidateQuantId = candidateQuant,
                    ReferenceStoredSlot = referenceStored,
                    CandidateStoredSlot = candidateStored,
                    Movement = movement
                });
            }
        }

        AnomalyMovementClassification classification;
        if (downgrades > 0 && upgrades == 0 && unknown == 0)
            classification = AnomalyMovementClassification.MonotoneDowngrade;
        else if (downgrades > 0 && upgrades > 0)
            classification = AnomalyMovementClassification.MixedTrade;
        else if (upgrades > 0 && downgrades == 0)
            classification = AnomalyMovementClassification.MonotoneUpgrade;
        else if (lateral > 0 && downgrades == 0 && upgrades == 0)
            classification = AnomalyMovementClassification.LateralOrProviderEquivalent;
        else if (changed.Count == 0)
            classification = AnomalyMovementClassification.NoMovement;
        else
            classification = AnomalyMovementClassification.Unknown;

        return new AnomalyMovementAnalysis
        {
            Classification = classification,
            ChangedGroups = changed,
            UpgradeCount = upgrades,
            DowngradeCount = downgrades,
            SameCount = same,
            UnknownCount = unknown,
            LateralCount = lateral,
            NetBitDelta = net
        };
    }

    public QuantMovementKind Compare(byte referenceQuantId, byte candidateQuantId)
    {
        if (referenceQuantId == candidateQuantId)
            return QuantMovementKind.Same;

        int referenceTier = EffectiveTier(referenceQuantId);
        int candidateTier = EffectiveTier(candidateQuantId);

        if (referenceTier < 0 || candidateTier < 0)
            return QuantMovementKind.Unknown;

        if (candidateTier == referenceTier)
            return QuantMovementKind.LateralOrEquivalent;

        return candidateTier < referenceTier
            ? QuantMovementKind.Downgrade
            : QuantMovementKind.Upgrade;
    }

    /// <summary>
    /// Builds an explicit contextual quantized blanket: base=referenceQuantId and every
    /// active tensor group is explicitly stored as that same learned quant. Inactive
    /// groups remain NULL so dense/MoE architecture differences are preserved.
    ///
    /// This is the anomaly-world equivalent of the old exact blanket, except it is
    /// intentionally quantized context, not BF16/F16/native isolation truth.
    /// </summary>
    public TensorConfig CreateActivatedContextBlanket(byte referenceQuantId)
    {
        if (BaselineQuants.IsNativeExactAlias(referenceQuantId))
        {
            throw new InvalidOperationException(
                $"SkippedInvalidContextualAnomalyProbe: reason=BF16ExactIsolationSample referenceQuant={SafeName(referenceQuantId)}");
        }

        byte stored = BaselineQuants.EncodeTensorConfigGroupSlotBaselineId(referenceQuantId);
        var config = new TensorConfig(
            baseQuant: referenceQuantId,
            embeddings: BaselineQuants.TensorConfigNullSlotValue,
            lmHead: BaselineQuants.TensorConfigNullSlotValue,
            attnQ: BaselineQuants.TensorConfigNullSlotValue,
            attnKV: BaselineQuants.TensorConfigNullSlotValue,
            attnOutput: BaselineQuants.TensorConfigNullSlotValue,
            ffnUpGate: BaselineQuants.TensorConfigNullSlotValue,
            ffnDown: BaselineQuants.TensorConfigNullSlotValue,
            moeExperts: BaselineQuants.TensorConfigNullSlotValue,
            moeRouter: BaselineQuants.TensorConfigNullSlotValue);

        foreach (var group in ActiveGroups.OrderBy(g => g.UniqueId))
            config = WithStoredSlot(config, group, stored);

        return config;
    }

    /// <summary>
    /// Builds the contextual higher-bit twin for anomaly detection.
    ///
    /// This is intentionally NOT the BF16/exact isolation reference used by normal
    /// tensor-group learning. For anomaly smoke/probes, the reference is an explicit
    /// activated quantized context: base=Q8 means every active group is explicitly Q8;
    /// base=Q6 means every active group is explicitly Q6; and so on.
    /// </summary>
    public TensorConfig BuildBaseContextTwin(TensorConfig candidate) => CreateActivatedContextBlanket(candidate.BaseQuant);

    /// <summary>
    /// Converts a normal generated DuckDB row into the explicit anomaly context shape.
    /// This is allowed for prediction-space smoke only: sparse generated rows are not
    /// treated as historical truth and are never persisted as anomaly probes/rules.
    /// </summary>
    public bool TryNormalizeSparseDuckRowToActivatedContext(
        TensorConfig source,
        out TensorConfig activated,
        out bool hadSparseActiveGroups,
        out string reason)
    {
        activated = default;
        hadSparseActiveGroups = false;

        if (BaselineQuants.IsNativeExactAlias(source.BaseQuant))
        {
            reason = $"BF16ExactIsolationSample: base={SafeName(source.BaseQuant)}";
            return false;
        }

        if (HasIsolationDisplayMarker(source))
        {
            reason = $"BF16ExactIsolationSample: displayName={HybridBenchmarkRepository.BuildDisplayName((HybridQuant)source)}";
            return false;
        }

        TensorConfig result;
        try
        {
            result = CreateActivatedContextBlanket(source.BaseQuant);
        }
        catch (Exception ex)
        {
            reason = ex.Message;
            return false;
        }

        foreach (var group in ActiveGroups)
        {
            byte stored = GetStoredSlot(source, group);
            if (BaselineQuants.IsNullTensorConfigGroupSlot(stored))
            {
                hadSparseActiveGroups = true;
                continue;
            }

            byte quantId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(stored);
            if (BaselineQuants.IsNativeExactAlias(quantId))
            {
                reason = $"BF16ExactIsolationSample: {group.Name}={SafeName(quantId)}";
                return false;
            }

            result = WithStoredSlot(result, group, stored);
        }

        foreach (var group in InactiveGroups)
        {
            if (!BaselineQuants.IsNullTensorConfigGroupSlot(GetStoredSlot(result, group)))
            {
                result = WithStoredSlot(result, group, BaselineQuants.TensorConfigNullSlotValue);
            }
        }

        if (!TryValidateContextualAnomalyConfig(result, out reason))
            return false;

        activated = result;
        reason = hadSparseActiveGroups
            ? "Sparse DuckDB prediction row normalized into explicit activated contextual anomaly vector."
            : string.Empty;
        return true;
    }

    public bool IsNativeExactQuantId(byte quantId) => BaselineQuants.IsNativeExactAlias(quantId);

    public bool IsContextualQuantizedConfig(TensorConfig config) => TryValidateContextualAnomalyConfig(config, out _);

    public bool TryValidateContextualAnomalyConfig(TensorConfig config, out string reason)
    {
        if (BaselineQuants.IsNativeExactAlias(config.BaseQuant))
        {
            reason = $"BF16ExactIsolationSample: base={SafeName(config.BaseQuant)}";
            return false;
        }

        if (HasIsolationDisplayMarker(config))
        {
            reason = $"BF16ExactIsolationSample: display name contains BF16/F16/native/exact marker ({HybridBenchmarkRepository.BuildDisplayName((HybridQuant)config)})";
            return false;
        }

        foreach (var group in ActiveGroups)
        {
            byte stored = GetStoredSlot(config, group);
            if (BaselineQuants.IsNullTensorConfigGroupSlot(stored))
            {
                reason = $"SparseActiveGroup: group={group.Name} shortCode={group.ShortCode}";
                return false;
            }

            byte quantId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(stored);
            if (BaselineQuants.IsNativeExactAlias(quantId))
            {
                reason = $"BF16ExactIsolationSample: {group.Name}={SafeName(quantId)}";
                return false;
            }
        }

        foreach (var group in InactiveGroups)
        {
            byte stored = GetStoredSlot(config, group);
            if (!BaselineQuants.IsNullTensorConfigGroupSlot(stored))
            {
                byte quantId = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(stored);
                if (BaselineQuants.IsNativeExactAlias(quantId))
                {
                    reason = $"BF16ExactIsolationSample: inactive {group.Name}={SafeName(quantId)}";
                    return false;
                }
            }
        }

        reason = string.Empty;
        return true;
    }

    public void EnsureAllActiveGroupsExplicit(TensorConfig config, string purpose)
    {
        if (TryValidateContextualAnomalyConfig(config, out _))
            return;

        TryValidateContextualAnomalyConfig(config, out var reason);
        throw new InvalidOperationException(
            $"SkippedInvalidContextualAnomalyProbe: purpose={purpose} reason={reason} config={TensorConfigIdentity.ToKey(config)} name={HybridBenchmarkRepository.BuildDisplayName((HybridQuant)config)}");
    }

    public bool TryDescribeNativeExactActiveState(TensorConfig config, out string reason)
    {
        if (BaselineQuants.IsNativeExactAlias(config.BaseQuant))
        {
            reason = $"base={SafeName(config.BaseQuant)}";
            return true;
        }

        foreach (var group in ActiveGroups)
        {
            byte stored = GetStoredSlot(config, group);
            if (BaselineQuants.IsNullTensorConfigGroupSlot(stored))
                continue;

            byte effective = BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(stored);
            if (BaselineQuants.IsNativeExactAlias(effective))
            {
                reason = $"{group.ShortCode}={SafeName(effective)}";
                return true;
            }
        }

        reason = string.Empty;
        return false;
    }

    public bool HasIsolationDisplayMarker(TensorConfig config)
    {
        string displayName = HybridBenchmarkRepository.BuildDisplayName((HybridQuant)config);
        return displayName.Contains("-B16", StringComparison.OrdinalIgnoreCase) ||
               displayName.Contains("BF16", StringComparison.OrdinalIgnoreCase) ||
               displayName.Contains("F16", StringComparison.OrdinalIgnoreCase) ||
               displayName.Contains("NATIVE", StringComparison.OrdinalIgnoreCase) ||
               displayName.Contains("EXACT", StringComparison.OrdinalIgnoreCase);
    }

    public bool IsContextualQuantizedRule(AnomalyInteractionRule rule)
    {
        if (BaselineQuants.IsNativeExactAlias(rule.ReferenceQuantId))
            return false;

        foreach (var state in rule.GroupStates)
        {
            if (BaselineQuants.IsNativeExactAlias(state.CandidateQuantId) ||
                BaselineQuants.IsNativeExactAlias(state.ReferenceQuantId))
            {
                return false;
            }
        }

        return true;
    }

    public byte EffectiveQuantId(TensorConfig config, TensorGroup group)
    {
        byte stored = GetStoredSlot(config, group);
        return BaselineQuants.IsNullTensorConfigGroupSlot(stored)
            ? config.BaseQuant
            : BaselineQuants.DecodeTensorConfigGroupSlotToBaselineId(stored);
    }

    public byte GetStoredSlot(TensorConfig config, TensorGroup group)
    {
        return group.UniqueId switch
        {
            var id when id == TReg.Embeddings.UniqueId => config.Embeddings,
            var id when id == TReg.LmHead.UniqueId => config.LmHead,
            var id when id == TReg.AttnQ.UniqueId => config.AttnQ,
            var id when id == TReg.AttnKV.UniqueId => config.AttnKV,
            var id when id == TReg.AttnOutput.UniqueId => config.AttnOutput,
            var id when id == TReg.FfnUpGate.UniqueId => config.FfnUpGate,
            var id when id == TReg.FfnDown.UniqueId => config.FfnDown,
            var id when id == TReg.MoeExperts.UniqueId => config.MoeExperts,
            var id when id == TReg.MoeRouter.UniqueId => config.MoeRouter,
            _ => throw new InvalidOperationException($"Unknown tensor group id '{group.UniqueId}'.")
        };
    }

    public TensorConfig WithStoredSlot(TensorConfig config, TensorGroup group, byte storedSlot)
    {
        return new TensorConfig(
            config.BaseQuant,
            group.UniqueId == TReg.Embeddings.UniqueId ? storedSlot : config.Embeddings,
            group.UniqueId == TReg.LmHead.UniqueId ? storedSlot : config.LmHead,
            group.UniqueId == TReg.AttnQ.UniqueId ? storedSlot : config.AttnQ,
            group.UniqueId == TReg.AttnKV.UniqueId ? storedSlot : config.AttnKV,
            group.UniqueId == TReg.AttnOutput.UniqueId ? storedSlot : config.AttnOutput,
            group.UniqueId == TReg.FfnUpGate.UniqueId ? storedSlot : config.FfnUpGate,
            group.UniqueId == TReg.FfnDown.UniqueId ? storedSlot : config.FfnDown,
            group.UniqueId == TReg.MoeExperts.UniqueId ? storedSlot : config.MoeExperts,
            group.UniqueId == TReg.MoeRouter.UniqueId ? storedSlot : config.MoeRouter);
    }

    public string BuildChangedGroupHash(IEnumerable<AnomalyChangedGroup> groups)
    {
        string key = string.Join("|", groups
            .OrderBy(x => x.Group.UniqueId)
            .Select(x => $"{x.Group.UniqueId}:{x.ReferenceQuantId}->{x.CandidateQuantId}"));

        return TensorConfigIdentity.StableHash(key);
    }

    public string DescribeGroups(IEnumerable<AnomalyChangedGroup> groups)
    {
        return string.Join(" + ", groups
            .OrderBy(x => x.Group.UniqueId)
            .Select(x => $"{x.Group.ShortCode}={SafeName(x.CandidateQuantId)} from {SafeName(x.ReferenceQuantId)}"));
    }

    public string ReferenceContextKey(TensorConfig reference)
    {
        return string.Join("|", ActiveGroups.Select(g => $"{g.UniqueId}:{EffectiveQuantId(reference, g)}"));
    }

    public Dictionary<string, string> BuildEffectiveGroupVector(TensorConfig config)
    {
        return ActiveGroups
            .OrderBy(g => g.UniqueId)
            .ToDictionary(
                g => g.Name,
                g => SafeName(EffectiveQuantId(config, g)),
                StringComparer.Ordinal);
    }

    public IReadOnlyList<string> BuildInactiveGroupList() => InactiveGroups
        .OrderBy(g => g.UniqueId)
        .Select(g => g.Name)
        .ToList();

    public bool HasAllActiveGroupsExplicit(TensorConfig config)
    {
        foreach (var group in ActiveGroups)
        {
            if (BaselineQuants.IsNullTensorConfigGroupSlot(GetStoredSlot(config, group)))
                return false;
        }

        return true;
    }

    public int EffectiveTier(byte quantId)
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
        if (name.Contains("Q1") || name.Contains("IQ1") || baseline.BitRange == 1) return 10;

        return baseline.BitRange > 0 ? baseline.BitRange * 10 : -1;
    }

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
