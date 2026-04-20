using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MagicQuant.Helpers;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;

namespace MagicQuant.Services;

public sealed class LearnedBaselinePruningResult
{
    public int GroupCandidateEliminations { get; set; }
    public int BaselinesSkippedWithoutLearnedRows { get; set; }
    public List<string> Notes { get; } = new();
}

public sealed class LearnedBaselinePruningService
{
    internal readonly record struct LearnedRow(
        byte BaselineQuantId,
        byte TensorWeightSchemeId,
        byte TensorGroupId,
        string FinalQuantType);

    public async Task<LearnedBaselinePruningResult> AnalyzeAndApplyAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        RuntimeSearchSpace.ClearLearnedBaselinePruneBookkeeping();

        var result = new LearnedBaselinePruningResult();

        await using var db = new MagicQuantContext();

        var aiModelHash = await db.AiModelHashes
            .AsNoTracking()
            .Where(x => x.UniqueHash == Cache.CurrentModelId)
            .Select(x => new { x.Id, x.UniqueHash })
            .FirstOrDefaultAsync(ct);

        if (aiModelHash == null)
            throw new InvalidOperationException(
                $"AiModelHash row was not found for current model id '{Cache.CurrentModelId}'.");

        result.Notes.Add(
            $"Learned-baseline pruning model resolution: Cache.CurrentModelId={Cache.CurrentModelId}, " +
            $"AiModelHash.Id={aiModelHash.Id}, AiModelHash.UniqueHash={aiModelHash.UniqueHash}");

        var learnedRows = await db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == aiModelHash.Id)
            .Select(x => new LearnedRow(
                x.BaselineQuantId,
                x.TensorWeightSchemeId,
                x.TensorGroupId,
                x.FinalQuantType))
            .ToListAsync(ct);

        if (learnedRows.Count == 0)
        {
            result.Notes.Add(
                "Learned-baseline pruning skipped because no LearnedBaselineTensorQuants rows existed for the current model.");
            return result;
        }

        var unusedIds = Cache.UnusedTensorGroups
            .Select(x => x.UniqueId)
            .ToHashSet();

        ApplyLearnedBaselinePruning(learnedRows, aiModelHash.Id, aiModelHash.UniqueHash, unusedIds, result);
        return result;
    }

    internal static void ApplyLearnedBaselinePruning(
        IReadOnlyList<LearnedRow> learnedRows,
        uint aiModelHashId,
        string aiModelHashUniqueHash,
        HashSet<byte> unusedGroupIds,
        LearnedBaselinePruningResult result)
    {
        RuntimeSearchSpace.ClearLearnedBaselinePruneBookkeeping();

        var aliasToSchemeIds = BuildAliasToSchemeIds();
        var effectiveSchemesByCandidateAndGroup = BuildEffectiveSchemesByBaselineAndGroup(learnedRows, aliasToSchemeIds);

        var explicitCandidates = BaselineQuants.GetGroupCombinationCandidates(RuntimeSearchSpace.HasUsableImatrix(), allowHighPrecisionHybrids: false)
            .Where(x => x.UniqueId != BaselineQuants.BF16_Hybrid.UniqueId)
            .Where(x => x.UniqueId != BaselineQuants.F16_Hybrid.UniqueId)
            .OrderBy(x => x.UniqueId)
            .ToList();

        foreach (var group in TReg.All.OrderBy(x => x.UniqueId))
        {
            if (unusedGroupIds.Contains(group.UniqueId))
                continue;

            foreach (var candidate in explicitCandidates)
            {
                if (RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate))
                    continue;

                var key = (candidate.UniqueId, group.UniqueId);
                bool hasEffectiveSet = effectiveSchemesByCandidateAndGroup.TryGetValue(key, out var effectiveForGroup);
                var expectedIds = candidate.TensorWeightSchemes.Select(x => x.UniqueId).Distinct().OrderBy(x => x).ToList();
                var effectiveIdsSet = hasEffectiveSet ? effectiveForGroup! : new HashSet<byte>();
                var matchedIds = expectedIds.Where(effectiveIdsSet.Contains).OrderBy(x => x).ToList();
                bool allow = matchedIds.Count > 0;
                string effectiveIds = hasEffectiveSet
                    ? string.Join(",", effectiveIdsSet.OrderBy(x => x))
                    : "<none>";
                string expected = string.Join(",", expectedIds);
                string matched = matchedIds.Count > 0 ? string.Join(",", matchedIds) : "<none>";

                result.Notes.Add(
                    $"Learned-prune check: model={aiModelHashId}/{aiModelHashUniqueHash}, group={group.Name}, " +
                    $"candidate={candidate.Names[0]}, expected=[{expected}], effective=[{effectiveIds}], matched=[{matched}], decision={(allow ? "ALLOW" : "BAN")}");

                if (!allow)
                {
                    RuntimeSearchSpace.BanCombinationCandidateForGroupDueToLearnedSchemeMismatch(
                        group,
                        candidate,
                        expectedTensorWeightSchemeIds: expectedIds,
                        matchedTensorWeightSchemeIds: matchedIds,
                        note: "No matching learned tensor-weight schemes for candidate/group.");
                    result.GroupCandidateEliminations++;
                }
            }
        }
    }

    internal static Dictionary<(byte BaselineId, byte GroupId), HashSet<byte>> BuildEffectiveSchemesByBaselineAndGroup(
        IReadOnlyList<LearnedRow> learnedRows,
        Dictionary<string, HashSet<byte>> aliasToSchemeIds)
    {
        var effectiveSchemesByBaselineAndGroup = new Dictionary<(byte BaselineId, byte GroupId), HashSet<byte>>();

        foreach (var row in learnedRows)
        {
            var key = (row.BaselineQuantId, row.TensorGroupId);

            if (!effectiveSchemesByBaselineAndGroup.TryGetValue(key, out var set))
            {
                set = new HashSet<byte>();
                effectiveSchemesByBaselineAndGroup[key] = set;
            }

            if (aliasToSchemeIds.TryGetValue(CanonicalizeQuantToken(row.FinalQuantType), out var resolvedIds))
            {
                foreach (var resolvedId in resolvedIds)
                    set.Add(resolvedId);
            }
        }

        return effectiveSchemesByBaselineAndGroup;
    }

    internal static Dictionary<string, HashSet<byte>> BuildAliasToSchemeIds()
    {
        var map = new Dictionary<string, HashSet<byte>>(StringComparer.Ordinal);

        foreach (var scheme in TensorWeightScheme.All)
        {
            if (scheme.Names.IsDefaultOrEmpty)
                continue;

            foreach (var alias in scheme.Names)
            {
                var token = CanonicalizeQuantToken(alias);

                if (!map.TryGetValue(token, out var ids))
                {
                    ids = new HashSet<byte>();
                    map[token] = ids;
                }

                ids.Add(scheme.UniqueId);
            }
        }

        return map;
    }

    private static string CanonicalizeQuantToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "UNKNOWN";

        return value
            .Trim()
            .Replace("-", "_")
            .Replace(" ", string.Empty)
            .ToUpperInvariant();
    }
}
