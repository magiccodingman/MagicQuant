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
    public int GroupSchemeEliminations { get; set; }
    public int BaselinesSkippedWithoutLearnedRows { get; set; }
    public List<string> Notes { get; } = new();
}

public sealed class LearnedBaselinePruningService
{
    private readonly record struct LearnedRow(
        byte BaselineQuantId,
        byte TensorWeightSchemeId,
        byte TensorGroupId,
        string FinalQuantType);

    public async Task<LearnedBaselinePruningResult> AnalyzeAndApplyAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var result = new LearnedBaselinePruningResult();

        await using var db = new MagicQuantContext();

        var aiModelHashId = await db.AiModelHashes
            .AsNoTracking()
            .Where(x => x.UniqueHash == Cache.CurrentModelId)
            .Select(x => (uint?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (aiModelHashId == null)
            throw new InvalidOperationException(
                $"AiModelHash row was not found for current model id '{Cache.CurrentModelId}'.");

        var learnedRows = await db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == aiModelHashId.Value)
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

        var baselinesWithAnyLearnedRows = learnedRows
            .Select(x => x.BaselineQuantId)
            .ToHashSet();

        var aliasToSchemeIds = BuildAliasToSchemeIds();

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

        var skippedBaselineNotes = new HashSet<byte>();
        var unusedIds = Cache.UnusedTensorGroups
            .Select(x => x.UniqueId)
            .ToHashSet();

        var schemeOwnerById = BaselineQuants.All
            .SelectMany(b => b.TensorWeightSchemes.Select(s => new
            {
                SchemeId = s.UniqueId,
                Baseline = b
            }))
            .ToDictionary(x => x.SchemeId, x => x.Baseline);

        var explicitSchemes = TensorWeightScheme.All_Allowed_Hybrid_Quants
            .Where(x => x.UniqueId != TensorWeightScheme.NULL.UniqueId)
            .Where(x => x.UniqueId != TensorWeightScheme.BF16_F16.UniqueId)
            .OrderBy(x => x.UniqueId)
            .ToList();

        foreach (var group in TReg.All.OrderBy(x => x.UniqueId))
        {
            if (unusedIds.Contains(group.UniqueId))
                continue;

            foreach (var scheme in explicitSchemes)
            {
                if (RuntimeSearchSpace.IsSchemeRuntimeBannedForGroup(group, scheme))
                    continue;

                if (!schemeOwnerById.TryGetValue(scheme.UniqueId, out var owningBaseline))
                    continue;

                if (!baselinesWithAnyLearnedRows.Contains(owningBaseline.UniqueId))
                {
                    if (skippedBaselineNotes.Add(owningBaseline.UniqueId))
                    {
                        result.BaselinesSkippedWithoutLearnedRows++;

                        result.Notes.Add(
                            $"Learned-baseline pruning skipped for baseline '{owningBaseline.Names[0]}' because no learned rows existed for the current model.");
                    }

                    continue;
                }

                var key = (owningBaseline.UniqueId, group.UniqueId);
                var effectiveForGroup = effectiveSchemesByBaselineAndGroup.TryGetValue(key, out var found)
                    ? found
                    : null;

                bool hasAnyConnectedMapping = effectiveForGroup != null &&
                                              owningBaseline.TensorWeightSchemes.Any(connectedScheme =>
                                                  effectiveForGroup.Contains(connectedScheme.UniqueId));

                if (hasAnyConnectedMapping)
                    continue;

                RuntimeSearchSpace.BanSchemeForGroupByLearnedBaselineAbsence(group, scheme, owningBaseline);
                result.GroupSchemeEliminations++;

                result.Notes.Add(
                    $"Learned-baseline prune: '{scheme.Names[0]}' removed for '{group.Name}' because baseline '{owningBaseline.Names[0]}' learned zero matching tensors in that group.");
            }
        }

        return result;
    }

    private static Dictionary<string, HashSet<byte>> BuildAliasToSchemeIds()
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
