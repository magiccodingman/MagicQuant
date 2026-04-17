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
    private readonly record struct LearnedRowKey(byte BaselineQuantId, byte TensorWeightSchemeId, byte TensorGroupId);

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
            .Select(x => new LearnedRowKey(
                x.BaselineQuantId,
                x.TensorWeightSchemeId,
                x.TensorGroupId))
            .ToListAsync(ct);

        if (learnedRows.Count == 0)
        {
            result.Notes.Add(
                "Learned-baseline pruning skipped because no LearnedBaselineTensorQuants rows existed for the current model.");
            return result;
        }

        var learnedRowSet = learnedRows.ToHashSet();
        var baselinesWithAnyLearnedRows = learnedRows
            .Select(x => x.BaselineQuantId)
            .ToHashSet();

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

                var lookup = new LearnedRowKey(
                    owningBaseline.UniqueId,
                    scheme.UniqueId,
                    group.UniqueId);

                if (learnedRowSet.Contains(lookup))
                    continue;

                RuntimeSearchSpace.BanSchemeForGroupByLearnedBaselineAbsence(group, scheme, owningBaseline);
                result.GroupSchemeEliminations++;

                result.Notes.Add(
                    $"Learned-baseline prune: '{scheme.Names[0]}' removed for '{group.Name}' because baseline '{owningBaseline.Names[0]}' learned zero matching tensors in that group.");
            }
        }

        return result;
    }
}