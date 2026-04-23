using MagicQuant.Helpers;
using MagicQuant.Models;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;

namespace MagicQuant.Services;

public sealed class HybridBenchmarkRepository
{
    public async Task<Dictionary<string, BenchmarkSnapshotRecord>> LoadBenchmarkSnapshotsAsync(
        IEnumerable<TensorConfig> configs,
        CancellationToken ct = default)
    {
        var result = new Dictionary<string, BenchmarkSnapshotRecord>(StringComparer.Ordinal);

        foreach (var config in configs.DistinctBy(TensorConfigIdentity.ToKey))
        {
            var snapshot = await LoadBenchmarkSnapshotAsync(config, ct);
            if (snapshot != null)
                result[TensorConfigIdentity.ToKey(config)] = snapshot;
        }

        return result;
    }

    public async Task<BenchmarkSnapshotRecord?> LoadBenchmarkSnapshotAsync(
        TensorConfig config,
        CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            return null;

        int? activeImatrixId = await ResolveActiveImatrixIdAsync(db, scopedAiModelHashId.Value, ct);

        var query = db.AiBenchmarks
            .AsNoTracking()
            .Include(x => x.TensorCombo)
            .Include(x => x.CategorBenchmarks)
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.TensorCombo.BaseQuant == config.BaseQuant)
            .Where(x => x.TensorCombo.Embeddings == config.Embeddings)
            .Where(x => x.TensorCombo.LmHead == config.LmHead)
            .Where(x => x.TensorCombo.AttnQ == config.AttnQ)
            .Where(x => x.TensorCombo.AttnKV == config.AttnKV)
            .Where(x => x.TensorCombo.AttnOutput == config.AttnOutput)
            .Where(x => x.TensorCombo.FfnUpGate == config.FfnUpGate)
            .Where(x => x.TensorCombo.FfnDown == config.FfnDown)
            .Where(x => x.TensorCombo.MoeExperts == config.MoeExperts)
            .Where(x => x.TensorCombo.MoeRouter == config.MoeRouter);

        var rows = await query.ToListAsync(ct);
        if (rows.Count == 0)
            return null;

        AiBenchmark chosen = rows
            .OrderByDescending(x => activeImatrixId != null && x.ImatrixDefinitionId == activeImatrixId.Value)
            .ThenByDescending(x => x.ImatrixDefinitionId != null)
            .ThenBy(x => x.Id)
            .First();

        var general = chosen.CategorBenchmarks.FirstOrDefault(x => x.Category == (byte)BenchmarkCategory.General)
            ?? chosen.CategorBenchmarks.OrderBy(x => x.Category).FirstOrDefault();

        if (general == null)
            return null;

        var quant = (HybridQuant)config;
        var baseQuant = quant.BaseQuant;

        return new BenchmarkSnapshotRecord
        {
            Config = config,
            Quant = quant,
            DisplayName = BuildDisplayName(quant),
            ProviderName = ResolveProviderName(quant, exportNaming: false),
            BaselineFamily = baseQuant.Names[0],
            IsHybrid = quant.Tensors.Count > 0,
            IsExternalPureBaseline = quant.Tensors.Count == 0 && baseQuant.IsExternalRepositoryBaseline,
            SizeBytes = chosen.SizeBytes,
            Kld = general.Kld,
            Ppl = general.Ppl,
            OutputModelPath = await FindLatestSuccessfulOutputPathAsync(config, ct),
            ExternalRepositoryUrl = BuildExternalRepositoryUrl(baseQuant)
        };
    }

    public async Task<List<BenchmarkSnapshotRecord>> LoadPureBaselineSnapshotsAsync(CancellationToken ct = default)
    {
        var result = new List<BenchmarkSnapshotRecord>();

        foreach (var baseline in BaselineQuants.GetAllRecognizedBaselines())
        {
            var snapshot = await LoadBenchmarkSnapshotAsync((TensorConfig)HybridQuant.CreatePureBaseline(baseline), ct);
            if (snapshot != null)
                result.Add(snapshot);
        }

        return result
            .GroupBy(x => TensorConfigIdentity.ToKey(x.Config), StringComparer.Ordinal)
            .Select(g => g.First())
            .OrderBy(x => x.Kld)
            .ThenBy(x => x.SizeBytes)
            .ToList();
    }

    public async Task<string?> FindLatestSuccessfulOutputPathAsync(TensorConfig config, CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            return null;

        var tensorComboId = await db.TensorCombos
            .AsNoTracking()
            .Where(x => x.BaseQuant == config.BaseQuant)
            .Where(x => x.Embeddings == config.Embeddings)
            .Where(x => x.LmHead == config.LmHead)
            .Where(x => x.AttnQ == config.AttnQ)
            .Where(x => x.AttnKV == config.AttnKV)
            .Where(x => x.AttnOutput == config.AttnOutput)
            .Where(x => x.FfnUpGate == config.FfnUpGate)
            .Where(x => x.FfnDown == config.FfnDown)
            .Where(x => x.MoeExperts == config.MoeExperts)
            .Where(x => x.MoeRouter == config.MoeRouter)
            .Select(x => (Guid?)x.Id)
            .FirstOrDefaultAsync(ct);

        if (tensorComboId == null)
            return null;

        return await db.QuantizationRuns
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.TensorComboId == tensorComboId.Value)
            .Where(x => x.Succeeded)
            .OrderByDescending(x => x.CompletedUtc)
            .Select(x => x.OutputModelPath)
            .FirstOrDefaultAsync(ct);
    }

    public async Task<Dictionary<string, string>> LoadLearnedTensorMappingsAsync(
        string canonicalBaselineKey,
        byte? groupId = null,
        TensorWeightScheme? preferredSourceScheme = null,
        bool allowDominantFallback = true,
        CancellationToken ct = default)
    {
        await using var db = new MagicQuantContext();
        var scopedAiModelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (scopedAiModelHashId == null)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var query = db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId.Value)
            .Where(x => x.BaselineCanonicalKey == canonicalBaselineKey);

        if (groupId != null)
            query = query.Where(x => x.TensorGroupId == groupId.Value);

        var allRows = await query.OrderBy(x => x.TensorName).ToListAsync(ct);
        if (allRows.Count == 0)
            return new Dictionary<string, string>(StringComparer.Ordinal);

        var rows = allRows;

        if (preferredSourceScheme != null)
        {
            var preferred = allRows.Where(x => x.TensorWeightSchemeId == preferredSourceScheme.UniqueId).ToList();
            if (preferred.Count > 0)
                rows = preferred;
            else if (!allowDominantFallback)
                return new Dictionary<string, string>(StringComparer.Ordinal);
        }

        if (rows.Select(x => x.TensorWeightSchemeId).Distinct().Count() > 1)
        {
            if (!allowDominantFallback)
                return new Dictionary<string, string>(StringComparer.Ordinal);

            var dominant = rows.GroupBy(x => x.TensorWeightSchemeId)
                .OrderByDescending(g => g.Count())
                .ThenBy(g => g.Key)
                .First()
                .Key;

            rows = rows.Where(x => x.TensorWeightSchemeId == dominant).ToList();
        }

        return rows.ToDictionary(
            x => x.TensorName,
            x =>
            {
                var normalized = NativePrecisionNormalization.NormalizeLearnedFinalQuantTypeForApplication(x.FinalQuantType);
                return string.IsNullOrWhiteSpace(normalized) ? x.FinalQuantType : normalized;
            },
            StringComparer.Ordinal);
    }

    public static string ResolveProviderName(HybridQuant quant, bool exportNaming)
    {
        if (exportNaming && quant.Tensors.Count > 0)
            return "MagicHybrid";

        var baseline = quant.BaseQuant;
        if (baseline.IsExternalRepositoryBaseline)
            return string.IsNullOrWhiteSpace(baseline.ShortSourceName)
                ? "External"
                : baseline.ShortSourceName!;

        return string.IsNullOrWhiteSpace(baseline.ShortSourceName)
            ? "llama.cpp"
            : baseline.ShortSourceName!;
    }

    public static string BuildDisplayName(HybridQuant quant)
    {
        string modelName = string.IsNullOrWhiteSpace(Cache.ModelDirectory)
            ? "model"
            : new DirectoryInfo(Cache.ModelDirectory!).Name;

        if (quant.Tensors.Count == 0)
            return $"{modelName}-{quant.BaseQuant.Names[0]}";

        var parts = quant.Tensors
            .Where(x => x?.TGroup != null)
            .OrderBy(x => x.TGroup.ShortCode)
            .Select(x =>
            {
                if (x.OverrideMode == HybridTensorOverrideMode.ExactTensorScheme)
                    return $"{x.TGroup.ShortCode}-{x.ExactTensorScheme!.Names[0]}";

                return $"{x.TGroup.ShortCode}-{x.CandidateBaseline!.Names[0]}";
            });

        return $"{modelName}-{quant.BaseQuant.Names[0]}-{string.Join("-", parts)}";
    }

    public static string? BuildExternalRepositoryUrl(BaselineQuants baseline)
    {
        if (!baseline.IsExternalRepositoryBaseline)
            return null;

        var resolved = Config.GetResolvedCustomBaseline(baseline.CanonicalKey);
        if (resolved != null && !string.IsNullOrWhiteSpace(resolved.RepoId))
            return $"https://huggingface.co/{resolved.RepoId}";

        if (!string.IsNullOrWhiteSpace(baseline.SourceRepository))
            return $"https://huggingface.co/{baseline.SourceRepository}";

        return null;
    }

    private static async Task<int?> ResolveActiveImatrixIdAsync(MagicQuantContext db, uint scopedAiModelHashId, CancellationToken ct)
    {
        if (!Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixIdentityHash))
            return null;

        return await db.ImatrixDefinitions
            .AsNoTracking()
            .Where(x => x.AiModelHashId == scopedAiModelHashId)
            .Where(x => x.IdentityHash == Cache.ActiveImatrixIdentityHash)
            .Select(x => (int?)x.Id)
            .FirstOrDefaultAsync(ct);
    }
}
