using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;

namespace MagicQuant.Services;

public static class BaselineDefinitionResolver
{
    public static string NormalizeRepoId(string value) => (value ?? string.Empty).Trim().ToLowerInvariant();

    public static string NormalizeFileName(string value) => (value ?? string.Empty).Trim().Replace('\\', '/').ToLowerInvariant();

    public static string NormalizeCanonicalKey(string value) => (value ?? string.Empty).Trim().ToLowerInvariant();

    public static string BuildCustomCanonicalKey(string architectureFamilyName, string repoId, string fileName) =>
        $"custom:{NormalizeRepoId(architectureFamilyName)}:{NormalizeRepoId(repoId)}:{NormalizeFileName(fileName)}";

    public static async Task<BaselineQuantDefinition> ResolveRequiredDefinitionAsync(
        MagicQuantContext db,
        BaselineQuants baseline,
        CancellationToken ct = default)
    {
        int? familyId = baseline.IsCustomBaseline
            ? TensorGroupProfileService.RequireCurrentArchitectureFamilyId()
            : null;

        string normalizedKey = NormalizeCanonicalKey(baseline.CanonicalKey);

        var definition = await db.BaselineQuantDefinitions
            .AsNoTracking()
            .FirstOrDefaultAsync(x =>
                    x.ArchitectureFamilyId == familyId &&
                    x.RuntimeBaselineId == baseline.UniqueId,
                ct)
            ?? await db.BaselineQuantDefinitions
                .AsNoTracking()
                .FirstOrDefaultAsync(x =>
                        x.ArchitectureFamilyId == familyId &&
                        x.NormalizedCanonicalKey == normalizedKey,
                    ct);

        if (definition == null)
        {
            throw new InvalidOperationException(
                $"Baseline definition was not found in SQLite for runtime id {baseline.UniqueId} / key '{baseline.CanonicalKey}'. " +
                "Run custom baseline precheck/sync before using learned or historical tensor-combo truth.");
        }

        return definition;
    }

    public static async Task<BaselineQuantDefinition?> TryResolveDefinitionByRuntimeIdAsync(
        MagicQuantContext db,
        byte runtimeBaselineId,
        int? architectureFamilyId,
        CancellationToken ct = default)
    {
        if (architectureFamilyId != null)
        {
            var custom = await db.BaselineQuantDefinitions
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.ArchitectureFamilyId == architectureFamilyId.Value && x.RuntimeBaselineId == runtimeBaselineId, ct);

            if (custom != null)
                return custom;
        }

        return await db.BaselineQuantDefinitions
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.ArchitectureFamilyId == null && x.RuntimeBaselineId == runtimeBaselineId, ct);
    }

    public static BaselineQuants ToRuntimeBaseline(BaselineQuantDefinition definition, bool forceInactiveRegistration = false)
    {
        if (!definition.IsCustomBaseline)
            return BaselineQuants.FromId(definition.RuntimeBaselineId);

        var scheme = TensorWeightScheme.FromId(definition.DefaultTensorSchemeId);
        var baseline = BaselineQuants.CreateDynamicCustomBaseline(
            uniqueId: definition.RuntimeBaselineId,
            displayName: string.IsNullOrWhiteSpace(definition.DisplayName) ? definition.BaselineName : definition.DisplayName,
            quantizeBaseArgumentName: definition.QuantizeBaseArgumentName,
            sourceRepository: definition.SourceRepository ?? string.Empty,
            sourceFileName: definition.SourceFileName ?? string.Empty,
            shortSourceName: definition.ShortSourceName ?? "External",
            sourceOwner: definition.SourceOwner ?? string.Empty,
            sourceKind: definition.SourceKind,
            canonicalKey: definition.CanonicalKey,
            primaryTensorWeightScheme: scheme,
            learnedMatchTensorWeightSchemes: [scheme],
            bannedGroupIds: Array.Empty<byte>(),
            requiresImatrix: definition.RequiresImatrix,
            isLearningBaseline: definition.IsActiveInCurrentConfig && definition.IsLearningBaseline,
            isCombinationCarrierCandidate: definition.IsActiveInCurrentConfig && definition.IsCombinationCarrierCandidate,
            isExplicitGroupCombinationCandidate: definition.IsActiveInCurrentConfig && definition.IsExplicitGroupCombinationCandidate,
            bitRange: definition.BitRange,
            explicitCandidateSortOrder: definition.ExplicitCandidateSortOrder);

        if (definition.IsActiveInCurrentConfig || forceInactiveRegistration)
            return baseline;

        return baseline with
        {
            IsLearningBaseline = false,
            IsCombinationCarrierCandidate = false,
            IsExplicitGroupCombinationCandidate = false
        };
    }
}
