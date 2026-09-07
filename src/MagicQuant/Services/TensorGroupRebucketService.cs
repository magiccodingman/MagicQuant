using MagicQuant.Models.Learning;
using MagicQuant.Services.Learning;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class TensorGroupRebucketService
{
    private const byte UnknownTensorGroupId = 255;
    private readonly TensorGroupingAuditService _auditService = new();

    public async Task<TensorGroupRebucketSummary> RebucketFromExistingProfileTruthAsync(CancellationToken ct = default)
    {
        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int currentProfileId = TensorGroupProfileService.RequireCurrentProfileId();

        await using var db = new MagicQuantContext();

        var sourceProfiles = await db.TensorGroupProfiles
            .AsNoTracking()
            .Where(x => x.ArchitectureFamilyId == architectureFamilyId && x.Id != currentProfileId)
            .OrderByDescending(x => x.CreatedUtc)
            .ThenByDescending(x => x.Id)
            .Select(x => new { x.Id, x.FingerprintHash, x.CreatedUtc })
            .ToListAsync(ct);

        if (sourceProfiles.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]Tensor-group rebucket requested, but no previous tensor group profiles exist for this architecture family.[/]");
            return TensorGroupRebucketSummary.Empty;
        }

        uint scopedModelHashId = await ArchitectureFamilyService.ResolveScopedAiModelHashIdAsync(db, ct);
        int? currentImatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(
            db,
            scopedModelHashId,
            createIfMissing: false,
            ct);

        var candidateKeys = await db.LearnedBaselineTensorQuants
            .AsNoTracking()
            .Where(x => x.ArchitectureFamilyId == architectureFamilyId && x.TensorGroupProfileId != currentProfileId)
            .GroupBy(x => new { x.BaselineQuantDefinitionId, x.TensorWeightSchemeId })
            .Select(g => new
            {
                g.Key.BaselineQuantDefinitionId,
                g.Key.TensorWeightSchemeId,
                RowCount = g.Count(),
                LatestProfileId = g.Max(x => x.TensorGroupProfileId)
            })
            .ToListAsync(ct);

        if (candidateKeys.Count == 0)
        {
            AnsiConsole.MarkupLine("[grey]Tensor-group rebucket requested, but no previous learned tensor truth exists for this architecture family.[/]");
            return TensorGroupRebucketSummary.Empty;
        }

        int copiedBaselines = 0;
        int copiedRows = 0;
        int clonedBenchmarks = 0;
        int skippedExisting = 0;
        int fatalSkipped = 0;

        AnsiConsole.Write(new Rule("[yellow]Tensor Group Rebucket From Existing Truth[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine("[grey]Rebuilding learned tensor/group rows for the active regex profile from prior DB truth. Pure baseline benchmark rows are cloned when available; group-override isolation truth is intentionally not cloned.[/]");

        foreach (var key in candidateKeys
                     .OrderBy(x => x.BaselineQuantDefinitionId)
                     .ThenBy(x => x.TensorWeightSchemeId))
        {
            bool alreadyExists = await db.LearnedBaselineTensorQuants
                .AsNoTracking()
                .AnyAsync(x => x.ArchitectureFamilyId == architectureFamilyId &&
                               x.TensorGroupProfileId == currentProfileId &&
                               x.BaselineQuantDefinitionId == key.BaselineQuantDefinitionId &&
                               x.TensorWeightSchemeId == key.TensorWeightSchemeId, ct);

            if (alreadyExists)
            {
                skippedExisting++;
                continue;
            }

            var sourceProfileId = await PickBestSourceProfileForKeyAsync(
                db,
                architectureFamilyId,
                currentProfileId,
                key.BaselineQuantDefinitionId,
                key.TensorWeightSchemeId,
                sourceProfiles.Select(x => x.Id).ToList(),
                ct);

            if (!sourceProfileId.HasValue)
                continue;

            var sourceRows = await db.LearnedBaselineTensorQuants
                .AsNoTracking()
                .Where(x => x.ArchitectureFamilyId == architectureFamilyId &&
                            x.TensorGroupProfileId == sourceProfileId.Value &&
                            x.BaselineQuantDefinitionId == key.BaselineQuantDefinitionId &&
                            x.TensorWeightSchemeId == key.TensorWeightSchemeId)
                .OrderBy(x => x.TensorName)
                .ToListAsync(ct);

            if (sourceRows.Count == 0)
                continue;

            var truth = sourceRows.ToDictionary(
                x => x.TensorName,
                x => new LearnedTensorTruth(x.TensorName, x.FinalQuantType, LearningSource.GgufOnly),
                StringComparer.Ordinal);

            var audit = _auditService.Audit(truth.Keys.ToList(), truth);
            if (audit.HasFatalIssues)
            {
                fatalSkipped++;
                AnsiConsole.MarkupLine(
                    $"[red]Skipped rebucket for BaselineDefinitionId={key.BaselineQuantDefinitionId}, scheme={key.TensorWeightSchemeId}:[/] ambiguous={audit.Ambiguous.Count}, unresolved={audit.IllegalUnresolved.Count}. Fix tensor_groups.yaml first.");
                continue;
            }

            var targetBenchmarkId = await EnsurePureBaselineBenchmarkCloneAsync(
                db,
                sourceRows,
                architectureFamilyId,
                currentProfileId,
                scopedModelHashId,
                currentImatrixDefinitionId,
                ct);

            if (targetBenchmarkId.Cloned)
                clonedBenchmarks++;

            await db.LearnedBaselineTensorQuants
                .Where(x => x.ArchitectureFamilyId == architectureFamilyId &&
                            x.TensorGroupProfileId == currentProfileId &&
                            x.BaselineQuantDefinitionId == key.BaselineQuantDefinitionId &&
                            x.TensorWeightSchemeId == key.TensorWeightSchemeId)
                .ExecuteDeleteAsync(ct);

            var targetRows = sourceRows.Select(row => new LearnedBaselineTensorQuant
            {
                Id = Guid.NewGuid(),
                ArchitectureFamilyId = architectureFamilyId,
                TensorGroupProfileId = currentProfileId,
                BaselineQuantDefinitionId = row.BaselineQuantDefinitionId,
                TensorComboId = row.TensorComboId,
                AiBenchmarkId = targetBenchmarkId.BenchmarkId,
                AiModelHashId = scopedModelHashId,
                BaselineQuantId = row.BaselineQuantId,
                BaselineCanonicalKey = row.BaselineCanonicalKey,
                BaselineSourceKind = row.BaselineSourceKind,
                BaselineSourceRepository = row.BaselineSourceRepository,
                BaselineSourceFileName = row.BaselineSourceFileName,
                TensorWeightSchemeId = row.TensorWeightSchemeId,
                TensorGroupId = ResolveRebucketedTensorGroupId(row.TensorName, audit),
                TensorName = row.TensorName,
                FinalQuantType = row.FinalQuantType
            }).ToList();

            db.LearnedBaselineTensorQuants.AddRange(targetRows);
            await db.SaveChangesAsync(ct);

            copiedBaselines++;
            copiedRows += targetRows.Count;

            string baselineName = await db.BaselineQuantDefinitions
                .AsNoTracking()
                .Where(x => x.Id == key.BaselineQuantDefinitionId)
                .Select(x => x.DisplayName)
                .FirstOrDefaultAsync(ct) ?? key.BaselineQuantDefinitionId.ToString();

            AnsiConsole.MarkupLine(
                $"[green]Rebucketed learned truth:[/] {Markup.Escape(baselineName)} scheme={key.TensorWeightSchemeId} tensors={targetRows.Count:N0} fromProfile={sourceProfileId.Value} -> currentProfile={currentProfileId}");
        }

        var summary = new TensorGroupRebucketSummary
        {
            BaselineSchemeSetsCopied = copiedBaselines,
            LearnedRowsCopied = copiedRows,
            PureBenchmarkRowsCloned = clonedBenchmarks,
            ExistingCurrentProfileSetsSkipped = skippedExisting,
            FatalSetsSkipped = fatalSkipped
        };

        AnsiConsole.MarkupLine(
            $"[green]Tensor-group rebucket summary:[/] baseline/scheme sets={summary.BaselineSchemeSetsCopied:N0}, rows={summary.LearnedRowsCopied:N0}, pure benchmarks cloned={summary.PureBenchmarkRowsCloned:N0}, already-current skipped={summary.ExistingCurrentProfileSetsSkipped:N0}, fatal skipped={summary.FatalSetsSkipped:N0}");

        return summary;
    }

    private static byte ResolveRebucketedTensorGroupId(
        string tensorName,
        TensorGroupingAuditResult audit)
    {
        if (!audit.GroupedByTensor.TryGetValue(tensorName, out var grouped))
        {
            throw new InvalidOperationException(
                $"Cannot rebucket tensor '{tensorName}' because it was not present in the active grouping audit.");
        }

        if (grouped.PrimaryGroup != null)
            return grouped.PrimaryGroup.UniqueId;

        if (grouped.IsBaseQuantException)
            return UnknownTensorGroupId;

        throw new InvalidOperationException(
            $"Cannot rebucket tensor '{tensorName}' because it is unresolved under the active tensor_groups.yaml profile. " +
            "BaseQuant fallback is only allowed when the tensor matches base_quant_exceptions.");
    }

    private static async Task<int?> PickBestSourceProfileForKeyAsync(
        MagicQuantContext db,
        int architectureFamilyId,
        int currentProfileId,
        int baselineDefinitionId,
        byte tensorWeightSchemeId,
        IReadOnlyList<int> preferredProfileOrder,
        CancellationToken ct)
    {
        foreach (var profileId in preferredProfileOrder)
        {
            bool exists = await db.LearnedBaselineTensorQuants
                .AsNoTracking()
                .AnyAsync(x => x.ArchitectureFamilyId == architectureFamilyId &&
                               x.TensorGroupProfileId == profileId &&
                               x.TensorGroupProfileId != currentProfileId &&
                               x.BaselineQuantDefinitionId == baselineDefinitionId &&
                               x.TensorWeightSchemeId == tensorWeightSchemeId, ct);
            if (exists)
                return profileId;
        }

        return null;
    }

    private static async Task<(Guid BenchmarkId, bool Cloned)> EnsurePureBaselineBenchmarkCloneAsync(
        MagicQuantContext db,
        IReadOnlyList<LearnedBaselineTensorQuant> sourceRows,
        int architectureFamilyId,
        int currentProfileId,
        uint scopedModelHashId,
        int? currentImatrixDefinitionId,
        CancellationToken ct)
    {
        var sourceBenchmarkId = sourceRows.Select(x => x.AiBenchmarkId).FirstOrDefault(x => x != Guid.Empty);
        if (sourceBenchmarkId == Guid.Empty)
            throw new InvalidOperationException("Cannot rebucket learned rows because the source AiBenchmarkId snapshot is missing.");

        var sourceBenchmark = await db.AiBenchmarks
            .Include(x => x.CategorBenchmarks)
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.Id == sourceBenchmarkId, ct);

        if (sourceBenchmark == null)
            throw new InvalidOperationException($"Cannot rebucket learned rows because source AiBenchmarkId={sourceBenchmarkId} was not found.");

        var target = await db.AiBenchmarks
            .FirstOrDefaultAsync(x => x.ArchitectureFamilyId == architectureFamilyId &&
                                      x.TensorGroupProfileId == currentProfileId &&
                                      x.AiModelHashId == scopedModelHashId &&
                                      x.ImatrixDefinitionId == currentImatrixDefinitionId &&
                                      x.TensorComboId == sourceBenchmark.TensorComboId, ct);

        if (target != null)
            return (target.Id, false);

        target = new AiBenchmark
        {
            Id = Guid.NewGuid(),
            ArchitectureFamilyId = architectureFamilyId,
            TensorGroupProfileId = currentProfileId,
            AiModelHashId = scopedModelHashId,
            ImatrixDefinitionId = currentImatrixDefinitionId,
            TensorComboId = sourceBenchmark.TensorComboId,
            Ngl = sourceBenchmark.Ngl,
            SizeBytes = sourceBenchmark.SizeBytes,
            TokensPerSecond = sourceBenchmark.TokensPerSecond
        };

        db.AiBenchmarks.Add(target);
        await db.SaveChangesAsync(ct);

        if (sourceBenchmark.CategorBenchmarks.Count > 0)
        {
            db.Set<CategoryBenchmark>().AddRange(sourceBenchmark.CategorBenchmarks.Select(x => new CategoryBenchmark
            {
                Id = Guid.NewGuid(),
                AiBenchmarkId = target.Id,
                Category = x.Category,
                Kld = x.Kld,
                Ppl = x.Ppl,
                PplError = x.PplError
            }));

            await db.SaveChangesAsync(ct);
        }

        return (target.Id, true);
    }
}

public sealed class TensorGroupRebucketSummary
{
    public int BaselineSchemeSetsCopied { get; init; }
    public int LearnedRowsCopied { get; init; }
    public int PureBenchmarkRowsCloned { get; init; }
    public int ExistingCurrentProfileSetsSkipped { get; init; }
    public int FatalSetsSkipped { get; init; }

    public static TensorGroupRebucketSummary Empty { get; } = new();
}
