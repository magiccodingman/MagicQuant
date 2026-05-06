using System.Text.Json;
using MagicQuant.Helpers;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class IsolationDiagnosticsManifestService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public async Task<string> GenerateIsolationSamplesAsync(
        string outputDirectory,
        RequiredSampleGenerationResult samplePlan,
        CancellationToken ct = default)
    {
        string manifestDirectory = MagicQuantManifestPathService.EnsureManifestDirectory(outputDirectory);
        string path = Path.Combine(manifestDirectory, MagicQuantManifestPathService.IsolationSamplesFileName);

        var samples = await BuildIsolationSamplePayloadAsync(samplePlan, ct);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(samples, JsonOptions), ct);
        AnsiConsole.MarkupLine($"[green]Isolation sample JSON generated:[/] {Markup.Escape(path)} samples={samples.Count:N0}");
        return path;
    }

    public async Task<string> GenerateBadTradesAsync(
        string outputDirectory,
        IsolationOptimizationResult isolationResult,
        CancellationToken ct = default)
    {
        string manifestDirectory = MagicQuantManifestPathService.EnsureManifestDirectory(outputDirectory);
        string path = Path.Combine(manifestDirectory, MagicQuantManifestPathService.BadTradesFileName);

        var payload = new
        {
            generatedUtc = DateTime.UtcNow,
            modelId = Cache.CurrentModelId,
            architectureFamily = Cache.CurrentArchitectureFamilyName,
            summary = new
            {
                badTradeEliminations = isolationResult.BadTradeEliminations,
                finalKldCleanupEliminations = isolationResult.FinalKldCleanupEliminations,
                disabledBaselines = isolationResult.DisabledBaselines,
                structuredBadTradeRows = isolationResult.BadTradeDetails.Count
            },
            thresholds = new
            {
                maxSizeDeltaPercent = IsolationPruningConfig.BadTradeMaxSizeDeltaPercent,
                kldMultiplier = IsolationPruningConfig.BadTradeKldMultiplier,
                pplMultiplier = IsolationPruningConfig.BadTradePplMultiplier,
                floatingPointEpsilon = IsolationPruningConfig.FloatingPointEpsilon
            },
            badTrades = isolationResult.BadTradeDetails,
            notes = isolationResult.Notes
                .Where(x => x.Contains("bad trade", StringComparison.OrdinalIgnoreCase) ||
                            x.Contains("carrier anchor", StringComparison.OrdinalIgnoreCase) ||
                            x.Contains("combination baseline", StringComparison.OrdinalIgnoreCase))
                .Distinct(StringComparer.Ordinal)
                .ToList()
        };

        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(payload, JsonOptions), ct);
        AnsiConsole.MarkupLine($"[green]Bad trade JSON generated:[/] {Markup.Escape(path)} records={isolationResult.BadTradeDetails.Count:N0}");
        return path;
    }

    private static async Task<List<object>> BuildIsolationSamplePayloadAsync(
        RequiredSampleGenerationResult samplePlan,
        CancellationToken ct)
    {
        await using var db = new MagicQuantContext();

        var exactAiModelHashId = await ArchitectureFamilyService.ResolveExactCurrentAiModelHashIdOrNullAsync(db, ct);
        if (exactAiModelHashId == null)
            throw new InvalidOperationException("Cannot export isolation sample manifest because the current exact AiModelHashId could not be resolved.");

        var imatrixDefinitionId = await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(
            db,
            exactAiModelHashId.Value,
            createIfMissing: false,
            ct: ct);

        var nativeSnapshot = await LoadSnapshotAsync(db, exactAiModelHashId.Value, imatrixDefinitionId, (TensorConfig)HybridQuant.CreatePureBaseline(BaselineQuants.GetBF16Quant()), ct);
        var nativeByCategory = nativeSnapshot?.Categories.ToDictionary(x => x.CategoryId) ?? new Dictionary<byte, CategoryPayload>();

        var output = new List<object>();

        foreach (var plan in samplePlan.Plans
                     .Where(x => x.Kind is RequiredSampleKind.BaseOnlyIsolation or RequiredSampleKind.GroupIsolationProbe or RequiredSampleKind.GroupIsolationContinuation)
                     .OrderBy(x => x.Kind)
                     .ThenBy(x => x.TargetGroupId ?? 0)
                     .ThenBy(x => x.TestedBaselineId ?? 0)
                     .ThenBy(x => x.TestedCandidateId ?? 0)
                     .ThenBy(x => x.Key, StringComparer.Ordinal))
        {
            ct.ThrowIfCancellationRequested();

            var config = (TensorConfig)plan.Quant;
            var snapshot = await LoadSnapshotAsync(db, exactAiModelHashId.Value, imatrixDefinitionId, config, ct);
            var categories = snapshot?.Categories ?? new List<CategoryPayload>();

            double? kld = categories.Where(x => x.Kld.HasValue).Select(x => x.Kld!.Value).AverageOrNull();
            double? ppl = categories.Select(x => x.Ppl).AverageOrNull();
            double? pplDelta = CalculateAggregatePplDelta(categories, nativeByCategory);

            output.Add(new
            {
                key = plan.Key,
                description = plan.Description,
                kind = plan.Kind.ToString(),
                tensorConfigKey = MagicQuant.Models.TensorConfigIdentity.ToKey(config),
                group = ResolveGroupName(plan.TargetGroupId),
                testedBaseline = ResolveBaselineName(plan.TestedBaselineId),
                testedCandidate = ResolveBaselineName(plan.TestedCandidateId),
                testedBaselineCanonicalKey = plan.TestedBaselineCanonicalKey,
                testedCandidateCanonicalKey = plan.TestedCandidateCanonicalKey,
                isSmallestProbe = plan.IsSmallestProbe,
                sizeBytes = snapshot?.SizeBytes,
                sizeGB = snapshot?.SizeBytes is { } bytes ? ToGBNumber(bytes) : (double?)null,
                sizeGiB = snapshot?.SizeBytes is { } gibBytes ? ToGiBNumber(gibBytes) : (double?)null,
                kld,
                ppl,
                pplDeltaPercent = pplDelta,
                foundBenchmark = snapshot != null,
                categories,
                config = new
                {
                    config.BaseQuant,
                    config.Embeddings,
                    config.LmHead,
                    config.AttnQ,
                    config.AttnKV,
                    config.AttnOutput,
                    config.FfnUpGate,
                    config.FfnDown,
                    config.MoeExperts,
                    config.MoeRouter
                }
            });
        }

        return output;
    }

    private static async Task<BenchmarkPayload?> LoadSnapshotAsync(
        MagicQuantContext db,
        uint aiModelHashId,
        int? imatrixDefinitionId,
        TensorConfig lookup,
        CancellationToken ct)
    {
        int architectureFamilyId = TensorGroupProfileService.RequireCurrentArchitectureFamilyId();
        int tensorGroupProfileId = TensorGroupProfileService.RequireCurrentProfileId();

        var row = await db.AiBenchmarks
            .AsNoTracking()
            .Include(x => x.CategorBenchmarks)
            .Join(db.TensorCombos,
                b => b.TensorComboId,
                c => c.Id,
                (b, c) => new { b, c })
            .FirstOrDefaultAsync(x =>
                    x.b.ArchitectureFamilyId == architectureFamilyId &&
                    x.b.TensorGroupProfileId == tensorGroupProfileId &&
                    x.b.AiModelHashId == aiModelHashId &&
                    x.b.ImatrixDefinitionId == imatrixDefinitionId &&
                    x.c.BaseQuant == lookup.BaseQuant &&
                    x.c.Embeddings == lookup.Embeddings &&
                    x.c.LmHead == lookup.LmHead &&
                    x.c.AttnQ == lookup.AttnQ &&
                    x.c.AttnKV == lookup.AttnKV &&
                    x.c.AttnOutput == lookup.AttnOutput &&
                    x.c.FfnUpGate == lookup.FfnUpGate &&
                    x.c.FfnDown == lookup.FfnDown &&
                    x.c.MoeExperts == lookup.MoeExperts &&
                    x.c.MoeRouter == lookup.MoeRouter,
                ct);

        if (row == null)
            return null;

        return new BenchmarkPayload
        {
            SizeBytes = row.b.SizeBytes,
            Categories = row.b.CategorBenchmarks
                .OrderBy(x => x.Category)
                .Select(x => new CategoryPayload
                {
                    CategoryId = x.Category,
                    Category = Enum.IsDefined(typeof(BenchmarkCategory), (int)x.Category)
                        ? ((BenchmarkCategory)x.Category).ToString()
                        : x.Category.ToString(),
                    Kld = x.Kld,
                    Ppl = x.Ppl,
                    PplError = x.PplError
                })
                .ToList()
        };
    }

    private static double? CalculateAggregatePplDelta(
        IReadOnlyCollection<CategoryPayload> categories,
        IReadOnlyDictionary<byte, CategoryPayload> nativeByCategory)
    {
        var deltas = new List<double>();

        foreach (var category in categories)
        {
            if (!nativeByCategory.TryGetValue(category.CategoryId, out var native))
                continue;

            if (native.Ppl <= 0d || category.Ppl <= 0d)
                continue;

            deltas.Add(((category.Ppl - native.Ppl) / native.Ppl) * 100d);
        }

        return deltas.Count == 0 ? null : deltas.Average();
    }

    private static string? ResolveGroupName(byte? groupId)
    {
        if (groupId == null)
            return null;

        return TReg.All.FirstOrDefault(x => x.UniqueId == groupId.Value)?.Name ?? groupId.Value.ToString();
    }

    private static string? ResolveBaselineName(byte? baselineId)
    {
        if (baselineId == null)
            return null;

        try
        {
            return BaselineQuants.FromId(baselineId.Value).Names[0];
        }
        catch
        {
            return baselineId.Value.ToString();
        }
    }

    private static double ToGBNumber(ulong bytes) => bytes / 1000d / 1000d / 1000d;
    private static double ToGiBNumber(ulong bytes) => bytes / 1024d / 1024d / 1024d;

    private sealed class BenchmarkPayload
    {
        public ulong SizeBytes { get; init; }
        public List<CategoryPayload> Categories { get; init; } = new();
    }

    private sealed class CategoryPayload
    {
        public byte CategoryId { get; init; }
        public string Category { get; init; } = string.Empty;
        public double? Kld { get; init; }
        public double Ppl { get; init; }
        public double PplError { get; init; }
    }
}

internal static class MagicQuantEnumerableExtensions
{
    public static double? AverageOrNull(this IEnumerable<double> values)
    {
        var list = values.Where(x => !double.IsNaN(x) && !double.IsInfinity(x)).ToList();
        return list.Count == 0 ? null : list.Average();
    }
}