using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using MagicQuant.Helpers;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class ArchitectureFamilyService
{
    private readonly PythonManager _python;

    public ArchitectureFamilyService(PythonManager python)
    {
        _python = python;
    }

    public async Task EnsureCurrentArchitectureFamilyAsync(string bf16GgufPath, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentArchitectureFamilyName))
            throw new InvalidOperationException("Architecture family is required. Provide --architecture-family or set identity.architecture_family_name in YAML.");

        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            throw new InvalidOperationException("Cache.CurrentModelId is not set.");

        var tensorNames = await ReadTensorNamesFromGgufAsync(bf16GgufPath, ct);
        if (tensorNames.Count == 0)
            throw new InvalidOperationException("Architecture-family validation could not read any tensor names from the BF16 GGUF.");

        string signatureHash = ComputeTensorSignatureHash(tensorNames);
        int tensorCount = tensorNames.Count;
        string normalized = NormalizeFamilyName(Cache.CurrentArchitectureFamilyName);

        await using var db = new MagicQuantContext();

        var aiModelHash = await db.AiModelHashes.FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (aiModelHash == null)
        {
            aiModelHash = new AiModelHash { UniqueHash = Cache.CurrentModelId };
            db.AiModelHashes.Add(aiModelHash);
            await db.SaveChangesAsync(ct);
        }

        var existingMapping = await db.Set<ArchitectureFamilyModelHash>()
            .Include(x => x.ArchitectureFamily)
            .FirstOrDefaultAsync(x => x.AiModelHashId == aiModelHash.Id, ct);

        if (existingMapping != null)
        {
            if (!string.Equals(existingMapping.ArchitectureFamily.NormalizedName, normalized, StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"Current model hash is already mapped to architecture family '{existingMapping.ArchitectureFamily.DisplayName}'.");
            }

            Cache.CurrentArchitectureFamilyId = existingMapping.ArchitectureFamilyId;
            Cache.CurrentArchitectureFamilyName = existingMapping.ArchitectureFamily.DisplayName;
            return;
        }

        var matchingName = await db.Set<ArchitectureFamily>()
            .FirstOrDefaultAsync(x => x.NormalizedName == normalized, ct);

        if (matchingName != null)
        {
            if (matchingName.TensorCount != tensorCount || !string.Equals(matchingName.TensorSignatureHash, signatureHash, StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"Architecture family '{matchingName.DisplayName}' already exists, but the current model tensor names/count do not match the previously registered architecture. Expected count={matchingName.TensorCount}, actual count={tensorCount}.");
            }

            bool familyAlreadyHasHashes = await db.Set<ArchitectureFamilyModelHash>()
                .AnyAsync(x => x.ArchitectureFamilyId == matchingName.Id, ct);

            if (familyAlreadyHasHashes && !Cache.AllowArchitectureFamilyAliasOverride)
            {
                throw new InvalidOperationException(
                    $"Architecture family '{matchingName.DisplayName}' already has one or more model hashes attached. " +
                    "Adding the current hash means you are manually asserting these different model hashes share the same tensor architecture/truth. " +
                    "Rerun with --allow-architecture-family-alias-override only if you intentionally approve this shared-family linkage.");
            }

            db.Add(new ArchitectureFamilyModelHash
            {
                ArchitectureFamilyId = matchingName.Id,
                AiModelHashId = aiModelHash.Id,
                IsCanonical = false
            });
            await db.SaveChangesAsync(ct);

            Cache.CurrentArchitectureFamilyId = matchingName.Id;
            Cache.CurrentArchitectureFamilyName = matchingName.DisplayName;
            AnsiConsole.MarkupLine($"[green]Architecture family linked:[/] [cyan]{Markup.Escape(matchingName.DisplayName)}[/] -> model hash [grey]{Markup.Escape(Cache.CurrentModelId)}[/]");
            return;
        }

        var sameSignatureFamilies = await db.Set<ArchitectureFamily>()
            .Where(x => x.TensorCount == tensorCount && x.TensorSignatureHash == signatureHash)
            .OrderBy(x => x.DisplayName)
            .ToListAsync(ct);

        if (sameSignatureFamilies.Count > 0 && !Cache.AllowArchitectureFamilyAliasOverride)
        {
            throw new InvalidOperationException(
                $"The provided architecture family '{Cache.CurrentArchitectureFamilyName}' matches an existing architecture signature already registered under: {string.Join(", ", sameSignatureFamilies.Select(x => x.DisplayName))}. Use one of those names or rerun with --allow-architecture-family-alias-override if you intentionally want a separate family namespace.");
        }

        var family = new ArchitectureFamily
        {
            NormalizedName = normalized,
            DisplayName = Cache.CurrentArchitectureFamilyName.Trim(),
            TensorSignatureHash = signatureHash,
            TensorCount = tensorCount,
            CreatedUtc = DateTime.UtcNow
        };
        db.Add(family);
        await db.SaveChangesAsync(ct);

        db.Add(new ArchitectureFamilyModelHash
        {
            ArchitectureFamilyId = family.Id,
            AiModelHashId = aiModelHash.Id,
            IsCanonical = true
        });
        await db.SaveChangesAsync(ct);

        Cache.CurrentArchitectureFamilyId = family.Id;
        Cache.CurrentArchitectureFamilyName = family.DisplayName;
        AnsiConsole.MarkupLine($"[green]Architecture family created:[/] [cyan]{Markup.Escape(family.DisplayName)}[/] tensors={tensorCount:N0}");
    }

    public static async Task<uint?> ResolveExactCurrentAiModelHashIdOrNullAsync(MagicQuantContext db, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            return null;

        return await db.AiModelHashes
            .AsNoTracking()
            .Where(x => x.UniqueHash == Cache.CurrentModelId)
            .Select(x => (uint?)x.Id)
            .FirstOrDefaultAsync(ct);
    }

    public static async Task<uint> ResolveExactCurrentAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct = default)
    {
        var id = await ResolveExactCurrentAiModelHashIdOrNullAsync(db, ct);
        if (id == null)
            throw new InvalidOperationException("Unable to resolve the exact current AiModelHashId.");
        return id.Value;
    }

    public static async Task<uint?> ResolveScopedAiModelHashIdOrNullAsync(MagicQuantContext db, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.CurrentModelId))
            return null;

        var current = await db.AiModelHashes.AsNoTracking().FirstOrDefaultAsync(x => x.UniqueHash == Cache.CurrentModelId, ct);
        if (current == null)
            return null;

        if (Cache.CurrentArchitectureFamilyId == null)
            return current.Id;

        var canonical = await db.Set<ArchitectureFamilyModelHash>()
            .AsNoTracking()
            .Where(x => x.ArchitectureFamilyId == Cache.CurrentArchitectureFamilyId.Value)
            .OrderByDescending(x => x.IsCanonical)
            .ThenBy(x => x.AiModelHashId)
            .Select(x => (uint?)x.AiModelHashId)
            .FirstOrDefaultAsync(ct);

        return canonical ?? current.Id;
    }

    public static async Task<uint> ResolveScopedAiModelHashIdAsync(MagicQuantContext db, CancellationToken ct = default)
    {
        var id = await ResolveScopedAiModelHashIdOrNullAsync(db, ct);
        if (id == null)
            throw new InvalidOperationException("Unable to resolve the current scoped AiModelHashId.");
        return id.Value;
    }

    private static string NormalizeFamilyName(string value) => value.Trim().ToLowerInvariant();

    private static string ComputeTensorSignatureHash(IReadOnlyCollection<string> tensorNames)
    {
        using var sha = SHA256.Create();
        var payload = string.Join("\n", tensorNames.OrderBy(x => x, StringComparer.Ordinal));
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(payload));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private async Task<List<string>> ReadTensorNamesFromGgufAsync(string ggufPath, CancellationToken ct)
    {
        string workingDir = Path.Combine(Cache.ModelMagicQuantDirectory ?? Path.GetTempPath(), "_architecture_family");
        Directory.CreateDirectory(workingDir);
        string unique = Guid.NewGuid().ToString("N");
        string payloadPath = Path.Combine(workingDir, $"arch_payload_{unique}.json");
        string resultPath = Path.Combine(workingDir, $"arch_result_{unique}.json");
        string scriptPath = Path.Combine(workingDir, $"arch_script_{unique}.py");

        try
        {
            await File.WriteAllTextAsync(payloadPath, JsonSerializer.Serialize(new { gguf_path = ggufPath, output_path = resultPath }), ct);
            const string py = """
import json
import sys
payload_path = sys.argv[1]
with open(payload_path, 'r', encoding='utf-8') as f:
    payload = json.load(f)
try:
    import gguf
    reader = gguf.GGUFReader(payload['gguf_path'])
    names = [t.name for t in reader.tensors]
    result = {'TensorNames': names, 'Error': None}
except Exception as e:
    result = {'TensorNames': [], 'Error': str(e)}
with open(payload['output_path'], 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
""";
            await File.WriteAllTextAsync(scriptPath, py, ct);
            await _python.RunPythonScriptAsync(scriptPath, $"\"{payloadPath}\"");
            using var stream = File.OpenRead(resultPath);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
            var root = doc.RootElement;
            var err = root.TryGetProperty("Error", out var e) && e.ValueKind != JsonValueKind.Null ? e.GetString() : null;
            if (!string.IsNullOrWhiteSpace(err))
                throw new InvalidOperationException($"Failed to read GGUF tensor names for architecture family validation: {err}");
            return root.GetProperty("TensorNames").EnumerateArray()
                .Select(x => x.GetString() ?? string.Empty)
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToList();
        }
        finally
        {
            TryDelete(payloadPath);
            TryDelete(resultPath);
            TryDelete(scriptPath);
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { }
    }
}