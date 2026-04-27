using System.Security.Cryptography;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models.DbModels;
using Microsoft.EntityFrameworkCore;

namespace MagicQuant.Services;

public static class ImatrixIdentityService
{
    public static async Task<string?> EnsureActiveImatrixIdentityHashAsync(CancellationToken ct = default)
    {
        if (!Cache.IsImatrixAvailable || string.IsNullOrWhiteSpace(Cache.ActiveImatrixPath))
        {
            Cache.ActiveImatrixIdentityHash = null;
            return null;
        }

        if (!string.IsNullOrWhiteSpace(Cache.ActiveImatrixIdentityHash))
            return Cache.ActiveImatrixIdentityHash;

        await using var stream = File.OpenRead(Cache.ActiveImatrixPath);
        var hash = await SHA256.HashDataAsync(stream, ct);
        Cache.ActiveImatrixIdentityHash = Convert.ToHexString(hash).ToLowerInvariant();
        return Cache.ActiveImatrixIdentityHash;
    }

    public static async Task<int?> ResolveCurrentImatrixDefinitionIdAsync(
        MagicQuantContext db,
        uint aiModelHashId,
        bool createIfMissing,
        CancellationToken ct = default)
    {
        var identityHash = await EnsureActiveImatrixIdentityHashAsync(ct);
        if (string.IsNullOrWhiteSpace(identityHash))
            return null;

        var existing = await db.ImatrixDefinitions
            .FirstOrDefaultAsync(x => x.AiModelHashId == aiModelHashId && x.IdentityHash == identityHash, ct);

        if (existing != null)
            return existing.Id;

        if (!createIfMissing)
            return null;

        var row = new ImatrixDefinition
        {
            AiModelHashId = aiModelHashId,
            IdentityHash = identityHash,
            CanonicalPath = Cache.ActiveImatrixPath,
            SourceKind = "runtime-active",
            MetadataJson = null,
            BuildFingerprint = null,
            CreatedUtc = DateTime.UtcNow
        };

        db.ImatrixDefinitions.Add(row);
        await db.SaveChangesAsync(ct);
        return row.Id;
    }

    public static async Task ValidateOwnershipAsync(
        MagicQuantContext db,
        uint aiModelHashId,
        int? imatrixDefinitionId,
        CancellationToken ct = default)
    {
        if (!imatrixDefinitionId.HasValue)
            return;

        var ownerHashId = await db.ImatrixDefinitions
            .AsNoTracking()
            .Where(x => x.Id == imatrixDefinitionId.Value)
            .Select(x => (uint?)x.AiModelHashId)
            .FirstOrDefaultAsync(ct);

        if (ownerHashId == null || ownerHashId.Value != aiModelHashId)
        {
            throw new InvalidOperationException(
                $"ImatrixDefinitionId {imatrixDefinitionId.Value} does not belong to AiModelHashId {aiModelHashId}. " +
                $"Owner AiModelHashId={(ownerHashId.HasValue ? ownerHashId.Value.ToString() : "missing")}. " +
                "Imatrix identity is exact-model-hash scoped and must not be resolved through architecture family scope.");
        }
    }
}
