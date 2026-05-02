using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class TensorGroupProfileService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public async Task<TensorGroupProfile> EnsureCurrentProfileAsync(CancellationToken ct = default)
    {
        int architectureFamilyId = Cache.CurrentArchitectureFamilyId
            ?? throw new InvalidOperationException("Architecture family must be resolved before resolving tensor group profile.");

        string snapshotJson = BuildSnapshotJson();
        string fingerprint = ComputeSha256(snapshotJson);

        await using var db = new MagicQuantContext();

        var existing = await db.TensorGroupProfiles
            .FirstOrDefaultAsync(x => x.ArchitectureFamilyId == architectureFamilyId && x.FingerprintHash == fingerprint, ct);

        if (existing == null)
        {
            existing = new TensorGroupProfile
            {
                ArchitectureFamilyId = architectureFamilyId,
                FingerprintHash = fingerprint,
                SnapshotJson = snapshotJson,
                CreatedUtc = DateTime.UtcNow,
                IsActive = true
            };
            db.TensorGroupProfiles.Add(existing);
        }

        var activeProfiles = await db.TensorGroupProfiles
            .Where(x => x.ArchitectureFamilyId == architectureFamilyId && x.Id != existing.Id && x.IsActive)
            .ToListAsync(ct);

        foreach (var profile in activeProfiles)
            profile.IsActive = false;

        existing.IsActive = true;
        await db.SaveChangesAsync(ct);

        Cache.CurrentTensorGroupProfileId = existing.Id;
        Cache.CurrentTensorGroupProfileFingerprintHash = existing.FingerprintHash;

        AnsiConsole.MarkupLine($"[green]Tensor group profile active:[/] id=[cyan]{existing.Id}[/] hash=[grey]{Markup.Escape(existing.FingerprintHash[..Math.Min(12, existing.FingerprintHash.Length)])}[/]");
        return existing;
    }

    public static int RequireCurrentProfileId() =>
        Cache.CurrentTensorGroupProfileId
        ?? throw new InvalidOperationException("Current tensor group profile is not set. Call TensorGroupProfileService.EnsureCurrentProfileAsync after architecture-family resolution.");

    public static int RequireCurrentArchitectureFamilyId() =>
        Cache.CurrentArchitectureFamilyId
        ?? throw new InvalidOperationException("Current architecture family is not set.");

    public static string BuildSnapshotJson()
    {
        var snapshot = new
        {
            schema = 1,
            groups = TReg.All
                .OrderBy(x => x.UniqueId)
                .Select(x => new
                {
                    id = x.UniqueId,
                    name = x.Name,
                    patterns = x.Tensors
                        .Where(p => !string.IsNullOrWhiteSpace(p))
                        .Select(p => p.Trim())
                        .ToArray()
                })
                .ToArray(),
            baseQuantExceptions = TReg.GetBaseQuantExceptionPatterns()
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => p.Trim())
                .ToArray()
        };

        return JsonSerializer.Serialize(snapshot, JsonOptions);
    }

    private static string ComputeSha256(string value)
    {
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();
    }
}
