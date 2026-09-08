using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class ImatrixDefinition : ISQLiteEntity<ImatrixDefinition>
{
    public int Id { get; set; }
    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;
    public string IdentityHash { get; set; } = string.Empty;
    public string? CanonicalPath { get; set; }
    public string SourceKind { get; set; } = "none";
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public string? MetadataJson { get; set; }
    public int? TokenCount { get; set; }
    public string? BuildFingerprint { get; set; }

    public void Configure(EntityTypeBuilder<ImatrixDefinition> builder)
    {
        builder.HasKey(x => x.Id);
        builder.HasIndex(x => new { x.AiModelHashId, x.IdentityHash }).IsUnique();
        builder.Property(x => x.IdentityHash).HasMaxLength(128);
        builder.Property(x => x.CanonicalPath).HasMaxLength(2048);
        builder.Property(x => x.SourceKind).HasMaxLength(64);
        builder.Property(x => x.MetadataJson).HasMaxLength(8000);
        builder.Property(x => x.BuildFingerprint).HasMaxLength(512);

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}