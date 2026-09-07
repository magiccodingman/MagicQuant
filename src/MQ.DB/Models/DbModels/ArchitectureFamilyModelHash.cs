using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class ArchitectureFamilyModelHash : ISQLiteEntity<ArchitectureFamilyModelHash>
{
    public int Id { get; set; }
    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;
    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;
    public bool IsCanonical { get; set; }
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<ArchitectureFamilyModelHash> builder)
    {
        builder.HasKey(x => x.Id);
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.AiModelHashId }).IsUnique();
        builder.HasIndex(x => x.AiModelHashId).IsUnique();
        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
