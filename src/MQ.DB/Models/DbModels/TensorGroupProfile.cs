using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class TensorGroupProfile : ISQLiteEntity<TensorGroupProfile>
{
    public int Id { get; set; }
    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;
    public string FingerprintHash { get; set; } = string.Empty;
    public string SnapshotJson { get; set; } = string.Empty;
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public bool IsActive { get; set; } = true;

    public void Configure(EntityTypeBuilder<TensorGroupProfile> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.FingerprintHash).HasMaxLength(128).IsRequired();
        builder.Property(x => x.SnapshotJson).IsRequired();
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.FingerprintHash }).IsUnique();
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.IsActive });

        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
