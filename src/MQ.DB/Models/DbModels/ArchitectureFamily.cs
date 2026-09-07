using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class ArchitectureFamily : ISQLiteEntity<ArchitectureFamily>
{
    public int Id { get; set; }
    public string NormalizedName { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string TensorSignatureHash { get; set; } = string.Empty;
    public int TensorCount { get; set; }
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<ArchitectureFamily> builder)
    {
        builder.HasKey(x => x.Id);
        builder.HasIndex(x => x.NormalizedName).IsUnique();
        builder.HasIndex(x => new { x.TensorSignatureHash, x.TensorCount });
        builder.Property(x => x.NormalizedName).HasMaxLength(256).IsRequired();
        builder.Property(x => x.DisplayName).HasMaxLength(256).IsRequired();
        builder.Property(x => x.TensorSignatureHash).HasMaxLength(128).IsRequired();
    }
}
