using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class BaselineQuantDefinition : ISQLiteEntity<BaselineQuantDefinition>
{
    public int Id { get; set; }

    /// <summary>
    /// Null for built-in standard/exact aliases. Non-null for architecture-family-scoped custom baselines.
    /// </summary>
    public int? ArchitectureFamilyId { get; set; }
    public ArchitectureFamily? ArchitectureFamily { get; set; }

    /// <summary>
    /// Compact id used inside TensorCombo slots.
    /// </summary>
    public byte RuntimeBaselineId { get; set; }

    public string CanonicalKey { get; set; } = string.Empty;
    public string NormalizedCanonicalKey { get; set; } = string.Empty;
    public string BaselineName { get; set; } = string.Empty;
    public string DisplayName { get; set; } = string.Empty;
    public string QuantizeBaseArgumentName { get; set; } = string.Empty;
    public byte DefaultTensorSchemeId { get; set; }
    public string DefaultTensorSchemeName { get; set; } = string.Empty;
    public string SourceKind { get; set; } = string.Empty;
    public string? SourceOwner { get; set; }
    public string? SourceRepository { get; set; }
    public string? NormalizedSourceRepository { get; set; }
    public string? SourceFileName { get; set; }
    public string? NormalizedSourceFileName { get; set; }
    public string? ShortSourceName { get; set; }
    public string? BaselineFamily { get; set; }
    public bool IsCustomBaseline { get; set; }
    public bool IsLearningBaseline { get; set; }
    public bool IsCombinationCarrierCandidate { get; set; }
    public bool IsExplicitGroupCombinationCandidate { get; set; }
    public bool RequiresImatrix { get; set; }
    public byte BitRange { get; set; }
    public int ExplicitCandidateSortOrder { get; set; }
    public bool IsActiveInCurrentConfig { get; set; } = true;
    public DateTime FirstSeenUtc { get; set; } = DateTime.UtcNow;
    public DateTime LastSeenUtc { get; set; } = DateTime.UtcNow;
    public DateTime LastUpdatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<BaselineQuantDefinition> builder)
    {
        builder.HasKey(x => x.Id);

        builder.Property(x => x.CanonicalKey).HasMaxLength(512).IsRequired();
        builder.Property(x => x.NormalizedCanonicalKey).HasMaxLength(512).IsRequired();
        builder.Property(x => x.BaselineName).HasMaxLength(128).IsRequired();
        builder.Property(x => x.DisplayName).HasMaxLength(128).IsRequired();
        builder.Property(x => x.QuantizeBaseArgumentName).HasMaxLength(64).IsRequired();
        builder.Property(x => x.DefaultTensorSchemeName).HasMaxLength(64).IsRequired();
        builder.Property(x => x.SourceKind).HasMaxLength(64).IsRequired();
        builder.Property(x => x.SourceOwner).HasMaxLength(128);
        builder.Property(x => x.SourceRepository).HasMaxLength(256);
        builder.Property(x => x.NormalizedSourceRepository).HasMaxLength(256);
        builder.Property(x => x.SourceFileName).HasMaxLength(512);
        builder.Property(x => x.NormalizedSourceFileName).HasMaxLength(512);
        builder.Property(x => x.ShortSourceName).HasMaxLength(64);
        builder.Property(x => x.BaselineFamily).HasMaxLength(128);

        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.RuntimeBaselineId }).IsUnique();
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.NormalizedCanonicalKey }).IsUnique();
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.NormalizedSourceRepository, x.NormalizedSourceFileName }).IsUnique();
        builder.HasIndex(x => new { x.RuntimeBaselineId, x.ArchitectureFamilyId });
        builder.HasIndex(x => x.IsActiveInCurrentConfig);

        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
