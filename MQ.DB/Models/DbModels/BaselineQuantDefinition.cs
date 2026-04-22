using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class BaselineQuantDefinition : ISQLiteEntity<BaselineQuantDefinition>
{
    public byte BaselineQuantId { get; set; }
    public string CanonicalKey { get; set; } = string.Empty;
    public string BaselineName { get; set; } = string.Empty;
    public string QuantizeBaseArgumentName { get; set; } = string.Empty;
    public byte DefaultTensorSchemeId { get; set; }
    public string DefaultTensorSchemeName { get; set; } = string.Empty;
    public string SourceKind { get; set; } = string.Empty;
    public string? SourceOwner { get; set; }
    public string? SourceRepository { get; set; }
    public string? SourceFileName { get; set; }
    public string? ShortSourceName { get; set; }
    public bool IsCustomBaseline { get; set; }
    public bool IsLearningBaseline { get; set; }
    public bool IsCombinationCarrierCandidate { get; set; }
    public bool IsExplicitGroupCombinationCandidate { get; set; }
    public bool RequiresImatrix { get; set; }
    public byte BitRange { get; set; }
    public int ExplicitCandidateSortOrder { get; set; }

    public void Configure(EntityTypeBuilder<BaselineQuantDefinition> builder)
    {
        builder.HasKey(x => x.BaselineQuantId);

        builder.Property(x => x.CanonicalKey)
            .HasMaxLength(256)
            .IsRequired();

        builder.Property(x => x.BaselineName)
            .HasMaxLength(128)
            .IsRequired();

        builder.Property(x => x.QuantizeBaseArgumentName)
            .HasMaxLength(64)
            .IsRequired();

        builder.Property(x => x.DefaultTensorSchemeName)
            .HasMaxLength(64)
            .IsRequired();

        builder.Property(x => x.SourceKind)
            .HasMaxLength(64)
            .IsRequired();

        builder.Property(x => x.SourceOwner)
            .HasMaxLength(128);

        builder.Property(x => x.SourceRepository)
            .HasMaxLength(256);

        builder.Property(x => x.SourceFileName)
            .HasMaxLength(512);

        builder.Property(x => x.ShortSourceName)
            .HasMaxLength(64);

        builder.HasIndex(x => x.CanonicalKey).IsUnique();
        builder.HasIndex(x => new { x.SourceRepository, x.SourceFileName });
    }
}
