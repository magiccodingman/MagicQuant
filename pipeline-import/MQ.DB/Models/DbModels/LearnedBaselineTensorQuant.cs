using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class LearnedBaselineTensorQuant : ISQLiteEntity<LearnedBaselineTensorQuant>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;

    public int TensorGroupProfileId { get; set; }
    public TensorGroupProfile TensorGroupProfile { get; set; } = default!;

    public int BaselineQuantDefinitionId { get; set; }
    public BaselineQuantDefinition BaselineQuantDefinition { get; set; } = default!;

    public Guid TensorComboId { get; set; }
    public TensorCombo TensorCombo { get; set; } = default!;

    public Guid AiBenchmarkId { get; set; }
    public AiBenchmark AiBenchmark { get; set; } = default!;

    /// <summary>
    /// Exact source model hash used when learning happened. Reuse is scoped by architecture family/profile.
    /// </summary>
    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;

    /// <summary>
    /// Snapshot of the compact runtime id at learning time.
    /// </summary>
    public byte BaselineQuantId { get; set; }

    public string BaselineCanonicalKey { get; set; } = string.Empty;
    public string BaselineSourceKind { get; set; } = string.Empty;
    public string? BaselineSourceRepository { get; set; }
    public string? BaselineSourceFileName { get; set; }

    public byte TensorWeightSchemeId { get; set; }
    public byte TensorGroupId { get; set; }

    public string TensorName { get; set; } = string.Empty;
    public string FinalQuantType { get; set; } = string.Empty;

    public void Configure(EntityTypeBuilder<LearnedBaselineTensorQuant> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();

        builder.Property(x => x.BaselineCanonicalKey).HasMaxLength(512).IsRequired();
        builder.Property(x => x.BaselineSourceKind).HasMaxLength(64).IsRequired();
        builder.Property(x => x.BaselineSourceRepository).HasMaxLength(256);
        builder.Property(x => x.BaselineSourceFileName).HasMaxLength(512);
        builder.Property(x => x.TensorName).HasMaxLength(512).IsRequired();
        builder.Property(x => x.FinalQuantType).HasMaxLength(32).IsRequired();

        builder.HasIndex(x => new
            {
                x.ArchitectureFamilyId,
                x.TensorGroupProfileId,
                x.BaselineQuantDefinitionId,
                x.TensorWeightSchemeId,
                x.TensorName
            })
            .IsUnique();

        builder.HasIndex(x => new
        {
            x.ArchitectureFamilyId,
            x.TensorGroupProfileId,
            x.BaselineQuantDefinitionId,
            x.TensorGroupId
        });

        builder.HasIndex(x => x.AiBenchmarkId);
        builder.HasIndex(x => x.AiModelHashId);
        builder.HasIndex(x => x.TensorComboId);

        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.TensorGroupProfile)
            .WithMany()
            .HasForeignKey(x => x.TensorGroupProfileId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.BaselineQuantDefinition)
            .WithMany()
            .HasForeignKey(x => x.BaselineQuantDefinitionId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.AiBenchmark)
            .WithMany()
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
