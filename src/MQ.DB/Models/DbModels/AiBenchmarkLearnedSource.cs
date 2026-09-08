using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class AiBenchmarkLearnedSource : ISQLiteEntity<AiBenchmarkLearnedSource>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid AiBenchmarkId { get; set; }
    public AiBenchmark AiBenchmark { get; set; } = default!;

    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;

    public int TensorGroupProfileId { get; set; }
    public TensorGroupProfile TensorGroupProfile { get; set; } = default!;

    public Guid TensorComboId { get; set; }
    public TensorCombo TensorCombo { get; set; } = default!;

    public byte TensorGroupId { get; set; }

    public int BaselineQuantDefinitionId { get; set; }
    public BaselineQuantDefinition BaselineQuantDefinition { get; set; } = default!;

    public Guid? SourceLearningBenchmarkId { get; set; }
    public AiBenchmark? SourceLearningBenchmark { get; set; }

    public string BaselineCanonicalKey { get; set; } = string.Empty;
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<AiBenchmarkLearnedSource> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();
        builder.Property(x => x.BaselineCanonicalKey).HasMaxLength(512).IsRequired();

        builder.HasIndex(x => new { x.AiBenchmarkId, x.TensorGroupId }).IsUnique();
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.BaselineQuantDefinitionId });
        builder.HasIndex(x => x.TensorComboId);

        builder.HasOne(x => x.AiBenchmark)
            .WithMany(x => x.LearnedSources)
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.TensorGroupProfile)
            .WithMany()
            .HasForeignKey(x => x.TensorGroupProfileId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.BaselineQuantDefinition)
            .WithMany()
            .HasForeignKey(x => x.BaselineQuantDefinitionId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.SourceLearningBenchmark)
            .WithMany()
            .HasForeignKey(x => x.SourceLearningBenchmarkId)
            .OnDelete(DeleteBehavior.SetNull);
    }
}
