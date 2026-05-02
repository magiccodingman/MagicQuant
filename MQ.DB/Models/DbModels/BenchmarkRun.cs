using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class BenchmarkRun : ISQLiteEntity<BenchmarkRun>
{
    public Guid Id { get; set; }

    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;

    public int TensorGroupProfileId { get; set; }
    public TensorGroupProfile TensorGroupProfile { get; set; } = default!;

    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;

    public int? ImatrixDefinitionId { get; set; }
    public ImatrixDefinition? ImatrixDefinition { get; set; }

    public Guid TensorComboId { get; set; }
    public TensorCombo TensorCombo { get; set; } = default!;

    public Guid AiBenchmarkId { get; set; }
    public AiBenchmark AiBenchmark { get; set; } = default!;

    /// <summary>
    /// Nullable until the CategoryBenchmark row is created/persisted.
    /// </summary>
    public Guid? CategoryBenchmarkId { get; set; }
    public CategoryBenchmark? CategoryBenchmark { get; set; }

    /// <summary>
    /// Snapshot of the category for convenience and resilience.
    /// Stored as the byte value of BenchmarkCategory.
    /// </summary>
    public byte Category { get; set; }

    public DateTime StartedUtc { get; set; }
    public DateTime CompletedUtc { get; set; }

    public long DurationMs { get; set; }

    public bool Succeeded { get; set; }

    public string? Error { get; set; }

    public void Configure(EntityTypeBuilder<BenchmarkRun> builder)
    {
        builder.HasKey(x => x.Id);

        builder.Property(x => x.Id)
            .ValueGeneratedNever();

        builder.HasIndex(x => x.ArchitectureFamilyId);
        builder.HasIndex(x => x.TensorGroupProfileId);
        builder.HasIndex(x => x.AiModelHashId);
        builder.HasIndex(x => x.ImatrixDefinitionId);
        builder.HasIndex(x => x.TensorComboId);
        builder.HasIndex(x => x.AiBenchmarkId);
        builder.HasIndex(x => x.CategoryBenchmarkId);
        builder.HasIndex(x => x.StartedUtc);
        builder.HasIndex(x => new { x.AiBenchmarkId, x.Category });

        builder.Property(x => x.Error)
            .HasMaxLength(4000);

        builder.HasOne(x => x.ArchitectureFamily)
            .WithMany()
            .HasForeignKey(x => x.ArchitectureFamilyId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.TensorGroupProfile)
            .WithMany()
            .HasForeignKey(x => x.TensorGroupProfileId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.ImatrixDefinition)
            .WithMany()
            .HasForeignKey(x => x.ImatrixDefinitionId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.AiBenchmark)
            .WithMany()
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.CategoryBenchmark)
            .WithMany()
            .HasForeignKey(x => x.CategoryBenchmarkId)
            .OnDelete(DeleteBehavior.SetNull);
    }
}
