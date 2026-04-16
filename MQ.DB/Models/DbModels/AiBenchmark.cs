using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public enum BenchmarkCategory
{
    General = 1,
    Math = 2,
    Code = 3,
}

public class AiBenchmark : ISQLiteEntity<AiBenchmark>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>
    /// n-N gpu layers
    /// </summary>
    public byte Ngl { get; set; }

    /// <summary>
    /// Size of the model in bytes at this combination.
    /// </summary>
    public ulong SizeBytes { get; set; }

    public double TokensPerSecond { get; set; }

    /// <summary>
    /// foreign key
    /// </summary>
    public Guid TensorComboId { get; set; }

    public TensorCombo TensorCombo { get; set; } = default!;

    /// <summary>
    /// foreign key
    /// </summary>
    public uint AiModelHashId { get; set; }

    public AiModelHash AiModelHash { get; set; } = default!;

    public List<CategoryBenchmark> CategorBenchmarks { get; set; } = new();

    public void Configure(EntityTypeBuilder<AiBenchmark> builder)
    {
        builder.HasKey(x => x.Id);
        
        builder.Property(x => x.Id)
            .ValueGeneratedNever();

        builder.HasIndex(x => new { x.AiModelHashId, x.TensorComboId })
            .IsUnique();

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasMany(x => x.CategorBenchmarks)
            .WithOne(x => x.AiBenchmark)
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}

public class CategoryBenchmark : ISQLiteEntity<CategoryBenchmark>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>
    /// foreign key to AiBenchmark
    /// </summary>
    public Guid AiBenchmarkId { get; set; }

    public AiBenchmark AiBenchmark { get; set; } = default!;

    /// <summary>
    /// Byte version to BenchmarkCategory enum in C#
    /// </summary>
    public byte Category { get; set; }

    public double Kld { get; set; }
    public double Ppl { get; set; }
    public double PplError { get; set; }

    public void Configure(EntityTypeBuilder<CategoryBenchmark> builder)
    {
        builder.HasKey(x => x.Id);
        
        builder.Property(x => x.Id)
            .ValueGeneratedNever();

        builder.HasIndex(x => x.AiBenchmarkId);

        builder.HasOne(x => x.AiBenchmark)
            .WithMany(x => x.CategorBenchmarks)
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}