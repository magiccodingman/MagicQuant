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

public class AiBenchmark: ISQLiteEntity<AiBenchmark>
{
    public uint Id { get; set; }

    /// <summary>
    /// n-N gpu layers
    /// </summary>
    public byte Ngl { get; set; }

    /// <summary>
    /// Size of the model in bytes at this combination. 
    /// </summary>
    /// <returns></returns>
    public ulong SizeBytes { get; set; }

    public double TokensPerSecond { get; set; }

    // both the AiModelHash and the TensorComboId combined
    // must be unique in the table. 

    /// <summary>
    /// foreign key
    /// </summary>
    public uint TensorComboId { get; set; }

    public TensorCombo TensorCombo { get; set; }

    /// <summary>
    /// foreign key
    /// </summary>
    public uint AiModelHashId { get; set; }

    public AiModelHash AiModelHash { get; set; }
    
    public List<CategoryBenchmark>  CategorBenchmarks { get; set; }
    
    public void Configure(EntityTypeBuilder<AiBenchmark> builder)
    {
        builder.HasKey(x => x.Id);

        builder.HasIndex(x => new { x.AiModelHashId, x.TensorComboId })
            .IsUnique();

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict); // Prevent deleting a combo if benchmarks exist

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);
    }
    
}

public class CategoryBenchmark: ISQLiteEntity<CategoryBenchmark>
{
    public uint Id { get; set; }
    
    /// <summary>
    /// foreign key to AiBenchmark
    /// </summary>
    public uint AiBenchmarkId { get; set; }
    public AiBenchmark AiBenchmark { get; set; }
    
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

        // Foreign Key Configuration
        builder.HasOne(x => x.AiBenchmark)
            .WithMany() // One Benchmark has Many Category Scores
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade); // If you delete the Benchmark, delete its scores

        builder.HasIndex(x => x.AiBenchmarkId);
        
        builder.HasOne<AiBenchmark>() 
            .WithMany(p => p.CategorBenchmarks) 
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}