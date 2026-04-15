using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class QuantizationRun : ISQLiteEntity<QuantizationRun>
{
    public Guid Id { get; set; }

    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;

    public uint TensorComboId { get; set; }
    public TensorCombo TensorCombo { get; set; } = default!;

    /// <summary>
    /// Nullable because a quantization can fail before a benchmark row exists.
    /// </summary>
    public uint? AiBenchmarkId { get; set; }
    public AiBenchmark? AiBenchmark { get; set; }

    public DateTime StartedUtc { get; set; }
    public DateTime CompletedUtc { get; set; }

    /// <summary>
    /// Total wall clock duration in milliseconds.
    /// </summary>
    public long DurationMs { get; set; }

    public bool Succeeded { get; set; }

    public string? Error { get; set; }

    public string? OutputModelPath { get; set; }

    public void Configure(EntityTypeBuilder<QuantizationRun> builder)
    {
        builder.HasKey(x => x.Id);

        builder.Property(x => x.Id)
            .ValueGeneratedNever();

        builder.HasIndex(x => x.AiModelHashId);
        builder.HasIndex(x => x.TensorComboId);
        builder.HasIndex(x => x.AiBenchmarkId);
        builder.HasIndex(x => x.StartedUtc);

        builder.Property(x => x.Error)
            .HasMaxLength(4000);

        builder.Property(x => x.OutputModelPath)
            .HasMaxLength(2048);

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.TensorCombo)
            .WithMany()
            .HasForeignKey(x => x.TensorComboId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(x => x.AiBenchmark)
            .WithMany()
            .HasForeignKey(x => x.AiBenchmarkId)
            .OnDelete(DeleteBehavior.SetNull);
    }
}