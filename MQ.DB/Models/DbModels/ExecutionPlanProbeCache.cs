using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class ExecutionPlanProbeCache : ISQLiteEntity<ExecutionPlanProbeCache>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;

    public int? ImatrixDefinitionId { get; set; }
    public ImatrixDefinition? ImatrixDefinition { get; set; }

    public string HardwareFingerprint { get; set; } = string.Empty;
    public string QuantizedModelFingerprint { get; set; } = string.Empty;
    public string QuantizationKey { get; set; } = string.Empty;
    public int DiscoveryTokenTarget { get; set; }

    public int StaticNgl { get; set; }
    public bool UsesGpu { get; set; }
    public int GroupSize { get; set; }
    public string SlotsJson { get; set; } = "[]";

    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<ExecutionPlanProbeCache> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();

        builder.Property(x => x.HardwareFingerprint).HasMaxLength(1024);
        builder.Property(x => x.QuantizedModelFingerprint).HasMaxLength(2048);
        builder.Property(x => x.QuantizationKey).HasMaxLength(128);
        builder.Property(x => x.SlotsJson).HasMaxLength(8000);

        builder.HasIndex(x => x.AiModelHashId);
        builder.HasIndex(x => x.ImatrixDefinitionId);
        builder.HasIndex(x => new
        {
            x.AiModelHashId,
            x.ImatrixDefinitionId,
            x.HardwareFingerprint,
            x.QuantizedModelFingerprint,
            x.QuantizationKey,
            x.DiscoveryTokenTarget
        }).IsUnique();

        builder.HasOne(x => x.AiModelHash)
            .WithMany()
            .HasForeignKey(x => x.AiModelHashId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasOne(x => x.ImatrixDefinition)
            .WithMany()
            .HasForeignKey(x => x.ImatrixDefinitionId)
            .OnDelete(DeleteBehavior.Restrict);
    }
}
