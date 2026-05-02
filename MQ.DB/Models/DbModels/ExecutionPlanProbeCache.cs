using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class ExecutionPlanProbeCache : ISQLiteEntity<ExecutionPlanProbeCache>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;

    public int TensorGroupProfileId { get; set; }
    public TensorGroupProfile TensorGroupProfile { get; set; } = default!;

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

    public int ProbeSchemaVersion { get; set; } = 2;
    public ulong Q8ModelSizeBytes { get; set; }
    public int Q8StableNgl { get; set; }
    public ulong NativeModelSizeBytes { get; set; }
    public int NativeStableNgl { get; set; }
    public string NativeQuantizationKey { get; set; } = string.Empty;
    public int MaxCandidateNgl { get; set; }
    public string GpuMemoryLimitsJson { get; set; } = "{}";
    public string TensorSplitJson { get; set; } = "{}";

    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<ExecutionPlanProbeCache> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();

        builder.Property(x => x.HardwareFingerprint).HasMaxLength(1024);
        builder.Property(x => x.QuantizedModelFingerprint).HasMaxLength(2048);
        builder.Property(x => x.QuantizationKey).HasMaxLength(128);
        builder.Property(x => x.NativeQuantizationKey).HasMaxLength(128);
        builder.Property(x => x.SlotsJson).HasMaxLength(8000);
        builder.Property(x => x.GpuMemoryLimitsJson).HasMaxLength(4000);
        builder.Property(x => x.TensorSplitJson).HasMaxLength(4000);

        builder.HasIndex(x => x.ArchitectureFamilyId);
        builder.HasIndex(x => x.TensorGroupProfileId);
        builder.HasIndex(x => x.AiModelHashId);
        builder.HasIndex(x => x.ImatrixDefinitionId);
        builder.HasIndex(x => new
        {
            x.ArchitectureFamilyId,
            x.TensorGroupProfileId,
            x.AiModelHashId,
            x.ImatrixDefinitionId,
            x.HardwareFingerprint,
            x.QuantizedModelFingerprint,
            x.QuantizationKey,
            x.DiscoveryTokenTarget
        }).IsUnique();

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
    }
}
