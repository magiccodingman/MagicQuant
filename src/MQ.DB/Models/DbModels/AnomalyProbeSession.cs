using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class AnomalyProbeSession : ISQLiteEntity<AnomalyProbeSession>
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
    public byte BenchmarkCategory { get; set; } = (byte)MQ.DB.Models.DbModels.BenchmarkCategory.General;
    public DateTime StartedUtc { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedUtc { get; set; }
    public string SourceRunLabel { get; set; } = string.Empty;
    public string ConfigJson { get; set; } = string.Empty;
    public List<AnomalyProbeObservation> Observations { get; set; } = new();

    public void Configure(EntityTypeBuilder<AnomalyProbeSession> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();
        builder.Property(x => x.SourceRunLabel).HasMaxLength(256);
        builder.Property(x => x.ConfigJson).HasColumnType("TEXT");
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.AiModelHashId, x.ImatrixDefinitionId, x.BenchmarkCategory, x.StartedUtc });
        builder.HasOne(x => x.ArchitectureFamily).WithMany().HasForeignKey(x => x.ArchitectureFamilyId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.TensorGroupProfile).WithMany().HasForeignKey(x => x.TensorGroupProfileId).OnDelete(DeleteBehavior.Restrict);
        builder.HasOne(x => x.AiModelHash).WithMany().HasForeignKey(x => x.AiModelHashId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.ImatrixDefinition).WithMany().HasForeignKey(x => x.ImatrixDefinitionId).OnDelete(DeleteBehavior.Restrict);
        builder.HasMany(x => x.Observations).WithOne(x => x.Session).HasForeignKey(x => x.SessionId).OnDelete(DeleteBehavior.Cascade);
    }
}

public class AnomalyProbeObservation : ISQLiteEntity<AnomalyProbeObservation>
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid SessionId { get; set; }
    public AnomalyProbeSession Session { get; set; } = default!;
    public int ArchitectureFamilyId { get; set; }
    public ArchitectureFamily ArchitectureFamily { get; set; } = default!;
    public int TensorGroupProfileId { get; set; }
    public TensorGroupProfile TensorGroupProfile { get; set; } = default!;
    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;
    public int? ImatrixDefinitionId { get; set; }
    public ImatrixDefinition? ImatrixDefinition { get; set; }
    public byte BenchmarkCategory { get; set; } = (byte)MQ.DB.Models.DbModels.BenchmarkCategory.General;
    public Guid? ReferenceTensorComboId { get; set; }
    public TensorCombo? ReferenceTensorCombo { get; set; }
    public Guid? ProbeTensorComboId { get; set; }
    public TensorCombo? ProbeTensorCombo { get; set; }
    public string ProbeType { get; set; } = string.Empty;
    public string Classification { get; set; } = string.Empty;
    public string HypothesisLabel { get; set; } = string.Empty;
    public string MovementClassification { get; set; } = string.Empty;
    public string ChangedGroupSetHash { get; set; } = string.Empty;
    public string ChangedGroupsJson { get; set; } = string.Empty;
    public string CandidateQuantsJson { get; set; } = string.Empty;
    public string ReferenceEffectiveGroupsJson { get; set; } = string.Empty;
    public string CandidateEffectiveGroupsJson { get; set; } = string.Empty;
    public string InactiveGroupsJson { get; set; } = string.Empty;
    public string ReferenceTensorConfigKey { get; set; } = string.Empty;
    public string ProbeTensorConfigKey { get; set; } = string.Empty;
    public string ReferenceDisplayName { get; set; } = string.Empty;
    public string ProbeDisplayName { get; set; } = string.Empty;
    public string ReferenceInternalName { get; set; } = string.Empty;
    public string ProbeInternalName { get; set; } = string.Empty;
    public string SeedClass { get; set; } = string.Empty;
    public int SeedPriority { get; set; }
    public string ProbePlanClass { get; set; } = string.Empty;
    public bool IsContextualAnomalyProbe { get; set; }
    public bool OldBf16Isolation { get; set; }
    public bool AllActiveGroupsExplicit { get; set; }
    public byte ReferenceQuantId { get; set; }
    public double ActualKld { get; set; }
    public double PredictedKld { get; set; }
    public double ReferenceActualKld { get; set; }
    public double ReferencePredictedKld { get; set; }
    public double ActualGainVsTwin { get; set; }
    public double PredictionSpaceGapVsTwin { get; set; }
    public ulong SizeSavingsBytes { get; set; }
    public int UpgradeCount { get; set; }
    public int DowngradeCount { get; set; }
    public int SameCount { get; set; }
    public int UnknownCount { get; set; }
    public int NetBitDelta { get; set; }
    public string RuleDirection { get; set; } = string.Empty;
    public bool Accepted { get; set; }
    public string FailureCode { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;

    public void Configure(EntityTypeBuilder<AnomalyProbeObservation> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();
        builder.Property(x => x.ProbeType).HasMaxLength(64);
        builder.Property(x => x.Classification).HasMaxLength(64);
        builder.Property(x => x.HypothesisLabel).HasMaxLength(128);
        builder.Property(x => x.MovementClassification).HasMaxLength(64);
        builder.Property(x => x.ChangedGroupSetHash).HasMaxLength(128);
        builder.Property(x => x.ChangedGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.CandidateQuantsJson).HasColumnType("TEXT");
        builder.Property(x => x.ReferenceEffectiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.CandidateEffectiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.InactiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.ReferenceTensorConfigKey).HasMaxLength(128);
        builder.Property(x => x.ProbeTensorConfigKey).HasMaxLength(128);
        builder.Property(x => x.ReferenceDisplayName).HasMaxLength(512);
        builder.Property(x => x.ProbeDisplayName).HasMaxLength(512);
        builder.Property(x => x.ReferenceInternalName).HasMaxLength(512);
        builder.Property(x => x.ProbeInternalName).HasMaxLength(512);
        builder.Property(x => x.SeedClass).HasMaxLength(64);
        builder.Property(x => x.ProbePlanClass).HasMaxLength(64);
        builder.HasIndex(x => x.ReferenceTensorConfigKey);
        builder.HasIndex(x => x.ProbeTensorConfigKey);
        builder.Property(x => x.RuleDirection).HasMaxLength(64);
        builder.Property(x => x.FailureCode).HasMaxLength(128);
        builder.Property(x => x.Message).HasMaxLength(4000);
        builder.HasIndex(x => x.SessionId);
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.AiModelHashId, x.ImatrixDefinitionId, x.BenchmarkCategory, x.ChangedGroupSetHash });
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.AiModelHashId, x.ImatrixDefinitionId, x.BenchmarkCategory, x.ReferenceTensorComboId, x.ProbeTensorComboId, x.ProbeType });
        builder.HasOne(x => x.Session).WithMany(x => x.Observations).HasForeignKey(x => x.SessionId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.ArchitectureFamily).WithMany().HasForeignKey(x => x.ArchitectureFamilyId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.TensorGroupProfile).WithMany().HasForeignKey(x => x.TensorGroupProfileId).OnDelete(DeleteBehavior.Restrict);
        builder.HasOne(x => x.AiModelHash).WithMany().HasForeignKey(x => x.AiModelHashId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.ImatrixDefinition).WithMany().HasForeignKey(x => x.ImatrixDefinitionId).OnDelete(DeleteBehavior.Restrict);
        builder.HasOne(x => x.ReferenceTensorCombo).WithMany().HasForeignKey(x => x.ReferenceTensorComboId).OnDelete(DeleteBehavior.Restrict);
        builder.HasOne(x => x.ProbeTensorCombo).WithMany().HasForeignKey(x => x.ProbeTensorComboId).OnDelete(DeleteBehavior.Restrict);
    }
}

public class AnomalyInteractionRule : ISQLiteEntity<AnomalyInteractionRule>
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
    public byte BenchmarkCategory { get; set; } = (byte)MQ.DB.Models.DbModels.BenchmarkCategory.General;
    public byte ReferenceQuantId { get; set; }
    public string ReferenceContextKey { get; set; } = string.Empty;
    public string ReferenceEffectiveGroupsJson { get; set; } = string.Empty;
    public string CandidateEffectiveGroupsJson { get; set; } = string.Empty;
    public string InactiveGroupsJson { get; set; } = string.Empty;
    public string FullTensorConfigKey { get; set; } = string.Empty;
    public string ReferenceDisplayName { get; set; } = string.Empty;
    public string CandidateDisplayName { get; set; } = string.Empty;
    public string ReferenceInternalName { get; set; } = string.Empty;
    public string CandidateInternalName { get; set; } = string.Empty;
    public string RuleType { get; set; } = string.Empty;
    public string RuleDirection { get; set; } = string.Empty;
    public string RuleStatus { get; set; } = string.Empty;
    public string MovementClassification { get; set; } = string.Empty;
    public string GroupSetHash { get; set; } = string.Empty;
    public int GroupCount { get; set; }
    public double MeanActualGainVsTwin { get; set; }
    public double BestActualGainVsTwin { get; set; }
    public double MeanPredictionSpaceGap { get; set; }
    public double BestPredictionSpaceGap { get; set; }
    public double AppliedPredictionSpaceAdjustmentKld { get; set; }
    public int EvidenceCount { get; set; }
    public double Confidence { get; set; }
    public double ShrinkFactor { get; set; }
    public string Status { get; set; } = string.Empty;
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedUtc { get; set; } = DateTime.UtcNow;
    public string MetadataJson { get; set; } = string.Empty;
    public List<AnomalyInteractionRuleGroupState> GroupStates { get; set; } = new();

    public void Configure(EntityTypeBuilder<AnomalyInteractionRule> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();
        builder.Property(x => x.ReferenceContextKey).HasMaxLength(512);
        builder.Property(x => x.ReferenceEffectiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.CandidateEffectiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.InactiveGroupsJson).HasColumnType("TEXT");
        builder.Property(x => x.FullTensorConfigKey).HasMaxLength(128);
        builder.Property(x => x.ReferenceDisplayName).HasMaxLength(512);
        builder.Property(x => x.CandidateDisplayName).HasMaxLength(512);
        builder.Property(x => x.ReferenceInternalName).HasMaxLength(512);
        builder.Property(x => x.CandidateInternalName).HasMaxLength(512);
        builder.HasIndex(x => x.FullTensorConfigKey);
        builder.Property(x => x.RuleType).HasMaxLength(64);
        builder.Property(x => x.RuleDirection).HasMaxLength(64);
        builder.Property(x => x.RuleStatus).HasMaxLength(64);
        builder.Property(x => x.MovementClassification).HasMaxLength(64);
        builder.Property(x => x.GroupSetHash).HasMaxLength(128);
        builder.Property(x => x.Status).HasMaxLength(64);
        builder.Property(x => x.MetadataJson).HasColumnType("TEXT");
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.AiModelHashId, x.ImatrixDefinitionId, x.BenchmarkCategory, x.RuleDirection, x.RuleStatus });
        builder.HasIndex(x => new { x.ArchitectureFamilyId, x.TensorGroupProfileId, x.AiModelHashId, x.ImatrixDefinitionId, x.BenchmarkCategory, x.ReferenceQuantId, x.ReferenceContextKey, x.GroupSetHash, x.RuleDirection }).IsUnique();
        builder.HasOne(x => x.ArchitectureFamily).WithMany().HasForeignKey(x => x.ArchitectureFamilyId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.TensorGroupProfile).WithMany().HasForeignKey(x => x.TensorGroupProfileId).OnDelete(DeleteBehavior.Restrict);
        builder.HasOne(x => x.AiModelHash).WithMany().HasForeignKey(x => x.AiModelHashId).OnDelete(DeleteBehavior.Cascade);
        builder.HasOne(x => x.ImatrixDefinition).WithMany().HasForeignKey(x => x.ImatrixDefinitionId).OnDelete(DeleteBehavior.Restrict);
        builder.HasMany(x => x.GroupStates).WithOne(x => x.Rule).HasForeignKey(x => x.RuleId).OnDelete(DeleteBehavior.Cascade);
    }
}

public class AnomalyInteractionRuleGroupState : ISQLiteEntity<AnomalyInteractionRuleGroupState>
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public Guid RuleId { get; set; }
    public AnomalyInteractionRule Rule { get; set; } = default!;
    public byte TensorGroupId { get; set; }
    public byte CandidateQuantId { get; set; }
    public byte ReferenceQuantId { get; set; }
    public string Movement { get; set; } = string.Empty;
    public int SortOrder { get; set; }

    public void Configure(EntityTypeBuilder<AnomalyInteractionRuleGroupState> builder)
    {
        builder.HasKey(x => x.Id);
        builder.Property(x => x.Id).ValueGeneratedNever();
        builder.Property(x => x.Movement).HasMaxLength(64);
        builder.HasIndex(x => new { x.RuleId, x.TensorGroupId }).IsUnique();
        builder.HasOne(x => x.Rule).WithMany(x => x.GroupStates).HasForeignKey(x => x.RuleId).OnDelete(DeleteBehavior.Cascade);
    }
}