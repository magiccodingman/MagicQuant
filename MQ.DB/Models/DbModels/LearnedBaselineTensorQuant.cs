using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class LearnedBaselineTensorQuant : ISQLiteEntity<LearnedBaselineTensorQuant>
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid AiBenchmarkId { get; set; }
    public AiBenchmark AiBenchmark { get; set; } = default!;

    public uint AiModelHashId { get; set; }
    public AiModelHash AiModelHash { get; set; } = default!;

    public byte BaselineQuantId { get; set; }
    public byte TensorWeightSchemeId { get; set; }
    public byte TensorGroupId { get; set; }

    public string TensorName { get; set; } = string.Empty;
    public string FinalQuantType { get; set; } = string.Empty;

    public void Configure(EntityTypeBuilder<LearnedBaselineTensorQuant> builder)
    {
        builder.HasKey(x => x.Id);

        builder.Property(x => x.TensorName)
            .HasMaxLength(512)
            .IsRequired();

        builder.Property(x => x.FinalQuantType)
            .HasMaxLength(32)
            .IsRequired();

        builder.HasIndex(x => new
            {
                x.AiModelHashId,
                x.BaselineQuantId,
                x.TensorWeightSchemeId,
                x.TensorName
            })
            .IsUnique();

        builder.HasIndex(x => new
            {
                x.AiModelHashId,
                x.BaselineQuantId,
                x.TensorWeightSchemeId,
                x.TensorGroupId
            });

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
