using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class BaselineQuantDefinition : ISQLiteEntity<BaselineQuantDefinition>
{
    public byte BaselineQuantId { get; set; }
    public string BaselineName { get; set; } = string.Empty;
    public byte DefaultTensorSchemeId { get; set; }
    public string DefaultTensorSchemeName { get; set; } = string.Empty;

    public void Configure(EntityTypeBuilder<BaselineQuantDefinition> builder)
    {
        builder.HasKey(x => x.BaselineQuantId);

        builder.Property(x => x.BaselineName)
            .HasMaxLength(64)
            .IsRequired();

        builder.Property(x => x.DefaultTensorSchemeName)
            .HasMaxLength(64)
            .IsRequired();

        builder.HasIndex(x => x.BaselineName).IsUnique();
        builder.HasIndex(x => x.DefaultTensorSchemeId).IsUnique();
        builder.HasIndex(x => x.DefaultTensorSchemeName).IsUnique();
    }
}
