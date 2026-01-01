using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

/// <summary>
/// The created blake3 hash ID associated to a proper.
/// Additionally this has a cascading delete effect.
/// If one of these rows is ever deleted, all other table
/// rows that reference this Id as a foreign key must be deleted
/// alongside this and for it to occur safely.
/// </summary>
public class AiModelHash : ISQLiteEntity<AiModelHash>
{
    public uint Id { get; set; }
    public string UniqueHash { get; set; }
    
    public void Configure(EntityTypeBuilder<AiModelHash> builder)
    {
        builder.HasKey(x => x.Id);
        builder.HasIndex(h => h.UniqueHash);
    }
}