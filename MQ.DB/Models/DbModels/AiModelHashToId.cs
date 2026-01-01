using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

/// <summary>
/// The created blake3 hash ID associated to a proper 
/// </summary>
public class AiModelHashToId : ISQLiteEntity<AiModelHashToId>
{
    public int Id { get; set; }
    public string UniqueHash { get; set; }
    
    public void Configure(EntityTypeBuilder<AiModelHashToId> builder)
    {
        builder.HasKey(x => x.Id);
        builder.HasIndex(h => h.UniqueHash);
    }
}