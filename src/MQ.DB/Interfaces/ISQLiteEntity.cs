using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace MQ.DB.Interfaces;
using Microsoft.EntityFrameworkCore;

internal interface ISQLiteEntity<T> : IEntityTypeConfiguration<T>
    where T : class
{
    
}