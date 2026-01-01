using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Sqlite;
using System.IO;
using System.Reflection;
using MQ.DB.Interfaces;

namespace MQ.DB.Data;

public class AppDbContext : DbContext
{
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        var directory = Cache.MagicQuantDirectory;
        var dbPath = Path.Combine(directory, "MagicQuant_SQLite.db");
        optionsBuilder.UseSqlite($"Data Source={dbPath};Foreign Keys=True;");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // Fast and automatic EF config loading
        modelBuilder.ApplyConfigurationsFromAssembly(typeof(AppDbContext).Assembly);

        // Manual check to ensure all DbSet<T> have IAutoEntityTypeConfiguration<T>
        var dbSetTypes = this.GetType()
            .GetProperties()
            .Where(p => p.PropertyType.IsGenericType &&
                        p.PropertyType.GetGenericTypeDefinition() == typeof(DbSet<>))
            .Select(p => p.PropertyType.GetGenericArguments()[0])
            .ToList();

        var configuredTypes = Assembly.GetExecutingAssembly()
            .GetTypes()
            .Where(t => !t.IsInterface && !t.IsAbstract)
            .SelectMany(t =>
                t.GetInterfaces()
                    .Where(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(ISQLiteEntity<>))
                    .Select(i => i.GetGenericArguments()[0])
            ).ToHashSet();

        foreach (var dbSetType in dbSetTypes)
        {
            if (!configuredTypes.Contains(dbSetType))
            {
                throw new InvalidOperationException(
                    $"DbSet<{dbSetType.Name}> is declared but does not implement IAutoEntityTypeConfiguration<{dbSetType.Name}>."
                );
            }
        }

        base.OnModelCreating(modelBuilder);
    }
}