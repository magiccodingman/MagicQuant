using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Sqlite;
using System.IO;
using System.Reflection;
using MQ.DB.Interfaces;

namespace MQ.DB.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }
    
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        var directory = Cache.MagicQuantDirectory;
        var dbPath = Path.Combine(directory, "MagicQuant_SQLite.db");
        optionsBuilder.UseSqlite($"Data Source={dbPath};Foreign Keys=True;");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // Load all configs found in this assembly.
        // Pick up entities with ISQLiteEntity<T>
        modelBuilder.ApplyConfigurationsFromAssembly(typeof(AppDbContext).Assembly);

        // Ensure strict adherence to the pattern.
        ValidateDbSetsImplementInterface();

        base.OnModelCreating(modelBuilder);
    }

    private void ValidateDbSetsImplementInterface()
    {
        // Get all properties that are DbSet<T>
        var dbSetGenericTypes = this.GetType()
            .GetProperties()
            .Where(p => p.PropertyType.IsGenericType &&
                        p.PropertyType.GetGenericTypeDefinition() == typeof(DbSet<>))
            .Select(p => p.PropertyType.GetGenericArguments()[0])
            .ToHashSet();

        // Get all types in assembly that implement ISQLiteEntity<T>
        var configuredTypes = typeof(AppDbContext).Assembly
            .GetTypes()
            .Where(t => t.GetInterfaces().Any(i => 
                i.IsGenericType && i.GetGenericTypeDefinition() == typeof(ISQLiteEntity<>)))
            .ToHashSet();

        // Find the difference
        var missingConfigs = dbSetGenericTypes.Except(configuredTypes).ToList();

        if (missingConfigs.Any())
        {
            var names = string.Join(", ", missingConfigs.Select(t => t.Name));
            throw new InvalidOperationException(
                $"STRICT MODE ERROR: The following DbSets do not implement ISQLiteEntity<T>: [{names}]. " +
                "Please implement the interface to ensure configuration is centralized."
            );
        }
    }
}