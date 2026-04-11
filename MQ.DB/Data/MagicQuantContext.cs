using Microsoft.EntityFrameworkCore;
using MQ.DB.Interfaces;
using MQ.DB.Models;
using MQ.DB.Models.DbModels;

namespace MQ.DB.Data;

public class MagicQuantContext : DbContext
{
    // --------------------------------------------------------
    // Self-Initialization Logic
    // --------------------------------------------------------
    private static bool _isInitialized = false;
    private static readonly object _initLock = new();

    public MagicQuantContext()
    {
        // On the very first instantiation (e.g., first benchmark run),
        // we ensure the folder exists and migrations are applied.
        if (!_isInitialized)
        {
            lock (_initLock)
            {
                if (!_isInitialized)
                {
                    InitializeDatabase();
                    _isInitialized = true;
                }
            }
        }
    }

    private void InitializeDatabase()
    {
        var directory = Cache.MagicQuantDirectory;
        
        // Safety: fallback if Cache isn't set yet (rare, but good for stability)
        if (string.IsNullOrEmpty(directory)) 
            directory = Directory.GetCurrentDirectory();

        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Apply Migrations automatically
        Database.Migrate();
    }

    // --------------------------------------------------------
    // Standard DbContext Configuration
    // --------------------------------------------------------

    public MagicQuantContext(DbContextOptions<MagicQuantContext> options) : base(options) { }

    public DbSet<AiBenchmark> AiBenchmarks { get; set; }
    public DbSet<AiModelHash> AiModelHashes { get; set; }
    public DbSet<TensorCombo> TensorCombos { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        if (!optionsBuilder.IsConfigured)
        {
            var directory = Cache.MagicQuantDirectory;
            if (string.IsNullOrEmpty(directory))
            {
                directory = Directory.GetCurrentDirectory();
            }

            var dbPath = Path.Combine(directory, "MagicQuant_SQLite.db");
            optionsBuilder.UseSqlite($"Data Source={dbPath};Foreign Keys=True;");
        }
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfigurationsFromAssembly(typeof(MagicQuantContext).Assembly);
        ValidateDbSetsImplementInterface();
        base.OnModelCreating(modelBuilder);
    }

    private void ValidateDbSetsImplementInterface()
    {
        var dbSetGenericTypes = this.GetType()
            .GetProperties()
            .Where(p => p.PropertyType.IsGenericType &&
                        p.PropertyType.GetGenericTypeDefinition() == typeof(DbSet<>))
            .Select(p => p.PropertyType.GetGenericArguments()[0])
            .ToHashSet();

        var configuredTypes = typeof(MagicQuantContext).Assembly
            .GetTypes()
            .Where(t => t.GetInterfaces().Any(i =>
                i.IsGenericType && i.GetGenericTypeDefinition() == typeof(ISQLiteEntity<>)))
            .ToHashSet();

        var missingConfigs = dbSetGenericTypes.Except(configuredTypes).ToList();

        if (missingConfigs.Any())
        {
            var names = string.Join(", ", missingConfigs.Select(t => t.Name));
            throw new InvalidOperationException(
                $"STRICT MODE ERROR: The following DbSets do not implement ISQLiteEntity<T>: [{names}]. "
            );
        }
    }
}