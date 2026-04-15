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
        EnsureInitialized();
    }

    public MagicQuantContext(DbContextOptions<MagicQuantContext> options)
        : base(options)
    {
        EnsureInitialized();
    }

    private void EnsureInitialized()
    {
        // 🚫 Never run during EF tooling (migrations, etc.)
        if (IsDesignTime())
            return;

        if (_isInitialized)
            return;

        lock (_initLock)
        {
            if (_isInitialized)
                return;

            InitializeDatabase();
            _isInitialized = true;
        }
    }

    private void InitializeDatabase()
    {
        var directory = Cache.MagicQuantDirectory;

        // Safety fallback
        if (string.IsNullOrEmpty(directory))
            directory = Directory.GetCurrentDirectory();

        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // 🔥 Apply migrations automatically
        Database.Migrate();
    }

    private static bool IsDesignTime()
    {
        return AppDomain.CurrentDomain.GetAssemblies()
            .Any(a => a.FullName != null &&
                      a.FullName.Contains("EntityFrameworkCore.Design", StringComparison.OrdinalIgnoreCase));
    }

    // --------------------------------------------------------
    // DbSets
    // --------------------------------------------------------

    public DbSet<AiBenchmark> AiBenchmarks { get; set; }
    public DbSet<AiModelHash> AiModelHashes { get; set; }
    public DbSet<TensorCombo> TensorCombos { get; set; }
    public DbSet<QuantizationRun> QuantizationRuns { get; set; }
    public DbSet<BenchmarkRun> BenchmarkRuns { get; set; }

    // --------------------------------------------------------
    // Configuration
    // --------------------------------------------------------

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        if (!optionsBuilder.IsConfigured)
        {
            var directory = Cache.MagicQuantDirectory;

            if (string.IsNullOrEmpty(directory))
                directory = Directory.GetCurrentDirectory();

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

    // --------------------------------------------------------
    // Strict Validation
    // --------------------------------------------------------

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