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
        EnsureBaselineQuantDefinitions();
    }

    private void EnsureBaselineQuantDefinitions()
    {
        var expected = BaselineQuants.All
            .Select(x => new BaselineQuantDefinition
            {
                BaselineQuantId = x.UniqueId,
                BaselineName = x.Names[0],
                DefaultTensorSchemeId = x.DefaultTensorScheme!.UniqueId,
                DefaultTensorSchemeName = x.DefaultTensorScheme.Names[0]
            })
            .OrderBy(x => x.BaselineQuantId)
            .ToList();

        var current = BaselineQuantDefinitions
            .AsNoTracking()
            .OrderBy(x => x.BaselineQuantId)
            .ToList();

        if (current.Count == 0)
        {
            BaselineQuantDefinitions.AddRange(expected);
            SaveChanges();
            return;
        }

        var mismatch = current.Count != expected.Count ||
                       current.Zip(expected, (a, b) =>
                           a.BaselineQuantId == b.BaselineQuantId &&
                           a.DefaultTensorSchemeId == b.DefaultTensorSchemeId &&
                           string.Equals(a.BaselineName, b.BaselineName, StringComparison.Ordinal) &&
                           string.Equals(a.DefaultTensorSchemeName, b.DefaultTensorSchemeName, StringComparison.Ordinal))
                           .Any(equal => !equal);

        if (mismatch)
        {
            throw new InvalidOperationException(
                "BaselineQuantDefinitions table is out of sync with code-defined BaselineQuants/DefaultTensorScheme mappings. " +
                "Run migrations and regenerate the DB definitions.");
        }
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
    public DbSet<LearnedBaselineTensorQuant> LearnedBaselineTensorQuants { get; set; }
    public DbSet<BaselineQuantDefinition> BaselineQuantDefinitions { get; set; }
    public DbSet<ExecutionPlanProbeCache> ExecutionPlanProbeCaches { get; set; }
    public DbSet<ImatrixDefinition> ImatrixDefinitions { get; set; }

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
