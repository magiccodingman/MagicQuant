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
                CanonicalKey = x.CanonicalKey,
                BaselineName = x.Names[0],
                QuantizeBaseArgumentName = x.QuantizeBaseArgumentName,
                DefaultTensorSchemeId = x.DefaultTensorScheme!.UniqueId,
                DefaultTensorSchemeName = x.DefaultTensorScheme.Names[0],
                SourceKind = x.SourceKind,
                SourceOwner = x.SourceOwner,
                SourceRepository = x.SourceRepository,
                SourceFileName = x.SourceFileName,
                ShortSourceName = x.ShortSourceName,
                IsCustomBaseline = x.IsCustomBaseline,
                IsLearningBaseline = x.IsLearningBaseline,
                IsCombinationCarrierCandidate = x.IsCombinationCarrierCandidate,
                IsExplicitGroupCombinationCandidate = x.IsExplicitGroupCombinationCandidate,
                RequiresImatrix = x.RequiresImatrix,
                ExplicitCandidateSortOrder = x.ExplicitCandidateSortOrder
            })
            .OrderBy(x => x.BaselineQuantId)
            .ToList();

        var current = BaselineQuantDefinitions
            .ToList();

        if (current.Count == 0)
        {
            BaselineQuantDefinitions.AddRange(expected);
            SaveChanges();
            return;
        }

        var currentByCanonicalKey = current
            .Where(x => !string.IsNullOrWhiteSpace(x.CanonicalKey))
            .GroupBy(x => x.CanonicalKey, StringComparer.Ordinal)
            .ToDictionary(
                g => g.Key,
                g => g.OrderBy(x => x.BaselineQuantId).First(),
                StringComparer.Ordinal);

        var currentById = current.ToDictionary(x => x.BaselineQuantId);
        var changed = false;

        foreach (var expectedRow in expected)
        {
            BaselineQuantDefinition? target = null;

            if (!string.IsNullOrWhiteSpace(expectedRow.CanonicalKey) &&
                currentByCanonicalKey.TryGetValue(expectedRow.CanonicalKey, out var byCanonicalKey))
            {
                target = byCanonicalKey;
            }
            else if (currentById.TryGetValue(expectedRow.BaselineQuantId, out var byId))
            {
                target = byId;
            }

            if (target == null)
            {
                BaselineQuantDefinitions.Add(expectedRow);
                changed = true;
                continue;
            }

            if (!BaselineDefinitionEquals(target, expectedRow))
            {
                ApplyBaselineDefinitionUpdate(target, expectedRow);
                changed = true;
            }
        }

        if (changed)
            SaveChanges();
    }

    private static bool BaselineDefinitionEquals(BaselineQuantDefinition a, BaselineQuantDefinition b)
    {
        return a.BaselineQuantId == b.BaselineQuantId &&
               a.DefaultTensorSchemeId == b.DefaultTensorSchemeId &&
               a.IsCustomBaseline == b.IsCustomBaseline &&
               a.IsLearningBaseline == b.IsLearningBaseline &&
               a.IsCombinationCarrierCandidate == b.IsCombinationCarrierCandidate &&
               a.IsExplicitGroupCombinationCandidate == b.IsExplicitGroupCombinationCandidate &&
               a.RequiresImatrix == b.RequiresImatrix &&
               a.ExplicitCandidateSortOrder == b.ExplicitCandidateSortOrder &&
               string.Equals(a.CanonicalKey, b.CanonicalKey, StringComparison.Ordinal) &&
               string.Equals(a.BaselineName, b.BaselineName, StringComparison.Ordinal) &&
               string.Equals(a.QuantizeBaseArgumentName, b.QuantizeBaseArgumentName, StringComparison.Ordinal) &&
               string.Equals(a.DefaultTensorSchemeName, b.DefaultTensorSchemeName, StringComparison.Ordinal) &&
               string.Equals(a.SourceKind, b.SourceKind, StringComparison.Ordinal) &&
               string.Equals(a.SourceOwner, b.SourceOwner, StringComparison.Ordinal) &&
               string.Equals(a.SourceRepository, b.SourceRepository, StringComparison.Ordinal) &&
               string.Equals(a.SourceFileName, b.SourceFileName, StringComparison.Ordinal) &&
               string.Equals(a.ShortSourceName, b.ShortSourceName, StringComparison.Ordinal);
    }

    private static void ApplyBaselineDefinitionUpdate(BaselineQuantDefinition target, BaselineQuantDefinition source)
    {
        target.CanonicalKey = source.CanonicalKey;
        target.BaselineName = source.BaselineName;
        target.QuantizeBaseArgumentName = source.QuantizeBaseArgumentName;
        target.DefaultTensorSchemeId = source.DefaultTensorSchemeId;
        target.DefaultTensorSchemeName = source.DefaultTensorSchemeName;
        target.SourceKind = source.SourceKind;
        target.SourceOwner = source.SourceOwner;
        target.SourceRepository = source.SourceRepository;
        target.SourceFileName = source.SourceFileName;
        target.ShortSourceName = source.ShortSourceName;
        target.IsCustomBaseline = source.IsCustomBaseline;
        target.IsLearningBaseline = source.IsLearningBaseline;
        target.IsCombinationCarrierCandidate = source.IsCombinationCarrierCandidate;
        target.IsExplicitGroupCombinationCandidate = source.IsExplicitGroupCombinationCandidate;
        target.RequiresImatrix = source.RequiresImatrix;
        target.ExplicitCandidateSortOrder = source.ExplicitCandidateSortOrder;
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