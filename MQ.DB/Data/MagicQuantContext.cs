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
    var now = DateTime.UtcNow;
    var expected = BaselineQuants.GetBuiltInStandardBaselines()
        .Concat(BaselineQuants.GetExactHighPrecisionAliases(allowHighPrecisionHybrids: true))
        .Append(BaselineQuants.GetNativeQuant())
        .Select(x => new BaselineQuantDefinition
        {
            ArchitectureFamilyId = null,
            RuntimeBaselineId = x.UniqueId,
            CanonicalKey = x.CanonicalKey,
            NormalizedCanonicalKey = NormalizeKey(x.CanonicalKey),
            BaselineName = x.Names[0],
            DisplayName = x.Names[0],
            QuantizeBaseArgumentName = x.QuantizeBaseArgumentName,
            DefaultTensorSchemeId = x.DefaultTensorScheme!.UniqueId,
            DefaultTensorSchemeName = x.DefaultTensorScheme.Names[0],
            SourceKind = x.SourceKind,
            SourceOwner = x.SourceOwner,
            SourceRepository = x.SourceRepository,
            NormalizedSourceRepository = NormalizeNullable(x.SourceRepository),
            SourceFileName = x.SourceFileName,
            NormalizedSourceFileName = NormalizeFileNullable(x.SourceFileName),
            ShortSourceName = x.ShortSourceName,
            BaselineFamily = x.Names[0],
            IsCustomBaseline = x.IsCustomBaseline,
            IsLearningBaseline = x.IsLearningBaseline,
            IsCombinationCarrierCandidate = x.IsCombinationCarrierCandidate,
            IsExplicitGroupCombinationCandidate = x.IsExplicitGroupCombinationCandidate,
            RequiresImatrix = x.RequiresImatrix,
            BitRange = x.BitRange,
            ExplicitCandidateSortOrder = x.ExplicitCandidateSortOrder,
            IsActiveInCurrentConfig = true,
            FirstSeenUtc = now,
            LastSeenUtc = now,
            LastUpdatedUtc = now
        })
        .OrderBy(x => x.RuntimeBaselineId)
        .ToList();

    var current = BaselineQuantDefinitions
        .Where(x => x.ArchitectureFamilyId == null)
        .ToList();

    var currentByCanonicalKey = current
        .Where(x => !string.IsNullOrWhiteSpace(x.NormalizedCanonicalKey))
        .GroupBy(x => x.NormalizedCanonicalKey, StringComparer.Ordinal)
        .ToDictionary(g => g.Key, g => g.OrderBy(x => x.RuntimeBaselineId).First(), StringComparer.Ordinal);

    var currentByRuntimeId = current.ToDictionary(x => x.RuntimeBaselineId);
    var changed = false;

    foreach (var expectedRow in expected)
    {
        BaselineQuantDefinition? target = null;

        if (!string.IsNullOrWhiteSpace(expectedRow.NormalizedCanonicalKey) &&
            currentByCanonicalKey.TryGetValue(expectedRow.NormalizedCanonicalKey, out var byCanonicalKey))
        {
            target = byCanonicalKey;
        }
        else if (currentByRuntimeId.TryGetValue(expectedRow.RuntimeBaselineId, out var byId))
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
            ApplyBaselineDefinitionUpdate(target, expectedRow, preserveFirstSeen: true);
            target.LastUpdatedUtc = now;
            changed = true;
        }

        target.IsActiveInCurrentConfig = true;
        target.LastSeenUtc = now;
    }

    if (changed)
        SaveChanges();
}

private static bool BaselineDefinitionEquals(BaselineQuantDefinition a, BaselineQuantDefinition b)
{
    return a.ArchitectureFamilyId == b.ArchitectureFamilyId &&
           a.RuntimeBaselineId == b.RuntimeBaselineId &&
           a.DefaultTensorSchemeId == b.DefaultTensorSchemeId &&
           a.IsCustomBaseline == b.IsCustomBaseline &&
           a.IsLearningBaseline == b.IsLearningBaseline &&
           a.IsCombinationCarrierCandidate == b.IsCombinationCarrierCandidate &&
           a.IsExplicitGroupCombinationCandidate == b.IsExplicitGroupCombinationCandidate &&
           a.RequiresImatrix == b.RequiresImatrix &&
           a.BitRange == b.BitRange &&
           a.ExplicitCandidateSortOrder == b.ExplicitCandidateSortOrder &&
           a.IsActiveInCurrentConfig == b.IsActiveInCurrentConfig &&
           string.Equals(a.CanonicalKey, b.CanonicalKey, StringComparison.Ordinal) &&
           string.Equals(a.NormalizedCanonicalKey, b.NormalizedCanonicalKey, StringComparison.Ordinal) &&
           string.Equals(a.BaselineName, b.BaselineName, StringComparison.Ordinal) &&
           string.Equals(a.DisplayName, b.DisplayName, StringComparison.Ordinal) &&
           string.Equals(a.QuantizeBaseArgumentName, b.QuantizeBaseArgumentName, StringComparison.Ordinal) &&
           string.Equals(a.DefaultTensorSchemeName, b.DefaultTensorSchemeName, StringComparison.Ordinal) &&
           string.Equals(a.SourceKind, b.SourceKind, StringComparison.Ordinal) &&
           string.Equals(a.SourceOwner, b.SourceOwner, StringComparison.Ordinal) &&
           string.Equals(a.SourceRepository, b.SourceRepository, StringComparison.Ordinal) &&
           string.Equals(a.NormalizedSourceRepository, b.NormalizedSourceRepository, StringComparison.Ordinal) &&
           string.Equals(a.SourceFileName, b.SourceFileName, StringComparison.Ordinal) &&
           string.Equals(a.NormalizedSourceFileName, b.NormalizedSourceFileName, StringComparison.Ordinal) &&
           string.Equals(a.ShortSourceName, b.ShortSourceName, StringComparison.Ordinal) &&
           string.Equals(a.BaselineFamily, b.BaselineFamily, StringComparison.Ordinal);
}

public static void ApplyBaselineDefinitionUpdate(BaselineQuantDefinition target, BaselineQuantDefinition source, bool preserveFirstSeen = true)
{
    var firstSeen = target.FirstSeenUtc;
    target.ArchitectureFamilyId = source.ArchitectureFamilyId;
    target.RuntimeBaselineId = source.RuntimeBaselineId;
    target.CanonicalKey = source.CanonicalKey;
    target.NormalizedCanonicalKey = source.NormalizedCanonicalKey;
    target.BaselineName = source.BaselineName;
    target.DisplayName = source.DisplayName;
    target.QuantizeBaseArgumentName = source.QuantizeBaseArgumentName;
    target.DefaultTensorSchemeId = source.DefaultTensorSchemeId;
    target.DefaultTensorSchemeName = source.DefaultTensorSchemeName;
    target.SourceKind = source.SourceKind;
    target.SourceOwner = source.SourceOwner;
    target.SourceRepository = source.SourceRepository;
    target.NormalizedSourceRepository = source.NormalizedSourceRepository;
    target.SourceFileName = source.SourceFileName;
    target.NormalizedSourceFileName = source.NormalizedSourceFileName;
    target.ShortSourceName = source.ShortSourceName;
    target.BaselineFamily = source.BaselineFamily;
    target.IsCustomBaseline = source.IsCustomBaseline;
    target.IsLearningBaseline = source.IsLearningBaseline;
    target.IsCombinationCarrierCandidate = source.IsCombinationCarrierCandidate;
    target.IsExplicitGroupCombinationCandidate = source.IsExplicitGroupCombinationCandidate;
    target.RequiresImatrix = source.RequiresImatrix;
    target.BitRange = source.BitRange;
    target.ExplicitCandidateSortOrder = source.ExplicitCandidateSortOrder;
    target.IsActiveInCurrentConfig = source.IsActiveInCurrentConfig;
    target.LastSeenUtc = source.LastSeenUtc;
    target.LastUpdatedUtc = source.LastUpdatedUtc;
    if (!preserveFirstSeen)
        target.FirstSeenUtc = source.FirstSeenUtc;
    else if (firstSeen != default)
        target.FirstSeenUtc = firstSeen;
}

private static string NormalizeKey(string value) => (value ?? string.Empty).Trim().ToLowerInvariant();
private static string? NormalizeNullable(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim().ToLowerInvariant();
private static string? NormalizeFileNullable(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim().Replace('\\', '/').ToLowerInvariant();

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
    public DbSet<TensorGroupProfile> TensorGroupProfiles { get; set; }
    public DbSet<AiBenchmarkLearnedSource> AiBenchmarkLearnedSources { get; set; }
    public DbSet<ExecutionPlanProbeCache> ExecutionPlanProbeCaches { get; set; }
    public DbSet<ImatrixDefinition> ImatrixDefinitions { get; set; }
    public DbSet<ArchitectureFamily> ArchitectureFamilies { get; set; }
    public DbSet<ArchitectureFamilyModelHash> ArchitectureFamilyModelHashes { get; set; }
    public DbSet<AnomalyProbeSession> AnomalyProbeSessions { get; set; }
    public DbSet<AnomalyProbeObservation> AnomalyProbeObservations { get; set; }
    public DbSet<AnomalyInteractionRule> AnomalyInteractionRules { get; set; }
    public DbSet<AnomalyInteractionRuleGroupState> AnomalyInteractionRuleGroupStates { get; set; }


    // --------------------------------------------------------
    // Imatrix Ownership Guard
    // --------------------------------------------------------

    public override int SaveChanges()
    {
        ValidateImatrixOwnershipBeforeSaveAsync(CancellationToken.None).GetAwaiter().GetResult();
        return base.SaveChanges();
    }

    public override int SaveChanges(bool acceptAllChangesOnSuccess)
    {
        ValidateImatrixOwnershipBeforeSaveAsync(CancellationToken.None).GetAwaiter().GetResult();
        return base.SaveChanges(acceptAllChangesOnSuccess);
    }

    public override async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        await ValidateImatrixOwnershipBeforeSaveAsync(cancellationToken);
        return await base.SaveChangesAsync(cancellationToken);
    }

    public override async Task<int> SaveChangesAsync(bool acceptAllChangesOnSuccess, CancellationToken cancellationToken = default)
    {
        await ValidateImatrixOwnershipBeforeSaveAsync(cancellationToken);
        return await base.SaveChangesAsync(acceptAllChangesOnSuccess, cancellationToken);
    }

    private async Task ValidateImatrixOwnershipBeforeSaveAsync(CancellationToken ct)
    {
        var pairs = ChangeTracker.Entries()
            .Where(e => e.State is EntityState.Added or EntityState.Modified)
            .Select(e => e.Entity)
            .Select(entity => entity switch
            {
                AiBenchmark x => (EntityName: nameof(AiBenchmark), x.AiModelHashId, x.ImatrixDefinitionId),
                BenchmarkRun x => (EntityName: nameof(BenchmarkRun), x.AiModelHashId, x.ImatrixDefinitionId),
                QuantizationRun x => (EntityName: nameof(QuantizationRun), x.AiModelHashId, x.ImatrixDefinitionId),
                ExecutionPlanProbeCache x => (EntityName: nameof(ExecutionPlanProbeCache), x.AiModelHashId, x.ImatrixDefinitionId),
                AnomalyProbeSession x => (EntityName: nameof(AnomalyProbeSession), x.AiModelHashId, x.ImatrixDefinitionId),
                AnomalyProbeObservation x => (EntityName: nameof(AnomalyProbeObservation), x.AiModelHashId, x.ImatrixDefinitionId),
                AnomalyInteractionRule x => (EntityName: nameof(AnomalyInteractionRule), x.AiModelHashId, x.ImatrixDefinitionId),
                _ => default
            })
            .Where(x => !string.IsNullOrWhiteSpace(x.EntityName) && x.ImatrixDefinitionId.HasValue)
            .Distinct()
            .ToList();

        if (pairs.Count == 0)
            return;

        var ids = pairs
            .Select(x => x.ImatrixDefinitionId!.Value)
            .Distinct()
            .ToList();

        var owners = await ImatrixDefinitions
            .AsNoTracking()
            .Where(x => ids.Contains(x.Id))
            .Select(x => new { x.Id, x.AiModelHashId })
            .ToDictionaryAsync(x => x.Id, x => x.AiModelHashId, ct);

        foreach (var pair in pairs)
        {
            if (!owners.TryGetValue(pair.ImatrixDefinitionId!.Value, out var ownerHashId) ||
                ownerHashId != pair.AiModelHashId)
            {
                throw new InvalidOperationException(
                    $"{pair.EntityName} attempted to save AiModelHashId={pair.AiModelHashId} with " +
                    $"ImatrixDefinitionId={pair.ImatrixDefinitionId.Value}, but that imatrix belongs to " +
                    $"AiModelHashId={(owners.TryGetValue(pair.ImatrixDefinitionId.Value, out var found) ? found.ToString() : "missing")}. " +
                    "ImatrixDefinition ownership is exact-model-hash scoped.");
            }
        }
    }

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
