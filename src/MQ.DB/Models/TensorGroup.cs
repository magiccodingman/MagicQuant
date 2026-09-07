using System.Collections.Immutable;
using System.Text.RegularExpressions;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MQ.DB.Models;

public class TensorGroupInfo
{
    public TensorGroup Group { get; set; } = null!;
}

/// <summary>
/// Represents a categorized group of tensors with a unique name and matching patterns.
/// 
/// Important:
/// The group identity itself is intentionally owned by C#.
/// The regex patterns are loaded from tensor_groups.yaml.
/// 
/// This keeps MagicQuant's benchmark identity stable while allowing tensor-name
/// matching rules to evolve without recompiling the application.
/// </summary>
public record TensorGroup(byte UniqueId, string Name, ImmutableArray<string> Tensors)
{
    /// <summary>
    /// Helper to map the group name to a single-character identifier for CLI or UI display.
    /// </summary>
    public char ShortCode => Name switch
    {
        "embeddings" => 'E',
        "lm_head" => 'H',
        "attn_q" => 'Q',
        "attn_kv" => 'K',
        "attn_output" => 'O',
        "ffn_up_gate" => 'U',
        "ffn_down" => 'D',
        "moe_experts" => 'X',
        "moe_router" => 'R',
        _ => '?'
    };
}

/// <summary>
/// Tensor Registry.
/// 
/// The semantic tensor groups are fixed here on purpose.
/// The matching regex patterns are loaded from tensor_groups.yaml and cached on first use.
/// 
/// Design rule:
/// MagicQuant should not silently invent tensor-group behavior at runtime.
/// New groups should be deliberate architecture/benchmark decisions.
/// Pattern changes, however, are config/schema-level changes and belong in YAML.
/// </summary>
public static class TReg
{
    private const string DefaultYamlFileName = "tensor_groups.yaml";

    private static readonly object CacheLock = new();

    private static TensorGroupYamlFile? _yamlCache;

    private static readonly Dictionary<string, TensorGroup> GroupCache =
        new(StringComparer.OrdinalIgnoreCase);

    private static readonly Dictionary<string, ImmutableArray<Regex>> GroupRegexCache =
        new(StringComparer.OrdinalIgnoreCase);

    private static ImmutableArray<string>? _baseQuantExceptionPatternsCache;

    private static ImmutableArray<Regex>? _baseQuantExceptionRegexCache;

    /// <summary>
    /// Future extension point.
    /// 
    /// Right now this can remain null and the loader will resolve tensor_groups.yaml
    /// from the application output directory.
    /// 
    /// Later, CLI/config code can set this before first access if users provide
    /// an override location.
    /// </summary>
    public static string? TensorGroupsYamlPathOverride { get; set; }

    public static TensorGroup Embeddings => GetRequiredGroup(0, "embeddings");

    public static TensorGroup LmHead => GetRequiredGroup(1, "lm_head");

    public static TensorGroup AttnQ => GetRequiredGroup(2, "attn_q");

    public static TensorGroup AttnKV => GetRequiredGroup(3, "attn_kv");

    public static TensorGroup AttnOutput => GetRequiredGroup(4, "attn_output");

    public static TensorGroup FfnUpGate => GetRequiredGroup(5, "ffn_up_gate");

    public static TensorGroup FfnDown => GetRequiredGroup(6, "ffn_down");

    public static TensorGroup MoeExperts => GetRequiredGroup(7, "moe_experts");

    public static TensorGroup MoeRouter => GetRequiredGroup(8, "moe_router");

    /// <summary>
    /// Provides a complete list of all registered tensor groups.
    /// 
    /// This remains fixed by design. The regex patterns inside each group come
    /// from tensor_groups.yaml.
    /// </summary>
    public static ImmutableArray<TensorGroup> All =>
    [
        Embeddings,
        LmHead,
        AttnQ,
        AttnKV,
        AttnOutput,
        FfnUpGate,
        FfnDown,
        MoeExperts,
        MoeRouter
    ];

    /// <summary>
    /// Look up a group by its string name, useful when parsing external configs.
    /// </summary>
    public static TensorGroup? GetByName(string name) =>
        All.FirstOrDefault(g => g.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Gets compiled regexes for a semantic tensor group.
    /// 
    /// This is useful when matching many tensors repeatedly and avoids recompiling
    /// the same patterns over and over.
    /// </summary>
    public static ImmutableArray<Regex> GetRegexesForGroup(TensorGroup group)
    {
        lock (CacheLock)
        {
            if (GroupRegexCache.TryGetValue(group.Name, out var cached))
                return cached;

            var regexes = group.Tensors
                .Select(CreateRegex)
                .ToImmutableArray();

            GroupRegexCache[group.Name] = regexes;
            return regexes;
        }
    }

    /// <summary>
    /// Gets regex patterns for tensors that are explicitly allowed to fall back to BaseQuant.
    /// 
    /// These are not semantic tensor groups. They are only checked after a tensor fails to
    /// match any registered semantic group.
    /// </summary>
    public static ImmutableArray<string> GetBaseQuantExceptionPatterns()
    {
        lock (CacheLock)
        {
            if (_baseQuantExceptionPatternsCache is not null)
                return _baseQuantExceptionPatternsCache.Value;

            var yaml = LoadYamlIfNeeded();

            var patterns = yaml.BaseQuantExceptions?.Patterns ?? [];

            var cleanedPatterns = patterns
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => p.Trim())
                .Distinct(StringComparer.Ordinal)
                .ToImmutableArray();

            _baseQuantExceptionPatternsCache = cleanedPatterns;
            return cleanedPatterns;
        }
    }

    /// <summary>
    /// Gets compiled regexes for tensors that are explicitly allowed to fall back to BaseQuant.
    /// </summary>
    public static ImmutableArray<Regex> GetBaseQuantExceptionRegexes()
    {
        lock (CacheLock)
        {
            if (_baseQuantExceptionRegexCache is not null)
                return _baseQuantExceptionRegexCache.Value;

            var regexes = GetBaseQuantExceptionPatterns()
                .Select(CreateRegex)
                .ToImmutableArray();

            _baseQuantExceptionRegexCache = regexes;
            return regexes;
        }
    }

    /// <summary>
    /// Returns true when the tensor is explicitly allowed to remain outside all semantic
    /// tensor groups and fall back to the artifact BaseQuant.
    /// 
    /// Important:
    /// This should only be called after normal group matching returns zero matches.
    /// It must not be used to resolve group collisions.
    /// </summary>
    public static bool IsBaseQuantException(string tensorName)
    {
        foreach (var regex in GetBaseQuantExceptionRegexes())
        {
            if (regex.IsMatch(tensorName))
                return true;
        }

        return false;
    }

    /// <summary>
    /// Finds all semantic tensor groups that match the provided tensor name.
    /// 
    /// If this returns:
    /// - 0 groups: caller may then check IsBaseQuantException.
    /// - 1 group: tensor is safely categorized.
    /// - 2+ groups: caller should treat this as an ambiguity/collision error.
    /// </summary>
    public static ImmutableArray<TensorGroup> FindMatchingGroups(string tensorName)
    {
        var matches = ImmutableArray.CreateBuilder<TensorGroup>();

        foreach (var group in All)
        {
            var regexes = GetRegexesForGroup(group);

            foreach (var regex in regexes)
            {
                if (!regex.IsMatch(tensorName))
                    continue;

                matches.Add(group);
                break;
            }
        }

        return matches.ToImmutable();
    }

    /// <summary>
    /// Clears the loaded YAML and materialized group/regex caches.
    /// 
    /// This is mainly useful for tests or future reload behavior.
    /// Normal production runs should not need to call this.
    /// </summary>
    public static void ClearCache()
    {
        lock (CacheLock)
        {
            _yamlCache = null;
            GroupCache.Clear();
            GroupRegexCache.Clear();
            _baseQuantExceptionPatternsCache = null;
            _baseQuantExceptionRegexCache = null;
        }
    }

    private static TensorGroup GetRequiredGroup(byte uniqueId, string name)
    {
        lock (CacheLock)
        {
            if (GroupCache.TryGetValue(name, out var cached))
                return cached;

            var yaml = LoadYamlIfNeeded();

            if (yaml.Groups is null || yaml.Groups.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Tensor group YAML did not define any groups. File: {ResolveTensorGroupsYamlPath()}");
            }

            if (!yaml.Groups.TryGetValue(name, out var groupDef))
            {
                throw new InvalidOperationException(
                    $"Required tensor group '{name}' was not found in {ResolveTensorGroupsYamlPath()}.");
            }

            if (groupDef.Patterns is null || groupDef.Patterns.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Tensor group '{name}' exists in {ResolveTensorGroupsYamlPath()}, but it has no patterns.");
            }

            var cleanedPatterns = groupDef.Patterns
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(p => p.Trim())
                .Distinct(StringComparer.Ordinal)
                .ToImmutableArray();

            if (cleanedPatterns.Length == 0)
            {
                throw new InvalidOperationException(
                    $"Tensor group '{name}' exists in {ResolveTensorGroupsYamlPath()}, but all patterns were empty.");
            }

            var group = new TensorGroup(uniqueId, name, cleanedPatterns);
            GroupCache[name] = group;

            return group;
        }
    }

    private static TensorGroupYamlFile LoadYamlIfNeeded()
    {
        if (_yamlCache is not null)
            return _yamlCache;

        var path = ResolveTensorGroupsYamlPath();

        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"Could not find {DefaultYamlFileName}. Expected it at: {path}",
                path);
        }

        var yamlText = File.ReadAllText(path);

        if (string.IsNullOrWhiteSpace(yamlText))
        {
            throw new InvalidOperationException(
                $"Tensor group YAML file is empty: {path}");
        }

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        TensorGroupYamlFile? parsed;

        try
        {
            parsed = deserializer.Deserialize<TensorGroupYamlFile>(yamlText);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to parse tensor group YAML file: {path}",
                ex);
        }

        if (parsed is null)
        {
            throw new InvalidOperationException(
                $"Tensor group YAML parsed to null: {path}");
        }

        if (parsed.SchemaVersion <= 0)
        {
            throw new InvalidOperationException(
                $"Tensor group YAML must define a positive schema_version. File: {path}");
        }

        _yamlCache = parsed;
        return _yamlCache;
    }

    private static string ResolveTensorGroupsYamlPath()
    {
        if (!string.IsNullOrWhiteSpace(TensorGroupsYamlPathOverride))
            return Path.GetFullPath(TensorGroupsYamlPathOverride);

        return Path.Combine(AppContext.BaseDirectory, DefaultYamlFileName);
    }

    private static Regex CreateRegex(string pattern)
    {
        try
        {
            return new Regex(
                pattern,
                RegexOptions.Compiled |
                RegexOptions.CultureInvariant);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Invalid tensor group regex pattern: {pattern}",
                ex);
        }
    }

    private sealed class TensorGroupYamlFile
    {
        public int SchemaVersion { get; set; }

        public Dictionary<string, TensorGroupYamlDefinition> Groups { get; set; } =
            new(StringComparer.OrdinalIgnoreCase);

        public BaseQuantExceptionYamlDefinition? BaseQuantExceptions { get; set; }
    }

    private sealed class TensorGroupYamlDefinition
    {
        public string? Description { get; set; }

        public List<string> Patterns { get; set; } = [];
    }

    private sealed class BaseQuantExceptionYamlDefinition
    {
        public string? Description { get; set; }

        public List<string> Patterns { get; set; } = [];
    }
}
