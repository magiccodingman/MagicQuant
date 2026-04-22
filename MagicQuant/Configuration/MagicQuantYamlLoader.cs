using System.Diagnostics;
using System.Text.Json;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Configuration;

public static class MagicQuantYamlLoader
{
    public static MagicQuantYamlConfig LoadAndApply(string commandName, IReadOnlyList<CliArg> args)
    {
        string configPath = ResolveConfigPath(args);
        Cache.ActiveConfigPath = configPath;

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                $"MagicQuant config file was not found at '{configPath}'. " +
                "Ensure config.default.yaml or config.dev.yaml is copied next to the build output, or pass --config.");
        }

        var deserializer = new DeserializerBuilder()
            .IgnoreUnmatchedProperties()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build();

        var yaml = File.ReadAllText(configPath);
        var loaded = deserializer.Deserialize<MagicQuantYamlConfig>(yaml) ?? MagicQuantYamlConfig.CreateDefault();
      
        ApplyCliOverrides(loaded, args);
        NormalizeAndApply(loaded);

        Config.Load(loaded);

        AnsiConsole.MarkupLine($"[grey]Using config:[/] {Markup.Escape(configPath)}");
        return loaded;
    }

    public static string ResolveConfigPath(IReadOnlyList<CliArg> args)
    {
        string? explicitPath = args.FirstOrDefault(a => string.Equals(a.Name, "config", StringComparison.OrdinalIgnoreCase))?.Value;
        if (!string.IsNullOrWhiteSpace(explicitPath))
            return Path.GetFullPath(explicitPath);

#if DEBUG
        string preferred = Path.Combine(AppContext.BaseDirectory, "config.dev.yaml");
        if (File.Exists(preferred))
            return preferred;
#endif
        return Path.Combine(AppContext.BaseDirectory, "config.default.yaml");
    }

    private static void NormalizeAndApply(MagicQuantYamlConfig config)
    {
        config.Paths.MagicQuantRoot = ResolveMagicQuantRoot(config.Paths.MagicQuantRoot);
        Cache.MagicQuantDirectory = config.Paths.MagicQuantRoot;
        Cache.LlamaRoot = NormalizeNullOrFullPath(config.Paths.LlamaRoot);
        Cache.LlamaBin = NormalizeNullOrFullPath(config.Paths.LlamaBin);
        Cache.ConvertScript = NormalizeNullOrFullPath(config.Paths.ConvertScript);
        Cache.ExternalBaselineCacheDirectory = Path.Combine(
            config.Paths.MagicQuantRoot!,
            string.IsNullOrWhiteSpace(config.Paths.ExternalBaselineCacheDirName) ? "ExternalBaselines" : config.Paths.ExternalBaselineCacheDirName);

        Directory.CreateDirectory(Cache.MagicQuantDirectory!);
        Directory.CreateDirectory(Cache.ExternalBaselineCacheDirectory!);

        Cache.UseImatrix = config.Flags.UseImatrix;
        Cache.ForceImatrixRebuild = config.Flags.ForceImatrixRebuild;
        Cache.ForceRelearnBaselineTensorMappings = config.Flags.ForceRelearnBaselineTensorMappings;
        Cache.ForceRefreshHardwareProbe = config.Flags.ForceRefreshHardwareProbe;

        RuntimeSearchSpace.AllowHighPrecisionHybrids = config.Flags.AllowHighPrecisionHybrids;
        Cache.CurrentArchitectureFamilyName =
            config.Identity.ArchitectureFamilyName?.Trim() ?? string.Empty;

        Cache.AllowArchitectureFamilyAliasOverride =
            config.Identity.AllowArchitectureFamilyAliasOverride;
        
        Cache.CurrentArchitectureFamilyId = null;

        ApplyStandardBaselineFilters(config.Baselines);
        BaselineQuants.ResetDynamicCustomBaselines();
    }

    private static void ApplyStandardBaselineFilters(RuntimeBaselineConfig baselineConfig)
    {
        string mode = (baselineConfig.StandardBaselinesMode ?? "all").Trim().ToLowerInvariant();

        HashSet<byte>? learning = null;
        HashSet<byte>? carriers = null;
        HashSet<byte>? explicitCandidates = null;

        if (mode == "none")
        {
            learning = new HashSet<byte>();
            carriers = new HashSet<byte>();
            explicitCandidates = new HashSet<byte>();
        }
        else if (mode == "selected")
        {
            learning = ResolveStandardBaselineIds(baselineConfig.EnabledStandardLearningBaselines);
            carriers = ResolveStandardBaselineIds(baselineConfig.EnabledStandardCombinationCarriers);
            explicitCandidates = ResolveStandardBaselineIds(baselineConfig.EnabledStandardExplicitGroupCandidates);
        }

        BaselineQuants.ConfigureStandardRoleFilters(learning, carriers, explicitCandidates);
    }

    private static HashSet<byte> ResolveStandardBaselineIds(IEnumerable<string> names)
    {
        var result = new HashSet<byte>();

        foreach (var raw in names ?? Array.Empty<string>())
        {
            if (string.IsNullOrWhiteSpace(raw))
                continue;

            var baseline = BaselineQuants.ResolveBuiltInStandardBaseline(raw.Trim());
            if (baseline == null)
            {
                throw new InvalidOperationException(
                    $"Unknown built-in baseline '{raw}'. " +
                    $"Known values: {string.Join(", ", BaselineQuants.GetBuiltInStandardBaselines().Select(x => x.Names[0]))}");
            }

            result.Add(baseline.UniqueId);
        }

        return result;
    }

    private static void ApplyCliOverrides(MagicQuantYamlConfig config, IReadOnlyList<CliArg> args)
    {
        string? Get(string name) => args.FirstOrDefault(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase))?.Value;
        bool Has(string name) => args.Any(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase));

        config.Paths.ModelDir = Prefer(Get("model-dir"), config.Paths.ModelDir);
        config.Paths.LlamaRoot = Prefer(Get("llama-root"), config.Paths.LlamaRoot);
        config.Paths.LlamaBin = Prefer(Get("llama-bin"), config.Paths.LlamaBin);
        config.Paths.ConvertScript = Prefer(Get("convert-script"), config.Paths.ConvertScript);

        if (Has("use-imatrix")) config.Flags.UseImatrix = true;
        if (Has("imatrix-force-rebuild")) config.Flags.ForceImatrixRebuild = true;
        if (Has("relearn-baseline-mappings")) config.Flags.ForceRelearnBaselineTensorMappings = true;
        if (Has("recheck-hardware-probe")) config.Flags.ForceRefreshHardwareProbe = true;
        if (Has("allow-high-precision-hybrids")) config.Flags.AllowHighPrecisionHybrids = true;

        config.Imatrix.ImatrixUrl = Prefer(Get("imatrix-url"), config.Imatrix.ImatrixUrl);
        config.Imatrix.DatasetRepo = Prefer(Get("imatrix-dataset-repo"), config.Imatrix.DatasetRepo);
        config.Imatrix.DatasetSplit = Prefer(Get("imatrix-dataset-split"), config.Imatrix.DatasetSplit);
        config.Imatrix.DatasetConfig = Prefer(Get("imatrix-dataset-config"), config.Imatrix.DatasetConfig);
        config.Imatrix.DatasetLocalFile = Prefer(Get("imatrix-dataset-local-file"), config.Imatrix.DatasetLocalFile);

        if (int.TryParse(Get("brute-force-final-combination-threshold"), out var bruteForceThreshold) && bruteForceThreshold > 0)
            config.Evolution.BruteForceFinalCombinationThreshold = bruteForceThreshold;

        if (ulong.TryParse(Get("manual-max-predicted-size-bytes"), out var manualBytes))
            config.Prediction.ManualMaxPredictedSizeBytes = manualBytes;

        config.Identity.ArchitectureFamilyName = Prefer(Get("architecture-family"), config.Identity.ArchitectureFamilyName);
        if (Has("allow-architecture-family-alias-override")) config.Identity.AllowArchitectureFamilyAliasOverride = true;
    }

    private static string? Prefer(string? preferred, string? fallback)
        => string.IsNullOrWhiteSpace(preferred) ? fallback : preferred;

    private static string ResolveMagicQuantRoot(string? configured)
    {
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.GetFullPath(configured);

        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), MagicConstants.MagicQuantFolder);
    }

    private static string? NormalizeNullOrFullPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return Path.GetFullPath(value);
    }
}