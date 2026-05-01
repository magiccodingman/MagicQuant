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
        Cache.ScratchRoots = NormalizeScratchRoots(config.Paths.ScratchRoots);

        Directory.CreateDirectory(Cache.MagicQuantDirectory!);

        Cache.UseImatrix = config.Flags.UseImatrix;
        Cache.ForceImatrixRebuild = config.Flags.ForceImatrixRebuild;
        Cache.ForceRelearnBaselineTensorMappings = config.Flags.ForceRelearnBaselineTensorMappings;
        Cache.ForceRefreshHardwareProbe = config.Flags.ForceRefreshHardwareProbe;

        config.Hardware.GpuMemoryLimitsGb ??= new Dictionary<int, double>();
        config.Hardware.GpuMemoryLimitsGb = config.Hardware.GpuMemoryLimitsGb
            .Where(x => x.Key >= 0 && x.Value > 0d)
            .ToDictionary(x => x.Key, x => x.Value);

        Cache.GpuMemoryLimitsGb = config.Hardware.GpuMemoryLimitsGb
            .Where(x => x.Key >= 0 && x.Value > 0d)
            .ToDictionary(x => x.Key, x => x.Value);

        RuntimeSearchSpace.AllowHighPrecisionHybrids = config.Flags.AllowHighPrecisionHybrids;
        Cache.CurrentArchitectureFamilyName =
            config.Identity.ArchitectureFamilyName?.Trim() ?? string.Empty;

        Cache.AllowArchitectureFamilyAliasOverride =
            config.Identity.AllowArchitectureFamilyAliasOverride;
        
        Cache.CurrentArchitectureFamilyId = null;

        config.Output.OutputDir = string.IsNullOrWhiteSpace(config.Output.OutputDir)
            ? null
            : config.Output.OutputDir.Trim();

        config.Output.OutputNamePrefix = string.IsNullOrWhiteSpace(config.Output.OutputNamePrefix)
            ? "Model"
            : config.Output.OutputNamePrefix.Trim();

        config.Readme ??= new RuntimeReadmeConfig();

        config.Readme.TitleModelNameOverride = string.IsNullOrWhiteSpace(config.Readme.TitleModelNameOverride)
            ? null
            : config.Readme.TitleModelNameOverride.Trim();

        config.Readme.Frontmatter ??= new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        config.Readme.Frontmatter = config.Readme.Frontmatter
            .Where(x => !string.IsNullOrWhiteSpace(x.Key) && !IsEmptyFrontmatterValue(x.Value))
            .ToDictionary(x => x.Key.Trim(), x => x.Value, StringComparer.OrdinalIgnoreCase);

        if (config.Survival.MaxSelectedChoicesPerBucket <= 0)
            config.Survival.MaxSelectedChoicesPerBucket = 1;

        if (config.Prediction.BitStressThresholdCandidates.Count == 0)
            config.Prediction.BitStressThresholdCandidates.Add(config.Prediction.DefaultBitStressThreshold);

        config.Prediction.BitStressThresholdCandidates = config.Prediction.BitStressThresholdCandidates
            .Where(x => x > 0d)
            .Distinct()
            .OrderBy(x => x)
            .ToList();

        if (config.Prediction.MinimumFitRows < 2)
            config.Prediction.MinimumFitRows = 2;

        if (config.CandidateSelection.InteriorWindowFractions.Count == 0)
        {
            config.CandidateSelection.InteriorWindowFractions.Add(0.35d);
            config.CandidateSelection.InteriorWindowFractions.Add(0.35d);
        }

        config.CandidateSelection.InteriorWindowFractions = config.CandidateSelection.InteriorWindowFractions
            .Select(x => Math.Clamp(x, 0d, 1d))
            .Where(x => x > 0d)
            .ToList();

        config.CandidateSelection.MaxCandidatesPerInteriorWindow = Math.Max(1, config.CandidateSelection.MaxCandidatesPerInteriorWindow);
        config.CandidateSelection.MaxFallbackAttemptsPerAnchor = Math.Max(1, config.CandidateSelection.MaxFallbackAttemptsPerAnchor);
        config.CandidateSelection.NearBaselineMaxSizeGrowthPercent = Math.Max(0d, config.CandidateSelection.NearBaselineMaxSizeGrowthPercent);
        config.CandidateSelection.MinimumKldImprovementEpsilon = Math.Max(0d, config.CandidateSelection.MinimumKldImprovementEpsilon);

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
        if (Has("recheck-hardware-probe") || Has("force-refresh-hardware-probe") || Has("force_refresh_hardware_probe")) config.Flags.ForceRefreshHardwareProbe = true;
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

        if (double.TryParse(Get("prediction-default-bit-stress-threshold"), out var defaultBitStress) && defaultBitStress > 0d)
            config.Prediction.DefaultBitStressThreshold = defaultBitStress;

        if (int.TryParse(Get("prediction-minimum-fit-rows"), out var minFitRows) && minFitRows >= 2)
            config.Prediction.MinimumFitRows = minFitRows;

        var bitStressCandidates = ParseDoubleList(Get("prediction-bit-stress-threshold-candidates"));
        if (bitStressCandidates.Count > 0)
            config.Prediction.BitStressThresholdCandidates = bitStressCandidates;

        if (double.TryParse(Get("selection-near-baseline-max-size-growth-percent"), out var nearPct) && nearPct >= 0d)
            config.CandidateSelection.NearBaselineMaxSizeGrowthPercent = nearPct;

        var windows = ParseDoubleList(Get("selection-interior-window-fractions"));
        if (windows.Count > 0)
            config.CandidateSelection.InteriorWindowFractions = windows;

        if (int.TryParse(Get("selection-max-candidates-per-interior-window"), out var maxInterior) && maxInterior > 0)
            config.CandidateSelection.MaxCandidatesPerInteriorWindow = maxInterior;

        if (int.TryParse(Get("selection-max-fallback-attempts-per-anchor"), out var maxFallbacks) && maxFallbacks > 0)
            config.CandidateSelection.MaxFallbackAttemptsPerAnchor = maxFallbacks;

        if (double.TryParse(Get("selection-minimum-kld-improvement-epsilon"), out var minKldEpsilon) && minKldEpsilon >= 0d)
            config.CandidateSelection.MinimumKldImprovementEpsilon = minKldEpsilon;

        if (double.TryParse(Get("selection-minimum-neighbor-gap-fraction"), out var neighborGap) && neighborGap >= 0d)
            config.CandidateSelection.MinimumNeighborGapFractionOfGlobalSpan = neighborGap;

        if (double.TryParse(Get("selection-near-lower-anchor-brutal-zone-fraction"), out var brutalZone) && brutalZone >= 0d)
            config.CandidateSelection.NearLowerAnchorBrutalZoneFractionOfPairSpan = brutalZone;

        if (double.TryParse(Get("selection-near-anchor-required-kld-gain-fraction"), out var brutalGain) && brutalGain >= 0d)
            config.CandidateSelection.NearAnchorRequiredKldGainFractionOfPairGap = brutalGain;

        if (Has("allow-eight-bit-anchor-replacements"))
            config.CandidateSelection.AllowEightBitAnchorReplacements = true;

        config.Output.OutputDir = Prefer(Get("output-dir"), config.Output.OutputDir);
        config.Output.OutputNamePrefix = Prefer(Get("output-name-prefix"), config.Output.OutputNamePrefix);
        if (Has("export-external-learned-baselines")) config.Output.ExportExternalLearnedBaselines = true;
        if (Has("reuse-existing-final-artifacts")) config.Output.ReuseExistingFinalArtifacts = true;

        if (int.TryParse(Get("max-selected-choices-per-bucket"), out var maxSelectedChoicesPerBucket) && maxSelectedChoicesPerBucket > 0)
            config.Survival.MaxSelectedChoicesPerBucket = maxSelectedChoicesPerBucket;

        if (double.TryParse(Get("survival-meaningful-size-bias-percent"), out var sizeBiasPercent) && sizeBiasPercent >= 0d)
            config.Survival.MeaningfulSizeBiasPercent = sizeBiasPercent;

        if (double.TryParse(Get("survival-kld-close-call-absolute-epsilon"), out var kldCloseCallAbs) && kldCloseCallAbs >= 0d)
            config.Survival.KldCloseCallAbsoluteEpsilon = kldCloseCallAbs;

        if (double.TryParse(Get("survival-kld-close-call-relative-fraction"), out var kldCloseCallRelative) && kldCloseCallRelative >= 0d)
            config.Survival.KldCloseCallRelativeFraction = kldCloseCallRelative;

        if (double.TryParse(Get("survival-ppl-large-difference-percent"), out var pplLargeDiff) && pplLargeDiff >= 0d)
            config.Survival.PplLargeDifferencePercent = pplLargeDiff;

        if (double.TryParse(Get("survival-trade-score-size-bias-weight"), out var sizeWeight) && sizeWeight >= 0d)
            config.Survival.TradeScoreSizeBiasWeight = sizeWeight;

        if (double.TryParse(Get("survival-trade-score-ppl-weight"), out var pplWeight) && pplWeight >= 0d)
            config.Survival.TradeScorePplWeight = pplWeight;

        config.Identity.ArchitectureFamilyName = Prefer(Get("architecture-family"), config.Identity.ArchitectureFamilyName);
        if (Has("allow-architecture-family-alias-override")) config.Identity.AllowArchitectureFamilyAliasOverride = true;
    }

    private static List<double> ParseDoubleList(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return new List<double>();

        return value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(x => double.TryParse(x, out var parsed) ? (double?)parsed : null)
            .Where(x => x.HasValue)
            .Select(x => x!.Value)
            .ToList();
    }

    private static string? Prefer(string? preferred, string? fallback)
        => string.IsNullOrWhiteSpace(preferred) ? fallback : preferred;

    private static bool IsEmptyFrontmatterValue(object? value)
    {
        if (value == null)
            return true;

        if (value is string text)
            return string.IsNullOrWhiteSpace(text);

        if (value is System.Collections.IEnumerable sequence && value is not string)
        {
            foreach (var item in sequence)
            {
                if (!IsEmptyFrontmatterValue(item))
                    return false;
            }

            return true;
        }

        return false;
    }

    private static string ResolveMagicQuantRoot(string? configured)
    {
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.GetFullPath(configured);

        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), MagicConstants.MagicQuantFolder);
    }


    private static List<string> NormalizeScratchRoots(IEnumerable<string>? roots)
    {
        if (roots == null)
            return new List<string>();

        return roots
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(x => Path.GetFullPath(x.Trim()))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
    private static string? NormalizeNullOrFullPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        return Path.GetFullPath(value);
    }
}