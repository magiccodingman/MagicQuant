using System.Globalization;
using MagicQuant.Models;

namespace MagicQuant.Configuration;

/// <summary>Rejects typos and ambiguous CLI values before any configuration or runtime mutation.</summary>
public static class CliOptionValidator
{
    private static readonly HashSet<string> Flags = new(StringComparer.OrdinalIgnoreCase)
    {
        "allow-architecture-family-alias-override",
        "allow-eight-bit-anchor-replacements",
        "allow-high-precision-hybrids",
        "allow-missing-manifest-tensors",
        "check-config",
        "disable-tensor-group-rebucket",
        "export-external-learned-baselines",
        "force-refresh-hardware-probe",
        "force-relearn-baseline-tensor-mappings",
        "force_refresh_hardware_probe",
        "full-relearn-tensor-groups",
        "help",
        "imatrix-force-rebuild",
        "no-rebucket-learned-tensor-groups",
        "rebucket-learned-tensor-groups",
        "rebucket-tensor-groups-from-db",
        "recheck-hardware-probe",
        "relearn-baseline-mappings",
        "relearn-tensor-groups-from-db",
        "reuse-existing-final-artifacts",
        "skip-tensor-group-confirm",
        "strict-config",
        "update",
        "use-imatrix",
        "validate",
        "validate-all-anomaly-strict-candidates-after-success",
        "verify",
        "yes-tensor-groups",
    };
    private static readonly HashSet<string> Values = new(StringComparer.OrdinalIgnoreCase)
    {
        "architecture-family",
        "clone-json",
        "clone-repo",
        "config",
        "convert-script",
        "imatrix-dataset-config",
        "imatrix-dataset-local-file",
        "imatrix-dataset-repo",
        "imatrix-dataset-split",
        "imatrix-identity-hash",
        "imatrix-path",
        "imatrix-url",
        "llama-bin",
        "llama-root",
        "magic-quant-root",
        "manual-max-predicted-size-bytes",
        "missing-manifest-base-quant",
        "model-dir",
        "output-dir",
        "output-name-prefix",
        "prediction-bit-stress-threshold-candidates",
        "prediction-default-bit-stress-threshold",
        "prediction-minimum-fit-rows",
        "selection-diversify-validation-candidates",
        "selection-diversity-low-bit-only",
        "selection-diversity-scan-max-candidates",
        "selection-diversity-scan-min-candidates",
        "selection-diversity-scan-multiplier",
        "selection-interior-window-fractions",
        "selection-max-candidates-per-interior-window",
        "selection-max-fallback-attempts-per-anchor",
        "selection-minimum-kld-improvement-epsilon",
        "selection-minimum-neighbor-gap-fraction",
        "selection-near-anchor-required-kld-gain-fraction",
        "selection-near-baseline-max-size-growth-percent",
        "selection-near-lower-anchor-brutal-zone-fraction",
        "source-json",
        "source-repo",
    };

    public static void Validate(IReadOnlyList<CliArg> args)
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var arg in args)
        {
            string name = arg.Name ?? "";
            if (!seen.Add(name)) throw new ArgumentException($"Duplicate option --{name}. Supply it once.");
            if (Flags.Contains(name))
            {
                if (!string.IsNullOrEmpty(arg.Value)) throw new ArgumentException($"--{name} is a flag and takes no value. Set boolean policy in YAML when disabling it.");
            }
            else if (Values.Contains(name))
            {
                if (string.IsNullOrWhiteSpace(arg.Value)) throw new ArgumentException($"--{name} requires a value.");
                ValidateTypedValue(name.ToLowerInvariant(), arg.Value);
            }
            else throw new ArgumentException($"Unknown or removed option --{name}. See command --help and docs/configuration.md.");
        }
    }
    private static void ValidateTypedValue(string name, string value)
    {
        string[] integers = ["prediction-minimum-fit-rows", "selection-max-candidates-per-interior-window",
            "selection-max-fallback-attempts-per-anchor", "selection-diversity-scan-multiplier",
            "selection-diversity-scan-min-candidates", "selection-diversity-scan-max-candidates"];
        string[] numbers = ["prediction-default-bit-stress-threshold", "selection-near-baseline-max-size-growth-percent",
            "selection-minimum-kld-improvement-epsilon", "selection-minimum-neighbor-gap-fraction",
            "selection-near-lower-anchor-brutal-zone-fraction", "selection-near-anchor-required-kld-gain-fraction"];
        bool valid = true;
        if (integers.Contains(name))
            valid = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n >= (name == "prediction-minimum-fit-rows" ? 2 : 1);
        else if (numbers.Contains(name))
            valid = double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double n) && double.IsFinite(n) &&
                (name == "prediction-default-bit-stress-threshold" ? n > 0 : n >= 0);
        else if (name == "manual-max-predicted-size-bytes")
            valid = ulong.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out _);
        else if (name is "prediction-bit-stress-threshold-candidates" or "selection-interior-window-fractions")
            valid = value.Split(',').All(v => double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out double n) &&
                double.IsFinite(n) && n > 0 && (name != "selection-interior-window-fractions" || n <= 1));
        else if (name is "selection-diversify-validation-candidates" or "selection-diversity-low-bit-only")
            valid = new[] { "true", "false", "1", "0", "yes", "no", "y", "n", "on", "off" }.Contains(value, StringComparer.OrdinalIgnoreCase);
        if (!valid) throw new ArgumentException($"Invalid value '{value}' for --{name}. Check the option's type/range; decimals use a dot.");
    }

}
