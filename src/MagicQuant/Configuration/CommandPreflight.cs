using System.Text.Json;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB.Models;

namespace MagicQuant.Configuration;

/// <summary>Read-only checks before cleanup, dependency setup, model hashing, or database initialization.</summary>
public static class CommandPreflight
{
    public static void Validate(string command, MagicQuantYamlConfig config, IReadOnlyList<CliArg> args)
    {
        string? Get(string name) => args.FirstOrDefault(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase))?.Value;
        if (!new[] { "all", "none", "selected" }.Contains(config.Baselines.StandardBaselinesMode.Trim(), StringComparer.OrdinalIgnoreCase))
            throw new InvalidOperationException("baselines.standard_baselines_mode must be all, selected, or none.");
        PathSafety.ValidateFolderName(config.Paths.ExternalBaselineCacheDirName, "paths.external_baseline_cache_dir_name");
        PathSafety.ValidateFolderName(config.Output.OutputNamePrefix, "output.output_name_prefix");
        var custom = new[] { config.Paths.LlamaRoot, config.Paths.LlamaBin, config.Paths.ConvertScript };
        if (custom.Any(p => !string.IsNullOrWhiteSpace(p)))
        {
            if (custom.Any(string.IsNullOrWhiteSpace))
                throw new InvalidOperationException("Provide all three custom llama.cpp paths: llama_root, llama_bin, convert_script.");
            if (!Directory.Exists(custom[0]) || !Directory.Exists(custom[1]) || !File.Exists(custom[2]))
                throw new InvalidOperationException("One or more custom llama.cpp paths do not exist.");
        }
        if (command.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase)) return;
        string model = config.Paths.ModelDir ?? "";
        if (string.IsNullOrWhiteSpace(model) || !Directory.Exists(model))
            throw new InvalidOperationException("A valid model directory is required. Set paths.model_dir or --model-dir.");
        model = Path.GetFullPath(model);
        string work = Path.Combine(model, "MagicQuant");
        string runtime = string.IsNullOrWhiteSpace(config.Paths.MagicQuantRoot)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), MagicConstants.MagicQuantFolder)
            : Path.GetFullPath(config.Paths.MagicQuantRoot);
        bool validation = command.Equals("validate-predictions", StringComparison.OrdinalIgnoreCase);
        bool clone = command.Equals("clone-repository-quants", StringComparison.OrdinalIgnoreCase);
        if (!validation)
        {
            if (!Directory.EnumerateFiles(model, "*.safetensors").Any())
                throw new InvalidOperationException("The model directory has no top-level .safetensors files.");
            string modelConfig = Path.Combine(model, "config.json");
            if (!File.Exists(modelConfig)) throw new InvalidOperationException("The source model is missing config.json.");
            using var parsed = JsonDocument.Parse(File.ReadAllText(modelConfig));
            if (parsed.RootElement.ValueKind != JsonValueKind.Object) throw new InvalidOperationException("Model config.json must contain a JSON object.");
            if (string.IsNullOrWhiteSpace(config.Identity.ArchitectureFamilyName))
                throw new InvalidOperationException("Set identity.architecture_family_name or --architecture-family explicitly.");
        }
        string output = validation ? OutputPathService.PredictionValidation(work, Get("output-dir"), config.Output.OutputDir)
            : clone ? OutputPathService.Clone(work, Get("output-dir"), config.Output.OutputDir)
            : OutputPathService.Pipeline(work, config.Output.OutputDir);
        var managed = new List<string>
        {
            Path.Combine(work, "GGUF"), Path.Combine(work, "Benchmarks"), Path.Combine(work, "Logs"), Path.Combine(work, "Runs"),
            Path.Combine(work, config.Paths.ExternalBaselineCacheDirName), Path.Combine(work, ".MagicQuant_tmp"),
            Path.Combine(runtime, MagicConstants.LlamaRepoName), Path.Combine(runtime, MagicConstants.EnvName), Path.Combine(runtime, "Runs")
        };
        if (!string.IsNullOrWhiteSpace(config.Paths.LlamaRoot)) managed.Add(config.Paths.LlamaRoot);
        if (!string.IsNullOrWhiteSpace(config.Paths.LlamaBin)) managed.Add(config.Paths.LlamaBin);
        managed.AddRange(config.Paths.ScratchRoots.Select(s => Path.Combine(s, ".MagicQuant_tmp")));
        PathSafety.ValidateExportDirectory(output, model, runtime, managed.ToArray());
        if (clone)
        {
            string? repo = Get("source-repo") ?? Get("clone-repo");
            string? source = Get("source-json") ?? Get("clone-json");
            if (string.IsNullOrWhiteSpace(repo) == string.IsNullOrWhiteSpace(source))
                throw new InvalidOperationException("Clone requires exactly one of --source-repo or --source-json.");
            if (source != null && !IsHttpUrl(source) && !File.Exists(source))
                throw new InvalidOperationException("The local --source-json manifest does not exist.");
        }
        if (validation && Get("imatrix-path") is { } matrix && !File.Exists(matrix))
            throw new InvalidOperationException("The validation --imatrix-path does not exist.");
        if (config.Flags.UseImatrix && config.Imatrix.DatasetLocalFile is { Length: > 0 } dataset && !File.Exists(dataset))
            throw new InvalidOperationException("imatrix.dataset_local_file does not exist.");
    }

    private static bool IsHttpUrl(string source) => Uri.TryCreate(source, UriKind.Absolute, out var uri) && uri.Scheme is "http" or "https";
}
