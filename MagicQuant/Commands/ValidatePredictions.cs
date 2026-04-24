using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using Microsoft.EntityFrameworkCore;
using MQ.DB;
using MQ.DB.Data;
using Spectre.Console;

namespace MagicQuant.Commands;

public sealed class ValidatePredictions : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowHelp();
            return;
        }

        string? modelDirRaw = args.FirstOrDefault(a =>
            string.Equals(a.Name, "model-dir", StringComparison.OrdinalIgnoreCase))?.Value;

        modelDirRaw = string.IsNullOrWhiteSpace(modelDirRaw)
            ? Config.Current.Paths.ModelDir
            : modelDirRaw;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
            throw new InvalidOperationException("Missing model directory. Provide --model-dir or set paths.model_dir in YAML.");

        string modelDir = Path.GetFullPath(modelDirRaw);
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory does not exist: {modelDir}");

        Cache.ModelDirectory = modelDir;
        Cache.ModelMagicQuantDirectory = Path.Combine(modelDir, "MagicQuant");
        Directory.CreateDirectory(Cache.ModelMagicQuantDirectory);

        Cache.CurrentModelId = MagicQuantModelId.GetOrCreateModelId(modelDir);
        JsonHelper.DetectAndSetTorchType(Cache.ModelDirectory);

        await ResolveArchitectureFamilyFromConfigAsync();

        ApplyOptionalImatrixContext(args);

        string outputDir = ResolveOutputDirectory(args);
        Directory.CreateDirectory(outputDir);

        var repository = new HybridBenchmarkRepository();
        var effectiveResolver = new EffectiveCandidateStateResolverService(repository);
        var prediction = new RankSafeKldPredictionService(repository, effectiveResolver);
        var validator = new PredictionValidationService(repository, prediction);

        await validator.ExportAsync(outputDir);
    }

    private static async Task ResolveArchitectureFamilyFromConfigAsync()
    {
        Cache.CurrentArchitectureFamilyId = null;

        if (string.IsNullOrWhiteSpace(Cache.CurrentArchitectureFamilyName))
            return;

        string normalized = Cache.CurrentArchitectureFamilyNormalizedName;

        await using var db = new MagicQuantContext();
        var family = await db.ArchitectureFamilies
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.NormalizedName == normalized);

        if (family == null)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Architecture family '{Markup.Escape(Cache.CurrentArchitectureFamilyName)}' was configured but not found in SQLite. Validation will use the raw model hash scope.");
            return;
        }

        Cache.CurrentArchitectureFamilyId = family.Id;
        AnsiConsole.MarkupLine($"[green]Architecture family scope:[/] {Markup.Escape(family.DisplayName)} (Id={family.Id})");
    }

    private static void ApplyOptionalImatrixContext(IReadOnlyList<CliArg> args)
    {
        string? imatrixPath = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-path", StringComparison.OrdinalIgnoreCase))?.Value;
        string? imatrixHash = args.FirstOrDefault(a => string.Equals(a.Name, "imatrix-identity-hash", StringComparison.OrdinalIgnoreCase))?.Value;

        if (!string.IsNullOrWhiteSpace(imatrixPath))
        {
            string fullPath = Path.GetFullPath(imatrixPath);
            if (!File.Exists(fullPath))
                throw new FileNotFoundException($"Imatrix path does not exist: {fullPath}");

            Cache.IsImatrixAvailable = true;
            Cache.ActiveImatrixPath = fullPath;
            Cache.ActiveImatrixIdentityHash = null;
            ImatrixIdentityService.EnsureActiveImatrixIdentityHashAsync().GetAwaiter().GetResult();
            AnsiConsole.MarkupLine($"[green]Validation imatrix path:[/] {Markup.Escape(fullPath)}");
            return;
        }

        if (!string.IsNullOrWhiteSpace(imatrixHash))
        {
            Cache.IsImatrixAvailable = true;
            Cache.ActiveImatrixPath = null;
            Cache.ActiveImatrixIdentityHash = imatrixHash.Trim().ToLowerInvariant();
            AnsiConsole.MarkupLine($"[green]Validation imatrix identity:[/] {Markup.Escape(Cache.ActiveImatrixIdentityHash)}");
            return;
        }

        Cache.IsImatrixAvailable = false;
        Cache.ActiveImatrixPath = null;
        Cache.ActiveImatrixIdentityHash = null;

        if (Config.Current.Flags.UseImatrix)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] flags.use_imatrix is true, but validate-predictions was not given --imatrix-path or --imatrix-identity-hash. Strict validation will use the no-imatrix bucket.");
        }
    }

    private static string ResolveOutputDirectory(IReadOnlyList<CliArg> args)
    {
        string? explicitOutput = args.FirstOrDefault(a => string.Equals(a.Name, "output-dir", StringComparison.OrdinalIgnoreCase))?.Value;
        if (!string.IsNullOrWhiteSpace(explicitOutput))
            return Path.GetFullPath(explicitOutput);

        if (!string.IsNullOrWhiteSpace(Config.OutputDirectory))
            return Path.Combine(Path.GetFullPath(Config.OutputDirectory!), "PredictionValidation");

        return Path.Combine(Cache.ModelMagicQuantDirectory!, "PredictionValidation");
    }

    private static void ShowHelp()
    {
        AnsiConsole.MarkupLine("[bold]validate-predictions[/]");
        AnsiConsole.MarkupLine("Validates rank-safe isolation KLD predictions against existing SQLite category=General benchmark truth.");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Required:[/]");
        AnsiConsole.MarkupLine("  --model-dir <path>                  HuggingFace source model directory, or set paths.model_dir in YAML");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[yellow]Optional:[/]");
        AnsiConsole.MarkupLine("  --architecture-family <name>         Uses configured family scope if present in SQLite");
        AnsiConsole.MarkupLine("  --imatrix-path <path>                Hash this imatrix and validate that exact bucket");
        AnsiConsole.MarkupLine("  --imatrix-identity-hash <sha256>     Validate an already-known imatrix bucket");
        AnsiConsole.MarkupLine("  --output-dir <path>                  Report output directory");
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[grey]Outputs prediction_validation_general.csv and prediction_validation_general.md.[/]");
    }
}
