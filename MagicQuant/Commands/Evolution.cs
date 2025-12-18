using MagicQuant.Models;
using MagicQuant.Helpers;
using MagicQuant;
using Spectre.Console;

namespace MagicQuant.Commands;

public class Evolution : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        // 1. Handle Help Flag
        if (args.Any(a => a.Name?.ToLower() == "help"))
        {
            ShowEvolutionHelp();
            return;
        }

        // 2. Parse --model-dir
        string? modelDirRaw = args.FirstOrDefault(a => a.Name?.ToLower() == "model-dir")?.Value;

        if (string.IsNullOrWhiteSpace(modelDirRaw))
        {
            string msg = "[red]Error:[/] Missing required argument [yellow]--model-dir[/].";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new Exception(msg);
        }

        // 3. Normalize and Validate Path
        string fullModelPath = Path.GetFullPath(modelDirRaw);

        if (!Directory.Exists(fullModelPath))
        {
            string msg = $"[red]Error:[/] The directory [yellow]'{fullModelPath}'[/] does not exist.";
            AnsiConsole.MarkupLine(msg);
            ShowEvolutionHelp();
            throw new Exception(msg);
        }

        // 4. Validate Content (.safetensors existence)
        // We look for any .safetensors file in the top directory. 
        // If your models are often in subfolders, change SearchOption.TopDirectoryOnly to AllDirectories.
        var safeTensorFiles = Directory.GetFiles(fullModelPath, "*.safetensors", SearchOption.TopDirectoryOnly);

        if (safeTensorFiles.Length == 0)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] No [yellow].safetensors[/] files found in [blue]{fullModelPath}[/].");
            AnsiConsole.MarkupLine("[grey]Please ensure this is a valid HuggingFace model directory.[/]");
            throw new Exception();
        }

        // 5. Populate Cache
        Cache.ModelDirectory = fullModelPath;
        Cache.MagicQuantDirectory = Path.Combine(fullModelPath, "MagicQuant");

        // Create the MagicQuant directory immediately so it's ready for future steps
        if (!Directory.Exists(Cache.MagicQuantDirectory))
        {
            Directory.CreateDirectory(Cache.MagicQuantDirectory);
        }

        // 6. Success Output
        AnsiConsole.MarkupLine("[green]✔ Model Directory Validated[/]");
        AnsiConsole.Write(new Rule("[yellow]Evolution Configuration[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine($"Model Path:   [blue]{Cache.ModelDirectory}[/]");
        AnsiConsole.MarkupLine($"Output Path:  [blue]{Cache.MagicQuantDirectory}[/]");
        AnsiConsole.MarkupLine($"Files Found:  [green]{safeTensorFiles.Length}[/] safe tensors");
        
        // Ensure Llama paths are set (sanity check from InitializeLlamaCpp)
        if (string.IsNullOrEmpty(Cache.LlamaBin))
        {
            // Note: In a real run, Program.cs runs Init first, so this might be populated. 
            // If not, we might want to warn or rely on defaults.
            AnsiConsole.MarkupLine("[yellow]Warning: Llama binaries path not set in Cache. (Did Initialization run?)[/]");
        }

        // Next steps of evolution would go here...
    }

    private void ShowEvolutionHelp()
    {
        // Use MarkupLine for colors/styles
        AnsiConsole.MarkupLine("[bold yellow]Command: evolution[/]");
        AnsiConsole.WriteLine("Runs the full evolutionary quantization search algorithm on a target model.");
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[bold]Usage:[/]");
        // Use WriteLine here so "[options]" doesn't crash it
        AnsiConsole.WriteLine("  mq evolution --model-dir \"<path>\" [options]");
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[bold]Arguments:[/]");
        // Use MarkupLine here because we WANT the [green] color
        AnsiConsole.MarkupLine("  [green]--model-dir[/]    Path to the model directory containing .safetensors files (Required)");
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[bold]Example:[/]");
        // Use WriteLine here to avoid issues with paths (backslashes)
        AnsiConsole.WriteLine("  mq evolution --model-dir \"C:\\Models\\Mistral-7B\"");
    }
}