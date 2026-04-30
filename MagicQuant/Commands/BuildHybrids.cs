using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Commands;

public class BuildHybrids : ICommand
{
    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            ShowHelp();
            return;
        }

        AnsiConsole.MarkupLine("[grey]build-hybrids now routes through the centralized evolution/survival/export pipeline.[/]");
        await new Evolution().Run(args);
    }

    private static void ShowHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: build-hybrids[/]");
        AnsiConsole.MarkupLine("Runs the centralized survival/export flow over the active MagicQuant evolution pipeline.");
        AnsiConsole.MarkupLine("Usage: mq build-hybrids --model-dir \"<path>\" [--config \"./config.default.yaml\"] [--output-dir \"<path>\"] [--output-name-prefix \"Model\"] [--reuse-existing-final-artifacts] [--export-external-learned-baselines]");
    }
}