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

        await Task.Yield();

        throw new NotImplementedException(
            "The build-hybrids command is currently disabled. Use `evolution` for active hybrid generation workflows.");
    }

    private static void ShowHelp()
    {
        AnsiConsole.MarkupLine("[bold yellow]Command: build-hybrids[/]");
        AnsiConsole.MarkupLine("Builds/benchmarks remaining hybrid combinations from the current candidate-based search space.");
        AnsiConsole.MarkupLine("Usage: mq build-hybrids --model-dir \"<path>\" [--use-imatrix] [--allow-high-precision-hybrids]");
    }
}
