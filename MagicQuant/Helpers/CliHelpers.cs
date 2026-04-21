using System.Collections.Immutable;
using System.Diagnostics;
using System.Numerics;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Models;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class CliHelpers
{
    public static void ValidateCombinationLogicWorks(bool realResults = false)
    {
        PrintTotalCombinationCount();

        var expectedTotal = ComboCounter.CountAll();

        AnsiConsole.MarkupLine(
            realResults
                ? $"[bold cyan]Total real combinations after model detection:[/] [bold yellow]{expectedTotal:N0}[/]"
                : $"[bold cyan]Expected total combinations:[/] [bold yellow]{expectedTotal:N0}[/]");

        var sw = Stopwatch.StartNew();
        long actualTotal = 0;

        foreach (var baseline in RuntimeSearchSpace.GetActiveCombinationBaselines().ToImmutableArray())
        {
            AnsiConsole.MarkupLine(
                $"[cyan]Base:[/] [bold]{string.Join("/", baseline.Names)}[/]  [grey](RequiresImatrix={baseline.RequiresImatrix})[/]");

            long baseTotal = 0;

            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(baseline, batchSize: 10_000_000))
            {
                baseTotal += batch.Count;
                actualTotal += batch.Count;

                AnsiConsole.MarkupLine($"  [green]Batch:[/] {batch.Count:N0}  [grey]BaseRunning:[/] {baseTotal:N0}");
                batch.Clear();
            }

            AnsiConsole.MarkupLine($"[yellow]Base total:[/] {baseTotal:N0}");
        }

        sw.Stop();

        bool match = actualTotal == expectedTotal;

        AnsiConsole.MarkupLine($"[bold green]Generated total:[/] {actualTotal:N0}");
        AnsiConsole.MarkupLine(
            match
                ? "[bold green] Counts match expected total[/]"
                : $"[bold red] MISMATCH! Expected {expectedTotal:N0} but generated {actualTotal:N0}[/]");

        var t = sw.Elapsed;
        AnsiConsole.MarkupLine($"[bold]Elapsed:[/] {t.Hours}h {t.Minutes}m {t.Seconds}s {t.Milliseconds}ms");

        Console.WriteLine();
        Console.WriteLine("---------------");
        Console.WriteLine();

        var samplePlan = TensorConfigGenerator.GenerateInitialIsolationSamplePlan(
            realResults ? Cache.UnusedTensorGroups : null);

        AnsiConsole.MarkupLine($"[bold green]Required pure baselines:[/] {samplePlan.PureBaselineCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Required base-only isolations:[/] {samplePlan.BaseOnlyIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Required smallest-probe isolations:[/] {samplePlan.GroupIsolationCount:N0}");
        AnsiConsole.MarkupLine($"[bold green]Total required startup samples:[/] {samplePlan.TotalCount:N0}");
    }

    public static void PrintTotalCombinationCount()
    {
        const long MaxSupported = 4_000_000_000L;

        BigInteger total = ComboCounter.CountAll();

        if (total > MaxSupported)
            throw new InvalidOperationException(
                $"Total combinations ({total:N0}) exceed database primary ID limit ({MaxSupported:N0}).");

        AnsiConsole.MarkupLine($"[green]Total potential combinations:[/] [bold yellow]{total:N0}[/]");
    }

    public static List<CliArg> ParseArguments(string input)
    {
        var cliArgs = new List<CliArg>();
        var regex = new Regex(@"--(?<name>[^\s=]+)(?:[\s=]+(?:""(?<value>[^""]*)""|(?<value>[^\s-]*)))?", RegexOptions.IgnoreCase);
        var matches = regex.Matches(input);

        foreach (Match match in matches)
        {
            cliArgs.Add(new CliArg
            {
                Name = match.Groups["name"].Value,
                Value = match.Groups["value"].Value
            });
        }

        return cliArgs;
    }

    public static void ShowHelp(Dictionary<string, (string Description, Func<ICommand> Factory)> commands)
    {
        AnsiConsole.Write(new Rule("[yellow]MagicQuant CLI[/]") { Justification = Justify.Left, Style = "grey" });
        AnsiConsole.WriteLine();

        var table = new Table()
            .AddColumn("[blue]Command[/]")
            .AddColumn("[white]Description[/]")
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey15);

        foreach (var cmd in commands)
            table.AddRow($"[green]{cmd.Key}[/]", cmd.Value.Description);

        table.AddRow("[green]help[/]", "Show this help information");

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("Usage: [bold]mq[/] <command> [blue][[--option value]][/]");
        AnsiConsole.MarkupLine("Config: [green]--config[/] [grey]<path-to-yaml>[/]  (CLI flags override YAML)");
        AnsiConsole.WriteLine();
    }
}