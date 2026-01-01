using System.Collections.Immutable;
using System.Diagnostics;
using System.Numerics;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Models;
using Spectre.Console;
using MQ.DB;
using MQ.DB.Models;

namespace MagicQuant.Helpers;

public static class CliHelpers
{
    public static void ValidateCombinationLogicWorks(bool realResults = false)
    {
        CliHelpers.PrintTotalCombinationCount();

        // ----------------------------------------
        // Pre-compute expected total
        // ----------------------------------------
        var expectedTotal = ComboCounter.CountAll();

        if (!realResults)
        {
            AnsiConsole.MarkupLine(
                $"[bold cyan]Expected total combinations:[/] [bold yellow]{expectedTotal:N0}[/]");
        }
        else
        {
            AnsiConsole.MarkupLine(
                $"[bold cyan]Total real combinations after model detection:[/] [bold yellow]{expectedTotal:N0}[/]");
        }

        // ----------------------------------------
        // Generation + timing
        // ----------------------------------------
        var sw = Stopwatch.StartNew();

        long actualTotal = 0;

        var bases =
            BaselineQuants.All
                .Where(b => b.BaseConversionBase != null)
                .ToImmutableArray();

        foreach (var b in bases)
        {
            AnsiConsole.MarkupLine(
                $"[cyan]Base:[/] [bold]{string.Join("/", b.Names)}[/]  " +
                $"[grey](RequiresImatrix={b.RequiresImatrix})[/]");

            long baseTotal = 0;

            foreach (var batch in TensorConfigGenerator.GenerateTensorConfigBatches(
                         b, batchSize: 10_000_000))
            {
                baseTotal += batch.Count;
                actualTotal += batch.Count;

                AnsiConsole.MarkupLine(
                    $"  [green]Batch:[/] {batch.Count:N0}  " +
                    $"[grey]BaseRunning:[/] {baseTotal:N0}");

                // Release memory aggressively (unit-test mode)
                batch.Clear();
            }

            AnsiConsole.MarkupLine(
                $"[yellow]Base total:[/] {baseTotal:N0}");
        }

        sw.Stop();

        // ----------------------------------------
        // Verification
        // ----------------------------------------
        bool match = actualTotal == expectedTotal;

        AnsiConsole.MarkupLine(
            $"[bold green]Generated total:[/] {actualTotal:N0}");

        AnsiConsole.MarkupLine(
            match
                ? "[bold green] Counts match expected total[/]"
                : $"[bold red] MISMATCH! Expected {expectedTotal:N0} but generated {actualTotal:N0}[/]");

        // ----------------------------------------
        // Human-readable elapsed time
        // ----------------------------------------
        var t = sw.Elapsed;

        AnsiConsole.MarkupLine(
            $"[bold]Elapsed:[/] " +
            $"{t.Hours}h {t.Minutes}m {t.Seconds}s {t.Milliseconds}ms");
        Console.WriteLine();
        Console.WriteLine("---------------");
        Console.WriteLine();
        if (!realResults)
        {
            var MOE = TensorConfigGenerator.GenerateRequiredDataSampleCombos();

            var Dense = TensorConfigGenerator.GenerateRequiredDataSampleCombos( //);
                new List<TensorGroup>() { TReg.MoeRouter, TReg.MoeExperts });

            Console.WriteLine();
            Console.WriteLine("---------------");
            Console.WriteLine();
            AnsiConsole.MarkupLine(
                $"[bold green]Max MOE samples created:[/] {MOE.Count():N0}");
            AnsiConsole.MarkupLine(
                $"[bold green]Max Dense samples created:[/] {Dense.Count():N0}");
        }
        else
        {
            var RealBans = TensorConfigGenerator.GenerateRequiredDataSampleCombos(Cache.UnusedTensorGroups);

            Console.WriteLine();
            Console.WriteLine("---------------");
            Console.WriteLine();
            AnsiConsole.MarkupLine(
                $"[bold green]Max real samples to create:[/] {RealBans.Count():N0}");
        }
    }

    public static void PrintTotalCombinationCount()
    {
        const long MaxSupported = 4_000_000_000L;

        BigInteger total = ComboCounter.CountAll();

        if (total > MaxSupported)
            throw new InvalidOperationException(
                $"Total combinations ({total:N0}) exceed database primary ID limit ({MaxSupported:N0}).");

        AnsiConsole.MarkupLine(
            $"[green]Total potential combinations:[/] [bold yellow]{total:N0}[/]");
    }


    public static List<CliArg> ParseArguments(string input)
    {
        var cliArgs = new List<CliArg>();

        // Regex identifies --key value or --key "value with spaces"
        var regex = new Regex(@"--(?<name>[^\s=]+)(?:[\s=]+(?:""(?<value>[^""]*)""|(?<value>[^\s-]*)))?",
            RegexOptions.IgnoreCase);
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

        // Create a table for a clean, aligned UI
        var table = new Table()
            .AddColumn("[blue]Command[/]")
            .AddColumn("[white]Description[/]")
            .Border(TableBorder.Rounded)
            .BorderColor(Color.Grey15);

        foreach (var cmd in commands)
        {
            table.AddRow($"[green]{cmd.Key}[/]", cmd.Value.Description);
        }

        table.AddRow("[green]help[/]", "Show this help information");

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("Usage: [bold]mq[/] <command> [blue][[--option value]][/]");
        AnsiConsole.WriteLine();
    }
}