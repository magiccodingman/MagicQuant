using System.Collections.Immutable;
using System.Numerics;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class CliHelpers
{
    public static void PrintTotalCombinationCount()
    {
        const long MaxSupported = 4_000_000_000L;

        BigInteger grandTotal = BigInteger.Zero;

        // Valid base conversions
        var baseConversions =
            BaselineQuants.All
                .Where(b => b.AllowedAsBaseConversion)
                .ToImmutableArray();

        if (baseConversions.IsEmpty)
            throw new InvalidOperationException("No BaselineQuants are marked as AllowedAsBaseConversion.");

        // Iterate each base
        foreach (var baseQuant in baseConversions)
        {
            // Filter tensor weight schemes allowed by imatrix rule
            var allowedSchemesForBase =
                TensorWeightScheme.All
                    .Where(s =>
                        baseQuant.RequiresImatrix ||
                        !s.RequiresImatrix
                    )
                    .ToImmutableArray();

            if (allowedSchemesForBase.IsEmpty)
                continue;

            BigInteger perBaseTotal = BigInteger.One;

            // For each tensor group, count legal schemes
            foreach (var group in TReg.All)
            {
                int validCount =
                    allowedSchemesForBase.Count(s =>
                        s.BannedGroups.IsEmpty ||
                        !s.BannedGroups.Contains(group)
                    );

                if (validCount == 0)
                {
                    perBaseTotal = BigInteger.Zero;
                    break;
                }

                perBaseTotal *= validCount;
            }

            grandTotal += perBaseTotal;
        }

        // Enforce DB constraint
        if (grandTotal > MaxSupported)
        {
            throw new InvalidOperationException(
                $"Total combinations ({grandTotal:N0}) exceed database primary ID limit ({MaxSupported:N0})."
            );
        }

        AnsiConsole.MarkupLine(
            $"[green]Total potential combinations:[/] [bold yellow]{grandTotal:N0}[/]"
        );
    }

    
    public static List<CliArg> ParseArguments(string input)
    {
        var cliArgs = new List<CliArg>();
    
        // Regex identifies --key value or --key "value with spaces"
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