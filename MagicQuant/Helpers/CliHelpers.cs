using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Models;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class CliHelpers
{
    
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