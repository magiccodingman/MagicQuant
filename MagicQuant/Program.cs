using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

// 1. Define the Command Registry
var commands = new Dictionary<string, (string Description, Func<ICommand> Factory)>(StringComparer.OrdinalIgnoreCase)
{
    { "evolution", ("Run the full evolutionary quantization search", () => new Evolution()) },
    { "build-hybrids", ("Export specific hybrid models with polished README", () => new BuildHybrids()) }
};

// 2. Validate input - Show help if no args or "help" requested
if (args.Length == 0 || args[0].Equals("help", StringComparison.OrdinalIgnoreCase))
{
    CliHelpers.ShowHelp(commands);
    return;
}

string commandInput = args[0];

// 3. Check if command exists
if (!commands.TryGetValue(commandInput, out var commandInfo))
{
    AnsiConsole.MarkupLine($"[red]Error:[/] The command [yellow]'{commandInput}'[/] does not exist.");
    CliHelpers.ShowHelp(commands);
    return;
}

// 4. Parse the remaining arguments using Regex for quote-safety
string remainingArgsString = string.Join(" ", args.Skip(1));
List<CliArg> parsedArgs = CliHelpers.ParseArguments(remainingArgsString);

// 5. Execute the command
try 
{
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
}


#region Helpers


#endregion