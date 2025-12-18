using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

// 1. Define the Command Registry
var commands = new Dictionary<string, (string Description, Func<ICommand> Factory)>(StringComparer.OrdinalIgnoreCase)
{
    { "evolution", ("Run the full evolutionary quantization search", () => new Evolution()) },
    { "build-hybrids", ("Export specific hybrid models with polished README", () => new BuildHybrids()) },
    { "initialize-llama-cpp", ("Initialize or update llama.cpp", () => new InitializeLlamaCpp()) }
};

// 2. Validate input
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

// 4. Parse the arguments for the primary command
string remainingArgsString = string.Join(" ", args.Skip(1));
List<CliArg> parsedArgs = CliHelpers.ParseArguments(remainingArgsString);

try 
{
    // 5. Pre-run Validation logic
    // If the command is NOT "initialize-llama-cpp", we run initialization first with --validate
    if (!commandInput.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase))
    {
        AnsiConsole.MarkupLine("[grey]Checking environment dependencies...[/]");
        
        var initializer = new InitializeLlamaCpp();
        var validationArgs = new List<CliArg> { new CliArg { Name = "validate", Value = "" } };
        
        // Run the validation
        await initializer.Run(validationArgs);
        
        AnsiConsole.MarkupLine("[green]Environment validated.[/]");
        AnsiConsole.WriteLine();
    }

    // 6. Execute the actual requested command
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    // Spectre.Console handles the formatting of the error automatically
    AnsiConsole.WriteException(ex);
}