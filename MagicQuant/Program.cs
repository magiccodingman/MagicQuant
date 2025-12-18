using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;

// 1. OS & Permission Check
if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
{
    // On Linux, the effective user ID for root is 0
    // We check if we are running as root to ensure file/env access
    if (GetLinuxUserId() != 0)
    {
        AnsiConsole.Write(new Rule("[red]Permission Denied[/]") { Justification = Justify.Left });
        AnsiConsole.MarkupLine("[red]Error:[/] MagicQuant must be run with [bold]sudo[/] on Linux to manage environments and files.");
        AnsiConsole.MarkupLine("[grey]Please try:[/] [yellow]sudo dotnet MagicQuant.dll[/] (or your binary name)");
        return;
    }
}

// 2. Define the Command Registry
var commands = new Dictionary<string, (string Description, Func<ICommand> Factory)>(StringComparer.OrdinalIgnoreCase)
{
    { "evolution", ("Run the full evolutionary quantization search", () => new Evolution()) },
    { "build-hybrids", ("Export specific hybrid models with polished README", () => new BuildHybrids()) },
    { "initialize-llama-cpp", ("Initialize or update llama.cpp", () => new InitializeLlamaCpp()) }
};

// 3. Validate input
if (args.Length == 0 || args[0].Equals("help", StringComparison.OrdinalIgnoreCase))
{
    CliHelpers.ShowHelp(commands);
    return;
}

string commandInput = args[0];

// 4. Check if command exists
if (!commands.TryGetValue(commandInput, out var commandInfo))
{
    AnsiConsole.MarkupLine($"[red]Error:[/] The command [yellow]'{commandInput}'[/] does not exist.");
    CliHelpers.ShowHelp(commands);
    return;
}

// 5. Parse Arguments
string remainingArgsString = string.Join(" ", args.Skip(1));
List<CliArg> parsedArgs = CliHelpers.ParseArguments(remainingArgsString);

try 
{
    // 6. Mandatory Validation for non-init commands
    if (!commandInput.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase))
    {
        await AnsiConsole.Status()
            .StartAsync("[grey]Checking environment dependencies...[/]", async ctx => 
            {
                var initializer = new InitializeLlamaCpp();
                var validationArgs = new List<CliArg> { new CliArg { Name = "validate", Value = "" } };
                await initializer.Run(validationArgs);
            });
            
        AnsiConsole.MarkupLine("[bold green]✓[/] Environment validated.");
        AnsiConsole.WriteLine();
    }

    // 7. Execute Command
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
}

#region Linux Helpers

static uint GetLinuxUserId()
{
    // Standard Unix call to get effective user ID
    [DllImport("libc")]
    static extern uint geteuid();

    try
    {
        return geteuid();
    }
    catch
    {
        // Fallback for environments where libc isn't standard
        return 1; // Assume non-root
    }
}

#endregion