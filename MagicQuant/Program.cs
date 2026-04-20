using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using Spectre.Console;
using System.Collections.Immutable;
using MQ.DB.Models;

#if DEBUG
// If we are in Debug and no arguments were passed, default to "evolution"
if (args.Length == 0)
{
    args = new[] { "evolution" };
}

// OPTIONAL: Manually append hardcoded flags for testing specific scenarios
// Example: If you want to test "evolution --iterations 10" every time you debug
string manualFlags =
    @"--model-dir ""/mnt/world8/AI/Models/Qwen3-4B-Instruct-2507-unsloth/""
      --use-imatrix
      --imatrix-dataset-local-file ""/home/slurp/Documents/Output_Files/Dataset/artifacts/imatrix-general-v1-1m.jsonl""
      --imatrix-dataset-split ""text""";
args = args.Concat(manualFlags.Split(' ', StringSplitOptions.RemoveEmptyEntries)).ToArray();
#endif

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
    // strict startup integrity checks
    TensorWeightScheme.ValidateSmallestConfiguration();
    BaselineQuants.ValidateIntegrityOrThrow();
    QuantizationService.ValidateQuantNameNormalizationOrThrow();

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

        AnsiConsole.MarkupLine("[bold green][/] Environment validated.");
        AnsiConsole.WriteLine();
    }

    // Manditory Run combinations and DuckDB setup
    CliHelpers.ValidateCombinationLogicWorks();

    // 7. Execute Command
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
}
