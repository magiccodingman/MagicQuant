using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using MagicQuant.Commands;
using MagicQuant.Helpers;
using MagicQuant.Models;
using Spectre.Console;
using System.Collections.Immutable;

#if DEBUG
// If we are in Debug and no arguments were passed, default to "evolution"
if (args.Length == 0)
{
    args = new[] { "evolution" };
}

// OPTIONAL: Manually append hardcoded flags for testing specific scenarios
// Example: If you want to test "evolution --iterations 10" every time you debug
string manualFlags = @"--model-dir ""/mnt/world8/AI/ToBench/Qwen3-4B-Instruct-2507-unsloth/""";
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
    ValidateCombinationLogicWorks();

    // 7. Execute Command
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
}

#region Linux Helpers

static void ValidateCombinationLogicWorks()
{
    CliHelpers.PrintTotalCombinationCount();

    // ----------------------------------------
    // Pre-compute expected total
    // ----------------------------------------
    var expectedTotal = ComboCounter.CountAll();

    AnsiConsole.MarkupLine(
        $"[bold cyan]Expected total combinations:[/] [bold yellow]{expectedTotal:N0}[/]");

    // ----------------------------------------
    // Generation + timing
    // ----------------------------------------
    var sw = Stopwatch.StartNew();

    long actualTotal = 0;

    var bases =
        BaselineQuants.All
            .Where(b => b.AllowedAsBaseConversion)
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

    var MOE = TensorConfigGenerator.GenerateRequiredDataSampleCombos();
    
    var Dense = TensorConfigGenerator.GenerateRequiredDataSampleCombos(//);
        new List<TensorGroup>(){TReg.MoeRouter, TReg.MoeExperts});
    
    Console.WriteLine();
    Console.WriteLine("---------------");
    Console.WriteLine();
    AnsiConsole.MarkupLine(
        $"[bold green]Max MOE samples created:[/] {MOE.Count():N0}");
    AnsiConsole.MarkupLine(
        $"[bold green]Max Dense samples created:[/] {Dense.Count():N0}");
}

#endregion