using MagicQuant.Commands;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

#if DEBUG
if (args.Length == 0)
{
    const string debugMode = "evolution"; // switch to "evolution" to use the full learning/search pipeline again. Or use "Clone" for cloning mode.

    if (string.Equals(debugMode, "clone", StringComparison.OrdinalIgnoreCase))
    {
        args =
        [
            "clone-repository-quants",
            "--architecture-family", @"""Qwen3-4B-Instruct-2507""",
            "--source-repo", @"""magiccodingman/Qwen3-4B-Instruct-2507-Unsloth-MagicQuant-v2-GGUF"""
        ];
    }
    else
    {
        // Previous DEBUG harness kept intact for quick full-pipeline testing.
        args = ["evolution", "--architecture-family", @"""Qwen3.6-35B-A3B"""];
    }
}
else if (args.Length > 0 &&
         (string.Equals(args[0], "evolution", StringComparison.OrdinalIgnoreCase) ||
          string.Equals(args[0], "clone-repository-quants", StringComparison.OrdinalIgnoreCase)) &&
         !args.Any(x => string.Equals(x, "--architecture-family", StringComparison.OrdinalIgnoreCase)))
{
    args = args.Concat(["--architecture-family", @"""Qwen3-4B-Instruct-2507"""]).ToArray();
}
#endif

var commands = new Dictionary<string, (string Description, Func<ICommand> Factory)>(StringComparer.OrdinalIgnoreCase)
{
    { "evolution", ("Run the full evolutionary quantization search", () => new Evolution()) },
    { "validate-predictions", ("Validate rank-safe KLD predictions against existing SQLite benchmarks", () => new ValidatePredictions()) },
    { "build-hybrids", ("Export specific hybrid models with polished README", () => new BuildHybrids()) },
    { "clone-repository-quants", ("Clone final MagicQuant tensor configurations from a compatible repository/json", () => new CloneRepositoryQuants()) },
    { "initialize-llama-cpp", ("Initialize or update llama.cpp", () => new InitializeLlamaCpp()) }
};

if (args.Length == 0 || args[0].Equals("help", StringComparison.OrdinalIgnoreCase))
{
    CliHelpers.ShowHelp(commands);
    return;
}

string commandInput = args[0];

if (!commands.TryGetValue(commandInput, out var commandInfo))
{
    AnsiConsole.MarkupLine($"[red]Error:[/] The command [yellow]'{commandInput}'[/] does not exist.");
    CliHelpers.ShowHelp(commands);
    return;
}

string remainingArgsString = string.Join(" ", args.Skip(1));
List<CliArg> parsedArgs = CliHelpers.ParseArguments(remainingArgsString);

try
{
    var loadedConfig = MagicQuantYamlLoader.LoadAndApply(commandInput, parsedArgs);
    var startupScratch = new ScratchStorageService();
    await startupScratch.CleanupStaleScratchArtifactsAsync();

    TensorWeightScheme.ValidateSmallestConfiguration();
    BaselineQuants.ValidateIntegrityOrThrow();
    QuantizationService.ValidateQuantNameNormalizationOrThrow();

    if (!commandInput.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase))
    {
        await AnsiConsole.Status()
            .StartAsync("[grey]Checking environment dependencies...[/]", async _ =>
            {
                var initializer = new InitializeLlamaCpp();
                var validationArgs = new List<CliArg>
                {
                    new() { Name = "validate", Value = string.Empty }
                };

                if (!string.IsNullOrWhiteSpace(loadedConfig.Paths.LlamaRoot))
                    validationArgs.Add(new CliArg { Name = "llama-root", Value = loadedConfig.Paths.LlamaRoot });
                if (!string.IsNullOrWhiteSpace(loadedConfig.Paths.LlamaBin))
                    validationArgs.Add(new CliArg { Name = "llama-bin", Value = loadedConfig.Paths.LlamaBin });
                if (!string.IsNullOrWhiteSpace(loadedConfig.Paths.ConvertScript))
                    validationArgs.Add(new CliArg { Name = "convert-script", Value = loadedConfig.Paths.ConvertScript });

                await initializer.Run(validationArgs);
            });

        AnsiConsole.MarkupLine("[bold green][/] Environment validated.");
        AnsiConsole.WriteLine();
    }

    CliHelpers.ValidateCombinationLogicWorks();

    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
}

