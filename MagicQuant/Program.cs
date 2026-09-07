using MagicQuant.Commands;
using MagicQuant.Configuration;
using MagicQuant.Helpers;
using MagicQuant.Models;
using MagicQuant.Services;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

var commands = CommandCatalog.Create();

if (args.Length == 0 || CommandCatalog.IsHelp(args[0]))
{
    CliHelpers.ShowHelp(commands);
    return;
}

string commandInput = args[0];

if (!commands.TryGetValue(commandInput, out var commandInfo))
{
    AnsiConsole.MarkupLine($"[red]Error:[/] The command [yellow]'{Markup.Escape(commandInput)}'[/] does not exist.");
    CliHelpers.ShowHelp(commands);
    Environment.ExitCode = 2;
    return;
}

List<CliArg> parsedArgs = CliHelpers.ParseArguments(args.Skip(1));

try
{
    // Help is a read-only operation: do not load config, clean caches, install
    // dependencies, or open databases just to explain a command.
    if (parsedArgs.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)) ||
        args.Skip(1).Any(a => a == "-h"))
    {
        await commandInfo.Factory().Run([new CliArg { Name = "help", Value = string.Empty }]);
        return;
    }

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

    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
}
catch (Exception ex)
{
    AnsiConsole.WriteException(ex);
    Environment.ExitCode = 1;
}
