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

using var cancellation = new CancellationTokenSource();
using var cancellationScope = MagicQuant.Runtime.RunCancellation.Use(cancellation.Token);
ConsoleCancelEventHandler onCancel = (_, e) =>
{
    // First Ctrl+C cooperatively unwinds leases/processes; a second uses OS termination.
    e.Cancel = !cancellation.IsCancellationRequested;
    cancellation.Cancel();
};
Console.CancelKeyPress += onCancel;
RunProvenanceService? provenance = null;
string completionStatus = "failed";
string? completionError = null;

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

    CliOptionValidator.Validate(parsedArgs);
    var loaded = MagicQuantYamlLoader.Read(parsedArgs);
    CommandPreflight.Validate(commandInput, loaded.Settings, parsedArgs);
    if (parsedArgs.Any(a => string.Equals(a.Name, "check-config", StringComparison.OrdinalIgnoreCase)))
    {
        foreach (string warning in loaded.Warnings) AnsiConsole.WriteLine(warning);
        AnsiConsole.WriteLine("Configuration and input paths are valid. No runtime setup was performed.");
        return;
    }
    MagicQuantYamlLoader.Apply(loaded);
    var loadedConfig = loaded.Settings;
    provenance = new RunProvenanceService(commandInput, args, loaded);
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

    if (!commandInput.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase))
        await provenance.CaptureToolchainAsync();
    cancellation.Token.ThrowIfCancellationRequested();
    var commandInstance = commandInfo.Factory();
    await commandInstance.Run(parsedArgs);
    if (commandInput.Equals("initialize-llama-cpp", StringComparison.OrdinalIgnoreCase))
        await provenance.CaptureToolchainAsync();
    cancellation.Token.ThrowIfCancellationRequested();
    completionStatus = "completed";
}
catch (OperationCanceledException)
{
    AnsiConsole.WriteLine("Run canceled. Active native work has been stopped.");
    Environment.ExitCode = 130;
    completionStatus = "canceled";
}
catch (Exception ex)
{
    completionError = ex.Message;
    AnsiConsole.WriteException(ex);
    Environment.ExitCode = 1;
}
finally
{
    Console.CancelKeyPress -= onCancel;
    try { provenance?.Complete(completionStatus, completionError); }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
    {
        AnsiConsole.WriteLine($"Could not finalize local run provenance: {ex.Message}");
    }
}
