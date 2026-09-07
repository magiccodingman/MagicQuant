using System.Diagnostics;
using MagicQuant.Commands;
using Xunit;

namespace MagicQuant.Tests;

/// <summary>Exercise the real executable so startup cannot hide side effects behind command help.</summary>
public sealed class CliStartupTests
{
    [Theory]
    [InlineData("")]
    [InlineData("help")]
    [InlineData("--help")]
    [InlineData("-h")]
    [InlineData("pipeline")]
    [InlineData("evolution")]
    [InlineData("build-hybrids")]
    [InlineData("clone-repository-quants")]
    [InlineData("validate-predictions")]
    [InlineData("initialize-llama-cpp")]
    public async Task Help_succeeds_without_config_or_runtime_artifacts(string command)
    {
        string[] arguments = command switch
        {
            "" => [],
            "help" or "--help" or "-h" => [command],
            _ => [command, "--help", "--config", "does-not-exist.yaml"]
        };
        var result = await RunAsync(arguments);
        Assert.True(result.ExitCode == 0, result.Output);
        Assert.DoesNotContain("Using config:", result.Output);
        Assert.DoesNotContain("Checking environment", result.Output);
        Assert.Empty(result.CreatedFiles);
    }

    [Fact]
    public async Task Unknown_command_is_escaped_and_returns_usage_error()
    {
        var result = await RunAsync(["[invalid]"]);
        Assert.Equal(2, result.ExitCode);
        Assert.Contains("does not exist", result.Output);
        Assert.Empty(result.CreatedFiles);
    }

    [Fact]
    public async Task Missing_config_returns_failure_before_setup()
    {
        var result = await RunAsync(["pipeline", "--config", "missing.yaml"]);
        Assert.Equal(1, result.ExitCode);
        Assert.Contains("config file was not found", result.Output);
        Assert.Empty(result.CreatedFiles);
    }

    [Fact]
    public void Historical_alias_uses_the_same_pipeline_implementation()
    {
        var commands = CommandCatalog.Create();
        Assert.IsType<QuantizationPipeline>(commands["pipeline"].Factory());
        Assert.IsType<QuantizationPipeline>(commands["EVOLUTION"].Factory());
        Assert.IsAssignableFrom<QuantizationPipeline>(new Evolution());
    }

    [Fact]
    public async Task Invalid_model_fails_before_config_application_or_dependency_setup()
    {
        var result = await RunAsync(["pipeline", "--config", "bad.yaml"], directory =>
            File.WriteAllText(Path.Combine(directory, "bad.yaml"), "paths:\n  magic_quant_root: runtime-must-not-exist\n  model_dir: missing-model\n"));
        Assert.Equal(1, result.ExitCode);
        Assert.DoesNotContain("Using config:", result.Output);
        Assert.DoesNotContain("Checking environment", result.Output);
        Assert.Single(result.CreatedFiles);
    }

    [Fact]
    public async Task Check_config_does_not_initialize_or_clean_runtime_state()
    {
        var result = await RunAsync(["initialize-llama-cpp", "--config", "check.yaml", "--check-config", "--strict-config"], directory =>
            File.WriteAllText(Path.Combine(directory, "check.yaml"), "paths:\n  magic_quant_root: runtime-must-not-exist\n"));
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("No runtime setup was performed", result.Output);
        Assert.Single(result.CreatedFiles);
    }

    private static async Task<(int ExitCode, string Output, string[] CreatedFiles)> RunAsync(string[] args, Action<string>? setup = null)
    {
        string directory = Path.Combine(Path.GetTempPath(), $"mq-cli-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        setup?.Invoke(directory);
        try
        {
            var start = new ProcessStartInfo("dotnet")
            {
                WorkingDirectory = directory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            start.ArgumentList.Add(typeof(QuantizationPipeline).Assembly.Location);
            foreach (string arg in args)
                start.ArgumentList.Add(arg);
            using var process = Process.Start(start)!;
            Task<string> stdout = process.StandardOutput.ReadToEndAsync();
            Task<string> stderr = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            try
            {
                await process.WaitForExitAsync(timeout.Token);
            }
            catch (OperationCanceledException)
            {
                process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync();
                throw;
            }
            return (process.ExitCode, await stdout + await stderr, Directory.GetFileSystemEntries(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
