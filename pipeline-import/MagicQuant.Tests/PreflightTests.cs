using MagicQuant.Configuration;
using MagicQuant.Models;
using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class PreflightTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"mq-preflight-{Guid.NewGuid():N}");
    private readonly MagicQuantYamlConfig _config = MagicQuantYamlConfig.CreateDefault();

    public PreflightTests()
    {
        Directory.CreateDirectory(_root);
        _config.Paths.ModelDir = Path.Combine(_root, "model");
        _config.Paths.MagicQuantRoot = Path.Combine(_root, "runtime");
        _config.Identity.ArchitectureFamilyName = "test-family";
        Directory.CreateDirectory(_config.Paths.ModelDir);
        File.WriteAllText(Path.Combine(_config.Paths.ModelDir, "test.safetensors"), "fixture");
        File.WriteAllText(Path.Combine(_config.Paths.ModelDir, "config.json"), "{}");
    }

    [Fact]
    public void Valid_preflight_does_not_create_runtime_or_model_work_directories()
    {
        CommandPreflight.Validate("pipeline", _config, []);
        Assert.False(Directory.Exists(_config.Paths.MagicQuantRoot));
        Assert.False(Directory.Exists(Path.Combine(_config.Paths.ModelDir!, "MagicQuant")));
    }

    [Theory]
    [InlineData(".")]
    [InlineData("..")]
    [InlineData("../..")]
    [InlineData("GGUF")]
    [InlineData("GGUF/nested")]
    [InlineData("Benchmarks")]
    public void Output_cannot_destroy_source_or_managed_artifacts(string output)
    {
        _config.Output.OutputDir = output;
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
    }

    [Fact]
    public void Symlinked_output_is_checked_against_the_physical_target()
    {
        if (OperatingSystem.IsWindows()) return; // Windows CI does not grant symlink privilege.
        string link = Path.Combine(_root, "export-link");
        Directory.CreateSymbolicLink(link, _config.Paths.ModelDir!);
        _config.Output.OutputDir = link;
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
    }

    [Theory]
    [InlineData("config.json")]
    [InlineData("test.safetensors")]
    public void Incomplete_models_fail_before_work_starts(string missing)
    {
        File.Delete(Path.Combine(_config.Paths.ModelDir!, missing));
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
        Assert.False(Directory.Exists(_config.Paths.MagicQuantRoot));
    }

    [Fact]
    public void Clone_source_is_required_and_legacy_alias_is_accepted()
    {
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("clone-repository-quants", _config, []));
        CommandPreflight.Validate("clone-repository-quants", _config,
            [new CliArg { Name = "clone-json", Value = Path.Combine(_config.Paths.ModelDir!, "config.json") }]);
    }

    [Fact]
    public void Misspelled_baseline_mode_does_not_silently_enable_all_baselines()
    {
        _config.Baselines.StandardBaselinesMode = "selcted";
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
    }

    [Theory]
    [InlineData("../downloads")]
    [InlineData("/tmp/downloads")]
    [InlineData("a\\b")]
    public void External_cache_name_cannot_escape_model_storage(string name)
    {
        _config.Paths.ExternalBaselineCacheDirName = name;
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
    }

    [Fact]
    public void Export_cannot_clean_the_runtime_toolchain_or_a_scratch_parent()
    {
        _config.Output.OutputDir = Path.Combine(_config.Paths.MagicQuantRoot!, "llama.cpp");
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
        string scratchRoot = Path.Combine(_root, "scratch");
        _config.Paths.ScratchRoots = [scratchRoot];
        _config.Output.OutputDir = scratchRoot;
        Assert.Throws<InvalidOperationException>(() => CommandPreflight.Validate("pipeline", _config, []));
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);
}
