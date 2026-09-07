using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class OutputPathTests
{
    private static readonly string ModelWork = Path.Combine(Path.GetTempPath(), "model with spaces", "MagicQuant");

    [Fact]
    public void Default_destinations_preserve_existing_command_layouts()
    {
        Assert.Equal(Path.Combine(ModelWork, "Final_Outputs"), OutputPathService.Pipeline(ModelWork, null));
        Assert.Equal(Path.Combine(ModelWork, "FinalOutput"), OutputPathService.Clone(ModelWork, null, null));
        Assert.Equal(Path.Combine(ModelWork, "PredictionValidation"), OutputPathService.PredictionValidation(ModelWork, null, null));
    }

    [Fact]
    public void Relative_pipeline_output_is_model_local_but_clone_output_is_cwd_relative()
    {
        Assert.Equal(Path.Combine(ModelWork, "exports"), OutputPathService.Pipeline(ModelWork, "exports"));
        Assert.Equal(Path.GetFullPath("exports"), OutputPathService.Clone(ModelWork, null, "exports"));
    }

    [Fact]
    public void Absolute_pipeline_output_overrides_model_directory()
    {
        string output = Path.Combine(Path.GetTempPath(), "other exports");
        Assert.Equal(output, OutputPathService.Pipeline(ModelWork, output));
    }

    [Fact]
    public void Explicit_validation_destination_does_not_add_a_subdirectory()
    {
        Assert.Equal(Path.GetFullPath("reports"), OutputPathService.PredictionValidation(ModelWork, "reports", "ignored"));
        Assert.Equal(Path.Combine(Path.GetFullPath("exports"), "PredictionValidation"),
            OutputPathService.PredictionValidation(ModelWork, null, "exports"));
        Assert.Equal(Path.GetFullPath("exports"), OutputPathService.Clone(ModelWork, "exports", "ignored"));
    }

    [Theory]
    [InlineData("magicquant.final-survivors.json")]
    [InlineData("magicquant-manifest/magicquant.final-survivors.json")]
    [InlineData("magicquant-manifest\\magicquant.final-survivors.json")]
    public void Manifest_links_have_one_directory_prefix_and_forward_slashes(string file)
    {
        Assert.Equal("magicquant-manifest/magicquant.final-survivors.json", MagicQuantManifestPathService.RelativeManifestPath(file));
    }

    [Fact]
    public void Manifest_directory_creation_is_idempotent_with_trailing_separator()
    {
        string root = Path.Combine(Path.GetTempPath(), $"mq-manifest-{Guid.NewGuid():N}");
        try
        {
            string manifest = MagicQuantManifestPathService.EnsureManifestDirectory(root);
            Assert.Equal(manifest, MagicQuantManifestPathService.EnsureManifestDirectory(manifest + Path.DirectorySeparatorChar));
            Assert.Empty(Directory.GetDirectories(manifest));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
