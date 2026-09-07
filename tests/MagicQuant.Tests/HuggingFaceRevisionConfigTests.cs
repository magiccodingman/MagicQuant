using MagicQuant.Configuration;
using Xunit;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Tests;

public sealed class HuggingFaceRevisionConfigTests
{
    [Fact]
    public void CustomRepositoryRevision_DeserializesPinnedCommit()
    {
        const string yaml = """
            baselines:
              custom_repositories:
                - repo_id: owner/model-GGUF
                  revision: 313447f257f7ebde0b968e4778feef774546ed81
            """;

        var config = Deserialize(yaml);
        var repository = Assert.Single(config.Baselines.CustomRepositories);

        Assert.Equal("owner/model-GGUF", repository.RepoId);
        Assert.Equal("313447f257f7ebde0b968e4778feef774546ed81", repository.Revision);
    }

    [Fact]
    public void CustomRepositoryRevision_RemainsOptional()
    {
        const string yaml = """
            baselines:
              custom_repositories:
                - repo_id: owner/model-GGUF
            """;

        var repository = Assert.Single(Deserialize(yaml).Baselines.CustomRepositories);

        Assert.Null(repository.Revision);
    }

    private static MagicQuantYamlConfig Deserialize(string yaml)
    {
        return new DeserializerBuilder()
            .IgnoreUnmatchedProperties()
            .WithNamingConvention(UnderscoredNamingConvention.Instance)
            .Build()
            .Deserialize<MagicQuantYamlConfig>(yaml);
    }
}
