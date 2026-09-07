using MagicQuant.Models;

namespace MagicQuant.Services;

/// <summary>
/// Compatibility wrapper kept so existing call sites can move over gradually.
/// The actual README body/table/frontmatter logic is centralized in ReadmeGenerationService.
/// </summary>
public sealed class CloneReadmeGenerationService
{
    private readonly ReadmeGenerationService _readmeGenerationService = new();

    public Task<string> GenerateAsync(
        string outputDirectory,
        string modelName,
        string sourceDescription,
        bool sourceWasHuggingFaceRepo,
        IReadOnlyCollection<CloneArtifactBuildRecord> records,
        IReadOnlyCollection<string>? archivedManifestFileNames = null,
        CancellationToken ct = default)
    {
        return _readmeGenerationService.GenerateCloneAsync(
            outputDirectory,
            modelName,
            sourceDescription,
            sourceWasHuggingFaceRepo,
            records,
            archivedManifestFileNames,
            ct);
    }
}
