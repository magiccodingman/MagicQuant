using MagicQuant.Helpers;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class ExternalBaselineCacheCleanupService
{
    public async Task<bool> CleanupStaleArtifactsAsync(CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(Cache.ExternalBaselineCacheDirectory))
            return false;

        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new InvalidOperationException(
                "ModelMagicQuantDirectory must be set before cleaning external baseline artifacts.");

        string cacheRoot = Path.GetFullPath(Cache.ExternalBaselineCacheDirectory);
        string modelMagicQuantRoot = Path.GetFullPath(Cache.ModelMagicQuantDirectory);
        ValidateCleanupRoot(cacheRoot, modelMagicQuantRoot);

        if (!Directory.Exists(cacheRoot))
            return false;

        await HardDeleteHelper.DeleteDirectoryIfExistsAsync(cacheRoot, ct);
        AnsiConsole.MarkupLine(
            $"[green]Cleaned abandoned external-baseline artifacts:[/] {Markup.Escape(cacheRoot)}");
        return true;
    }

    internal static void ValidateCleanupRoot(string cacheRoot, string modelMagicQuantRoot)
    {
        string fullCacheRoot = Path.GetFullPath(cacheRoot)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string fullModelRoot = Path.GetFullPath(modelMagicQuantRoot)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string relative = Path.GetRelativePath(fullModelRoot, fullCacheRoot);

        bool escapesModelRoot = Path.IsPathRooted(relative) ||
                                relative.Equals("..", StringComparison.Ordinal) ||
                                relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) ||
                                relative.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal);

        if (relative.Equals(".", StringComparison.Ordinal) || escapesModelRoot)
        {
            throw new InvalidOperationException(
                $"Refusing to clean unsafe external baseline cache path '{fullCacheRoot}'. " +
                $"It must be a child directory of '{fullModelRoot}'.");
        }

        if (Directory.Exists(fullCacheRoot) &&
            (File.GetAttributes(fullCacheRoot) & FileAttributes.ReparsePoint) != 0)
        {
            throw new InvalidOperationException(
                $"Refusing to recursively clean external baseline cache symlink/reparse point '{fullCacheRoot}'.");
        }
    }
}
