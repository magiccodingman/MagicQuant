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

        bool preserveResumableDownloads = Config.Current.Baselines.CustomRepositories
            .Any(x => x.Enabled && x.ResumeOrRetryDownloads);

        if (preserveResumableDownloads)
        {
            int removedTransientFiles = await CleanupTransientTopLevelFilesAsync(cacheRoot, ct);
            AnsiConsole.MarkupLine(
                $"[green]Preserved resumable external-baseline downloads:[/] {Markup.Escape(cacheRoot)} " +
                $"[grey](removed transient files={removedTransientFiles:N0})[/]");
            return removedTransientFiles > 0;
        }

        await HardDeleteHelper.DeleteDirectoryIfExistsAsync(cacheRoot, ct);
        AnsiConsole.MarkupLine(
            $"[green]Cleaned abandoned external-baseline artifacts:[/] {Markup.Escape(cacheRoot)}");
        return true;
    }

    private static async Task<int> CleanupTransientTopLevelFilesAsync(string cacheRoot, CancellationToken ct)
    {
        int removed = 0;
        foreach (string file in Directory.EnumerateFiles(cacheRoot, "*", SearchOption.TopDirectoryOnly))
        {
            ct.ThrowIfCancellationRequested();

            string fileName = Path.GetFileName(file);
            bool isCompletedGguf = fileName.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            if (isCompletedGguf)
                continue;

            await HardDeleteHelper.DeleteFileIfExistsAsync(file);
            removed++;
        }

        return removed;
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
