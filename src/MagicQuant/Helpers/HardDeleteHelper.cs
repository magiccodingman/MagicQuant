namespace MagicQuant.Helpers;

public static class HardDeleteHelper
{
    public static async Task DeleteDirectoryIfExistsAsync(
        string? directory,
        CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory))
            return;

        foreach (var file in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
        {
            ct.ThrowIfCancellationRequested();
            await DeleteFileIfExistsAsync(file);
        }

        foreach (var sub in Directory.EnumerateDirectories(directory, "*", SearchOption.AllDirectories)
                     .OrderByDescending(x => x.Length))
        {
            ct.ThrowIfCancellationRequested();
            if (Directory.Exists(sub))
                Directory.Delete(sub, recursive: false);
        }

        if (Directory.Exists(directory))
            Directory.Delete(directory, recursive: false);
    }

    public static async Task DeleteFileIfExistsAsync(
        string? path,
        int maxAttempts = 6,
        int delayMs = 500)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;

        Exception? lastError = null;

        for (int attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                var attributes = File.GetAttributes(path);
                if ((attributes & FileAttributes.ReadOnly) != 0)
                {
                    File.SetAttributes(path, attributes & ~FileAttributes.ReadOnly);
                }

                File.Delete(path);

                if (!File.Exists(path))
                    return;
            }
            catch (IOException ex)
            {
                lastError = ex;
            }
            catch (UnauthorizedAccessException ex)
            {
                lastError = ex;
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            await Task.Delay(delayMs);
        }

        throw new IOException(
            $"Failed to hard delete file '{path}' after {maxAttempts} attempts.",
            lastError);
    }
}
