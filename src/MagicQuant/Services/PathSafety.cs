namespace MagicQuant.Services;

/// <summary>Filesystem-aware containment checks for directories that the program may clean.</summary>
public static class PathSafety
{
    private static StringComparison Comparison => OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    public static string ResolvePhysicalPath(string path)
    {
        string full = Path.GetFullPath(path);
        string current = Path.GetPathRoot(full)!;
        foreach (string part in full[current.Length..].Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries))
        {
            current = Path.Combine(current, part);
            var info = new DirectoryInfo(current);
            if (info.LinkTarget != null)
                current = info.ResolveLinkTarget(returnFinalTarget: true)?.FullName
                    ?? throw new IOException($"Cannot resolve directory link '{current}'.");
        }
        return Path.TrimEndingDirectorySeparator(current);
    }

    public static bool Contains(string parent, string child)
    {
        string root = ResolvePhysicalPath(parent);
        string candidate = ResolvePhysicalPath(child);
        return string.Equals(root, candidate, Comparison) ||
               candidate.StartsWith(Path.EndsInDirectorySeparator(root) ? root : root + Path.DirectorySeparatorChar, Comparison);
    }

    public static void ValidateExportDirectory(string output, string model, string runtimeRoot, params string[] managedDirectories)
    {
        // An export cannot contain source/runtime data, nor live inside managed working data.
        foreach (string protectedPath in new[] { model, runtimeRoot, Path.Combine(model, "MagicQuant") }.Concat(managedDirectories))
            if (Contains(output, protectedPath))
                throw new InvalidOperationException($"Output '{output}' contains protected data '{protectedPath}'. Choose a dedicated export directory.");
        foreach (string managed in managedDirectories)
            if (Contains(managed, output))
                throw new InvalidOperationException($"Output '{output}' overlaps managed artifacts '{managed}'. Choose a dedicated export directory.");
        if (File.Exists(output))
            throw new InvalidOperationException($"Output '{output}' is a file, not a directory.");
    }

    public static void ValidateFolderName(string name, string setting)
    {
        if (string.IsNullOrWhiteSpace(name) || name is "." or ".." || name.IndexOfAny(['/', '\\', ':']) >= 0 || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            throw new InvalidOperationException($"{setting} must be one folder/file name, not a path.");
    }
}
