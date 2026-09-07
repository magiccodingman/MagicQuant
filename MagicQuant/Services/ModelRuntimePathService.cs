using MQ.DB;

namespace MagicQuant.Services;

/// <summary>
/// Initializes model-scoped cache locations after the command has selected its source model.
/// </summary>
public static class ModelRuntimePathService
{
    public static void InitializeForCurrentModel()
    {
        if (string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            throw new InvalidOperationException("Cache.ModelMagicQuantDirectory must be set before initializing runtime model paths.");

        string externalName = Config.Current.Paths.ExternalBaselineCacheDirName;
        if (string.IsNullOrWhiteSpace(externalName))
            externalName = "ExternalBaselines";

        Cache.ExternalBaselineCacheDirectory = Path.Combine(Cache.ModelMagicQuantDirectory, externalName);
    }
}
