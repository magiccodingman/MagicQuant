using System.Text.Json;
using MagicQuant.Helpers;
using MQ.DB;

namespace MagicQuant.Services;

public enum ScratchArtifactKind
{
    QuantizedSample,
    PureQ8Probe,
    ExternalBaselineRebuild,
    ExternalBaselineNormalizedSample,
    ExportTemp,
    MetadataRead,
    Other
}

public sealed class ScratchArtifactLease : IAsyncDisposable
{
    private readonly Func<ScratchArtifactLease, ValueTask> _dispose;
    private int _disposed;

    internal ScratchArtifactLease(
        Guid leaseId,
        ScratchArtifactKind kind,
        string scratchRoot,
        string leaseDirectory,
        string ggufPath,
        string primaryLogPath,
        Func<ScratchArtifactLease, ValueTask> dispose)
    {
        LeaseId = leaseId;
        Kind = kind;
        ScratchRoot = scratchRoot;
        LeaseDirectory = leaseDirectory;
        GgufPath = ggufPath;
        PrimaryLogPath = primaryLogPath;
        OwnsGgufLifecycle = true;
        _dispose = dispose;
    }

    public Guid LeaseId { get; }
    public ScratchArtifactKind Kind { get; }
    public string ScratchRoot { get; }
    public string LeaseDirectory { get; }
    public string GgufPath { get; }
    public string PrimaryLogPath { get; }
    public bool OwnsGgufLifecycle { get; private set; }

    public void PreserveOutput() => OwnsGgufLifecycle = false;

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        await _dispose(this);
    }
}

public sealed class ScratchStorageService
{
    private const string ScratchFolderName = ".MagicQuant_tmp";
    private readonly ModelArtifactPathService? _paths;
    private readonly string _modelNamespace;
    private readonly string _quantLogDir;
    private readonly IReadOnlyList<string> _roots;
    private readonly SemaphoreSlim[] _rootLocks;
    private readonly SemaphoreSlim _availableRoots;
    private int _cursor = -1;

    public ScratchStorageService(ModelArtifactPathService? paths = null)
    {
        _paths = paths;
        _modelNamespace = paths?.ScratchModelNamespace ?? "global";
        _quantLogDir = paths?.QuantizationLogsDir ?? Path.Combine(Cache.MagicQuantDirectory ?? Path.GetTempPath(), "Logs", "Quantization");
        var configured = Cache.ScratchRoots ?? [];

        var fallbackRoot = paths != null
            ? Path.Combine(paths.ModelMagicQuantDirectory, ScratchFolderName)
            : Path.Combine(Cache.MagicQuantDirectory ?? Path.GetTempPath(), ScratchFolderName);

        _roots = configured.Count == 0
            ? [fallbackRoot]
            : configured.Select(x => Path.GetFullPath(x)).ToList();

        _rootLocks = _roots.Select(_ => new SemaphoreSlim(1, 1)).ToArray();
        _availableRoots = new SemaphoreSlim(_roots.Count, _roots.Count);
    }

    public int WriterCapacity => _roots.Count;
    public IReadOnlyList<string> ConfiguredScratchRoots => _roots;

    public async Task<ScratchArtifactLease> AcquireAsync(
        ScratchArtifactKind kind,
        string artifactBaseName,
        string extension = ".gguf",
        CancellationToken ct = default)
    {
        await _availableRoots.WaitAsync(ct);

        int rootIndex = -1;
        try
        {
            while (rootIndex < 0)
            {
                int start = (Interlocked.Increment(ref _cursor) % _roots.Count + _roots.Count) % _roots.Count;
                for (int i = 0; i < _roots.Count; i++)
                {
                    int idx = (start + i) % _roots.Count;
                    if (_rootLocks[idx].Wait(0))
                    {
                        rootIndex = idx;
                        break;
                    }
                }

                if (rootIndex < 0)
                    await Task.Delay(20, ct);
            }

            var leaseId = Guid.NewGuid();
            string root = _roots[rootIndex];
            string tmpRoot = root.EndsWith(ScratchFolderName, StringComparison.OrdinalIgnoreCase)
                ? root
                : Path.Combine(root, ScratchFolderName);

            string leaseDir = Path.Combine(tmpRoot, _modelNamespace, leaseId.ToString("N"));
            Directory.CreateDirectory(leaseDir);
            Directory.CreateDirectory(_quantLogDir);

            string safeBase = ModelArtifactPathService.MakeSafeFileComponent(artifactBaseName);
            string ggufPath = Path.Combine(leaseDir, safeBase + extension);
            string logPath = _paths?.GetQuantizationLogPath(safeBase, leaseId) ?? Path.Combine(_quantLogDir, $"{safeBase}-{leaseId:N}.quantize.log");

            var marker = new
            {
                lease_id = leaseId,
                process_id = Environment.ProcessId,
                started_utc = DateTime.UtcNow,
                kind = kind.ToString(),
                artifact_name = artifactBaseName,
                gguf_path = ggufPath
            };
            await File.WriteAllTextAsync(Path.Combine(leaseDir, "lease.json"), JsonSerializer.Serialize(marker), ct);

            return new ScratchArtifactLease(
                leaseId,
                kind,
                root,
                leaseDir,
                ggufPath,
                logPath,
                async lease =>
                {
                    try
                    {
                        if (lease.OwnsGgufLifecycle && Directory.Exists(lease.LeaseDirectory))
                            await HardDeleteHelper.DeleteDirectoryIfExistsAsync(lease.LeaseDirectory);
                    }
                    finally
                    {
                        _rootLocks[rootIndex].Release();
                        _availableRoots.Release();
                    }
                });
        }
        catch
        {
            if (rootIndex >= 0)
                _rootLocks[rootIndex].Release();
            _availableRoots.Release();
            throw;
        }
    }

    public async Task CleanupStaleScratchArtifactsAsync(CancellationToken ct = default)
    {
        foreach (var root in _roots)
        {
            ct.ThrowIfCancellationRequested();
            string tmpRoot = root.EndsWith(ScratchFolderName, StringComparison.OrdinalIgnoreCase)
                ? root
                : Path.Combine(root, ScratchFolderName);

            if (!Directory.Exists(tmpRoot))
                continue;

            foreach (var child in Directory.EnumerateDirectories(tmpRoot))
            {
                ct.ThrowIfCancellationRequested();
                await HardDeleteHelper.DeleteDirectoryIfExistsAsync(child);
            }
        }
    }
}
