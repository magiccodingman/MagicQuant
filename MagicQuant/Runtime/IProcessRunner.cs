using System.Diagnostics;

namespace MagicQuant.Runtime;

/// <summary>Inject native execution at IO boundaries without mocking numerical policy.</summary>
public interface IProcessRunner
{
    Task<ProcessResult> RunAsync(ProcessStartInfo start, string? logPath = null,
        Action<string, bool>? onLine = null, CancellationToken ct = default);
}
