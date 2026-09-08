using System.Diagnostics;
using System.Text;

namespace MagicQuant.Runtime;

public sealed record ProcessResult(int ExitCode, string StdOut, string StdErr)
{
    public bool Success => ExitCode == 0;
    public string CombinedOutput => StdOut + StdErr;
}

/// <summary>
/// Owns native process lifetime, drains both pipes concurrently, and closes logs on
/// every exit path. Cancellation kills and reaps the child tree before callers may
/// dispose scratch leases. Exit codes remain explicit so each caller owns retry policy.
/// </summary>
public sealed class ProcessRunner : IProcessRunner
{
    public async Task<ProcessResult> RunAsync(ProcessStartInfo start, string? logPath = null,
        Action<string, bool>? onLine = null, CancellationToken ct = default)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(ct, RunCancellation.Token);
        ct = cancellation.Token;
        ct.ThrowIfCancellationRequested();
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        using var log = logPath == null ? null : new StreamWriter(new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
        using var process = new Process { StartInfo = start };
        if (!process.Start()) throw new InvalidOperationException($"Failed to start '{start.FileName}'.");
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        object sync = new();

        async Task DrainAsync(StreamReader reader, StringBuilder buffer, bool error)
        {
            try
            {
                while (await reader.ReadLineAsync() is { } line)
                {
                    lock (sync)
                    {
                        buffer.AppendLine(line);
                        log?.WriteLine(line);
                        onLine?.Invoke(line, error);
                    }
                }
            }
            catch
            {
                // Wake the other drain when this pipe/callback fails.
                try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
                throw;
            }
        }

        Task drains = Task.WhenAll(DrainAsync(process.StandardOutput, stdout, false), DrainAsync(process.StandardError, stderr, true));
        try
        {
            // A logging callback or pipe failure must terminate the child as well,
            // rather than letting it hang forever with an undrained output pipe.
            Task exited = process.WaitForExitAsync(ct);
            Task completed = await Task.WhenAny(exited, drains);
            if (completed == drains) await drains;
            await exited;
            await drains.WaitAsync(ct);
            ct.ThrowIfCancellationRequested();
            return new ProcessResult(process.ExitCode, stdout.ToString(), stderr.ToString());
        }
        catch
        {
            if (!process.HasExited)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { /* The child exited between the check and kill. */ }
            }
            await process.WaitForExitAsync(CancellationToken.None);
            try { await drains; } catch { /* Preserve the original cancellation/pipe failure. */ }
            throw;
        }
    }
}
