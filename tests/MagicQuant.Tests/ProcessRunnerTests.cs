using System.Diagnostics;
using MagicQuant.Runtime;
using Xunit;

namespace MagicQuant.Tests;

public sealed class ProcessRunnerTests
{
    private static ProcessStartInfo Start(params string[] args)
    {
        var start = new ProcessStartInfo("dotnet");
        start.ArgumentList.Add(typeof(ProcessFixture.Program).Assembly.Location);
        foreach (string arg in args) start.ArgumentList.Add(arg);
        return start;
    }

    [Fact]
    public async Task Arguments_preserve_spaces_quotes_and_shell_metacharacters()
    {
        string[] values = ["folder with spaces", "file'with\"quotes", "$(touch not-a-command)", "C:\\models\\my model", "a;b&c"];
        var result = await new ProcessRunner().RunAsync(Start(["echo", .. values]));
        Assert.True(result.Success);
        Assert.Equal(string.Join(Environment.NewLine, values) + Environment.NewLine, result.StdOut);
    }

    [Fact]
    public async Task Drains_both_full_pipes_and_preserves_nonzero_exit_and_log()
    {
        string log = Path.GetTempFileName();
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            var result = await new ProcessRunner().RunAsync(Start("flood"), log, ct: timeout.Token);
            Assert.Equal(7, result.ExitCode);
            Assert.Contains("stdout-11999", result.StdOut);
            Assert.Contains("stderr-11999", result.StdErr);
            Assert.Contains("stderr-11999", File.ReadAllText(log));
            using var exclusive = new FileStream(log, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(log); }
    }

    [Fact]
    public async Task Cancellation_reaps_native_work_and_releases_log()
    {
        string log = Path.GetTempFileName();
        using var cancel = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        int pid = 0;
        try
        {
            var task = new ProcessRunner().RunAsync(Start("wait"), log, (line, _) =>
            {
                if (line.StartsWith("ready:")) { pid = int.Parse(line[6..]); cancel.Cancel(); }
            }, cancel.Token);
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
            Assert.True(pid > 0);
            Assert.False(IsRunning(pid));
            using var exclusive = new FileStream(log, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(log); }
    }

    [Fact]
    public async Task Command_scope_cancellation_stops_legacy_callers_without_an_explicit_token()
    {
        using var cancel = new CancellationTokenSource();
        using (RunCancellation.Use(cancel.Token))
        {
            cancel.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => new ProcessRunner().RunAsync(Start("wait")));
        }
        Assert.False(RunCancellation.Token.IsCancellationRequested);
    }

    [Fact]
    public async Task Callback_failure_does_not_leave_the_child_running()
    {
        int pid = 0;
        await Assert.ThrowsAsync<IOException>(() => new ProcessRunner().RunAsync(Start("wait"), onLine: (line, _) =>
        {
            pid = int.Parse(line[6..]);
            throw new IOException("Simulated log failure");
        }));
        Assert.False(IsRunning(pid));
    }

    private static bool IsRunning(int pid)
    {
        try { using var process = Process.GetProcessById(pid); return !process.HasExited; }
        catch (ArgumentException) { return false; }
    }
}
