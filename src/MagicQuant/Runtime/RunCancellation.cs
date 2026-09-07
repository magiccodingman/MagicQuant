namespace MagicQuant.Runtime;

/// <summary>
/// Cancellation for the current async command scope. Legacy service APIs without a
/// token still stop native work; new APIs should also accept explicit caller tokens.
/// This value flows into tasks and is restored when the command finishes.
/// </summary>
public static class RunCancellation
{
    private static readonly AsyncLocal<CancellationToken> Ambient = new();
    public static CancellationToken Token => Ambient.Value;

    public static IDisposable Use(CancellationToken token)
    {
        var previous = Ambient.Value;
        Ambient.Value = token;
        return new Scope(previous);
    }

    private sealed class Scope(CancellationToken previous) : IDisposable
    {
        public void Dispose() => Ambient.Value = previous;
    }
}
