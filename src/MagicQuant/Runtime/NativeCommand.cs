using System.Diagnostics;
using System.Text.Json;

namespace MagicQuant.Runtime;

/// <summary>Executable and literal argv; never interpreted by a shell.</summary>
public sealed record NativeCommand(string Executable, IReadOnlyList<string> Arguments)
{
    public ProcessStartInfo CreateStartInfo(IReadOnlyDictionary<string, string>? environment = null)
    {
        var start = new ProcessStartInfo(Executable);
        foreach (string arg in Arguments) start.ArgumentList.Add(arg);
        if (environment != null)
            foreach (var (key, value) in environment) start.Environment[key] = value;
        return start;
    }

    // Diagnostic representation only, not a command to execute or shell-escape.
    public override string ToString() => JsonSerializer.Serialize(new { Executable, Arguments });
}
