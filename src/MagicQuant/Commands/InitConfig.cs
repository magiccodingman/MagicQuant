using MagicQuant.Models;

namespace MagicQuant.Commands;

/// <summary>Copies the packaged profile to a user-owned file without initializing runtime state.</summary>
public sealed class InitConfig : ICommand
{
    public static void ValidateTokens(string[] tokens)
    {
        bool valid = tokens.Length == 0
            || (tokens.Length == 2 && tokens[0].Equals("--output", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(tokens[1]) && !tokens[1].StartsWith("--", StringComparison.Ordinal))
            || (tokens.Length == 1 && tokens[0].StartsWith("--output=", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(tokens[0][9..]));
        if (!valid) throw new ArgumentException("Usage: magicquant init-config [--output config.yaml]");
    }

    public async Task Run(List<CliArg> args)
    {
        if (args.Any(a => string.Equals(a.Name, "help", StringComparison.OrdinalIgnoreCase)))
        {
            Console.WriteLine("Usage: magicquant init-config [--output config.yaml]");
            Console.WriteLine("Copy the complete bundled profile. Existing files are never overwritten.");
            return;
        }
        if (args.Any(a => !string.Equals(a.Name, "output", StringComparison.OrdinalIgnoreCase)) || args.Count > 1)
            throw new ArgumentException("init-config accepts only one --output path.");
        if (args.Count == 1 && string.IsNullOrWhiteSpace(args[0].Value))
            throw new ArgumentException("--output requires a file path.");
        string destination = Path.GetFullPath(args.Count == 0 ? "config.yaml" : args[0].Value!);
        using var source = File.OpenRead(Path.Combine(AppContext.BaseDirectory, "config.default.yaml"));
        using var target = new FileStream(destination, FileMode.CreateNew, FileAccess.Write);
        await source.CopyToAsync(target);
        Console.WriteLine($"Created {destination}. Edit model, architecture, output and scratch settings before running.");
    }
}
