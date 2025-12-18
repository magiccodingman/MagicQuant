using System.Runtime.InteropServices;

namespace MagicQuant.Models;

public class LlamaBinaries
{
    public string Bench { get; }
    public string Ppl { get; }
    public string Cli { get; }

    public LlamaBinaries(string root)
    {
        var binDir = Path.Combine(root, "build", "bin");
        Bench = Path.Combine(binDir, "llama-bench");
        Ppl = Path.Combine(binDir, "llama-perplexity");
        Cli = Path.Combine(binDir, "llama-cli");

        // Windows check (.exe)
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Bench += ".exe"; Ppl += ".exe"; Cli += ".exe";
        }
    }

    public void Validate()
    {
        var missing = new List<string>();
        if (!File.Exists(Bench)) missing.Add(Bench);
        if (!File.Exists(Ppl)) missing.Add(Ppl);
        if (!File.Exists(Cli)) missing.Add(Cli);

        if (missing.Any())
            throw new FileNotFoundException($"Missing llama.cpp binaries:\n{string.Join("\n", missing)}");
    }
}