using System.Text;
using MagicQuant.Models;
using MQ.DB;
using Spectre.Console;

namespace MagicQuant.Services;

public sealed class CloneReadmeGenerationService
{
    public async Task<string> GenerateAsync(
        string outputDirectory,
        string modelName,
        string sourceDescription,
        bool sourceWasHuggingFaceRepo,
        IReadOnlyCollection<CloneArtifactBuildRecord> records,
        CancellationToken ct = default)
    {
        Directory.CreateDirectory(outputDirectory);
        string path = Path.Combine(outputDirectory, "README.md");

        var sb = new StringBuilder();

        sb.AppendLine($"# {modelName} - MagicQuant Clone Build");
        sb.AppendLine();
        sb.AppendLine("This repository was built in **MagicQuant repository clone mode**.");
        sb.AppendLine();
        sb.AppendLine("That means these GGUF files copied exact tensor quantization configurations from an existing MagicQuant-compatible release, then rebuilt and benchmarked those tensor maps against this model locally.");
        sb.AppendLine();
        sb.AppendLine("> Important: this model did **not** run through the full MagicQuant probing/evolution/search pipeline by itself. It reused tensor configurations from another MagicQuant release and then generated fresh local benchmark metadata for this output.");
        sb.AppendLine();
        sb.AppendLine("## Clone source");
        sb.AppendLine();
        if (sourceWasHuggingFaceRepo)
            sb.AppendLine($"- Source Hugging Face repository: `{sourceDescription}`");
        else
            sb.AppendLine($"- Source clone JSON: `{sourceDescription}`");
        sb.AppendLine($"- Clone config file: [`{CloneConfigManifestGenerationService.FileName}`](./../../resolve/main/{CloneConfigManifestGenerationService.FileName}?download=true)");
        sb.AppendLine();
        sb.AppendLine("## Downloadable outputs");
        sb.AppendLine();
        sb.AppendLine("| Name | Provider | Quant Family | KLD | PPL Δ % | Size (GB) | Download |");
        sb.AppendLine("|---|---|---|---:|---:|---:|---|");

        foreach (var record in records.OrderBy(x => x.Kld ?? double.MaxValue).ThenBy(x => x.ActualSizeBytes))
        {
            var a = record.ManifestArtifact;
            string name = EscapePipe(string.IsNullOrWhiteSpace(a.ShortName) ? Path.GetFileNameWithoutExtension(a.FileName) : a.ShortName);
            string provider = EscapePipe(string.IsNullOrWhiteSpace(a.Provider) ? "Cloned config" : a.Provider);
            string family = EscapePipe(string.IsNullOrWhiteSpace(a.QuantFamily) ? a.BaseQuant : a.QuantFamily);
            string kld = record.Kld.HasValue ? record.Kld.Value.ToString("0.000000") : "n/a";
            string ppl = record.PplDeltaPercent.HasValue ? $"{record.PplDeltaPercent.Value:0.000}%" : "n/a";
            string size = (record.ActualSizeBytes / 1024d / 1024d / 1024d).ToString("0.00");
            sb.AppendLine($"| {name} | {provider} | {family} | {kld} | {ppl} | {size} | [Link](./../../resolve/main/{Uri.EscapeDataString(a.FileName)}?download=true) |");
        }

        sb.AppendLine();
        sb.AppendLine("## Reproducibility");
        sb.AppendLine();
        sb.AppendLine($"The file `{CloneConfigManifestGenerationService.FileName}` stores the exact `tensor name -> quant type` map used to rebuild each GGUF. A future clone run can use that JSON directly with `--source-json`, or a Hugging Face repository containing that file with `--source-repo`.");
        sb.AppendLine();
        sb.AppendLine("## Support");
        sb.AppendLine();
        sb.AppendLine("I’m a solo developer working full time for myself to achieve my dream. If you like any of my work, buying me a coffee is always appreciated. Otherwise, good vibes are also accepted as legal tender.");
        sb.AppendLine();
        sb.AppendLine("[Click here to see ways to support](https://sayou.biz/support) - BTC, Paypal, GitHub sponsors.");
        sb.AppendLine();

        await File.WriteAllTextAsync(path, sb.ToString(), ct);
        AnsiConsole.MarkupLine($"[green]Clone README generated:[/] {Markup.Escape(path)}");
        return path;
    }

    private static string EscapePipe(string value)
        => (value ?? string.Empty).Replace("|", "\\|");
}
