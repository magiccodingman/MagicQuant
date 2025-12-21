using System.Text.Json;
using Spectre.Console;

namespace MagicQuant.Helpers;

public static class JsonHelper
{
    // Priority list of keys to look for
    private static readonly List<string> DtypeKeys = new() 
    { 
        "torch_dtype", 
        "dtype", 
        "prec", 
        "precision" 
    };

    public static void DetectAndSetTorchType(string modelDir)
    {
        string configPath = Path.Combine(modelDir, "config.json");

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"Could not find 'config.json' in {modelDir}");
        }

        try
        {
            string jsonContent = File.ReadAllText(configPath);
            using JsonDocument doc = JsonDocument.Parse(jsonContent);

            // Recursive search for the key
            string? dtypeValue = FindKeyRecursive(doc.RootElement, DtypeKeys);

            if (string.IsNullOrWhiteSpace(dtypeValue))
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Could not find 'torch_dtype' in config.json. Defaulting to [bold]BF16[/].");
                Cache.TorchType = Cache.MainTorchType.BF16;
                return;
            }

            // Parse the value
            Cache.TorchType = ParseTorchType(dtypeValue);
            AnsiConsole.MarkupLine($"[grey]Detected Model Type:[/] [cyan]{Cache.TorchType}[/] (from '{dtypeValue}')");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error parsing config.json:[/] {ex.Message}");
            // Fail safe or throw depending on strictness. 
            // Usually safe to default if we assume modern models.
            Cache.TorchType = Cache.MainTorchType.BF16; 
        }
    }

    private static string? FindKeyRecursive(JsonElement element, List<string> targetKeys)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            // 1. Check current level first (Optimization)
            foreach (var prop in element.EnumerateObject())
            {
                if (targetKeys.Contains(prop.Name, StringComparer.OrdinalIgnoreCase) && 
                    prop.Value.ValueKind == JsonValueKind.String)
                {
                    return prop.Value.GetString();
                }
            }

            // 2. Recurse into children
            foreach (var prop in element.EnumerateObject())
            {
                // Skip if not object or array to save time
                if (prop.Value.ValueKind == JsonValueKind.Object || prop.Value.ValueKind == JsonValueKind.Array)
                {
                    string? found = FindKeyRecursive(prop.Value, targetKeys);
                    if (found != null) return found;
                }
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in element.EnumerateArray())
            {
                string? found = FindKeyRecursive(item, targetKeys);
                if (found != null) return found;
            }
        }

        return null;
    }

    private static Cache.MainTorchType ParseTorchType(string value)
    {
        // Normalize
        string v = value.ToLowerInvariant().Trim();

        return v switch
        {
            "bfloat16" => Cache.MainTorchType.BF16,
            "float16"  => Cache.MainTorchType.F16,
            "float32"  => Cache.MainTorchType.F32,
            _ => Cache.MainTorchType.BF16 // Default fallback
        };
    }
}