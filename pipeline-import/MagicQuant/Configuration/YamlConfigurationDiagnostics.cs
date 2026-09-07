using System.Reflection;
using YamlDotNet.RepresentationModel;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Configuration;

/// <summary>Checks document keys against the typed schema while leaving free-form metadata alone.</summary>
public static class YamlConfigurationDiagnostics
{
    public static IReadOnlyList<string> Inspect(string yaml)
    {
        var stream = new YamlStream();
        stream.Load(new StringReader(yaml));
        if (stream.Documents.Count > 1)
            throw new InvalidOperationException("Expected one YAML configuration document.");
        var warnings = new List<string>();
        if (stream.Documents.Count == 1)
            InspectNode(stream.Documents[0].RootNode, typeof(MagicQuantYamlConfig), "", warnings);
        return warnings;
    }

    private static void InspectNode(YamlNode node, Type type, string path, List<string> warnings)
    {
        // Dictionary keys (frontmatter, GPU indices) belong to the user, not the C# schema.
        if (type == typeof(object) || type.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IDictionary<,>)))
            return;
        if (node is YamlSequenceNode sequence && type.IsGenericType)
        {
            for (int i = 0; i < sequence.Children.Count; i++)
                InspectNode(sequence.Children[i], type.GetGenericArguments()[0], $"{path}[{i}]", warnings);
            return;
        }
        if (node is not YamlMappingNode mapping)
            return;
        var properties = type.GetProperties()
            .Where(p => p.GetCustomAttribute<YamlIgnoreAttribute>() == null)
            .ToDictionary(p => UnderscoredNamingConvention.Instance.Apply(p.Name), StringComparer.Ordinal);
        foreach (var entry in mapping.Children)
        {
            string key = ((YamlScalarNode)entry.Key).Value ?? "";
            string fullKey = path.Length == 0 ? key : $"{path}.{key}";
            if (fullKey == "flags.force_relearn_baseline_tensor_mappings")
                throw new InvalidOperationException("Removed destructive option flags.force_relearn_baseline_tensor_mappings. Use targeted learning options instead.");
            if (!properties.TryGetValue(key, out var property))
                warnings.Add($"Unknown or inactive YAML setting '{fullKey}' (line {entry.Key.Start.Line}). It will be ignored.");
            else
                InspectNode(entry.Value, property.PropertyType, fullKey, warnings);
        }
    }
}
