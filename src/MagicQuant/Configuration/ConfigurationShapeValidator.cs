using System.Collections;
using System.Reflection;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MagicQuant.Configuration;

/// <summary>Rejects malformed state before normalization or global caches are changed.</summary>
public static class ConfigurationShapeValidator
{
    public static void Validate(MagicQuantYamlConfig config) => ValidateObject(config, "");

    private static void ValidateObject(object value, string path)
    {
        if (value is double number && !double.IsFinite(number))
            throw new InvalidOperationException($"Configuration '{path}' must be finite.");
        if (value is string || value.GetType().IsValueType)
            return;
        if (value is IDictionary dictionary)
        {
            foreach (DictionaryEntry entry in dictionary)
                if (entry.Value != null) ValidateObject(entry.Value, $"{path}[{entry.Key}]");
            return;
        }
        if (value is IEnumerable sequence)
        {
            foreach (var item in sequence)
            {
                if (item == null) throw new InvalidOperationException($"Configuration '{path}' cannot contain null items.");
                ValidateObject(item, path);
            }
            return;
        }
        var nullability = new NullabilityInfoContext();
        foreach (var property in value.GetType().GetProperties().Where(p => p.GetCustomAttribute<YamlIgnoreAttribute>() == null))
        {
            string key = UnderscoredNamingConvention.Instance.Apply(property.Name);
            string fullKey = path.Length == 0 ? key : $"{path}.{key}";
            // Model-card metadata is intentionally free-form and supports null values.
            if (fullKey == "readme.frontmatter") continue;
            var member = property.GetValue(value);
            if (member == null && nullability.Create(property).ReadState == NullabilityState.NotNull)
                throw new InvalidOperationException($"Configuration '{fullKey}' cannot be null. Omit it to use its default.");
            if (member != null) ValidateObject(member, fullKey);
        }
    }
}
