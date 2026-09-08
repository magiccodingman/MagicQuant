using MagicQuant.Helpers;
using MQ.DB;

namespace MagicQuant.Services;

/// <summary>
/// Shared path contract for the DuckDB writer and prediction reader. Changing this
/// filename opens a different candidate database; preserve it across refactors.
/// SQLite remains the authority for measured truth, while DuckDB is derived state.
/// </summary>
public static class CombinationDatabasePathService
{
    private const string DbFileNamePrefix = "MagicQuant_Combinations";

    public static string GetPath() => Path.Combine(GetDirectory(), GetFileName());

    public static string GetDirectory()
    {
        if (!string.IsNullOrWhiteSpace(Cache.ModelMagicQuantDirectory))
            return Cache.ModelMagicQuantDirectory!;

        if (!string.IsNullOrWhiteSpace(Cache.MagicQuantDirectory))
            return Cache.MagicQuantDirectory!;

        throw new InvalidOperationException(
            "Neither Cache.ModelMagicQuantDirectory nor Cache.MagicQuantDirectory is set.");
    }

    public static string GetFileName()
    {
        string model = string.IsNullOrWhiteSpace(Cache.CurrentModelId) ? "unknown-model" : Cache.CurrentModelId;
        string imatrix = Cache.IsImatrixAvailable ? (Cache.ActiveImatrixIdentityHash ?? "imatrix-unknown") : "no-imatrix";
        string hp = RuntimeSearchSpace.AllowHighPrecisionHybrids ? "hp-on" : "hp-off";
        return $"{DbFileNamePrefix}_{model}_{imatrix}_{hp}.duckdb";
    }

}
