using System.Linq;

namespace MagicQuant.Services;

internal static class CombinationDuckDbSchema
{
    public const string TableName = "tensor_configs";
    public const string SlotColumnList = "BaseQuant, Embeddings, LmHead, AttnQ, AttnKV, AttnOutput, FfnUpGate, FfnDown, MoeExperts, MoeRouter";
    public const string PredictionColumnList = "PredictedKld, PredictedSizeBytes, PredictionConfidence, PredictionRank";
    public const string HybridPredicateSql = "(Embeddings <> 0 OR LmHead <> 0 OR AttnQ <> 0 OR AttnKV <> 0 OR AttnOutput <> 0 OR FfnUpGate <> 0 OR FfnDown <> 0 OR MoeExperts <> 0 OR MoeRouter <> 0)";

    public static readonly string[] SlotColumns =
    [
        "BaseQuant",
        "Embeddings",
        "LmHead",
        "AttnQ",
        "AttnKV",
        "AttnOutput",
        "FfnUpGate",
        "FfnDown",
        "MoeExperts",
        "MoeRouter"
    ];

    public static readonly string[] ExpectedColumnTypes =
    [
        "utinyint","utinyint","utinyint","utinyint","utinyint","utinyint","utinyint","utinyint","utinyint","utinyint",
        "double","ubigint","double","ubigint"
    ];

    public static string CreateTableSql => $@"
DROP TABLE IF EXISTS {TableName};
CREATE TABLE {TableName} (
    BaseQuant UTINYINT,
    Embeddings UTINYINT,
    LmHead UTINYINT,
    AttnQ UTINYINT,
    AttnKV UTINYINT,
    AttnOutput UTINYINT,
    FfnUpGate UTINYINT,
    FfnDown UTINYINT,
    MoeExperts UTINYINT,
    MoeRouter UTINYINT,

    -- Transient DuckDB-only prediction metadata.
    -- SQLite remains the real benchmark truth source.
    PredictedKld DOUBLE,
    PredictedSizeBytes UBIGINT,
    PredictionConfidence DOUBLE,
    PredictionRank UBIGINT
);";

    public static string BuildSlotEqualityPredicate(string leftAlias, string rightAlias)
    {
        return string.Join(" AND ", SlotColumns.Select(c => $"{leftAlias}.{c} = {rightAlias}.{c}"));
    }

    public static string QualifySlotColumnList(string alias)
    {
        return string.Join(", ", SlotColumns.Select(c => $"{alias}.{c}"));
    }

    public static string QualifyHybridPredicate(string alias)
    {
        return HybridPredicateSql.Replace("Embeddings", $"{alias}.Embeddings")
            .Replace("LmHead", $"{alias}.LmHead")
            .Replace("AttnQ", $"{alias}.AttnQ")
            .Replace("AttnKV", $"{alias}.AttnKV")
            .Replace("AttnOutput", $"{alias}.AttnOutput")
            .Replace("FfnUpGate", $"{alias}.FfnUpGate")
            .Replace("FfnDown", $"{alias}.FfnDown")
            .Replace("MoeExperts", $"{alias}.MoeExperts")
            .Replace("MoeRouter", $"{alias}.MoeRouter");
    }
}
