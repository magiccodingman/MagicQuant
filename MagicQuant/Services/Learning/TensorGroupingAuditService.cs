using MagicQuant.Models.Learning;
using MQ.DB.Models;

namespace MagicQuant.Services.Learning;

public sealed class TensorGroupingAuditService
{
    public TensorGroupingAuditResult Audit(
        IReadOnlyCollection<string> tensorNames,
        IReadOnlyDictionary<string, LearnedTensorTruth> truthByTensor)
    {
        var grouped = new Dictionary<string, TensorGroupingResult>(StringComparer.Ordinal);
        var ambiguous = new List<TensorGroupingAuditIssue>();
        var illegalUnresolved = new List<TensorGroupingAuditIssue>();
        var baseQuantExceptions = new List<TensorGroupingAuditIssue>();

        foreach (var tensorName in tensorNames.OrderBy(x => x, StringComparer.Ordinal))
        {
            var matchedGroups = TReg.FindMatchingGroups(tensorName);

            if (matchedGroups.Length == 1)
            {
                grouped[tensorName] = new TensorGroupingResult
                {
                    PrimaryGroup = matchedGroups[0],
                    MatchedGroups = [matchedGroups[0].Name]
                };

                continue;
            }

            if (matchedGroups.Length > 1)
            {
                var names = matchedGroups.Select(x => x.Name).ToList();
                truthByTensor.TryGetValue(tensorName, out var truth);
                grouped[tensorName] = new TensorGroupingResult
                {
                    PrimaryGroup = null,
                    MatchedGroups = names
                };

                ambiguous.Add(new TensorGroupingAuditIssue
                {
                    TensorName = tensorName,
                    IssueKind = "AmbiguousSemanticGroupCollision",
                    MatchedGroups = names,
                    FinalQuantType = truth?.FinalQuantType,
                    LearningSource = truth?.Source.ToString()
                });

                continue;
            }

            var matchedPattern = FindMatchingBaseQuantExceptionPattern(tensorName);
            if (matchedPattern != null)
            {
                truthByTensor.TryGetValue(tensorName, out var truth);
                grouped[tensorName] = new TensorGroupingResult
                {
                    PrimaryGroup = null,
                    MatchedGroups = [],
                    IsBaseQuantException = true,
                    MatchedExceptionPattern = matchedPattern
                };

                baseQuantExceptions.Add(new TensorGroupingAuditIssue
                {
                    TensorName = tensorName,
                    IssueKind = "BaseQuantExceptionFallback",
                    MatchedExceptionPattern = matchedPattern,
                    FinalQuantType = truth?.FinalQuantType,
                    LearningSource = truth?.Source.ToString()
                });

                continue;
            }

            grouped[tensorName] = new TensorGroupingResult
            {
                PrimaryGroup = null,
                MatchedGroups = []
            };

            truthByTensor.TryGetValue(tensorName, out var unresolvedTruth);
            illegalUnresolved.Add(new TensorGroupingAuditIssue
            {
                TensorName = tensorName,
                IssueKind = "IllegalUnresolvedTensor",
                FinalQuantType = unresolvedTruth?.FinalQuantType,
                LearningSource = unresolvedTruth?.Source.ToString()
            });
        }

        return new TensorGroupingAuditResult
        {
            GroupedByTensor = grouped,
            Ambiguous = ambiguous,
            IllegalUnresolved = illegalUnresolved,
            BaseQuantExceptions = baseQuantExceptions
        };
    }

    private static string? FindMatchingBaseQuantExceptionPattern(string tensorName)
    {
        var patterns = TReg.GetBaseQuantExceptionPatterns();
        var regexes = TReg.GetBaseQuantExceptionRegexes();

        for (int i = 0; i < regexes.Length; i++)
        {
            if (regexes[i].IsMatch(tensorName))
                return i < patterns.Length ? patterns[i] : regexes[i].ToString();
        }

        return null;
    }
}
