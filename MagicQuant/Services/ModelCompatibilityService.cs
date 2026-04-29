using System.Text.Json;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Data;
using MQ.DB.Models;
using Microsoft.EntityFrameworkCore;
using Spectre.Console;

namespace MagicQuant.Services;

public class ModelCompatibilityService
{
    private const string CompatVerboseEnv = "MAGICQUANT_DIAG_VERBOSE_TENSOR_COMPATIBILITY";
    private const string CompatFocusCandidatesEnv = "MAGICQUANT_DIAG_FOCUS_CANDIDATES";
    private readonly PythonManager _pyManager;

    public ModelCompatibilityService(PythonManager pyManager)
    {
        _pyManager = pyManager;
    }

    public async Task RunCompatibilityCheckAsync(string ggufPath)
    {
        AnsiConsole.Write(new Rule("[yellow]Tensor Compatibility Check[/]") { Justification = Justify.Left });

        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Base model not found at {ggufPath}");

        TensorWeightScheme.ValidateSmallestConfiguration();

        RuntimeSearchSpace.ResetForCompatibilityPass();
        Cache.UnusedTensorGroups.Clear();

        string directory = Path.GetDirectoryName(ggufPath)!;
        string scriptPath = Path.Combine(directory, "check_compat.py");
        string resultPath = Path.Combine(directory, "compat_results.json");
        string debugPath = Path.Combine(directory, "compat_debug.txt");

        try
        {
            var groupDefinitions = TReg.All.ToDictionary(g => g.Name, g => g.Tensors);

            var candidateBlockRequirements = BaselineQuants.GetGroupCombinationCandidates(RuntimeSearchSpace.HasUsableImatrix(), allowHighPrecisionHybrids: false)
                .Where(c => c.DefaultTensorScheme?.BlockNeo.HasValue == true)
                .ToDictionary(c => c.Names[0], c => c.DefaultTensorScheme!.BlockNeo!.Value);

            var payload = new
            {
                gguf_path = ggufPath,
                output_path = resultPath,
                groups = groupDefinitions,
                schemes = candidateBlockRequirements
            };

            string pyCode = GeneratePythonScript(JsonSerializer.Serialize(payload));
            await File.WriteAllTextAsync(scriptPath, pyCode);

            AnsiConsole.MarkupLine("[grey]Inspecting GGUF structure...[/]");
            await _pyManager.RunPythonScriptAsync(scriptPath);

            if (!File.Exists(resultPath))
                throw new Exception("Compatibility script finished but produced no result file.");

            string jsonResult = await File.ReadAllTextAsync(resultPath);

            if (jsonResult.Contains("\"Error\"", StringComparison.Ordinal))
            {
                var errorRes = JsonSerializer.Deserialize<CompatResult>(jsonResult);
                if (!string.IsNullOrEmpty(errorRes?.Error))
                    throw new Exception($"Python Inspection Failed: {errorRes.Error}");
            }

            var result = JsonSerializer.Deserialize<CompatResult>(jsonResult);
            if (result == null)
                return;

            bool compatVerbose = IsCompatVerbose();
            var focusCandidates = GetFocusCandidates();
            var runtimeCandidates = BaselineQuants.GetGroupCombinationCandidates(RuntimeSearchSpace.HasUsableImatrix(), allowHighPrecisionHybrids: false).ToList();
            int unusedCount = 0;
            int usedCount = 0;
            int shapeBanCount = 0;
            int explicitQuantBannedCount = 0;

            var shapeTable = new Table().Border(TableBorder.Rounded).Title("[red]Shape Incompatibilities[/]");
            shapeTable.AddColumn("Group");
            shapeTable.AddColumn("Candidate");
            shapeTable.AddColumn("Reason");

            foreach (var group in TReg.All)
            {
                bool exists = result.FoundGroups.Contains(group.Name, StringComparer.OrdinalIgnoreCase);

                if (exists)
                {
                    usedCount++;
                    continue;
                }

                unusedCount++;
                Cache.UnusedTensorGroups.Add(group);

                RuntimeSearchSpace.BanAllExplicitCombinationCandidatesForGroup(group, phase: "TensorCompatibilityCheck", reason: "group missing in GGUF");
            }

            var failuresByGroupAndScheme = result.Failures
                .GroupBy(x => $"{x.Group}::{x.Scheme}", StringComparer.OrdinalIgnoreCase)
                .ToDictionary(x => x.Key, x => x.ToList(), StringComparer.OrdinalIgnoreCase);

            foreach (var failure in result.Incompatible)
            {
                var group = TReg.GetByName(failure.Group);
                var candidate = runtimeCandidates.FirstOrDefault(c => c.Names.Any(n => n.Equals(failure.Scheme, StringComparison.OrdinalIgnoreCase)));

                if (group == null || candidate == null)
                    continue;
                if (RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate))
                    continue;

                var beforeRuntimeBan = RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate);
                
                // something is wrong with this. It's not working and this is a luxury not requirement. It's causing down stream issues on moe_experts for Qwen3.6-35B-A3B
                //RuntimeSearchSpace.BanCombinationCandidateForGroup(group, candidate, phase: "TensorCompatibilityCheck", reason: "Block Alignment");
                shapeBanCount++;
                shapeTable.AddRow($"[blue]{group.Name}[/]", $"[yellow]{candidate.Names[0]}[/]",
                    "[grey]Block Alignment[/]");

                if (ShouldLogCompatDetail(compatVerbose, group, candidate, focusCandidates))
                {
                    MagicQuantDiagnostics.Log("compat:decision",
                        $"group={group.Name}(id={group.UniqueId}) candidate={candidate.Names[0]}(id={candidate.UniqueId}) scheme={candidate.DefaultTensorScheme?.Names[0] ?? "<none>"} block={candidate.DefaultTensorScheme?.BlockNeo?.ToString() ?? "<none>"} staticBanned={candidate.BannedGroupIds.Contains(group.UniqueId)} runtimeBannedBefore={beforeRuntimeBan} result=restricted reason=Block Alignment");
                }

                if (failuresByGroupAndScheme.TryGetValue($"{failure.Group}::{failure.Scheme}", out var details) && details.Count > 0)
                {
                    LogFailureSummary(group, candidate, details);
                }
            }

            if (compatVerbose)
                await LogGroupCompatibilityOutcomeAndTruthCrossCheckAsync(result, runtimeCandidates, focusCandidates, ct: CancellationToken.None);

            foreach (var group in TReg.All.Except(Cache.UnusedTensorGroups))
            {
                if (RuntimeSearchSpace.IsGroupExplicitCandidateBanned(group))
                    explicitQuantBannedCount++;
            }

            AnsiConsole.MarkupLine("[green]✔[/] Analysis Complete.");
            AnsiConsole.MarkupLine($"   Active Groups: [bold cyan]{usedCount}[/]");

            if (unusedCount > 0)
            {
                string unusedNames = string.Join(", ", Cache.UnusedTensorGroups.Select(g => g.Name));
                AnsiConsole.MarkupLine($"   Unused Groups: [grey]{unusedNames}[/] (Forced to NULL)");
            }

            if (explicitQuantBannedCount > 0)
            {
                string groups = string.Join(", ",
                    RuntimeSearchSpace.GetGroupsWithExplicitQuantBanned().Select(x => x.Name));

                AnsiConsole.MarkupLine($"   Explicit-Quant-Banned Groups: [yellow]{groups}[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[green]No groups were reduced to explicit-banned/NULL-only by compatibility checks.[/]");
            }

            if (shapeBanCount > 0)
            {
                AnsiConsole.Write(shapeTable);
                AnsiConsole.MarkupLine($"[yellow]Applied {shapeBanCount} shape-based restrictions.[/]");
            }
            else
            {
                AnsiConsole.MarkupLine("[green]No shape-based restrictions found.[/]");
            }

            AnsiConsole.WriteLine();
        }
        finally
        {
            if (File.Exists(scriptPath)) File.Delete(scriptPath);
            if (File.Exists(resultPath)) File.Delete(resultPath);
            _ = debugPath;
        }
    }

    private static bool IsCompatVerbose()
        => string.Equals(Environment.GetEnvironmentVariable(CompatVerboseEnv), "1", StringComparison.OrdinalIgnoreCase)
           || string.Equals(Environment.GetEnvironmentVariable(CompatVerboseEnv), "true", StringComparison.OrdinalIgnoreCase);

    private static HashSet<string> GetFocusCandidates()
        => (Environment.GetEnvironmentVariable(CompatFocusCandidatesEnv) ?? string.Empty)
            .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

    private static bool ShouldLogCompatDetail(bool compatVerbose, TensorGroup group, BaselineQuants candidate, HashSet<string> focusCandidates)
        => compatVerbose && (MagicQuantDiagnostics.ShouldLogGroup(group) || focusCandidates.Count == 0 || candidate.Names.Any(x => focusCandidates.Contains(x)));

    private static void LogFailureSummary(TensorGroup group, BaselineQuants candidate, List<CompatFailure> details)
    {
        var failCount = details.Count;
        var firstFive = details.Take(5).ToList();
        var shapes = details.GroupBy(x => $"[{string.Join(",", x.Shape)}]").Select(x => $"{x.Key} x{x.Count()}").ToList();
        var remainders = details.Select(x => x.Remainder).Distinct().OrderBy(x => x).ToList();
        MagicQuantDiagnostics.Log("compat:summary", $"group={group.Name} candidate={candidate.Names[0]} checkedTensors={details.Max(x => x.CheckedTensorCount)} failingTensors={failCount} shapePatterns={string.Join("; ", shapes)} remainders={string.Join(",", remainders)}");
        foreach (var f in firstFive)
        {
            MagicQuantDiagnostics.Log("compat:block-alignment-fail",
                $"group={group.Name}(id={group.UniqueId}) candidate={candidate.Names[0]}(id={candidate.UniqueId}) tensor={f.Tensor} tensorClass={f.TensorClass} dims=[{string.Join(",", f.Shape)}] nDims={f.Shape.Count} checkedDimension={f.CheckedDimension} checkedValue={f.CheckedValue} requiredMultiple={f.RequiredMultiple} remainder={f.Remainder} pass=false reason=Block Alignment");
        }
    }

    private static async Task LogGroupCompatibilityOutcomeAndTruthCrossCheckAsync(CompatResult result, List<BaselineQuants> candidates, HashSet<string> focusCandidates, CancellationToken ct)
    {
        await using var db = new MagicQuantContext();
        var modelHashId = await ArchitectureFamilyService.ResolveExactCurrentAiModelHashIdOrNullAsync(db, ct);
        var imatrixId = modelHashId == null ? (long?)null : await ImatrixIdentityService.ResolveCurrentImatrixDefinitionIdAsync(db, modelHashId.Value, createIfMissing: false, ct);
        foreach (var group in TReg.All.Except(Cache.UnusedTensorGroups))
        {
            if (!MagicQuantDiagnostics.ShouldLogGroup(group))
                continue;
            var raw = RuntimeSearchSpace.GetRealExplicitCombinationCandidatesForGroup(group);
            var allowed = RuntimeSearchSpace.GetAllowedRealExplicitCombinationCandidatesForGroup(group);
            var restricted = raw.Where(x => RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, x)).ToList();
            MagicQuantDiagnostics.Log("compat:group-result", $"group={group.Name}(id={group.UniqueId}) before={raw.Count} after={allowed.Count} allowed={string.Join(", ", allowed.Select(MagicQuantDiagnostics.CandidateLabel))} restricted={string.Join(", ", restricted.Select(MagicQuantDiagnostics.CandidateLabel))}");
            if (raw.Count > 5 && allowed.Count == 1 && allowed[0].UniqueId == BaselineQuants.Q8_0.UniqueId)
            {
                var focusRestricted = restricted.Where(x => x.Names.Any(n => focusCandidates.Contains(n)) || x.Names[0] is "Q6_K" or "Q5_K" or "Q4_K_M").Select(x => x.Names[0]);
                MagicQuantDiagnostics.Log("compat:collapse-warning", $"group={group.Name} collapsed to Q8_0 only restrictions={restricted.Count} topReason=Block Alignment focusCandidatesRestricted={string.Join(",", focusRestricted)}");
            }
        }
    }

    private string GeneratePythonScript(string jsonPayload)
    {
        return $@"
import sys
import json
import re

payload_str = r'''{jsonPayload}'''
config = json.loads(payload_str)
output_path = config['output_path']
debug_path = output_path.replace('compat_results.json', 'compat_debug.txt')

def write_error(msg):
    with open(output_path, 'w') as f:
        json.dump({{""FoundGroups"": [], ""Incompatible"": [], ""Error"": msg}}, f)
    sys.exit(0)

try:
    import gguf
except ImportError:
    write_error('gguf module not installed')

try:
    reader = gguf.GGUFReader(config['gguf_path'])
except Exception as e:
    write_error(str(e))

tensors_map = {{t.name: t for t in reader.tensors}}
tensor_names = list(tensors_map.keys())

found_groups = []
failures = []
failure_details = []
debug_lines = []

debug_lines.append('Inspecting ' + str(len(tensor_names)) + ' tensors against ' + str(len(config[""schemes""])) + ' block requirements.')

for g_name, patterns in config['groups'].items():
    matched = []
    first_reason = None

    for pat in patterns:
        try:
            regex = re.compile(pat)
            for t in tensor_names:
                if regex.fullmatch(t):
                    matched.append(t)
                    if not first_reason:
                        first_reason = ""Match: '"" + pat + ""' -> '"" + t + ""'""
        except:
            continue

    if matched:
        found_groups.append(g_name)
        debug_lines.append(""[FOUND] "" + g_name + "" ("" + str(len(matched)) + "" tensors). "" + str(first_reason))
        weights = [t for t in matched if t.endswith('.weight')]

        if weights:
            for scheme, block_size in config['schemes'].items():
                is_valid = True

                for w_name in weights:
                    t_obj = tensors_map[w_name]
                    ne0 = t_obj.shape[0]
                    n_dims = len(t_obj.shape)

                    if n_dims != 2:
                        is_valid = False
                        debug_lines.append(""  [FAIL] "" + g_name + "" vs "" + scheme + "": "" + w_name + "" is "" + str(n_dims) + ""D (Required 2D)"")
                        break

                    if ne0 % block_size != 0:
                        is_valid = False
                        tclass = 'unknown'
                        lower_name = w_name.lower()
                        if 'exps' in lower_name:
                            tclass = 'routed_expert'
                        elif 'router' in lower_name:
                            tclass = 'router'
                        elif 'ffn_' in lower_name:
                            tclass = 'dense_ffn'
                        failure_details.append({{
                            ""Group"": g_name,
                            ""Scheme"": scheme,
                            ""Tensor"": w_name,
                            ""Shape"": list(t_obj.shape),
                            ""CheckedDimension"": 0,
                            ""CheckedValue"": int(ne0),
                            ""RequiredMultiple"": int(block_size),
                            ""Remainder"": int(ne0 % block_size),
                            ""TensorClass"": tclass,
                            ""CheckedTensorCount"": len(weights)
                        }})
                        debug_lines.append(""  [FAIL] "" + g_name + "" vs "" + scheme + "" (Block "" + str(block_size) + ""): "" + w_name + "" ne0="" + str(ne0) + "". Remainder="" + str(ne0 % block_size))
                        break

                if not is_valid:
                    failures.append({{""Group"": g_name, ""Scheme"": scheme}})
    else:
        debug_lines.append(""[MISSING] "" + g_name)

try:
    with open(debug_path, 'w') as f:
        f.write('\\n'.join(debug_lines))
except:
    pass

with open(output_path, 'w') as f:
    json.dump({{
        ""FoundGroups"": found_groups,
        ""Incompatible"": failures,
        ""Failures"": failure_details,
        ""Error"": None
    }}, f, indent=2)
";
    }

    private class CompatResult
    {
        public List<string> FoundGroups { get; set; } = new();
        public List<CompatFailure> Incompatible { get; set; } = new();
        public List<CompatFailure> Failures { get; set; } = new();
        public string? Error { get; set; }
    }

    private class CompatFailure
    {
        public string Group { get; set; } = string.Empty;
        public string Scheme { get; set; } = string.Empty;
        public string Tensor { get; set; } = string.Empty;
        public List<long> Shape { get; set; } = new();
        public int CheckedDimension { get; set; }
        public long CheckedValue { get; set; }
        public int RequiredMultiple { get; set; }
        public long Remainder { get; set; }
        public string TensorClass { get; set; } = "unknown";
        public int CheckedTensorCount { get; set; }
    }
}
