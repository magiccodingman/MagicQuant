using System.Text.Json;
using MagicQuant.Helpers;
using MQ.DB;
using MQ.DB.Models;
using Spectre.Console;

namespace MagicQuant.Services;

public class ModelCompatibilityService
{
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

        RuntimeSearchSpace.ResetForNewModel();
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

                RuntimeSearchSpace.BanAllExplicitCombinationCandidatesForGroup(group);
            }

            foreach (var failure in result.Incompatible)
            {
                var group = TReg.GetByName(failure.Group);
                var candidate = BaselineQuants.GetGroupCombinationCandidates(RuntimeSearchSpace.HasUsableImatrix(), allowHighPrecisionHybrids: false)
                    .FirstOrDefault(c => c.Names.Any(n => n.Equals(failure.Scheme, StringComparison.OrdinalIgnoreCase)));

                if (group == null || candidate == null)
                    continue;
                if (RuntimeSearchSpace.IsCombinationCandidateRuntimeBannedForGroup(group, candidate))
                    continue;

                RuntimeSearchSpace.BanCombinationCandidateForGroup(group, candidate);
                shapeBanCount++;
                shapeTable.AddRow($"[blue]{group.Name}[/]", $"[yellow]{candidate.Names[0]}[/]",
                    "[grey]Block Alignment[/]");
            }

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
                AnsiConsole.MarkupLine("[green]No groups were reduced to BF16/NULL-only by compatibility checks.[/]");
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
        ""Error"": None
    }}, f, indent=2)
";
    }

    private class CompatResult
    {
        public List<string> FoundGroups { get; set; } = new();
        public List<CompatFailure> Incompatible { get; set; } = new();
        public string? Error { get; set; }
    }

    private class CompatFailure
    {
        public string Group { get; set; } = string.Empty;
        public string Scheme { get; set; } = string.Empty;
    }
}
