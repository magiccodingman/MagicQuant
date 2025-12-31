using System.Text.Json;
using MagicQuant.Helpers;
using MagicQuant.Models;
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

        string directory = Path.GetDirectoryName(ggufPath)!;
        string scriptPath = Path.Combine(directory, "check_compat.py");
        string resultPath = Path.Combine(directory, "compat_results.json");
        string debugPath = Path.Combine(directory, "compat_debug.txt");

        try
        {
            // 1. Prepare Data
            var groupDefinitions = TReg.All.ToDictionary(g => g.Name, g => g.Tensors);
            
            var blockRequirements = TensorWeightScheme.All
                .Where(s => s.BlockNeo.HasValue)
                .ToDictionary(s => s.Names[0], s => s.BlockNeo!.Value);

            var payload = new
            {
                gguf_path = ggufPath,
                output_path = resultPath,
                groups = groupDefinitions,
                schemes = blockRequirements
            };

            // 2. Generate Python Script (With Shape Debugging)
            string pyCode = GeneratePythonScript(JsonSerializer.Serialize(payload));
            await File.WriteAllTextAsync(scriptPath, pyCode);

            // 3. Run Inspection
            AnsiConsole.MarkupLine("[grey]Inspecting GGUF structure...[/]");
            await _pyManager.RunPythonScriptAsync(scriptPath);

            // 4. Validate Result
            if (!File.Exists(resultPath))
                throw new Exception("Compatibility script finished but produced no result file.");

            string jsonResult = await File.ReadAllTextAsync(resultPath);
            
            // Handle script errors
            if (jsonResult.Contains("\"Error\""))
            {
                var errorRes = JsonSerializer.Deserialize<CompatResult>(jsonResult);
                if (!string.IsNullOrEmpty(errorRes?.Error))
                    throw new Exception($"Python Inspection Failed: {errorRes.Error}");
            }

            var result = JsonSerializer.Deserialize<CompatResult>(jsonResult);
            if (result == null) return;

            // ---------------------------------------------------------
            // 5. Global State Update Logic
            // ---------------------------------------------------------
            
            TensorWeightScheme.NULL.BannedGroups.Clear();
            MagicQuant.Config.UnusedTensorGroups.Clear();

            int unusedCount = 0;
            int usedCount = 0;

            foreach (var group in TReg.All)
            {
                bool exists = result.FoundGroups.Contains(group.Name);

                if (exists)
                {
                    if (!TensorWeightScheme.NULL.BannedGroups.Contains(group))
                    {
                        TensorWeightScheme.NULL.BannedGroups.Add(group);
                    }
                    usedCount++;
                }
                else
                {
                    unusedCount++;
                    MagicQuant.Config.UnusedTensorGroups.Add(group);

                    foreach (var scheme in TensorWeightScheme.All)
                    {
                        if (scheme == TensorWeightScheme.NULL) continue;
                        if (!scheme.BannedGroups.Contains(group)) scheme.BannedGroups.Add(group);
                    }
                }
            }

            // ---------------------------------------------------------
            // 6. Handle Shape Restrictions
            // ---------------------------------------------------------
            int shapeBanCount = 0;
            var table = new Table().Border(TableBorder.Rounded).Title("[red]Shape Incompatibilities[/]");
            table.AddColumn("Group");
            table.AddColumn("Scheme");
            table.AddColumn("Reason");

            foreach (var failure in result.Incompatible)
            {
                var group = TReg.GetByName(failure.Group);
                var scheme = TensorWeightScheme.All.FirstOrDefault(s => s.Names.Contains(failure.Scheme));

                if (group != null && scheme != null)
                {
                    if (!scheme.BannedGroups.Contains(group))
                    {
                        scheme.BannedGroups.Add(group);
                        shapeBanCount++;
                        table.AddRow($"[blue]{group.Name}[/]", $"[yellow]{scheme.Names[0]}[/]", "[grey]Block Alignment[/]");
                    }
                }
            }

            // ---------------------------------------------------------
            // 7. Report
            // ---------------------------------------------------------
            AnsiConsole.MarkupLine($"[green]✔[/] Analysis Complete.");
            AnsiConsole.MarkupLine($"   Active Groups: [bold cyan]{usedCount}[/]");
            
            if (unusedCount > 0)
            {
                string unusedNames = string.Join(", ", MagicQuant.Config.UnusedTensorGroups.Select(g => g.Name));
                AnsiConsole.MarkupLine($"   Unused Groups: [grey]{unusedNames}[/] (Forced to NULL)");
            }

            if (shapeBanCount > 0)
            {
                AnsiConsole.Write(table);
                AnsiConsole.MarkupLine($"[yellow]Applied {shapeBanCount} restrictions due to tensor shapes.[/]");
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
            // Debug path is kept for inspection
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
        json.dump({{'FoundGroups': [], 'Incompatible': [], 'Error': msg}}, f)
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

debug_lines.append(f'Inspecting {{len(tensor_names)}} tensors against {{len(config[""schemes""])}} block requirements.')

# 1. Match Groups
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
                        first_reason = f""Match: '{{pat}}' -> '{{t}}'""
        except:
            continue
    
    if matched:
        found_groups.append(g_name)
        debug_lines.append(f""[FOUND] {{g_name}} ({{len(matched)}} tensors). {{first_reason}}"")
        
        # 2. Check Compatibility (Only if found)
        weights = [t for t in matched if t.endswith('.weight')]
        
        if weights:
            # Check against every scheme that has a block req
            for scheme, block_size in config['schemes'].items():
                is_valid = True
                
                for w_name in weights:
                    t_obj = tensors_map[w_name]
                    ne0 = t_obj.shape[0] # GGUF ne0
                    n_dims = len(t_obj.shape)

                    # Rule A: Non-2D
                    if n_dims != 2:
                        is_valid = False
                        debug_lines.append(f""  [FAIL] {{g_name}} vs {{scheme}}: {{w_name}} is {{n_dims}}D (Required 2D)"")
                        break
                    
                    # Rule B: Modulo
                    if ne0 % block_size != 0:
                        is_valid = False
                        # Explicit debug for math check
                        debug_lines.append(f""  [FAIL] {{g_name}} vs {{scheme}} (Block {{block_size}}): {{w_name}} ne0={{ne0}}. {{ne0}} % {{block_size}} = {{ne0 % block_size}}"")
                        break
                
                if not is_valid:
                    failures.append({{ 'Group': g_name, 'Scheme': scheme }})
    else:
        debug_lines.append(f""[MISSING] {{g_name}}"")

# Write Debug
try:
    with open(debug_path, 'w') as f:
        f.write('\n'.join(debug_lines))
except:
    pass

# Write Result
with open(output_path, 'w') as f:
    json.dump({{
        'FoundGroups': found_groups,
        'Incompatible': failures,
        'Error': None
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
        public string Group { get; set; } = "";
        public string Scheme { get; set; } = "";
    }
}