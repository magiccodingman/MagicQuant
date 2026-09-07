# Worked contributor examples

## Add a configuration option

Suppose a future change adds a limit to a selection stage. First find the owning policy (`RuntimeCandidateSelectionConfig` and its service), and decide whether the option actually affects behavior. Do not add another dormant knob.

1. Add a clearly named typed property with its default and a comment describing the decision it controls.
2. Add the underscored YAML key to `config.default.yaml`; decide explicitly whether the distributed tuning profile uses the same default.
3. If a CLI override is useful, add it to `MagicQuantYamlLoader.ApplyCliOverrides` and the value/flag contract in `CliOptionValidator`. Validate numeric input with invariant culture. Add preflight validation for constraints that should fail before work begins.
4. Add tests for omitted/default values, CLI precedence, and the observable stage behavior. Keep any changed `Config.Current`/`Cache` state scoped and restored.
5. Update command help and configuration documentation. Run the strict distributed-config tests and the complete suite.

`YamlConfigurationDiagnostics` derives known keys from the typed schema, so it does not need a duplicate property-name list. Free-form `readme.frontmatter` and dictionary keys remain user-defined.

## Change a path or process invocation

For a new benchmark option, update `BenchmarkCommands`, then assert the literal argv sequence in `BenchmarkContractTests`. Use `NativeCommand.CreateStartInfo`, not interpolated shell strings. Include a path with spaces and metacharacters in the test. Keep retries in the calling service; `ProcessRunner` returns a nonzero exit code and only throws for launch/IO/cancellation failures.

For conversion behavior, `NativeModelConversionService` accepts `IProcessRunner`. `NativeConversionTests` injects a small fake that writes a partial output and returns failure or cancellation. That verifies incomplete artifacts never acquire a reusable success marker without invoking Python or a model. Keep production code using the real runner by default.

For output destinations, update `OutputPathService` and `PathSafety`, preserving existing command semantics or documenting a deliberate migration. Test ordinary paths, parent/child collisions, similarly prefixed sibling directories, and linked directories. Run the optional smoke workflow if native argument or artifact lifecycle behavior changed.

## Work on numerical policy

Read the research wiki and the owning service before editing. Add a regression around the actual measured/predicted tradeoff and its context identity. Do not replace exact custom tensor assignments with a built-in family surrogate merely to make a test pass. Model hash, architecture/profile identity, and imatrix scope are part of the input.

Global configuration and registries still exist; this cleanup does not support multiple concurrent campaigns in one process. New helpers should accept explicit inputs, return results, and be testable without mutating those registries. Extract a responsibility with behavior tests rather than mechanically splitting a large class into partial files.
