namespace MagicQuant.Commands;

/// <summary>
/// Compatibility entry point for callers using the historical command class.
/// MagicQuant now performs benchmark-driven discovery rather than evolutionary search.
/// </summary>
public class Evolution : QuantizationPipeline;
