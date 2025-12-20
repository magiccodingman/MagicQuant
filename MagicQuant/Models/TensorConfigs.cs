using System.Runtime.InteropServices;

namespace MagicQuant.Models;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct TensorConfig
{
    public readonly sbyte Embeddings;
    public readonly sbyte LmHead;
    public readonly sbyte AttnQ;
    public readonly sbyte AttnKV;
    public readonly sbyte AttnOutput;
    public readonly sbyte FfnUpGate;
    public readonly sbyte FfnDown;
    public readonly sbyte MoeExperts;
    public readonly sbyte MoeRouter;

    public TensorConfig(
        sbyte embeddings,
        sbyte lmHead,
        sbyte attnQ,
        sbyte attnKV,
        sbyte attnOutput,
        sbyte ffnUpGate,
        sbyte ffnDown,
        sbyte moeExperts,
        sbyte moeRouter)
    {
        Embeddings  = embeddings;
        LmHead      = lmHead;
        AttnQ       = attnQ;
        AttnKV      = attnKV;
        AttnOutput  = attnOutput;
        FfnUpGate   = ffnUpGate;
        FfnDown     = ffnDown;
        MoeExperts  = moeExperts;
        MoeRouter   = moeRouter;
    }
    
    public sbyte GetValue(in TensorGroup group) => group.UniqueId switch
    {
        0 => Embeddings,
        1 => LmHead,
        2 => AttnQ,
        3 => AttnKV,
        4 => AttnOutput,
        5 => FfnUpGate,
        6 => FfnDown,
        7 => MoeExperts,
        8 => MoeRouter,
        _ => throw new ArgumentOutOfRangeException(nameof(group))
    };
}
