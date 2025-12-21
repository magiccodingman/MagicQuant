using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace MagicQuant.Models;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct TensorConfig
{
    public readonly sbyte BaseQuant;
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
        sbyte baseQuant,
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
        BaseQuant   = baseQuant;
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

    // Converting constructor: HybridQuant -> TensorConfig
    public TensorConfig(HybridQuant h)
        : this(
            baseQuant:  checked((sbyte)h.BaseQuant.UniqueId),
            embeddings: GetSchemeId(h, TReg.Embeddings),
            lmHead:     GetSchemeId(h, TReg.LmHead),
            attnQ:      GetSchemeId(h, TReg.AttnQ),
            attnKV:     GetSchemeId(h, TReg.AttnKV),
            attnOutput: GetSchemeId(h, TReg.AttnOutput),
            ffnUpGate:  GetSchemeId(h, TReg.FfnUpGate),
            ffnDown:    GetSchemeId(h, TReg.FfnDown),
            moeExperts: GetSchemeId(h, TReg.MoeExperts),
            moeRouter:  GetSchemeId(h, TReg.MoeRouter))
    { }

    private static sbyte GetSchemeId(HybridQuant h, TensorGroup group)
    {
        if (h.Tensors == null)
            throw new ArgumentNullException(nameof(h.Tensors));

        TensorWeightScheme? found = null;

        // Single pass: find the tensor type for the requested group
        for (int i = 0; i < h.Tensors.Count; i++)
        {
            var t = h.Tensors[i];
            if (t?.TGroup == null)
                continue;

            if (t.TGroup.UniqueId != group.UniqueId)
                continue;

            if (found != null)
                throw new InvalidOperationException(
                    $"HybridQuant contains duplicate entries for group '{group.Name}' (UniqueId={group.UniqueId}).");

            found = t.TensorType;
        }

        if (found == null)
            throw new InvalidOperationException(
                $"HybridQuant missing tensor entry for group '{group.Name}' (UniqueId={group.UniqueId}).");

        return checked((sbyte)found.UniqueId);
    }
    
    // Conversion operator: HybridQuant -> TensorConfig
    public static explicit operator TensorConfig(HybridQuant h) => new TensorConfig(h);
}


