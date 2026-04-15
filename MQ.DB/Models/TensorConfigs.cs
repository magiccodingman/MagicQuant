using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace MQ.DB.Models;

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public readonly struct TensorConfig
{
    public readonly byte BaseQuant;
    public readonly byte Embeddings;
    public readonly byte LmHead;
    public readonly byte AttnQ;
    public readonly byte AttnKV;
    public readonly byte AttnOutput;
    public readonly byte FfnUpGate;
    public readonly byte FfnDown;
    public readonly byte MoeExperts;
    public readonly byte MoeRouter;

    public TensorConfig(
        byte baseQuant,
        byte embeddings,
        byte lmHead,
        byte attnQ,
        byte attnKV,
        byte attnOutput,
        byte ffnUpGate,
        byte ffnDown,
        byte moeExperts,
        byte moeRouter)
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
            baseQuant:  checked((byte)h.BaseQuant.UniqueId),
            embeddings: GetSchemeIdOrDefault(h, TReg.Embeddings),
            lmHead:     GetSchemeIdOrDefault(h, TReg.LmHead),
            attnQ:      GetSchemeIdOrDefault(h, TReg.AttnQ),
            attnKV:     GetSchemeIdOrDefault(h, TReg.AttnKV),
            attnOutput: GetSchemeIdOrDefault(h, TReg.AttnOutput),
            ffnUpGate:  GetSchemeIdOrDefault(h, TReg.FfnUpGate),
            ffnDown:    GetSchemeIdOrDefault(h, TReg.FfnDown),
            moeExperts: GetSchemeIdOrDefault(h, TReg.MoeExperts),
            moeRouter:  GetSchemeIdOrDefault(h, TReg.MoeRouter))
    { }

    private static byte GetSchemeIdOrDefault(HybridQuant h, TensorGroup group)
    {
        if (h.Tensors == null || h.Tensors.Count == 0)
            return TensorWeightScheme.NULL.UniqueId;

        TensorWeightScheme? found = null;

        for (int i = 0; i < h.Tensors.Count; i++)
        {
            var t = h.Tensors[i];
            if (t?.TGroup == null)
                continue;

            if (t.TGroup.UniqueId != group.UniqueId)
                continue;

            if (found != null)
            {
                throw new InvalidOperationException(
                    $"HybridQuant contains duplicate entries for group '{group.Name}' (UniqueId={group.UniqueId}).");
            }

            found = t.TensorType;
        }

        return found == null
            ? TensorWeightScheme.NULL.UniqueId
            : checked((byte)found.UniqueId);
    }

    public static explicit operator TensorConfig(HybridQuant h) => new TensorConfig(h);
}