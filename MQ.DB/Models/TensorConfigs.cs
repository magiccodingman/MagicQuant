using System;
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
        BaseQuant = baseQuant;
        Embeddings = embeddings;
        LmHead = lmHead;
        AttnQ = attnQ;
        AttnKV = attnKV;
        AttnOutput = attnOutput;
        FfnUpGate = ffnUpGate;
        FfnDown = ffnDown;
        MoeExperts = moeExperts;
        MoeRouter = moeRouter;
    }

    // Converting constructor: HybridQuant -> TensorConfig
    public TensorConfig(HybridQuant h)
        : this(
            baseQuant: checked((byte)h.BaseQuant.UniqueId),
            embeddings: GetStoredIdOrDefault(h, TReg.Embeddings),
            lmHead: GetStoredIdOrDefault(h, TReg.LmHead),
            attnQ: GetStoredIdOrDefault(h, TReg.AttnQ),
            attnKV: GetStoredIdOrDefault(h, TReg.AttnKV),
            attnOutput: GetStoredIdOrDefault(h, TReg.AttnOutput),
            ffnUpGate: GetStoredIdOrDefault(h, TReg.FfnUpGate),
            ffnDown: GetStoredIdOrDefault(h, TReg.FfnDown),
            moeExperts: GetStoredIdOrDefault(h, TReg.MoeExperts),
            moeRouter: GetStoredIdOrDefault(h, TReg.MoeRouter))
    { }

    private static byte GetStoredIdOrDefault(HybridQuant h, TensorGroup group)
    {
        if (h.Tensors == null || h.Tensors.Count == 0)
            return BaselineQuants.TensorConfigNullSlotValue;

        HybridTensor? found = null;

        for (int i = 0; i < h.Tensors.Count; i++)
        {
            var t = h.Tensors[i];
            if (t?.TGroup == null)
                continue;

            if (t.TGroup.UniqueId != group.UniqueId)
                continue;

            if (found != null)
                throw new InvalidOperationException($"HybridQuant contains duplicate entries for group '{group.Name}' (UniqueId={group.UniqueId}).");

            found = t;
        }

        if (found == null)
            return BaselineQuants.TensorConfigNullSlotValue;

        found.ValidateOrThrow();

        return found.OverrideMode switch
        {
            HybridTensorOverrideMode.LearnedBaselineCandidate => BaselineQuants.EncodeTensorConfigGroupSlot(found.CandidateBaseline!),
            HybridTensorOverrideMode.ExactTensorScheme => BaselineQuants.EncodeTensorConfigGroupSlot(found.ExactTensorScheme!),
            _ => throw new InvalidOperationException($"Unknown HybridTensorOverrideMode '{found.OverrideMode}'.")
        };
    }

    public static explicit operator TensorConfig(HybridQuant h) => new TensorConfig(h);
}
