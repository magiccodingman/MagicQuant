using Microsoft.EntityFrameworkCore.Metadata.Builders;
using MQ.DB.Interfaces;

namespace MQ.DB.Models.DbModels;

public class TensorCombo : ISQLiteEntity<TensorCombo>
{
    public TensorCombo()
    {
    }
    
    // Connection to TensorConfig
    public TensorCombo(TensorConfig c)
    {
        BaseQuant = c.BaseQuant;
        Embeddings = c.Embeddings;
        LmHead = c.LmHead;
        AttnQ = c.AttnQ;
        AttnKV = c.AttnKV;
        AttnOutput = c.AttnOutput;
        FfnUpGate = c.FfnUpGate;
        FfnDown = c.FfnDown;
        MoeExperts = c.MoeExperts;
        MoeRouter = c.MoeRouter;
    }

    public uint Id { get; set; }
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
    
    public void Configure(EntityTypeBuilder<TensorCombo> builder)
    {
        builder.HasKey(x => x.Id);
        
        builder.HasIndex(x => new 
        { 
            x.BaseQuant, 
            x.Embeddings, 
            x.LmHead, 
            x.AttnQ, 
            x.AttnKV, 
            x.AttnOutput, 
            x.FfnUpGate, 
            x.FfnDown, 
            x.MoeExperts, 
            x.MoeRouter 
        }).IsUnique();;
    }
}