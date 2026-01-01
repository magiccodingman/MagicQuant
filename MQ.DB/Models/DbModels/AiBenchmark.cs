namespace MQ.DB.Models.DbModels;

public enum BenchmarkCategory
{
    General = 1,
    Math = 2,
    Code = 3,
}
public class AiBenchmark
{
    public uint Id { get; set; }
    
    /// <summary>
    /// n-N gpu layers
    /// </summary>
    public byte Ngl { get; set; }
    
    /// <summary>
    /// Size of the model in bytes at this combination. 
    /// </summary>
    /// <returns></returns>
    public ulong SizeBytes { get; set; }
    
    public double TokensPerSecond { get; set; }
    
    // both the AiModelHash and the TensorComboId combined
    // must be unique in the table. 
    
    /// <summary>
    /// foreign key
    /// </summary>
    public uint TensorComboId { get; set; }
    
    /// <summary>
    /// foreign key
    /// </summary>
    public uint AiModelHashId { get; set; }
}

public class CategoryBenchmarkToAiBenchmark
{
    public uint Id { get; set; }
    public uint AiBenchmarkId { get; set; }
    public uint CategoryBenchmarkId { get; set; }
}

public class CategoryBenchmark
{
    public uint Id { get; set; }
    
    /// <summary>
    /// foreign key to AiBenchmark
    /// </summary>
    public uint AiBenchmarkId { get; set; }
    
    /// <summary>
    /// Byte version to BenchmarkCategory enum in C#
    /// </summary>
    public byte Category { get; set; }
    
    public double Kld { get; set; }
    public double Ppl { get; set; }
    public double PplError { get; set; }
}