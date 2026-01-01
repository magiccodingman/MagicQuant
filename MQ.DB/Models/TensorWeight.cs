namespace MQ.DB.Models;

public class TensorWeight
{
    public TensorWeight(byte uniqueId, bool requiresImatrix, string[] names, TensorGroup[]? bannedGroups = null)
    {
        Names = names.ToList();
        UniqueId = uniqueId;
        RequiresImatrix = requiresImatrix;
        BannedGroups = bannedGroups?.ToList();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="name">Leave null for basically everything. Only provide Bf16 or F16 or
    /// so on for those that are categorized together, which is Unique.</param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    public string GetName(string? name = null)
    {
        if (Names != null && Names.Any())
        {
            if (Names.Count == 1)
            {
                return Names.First();
            }
            else if(Names.Count > 1 && !string.IsNullOrEmpty(name))
            {
                return Names.First(x => x.Equals(name, StringComparison.InvariantCultureIgnoreCase));
            }
            else
            {
                throw new Exception("Tensor Weight had more than one name, but provided override name was null or didn't match any stored.");
            }
        }
        else
        {
            throw new Exception("No strings in the TensorWeight Names variable.");
        }
    }
    
    public List<string>? Names { get; }
    public byte UniqueId { get; }
    
    public bool RequiresImatrix { get; }
    
    /// <summary>
    /// Which Tensor Groups this tensor weight CANNOT be attached too.
    /// </summary>
    public List<TensorGroup>? BannedGroups { get; }
}