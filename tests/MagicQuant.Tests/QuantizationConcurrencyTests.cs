using MagicQuant.Services;
using Xunit;

namespace MagicQuant.Tests;

public sealed class QuantizationConcurrencyTests
{
    [Theory]
    [InlineData(1, 4, 1, 1)]
    [InlineData(4, 4, 1, 3)]
    [InlineData(8, 4, 1, 6)]
    [InlineData(32, 2, 2, 15)]
    [InlineData(32, 10, 7, 4)]
    public void Writer_capacity_and_cpu_budget_bound_concurrency(int threads, int writers, int concurrent, int perProcess)
    {
        var plan = QuantizationConcurrencyPlan.Create(threads, writers);
        Assert.Equal(concurrent, plan.Concurrency);
        Assert.Equal(perProcess, plan.ThreadsPerProcess);
        Assert.True(plan.Concurrency * plan.ThreadsPerProcess <= Math.Max(1, threads));
    }
}
