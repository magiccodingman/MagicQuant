using System.Diagnostics;

namespace MagicQuant.ProcessFixture;

/// <summary>Offline child-process fixture. Used only by process lifetime regression tests.</summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        switch (args[0])
        {
            case "echo":
                foreach (string arg in args.Skip(1)) Console.WriteLine(arg);
                return 0;
            case "flood":
                for (int i = 0; i < 12000; i++)
                {
                    Console.WriteLine($"stdout-{i:D5}");
                    Console.Error.WriteLine($"stderr-{i:D5}");
                }
                return 7;
            case "tree":
                var start = new ProcessStartInfo("dotnet");
                start.ArgumentList.Add(typeof(Program).Assembly.Location);
                start.ArgumentList.Add("wait");
                using (var child = Process.Start(start)!)
                {
                    Console.WriteLine($"child:{child.Id}");
                    await Task.Delay(TimeSpan.FromMinutes(5));
                }
                return 0;
            case "wait":
                Console.WriteLine($"ready:{Environment.ProcessId}");
                await Task.Delay(TimeSpan.FromMinutes(5));
                return 0;
            default:
                return 2;
        }
    }
}
