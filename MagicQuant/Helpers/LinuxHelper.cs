using System.Diagnostics;
using Spectre.Console;

namespace MagicQuant.Helpers;

public class LinuxHelper
{
    public static async Task RefreshSudoCredentialsAsync()
    {
        AnsiConsole.MarkupLine("[grey]Verifying sudo access for system installs...[/]");
        
        // "sudo -v" updates the user's cached credentials.
        // It will prompt for a password if necessary.
        var psi = new ProcessStartInfo
        {
            FileName = "sudo",
            Arguments = "-v",
            UseShellExecute = false // Let standard input handle the password prompt
        };
        
        var p = Process.Start(psi);
        await p!.WaitForExitAsync();
        
        if (p.ExitCode != 0)
        {
            throw new Exception("Sudo access denied or cancelled.");
        }
    }
}