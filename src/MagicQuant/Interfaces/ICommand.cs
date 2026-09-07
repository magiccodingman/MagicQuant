namespace MagicQuant.Commands;
using MagicQuant.Models;

public interface ICommand
{
    Task Run(List<CliArg> args);
}