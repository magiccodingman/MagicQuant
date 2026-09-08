import os
import sys
import runpy

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths for key directories
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')

# Add the necessary paths to sys.path
sys.path.insert(0, lib_dir)
sys.path.insert(0, site_packages_dir)

# Check if pip is available
try:
    import pip
except ImportError:
    print("pip is not available in the current environment.", file=sys.stderr)
    sys.exit(1)

def run_pip_command(command):
    """
    Run pip commands dynamically using pip directly as a module.
    """
    # Prepare arguments for pip by splitting the command string
    sys.argv = ['pip'] + command.split()

    # Run pip using runpy to run pip as a module
    try:
        runpy.run_module('pip', run_name="__main__")
    except Exception as e:
        print(f"Failed to run pip command: {e}", file=sys.stderr)

# Main execution logic
if __name__ == "__main__":
    # If there are command-line arguments, use them
    if len(sys.argv) > 1:
        # Use arguments passed to the script (excluding the script name)
        run_pip_command(' '.join(sys.argv[1:]))
    else:
        # Default to checking pip version if no arguments are passed
        run_pip_command('--version')
