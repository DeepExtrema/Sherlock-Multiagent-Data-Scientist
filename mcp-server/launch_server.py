#!/usr/bin/env python3
"""
Launcher script for the MCP server with proper environment setup.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add debug output to stderr so Claude Desktop can see it
def debug_print(message):
    print(f"[DEBUG] {message}", file=sys.stderr)
    sys.stderr.flush()

debug_print("Starting MCP server launcher...")

# Get the current directory (where this script is located)
current_dir = Path(__file__).parent.absolute()
os.chdir(current_dir)

debug_print(f"Working directory: {current_dir}")

# Set the Python path to include our dependencies
# For Windows, we'll use the virtual environment if it exists
venv_path = current_dir / "venv"
if venv_path.exists():
    # Add virtual environment site-packages to Python path
    if os.name == 'nt':  # Windows
        site_packages = venv_path / "Lib" / "site-packages"
    else:  # Unix/Linux/macOS
        site_packages = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    if site_packages.exists():
        os.environ['PYTHONPATH'] = str(site_packages)
        debug_print(f"Using virtual environment: {venv_path}")
    else:
        debug_print(f"Warning: Virtual environment site-packages not found at {site_packages}")
else:
    debug_print("Warning: Virtual environment not found. Using system Python.")

# Import and run the server
try:
    debug_print("Importing MCP server module...")
    from server import mcp
    debug_print("MCP server imported successfully")
except ImportError as e:
    debug_print(f"Error importing server: {e}")
    debug_print("Please ensure all dependencies are installed:")
    debug_print("  pip install -r requirements-exact.txt")
    sys.exit(1)

if __name__ == "__main__":
    debug_print("Starting MCP server...")
    try:
        mcp.run()
    except Exception as e:
        debug_print(f"Error running MCP server: {e}")
        sys.exit(1) 