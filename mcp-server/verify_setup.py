#!/usr/bin/env python3
"""
Verify MCP server setup for Claude Desktop.
"""

import json
import os
import sys

def main():
    print("🔍 Verifying MCP Server Setup for Claude Desktop")
    print("=" * 60)
    
    # Check configuration file
    config_path = os.path.expandvars(r"%APPDATA%\Claude\claude_desktop_config.json")
    print(f"📁 Configuration file: {config_path}")
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found!")
        return 1
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration file is valid JSON")
        
        # Check if our server is configured
        if "mcpServers" in config and "deepline-eda" in config["mcpServers"]:
            server_config = config["mcpServers"]["deepline-eda"]
            print("✅ deepline-eda server configured")
            print(f"   Command: {server_config.get('command', 'Not set')}")
            print(f"   Args: {server_config.get('args', 'Not set')}")
            print(f"   Working Dir: {server_config.get('cwd', 'Not set')}")
        else:
            print("❌ deepline-eda server not found in configuration")
            return 1
            
    except json.JSONDecodeError as e:
        print(f"❌ Configuration file has invalid JSON: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return 1
    
    # Check if Python executable exists
    python_exe = server_config.get('command', '')
    if os.path.exists(python_exe):
        print("✅ Python executable found")
    else:
        print(f"❌ Python executable not found: {python_exe}")
        return 1
    
    # Check if launch script exists
    cwd = server_config.get('cwd', '')
    launch_script = os.path.join(cwd, 'launch_server.py')
    if os.path.exists(launch_script):
        print("✅ Launch script found")
    else:
        print(f"❌ Launch script not found: {launch_script}")
        return 1
    
    print("\n🎉 Setup verification complete!")
    print("\n📋 Next steps:")
    print("1. Close Claude Desktop completely")
    print("2. Restart Claude Desktop")
    print("3. Look for 'deepline-eda' in the MCP servers list")
    print("4. Test with: 'Load the iris.csv dataset'")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 