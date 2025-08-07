#!/usr/bin/env python3
"""
Simple test runner for the Dashboard E2E Test
This script sets up the environment and runs the comprehensive end-to-end test.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker is not installed or not accessible")
            return False
        print("✅ Docker is available")
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False
    
    # Check Docker Compose
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Docker Compose is not installed or not accessible")
            return False
        print("✅ Docker Compose is available")
    except FileNotFoundError:
        print("❌ Docker Compose is not installed")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Node.js is not installed or not accessible")
            return False
        print("✅ Node.js is available")
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        return False
    
    # Check npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ npm is not installed or not accessible")
            return False
        print("✅ npm is available")
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False
    
    return True

def install_test_dependencies():
    """Install test dependencies."""
    print("📦 Installing test dependencies...")
    
    try:
        # Install Python dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "test_requirements.txt"], check=True)
        print("✅ Python dependencies installed")
        
        # Install frontend dependencies
        frontend_dir = Path("dashboard-frontend")
        if frontend_dir.exists():
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("✅ Frontend dependencies installed")
        else:
            print("⚠️ Frontend directory not found, skipping npm install")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def run_test():
    """Run the end-to-end test."""
    print("🚀 Running Dashboard E2E Test...")
    print("="*60)
    
    try:
        # Run the test
        result = subprocess.run([sys.executable, "test_dashboard_e2e.py"], check=True)
        
        if result.returncode == 0:
            print("\n🎉 Test completed successfully!")
            return True
        elif result.returncode == 1:
            print("\n⚠️ Test completed with warnings")
            return True
        else:
            print("\n❌ Test failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return False

def cleanup():
    """Clean up after test."""
    print("🧹 Cleaning up...")
    
    try:
        # Stop docker-compose services
        subprocess.run(["docker-compose", "down"], check=True)
        print("✅ Docker services stopped")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to stop Docker services")
    except FileNotFoundError:
        print("⚠️ Docker Compose not found")

def main():
    """Main function."""
    print("Deepline Dashboard End-to-End Test Runner")
    print("="*50)
    
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Install dependencies
    if not install_test_dependencies():
        print("\n❌ Failed to install dependencies.")
        sys.exit(1)
    
    # Run test
    success = run_test()
    
    # Cleanup
    cleanup()
    
    # Exit with appropriate code
    if success:
        print("\n✅ All tests completed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 