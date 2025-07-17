"""
Master Orchestrator Installation Script

This script installs dependencies and sets up the Master Orchestrator system.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors."""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed:")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    logger.info("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def install_dependencies():
    """Install required Python packages."""
    logger.info("📦 Installing Master Orchestrator dependencies...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        logger.warning("⚠️  Not in a virtual environment. Consider creating one for isolation.")
    
    # Install dependencies
    requirements_file = Path("requirements-python313.txt")
    if requirements_file.exists():
        success = run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing dependencies from requirements-python313.txt"
        )
        if not success:
            logger.error("Failed to install dependencies. Please check the error messages above.")
            return False
    else:
        logger.error(f"Requirements file not found: {requirements_file}")
        logger.info("Installing core dependencies manually...")
        
        core_deps = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0", 
            "pydantic>=2.11.7",
            "pyyaml>=6.0.1",
            "redis>=5.0.0",
            "aioredis>=2.0.0",
            "httpx>=0.25.0",
            "tenacity>=8.2.0",
            "bleach>=6.1.0",
            "validators>=0.22.0"
        ]
        
        for dep in core_deps:
            success = run_command(
                f"{sys.executable} -m pip install {dep}",
                f"Installing {dep}"
            )
            if not success:
                logger.warning(f"Failed to install {dep}, continuing with others...")
    
    return True

def create_directories():
    """Create necessary directories."""
    logger.info("📁 Creating necessary directories...")
    
    directories = [
        "reports",
        "logs",
        "orchestrator/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return True

def check_infrastructure():
    """Check if optional infrastructure is available."""
    logger.info("🔍 Checking optional infrastructure...")
    
    # Check Redis
    redis_available = run_command("redis-cli ping", "Checking Redis connection")
    if not redis_available:
        logger.warning("⚠️  Redis not available. Master Orchestrator will use in-memory cache.")
    
    # Check MongoDB
    mongo_available = run_command("mongosh --eval 'db.adminCommand(\"ping\")'", "Checking MongoDB connection")
    if not mongo_available:
        logger.warning("⚠️  MongoDB not available. Some features may not work.")
    
    # Check Kafka
    kafka_available = False  # Would need more complex check
    logger.warning("⚠️  Kafka check skipped. Ensure Kafka is running for full functionality.")
    
    return {
        "redis": redis_available,
        "mongodb": mongo_available, 
        "kafka": kafka_available
    }

def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running Master Orchestrator tests...")
    
    test_file = Path("test_master_orchestrator.py")
    if test_file.exists():
        success = run_command(
            f"{sys.executable} {test_file}",
            "Running Master Orchestrator tests"
        )
        return success
    else:
        logger.warning("Test file not found, skipping tests")
        return True

def create_startup_script():
    """Create a startup script for the Master Orchestrator."""
    logger.info("📝 Creating startup script...")
    
    startup_script = """#!/usr/bin/env python3
\"\"\"
Master Orchestrator Startup Script
\"\"\"

import subprocess
import sys
import os
import time

def start_master_orchestrator():
    print("🚀 Starting Master Orchestrator API...")
    
    try:
        # Change to the correct directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "master_orchestrator_api.py"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\\n🛑 Master Orchestrator stopped by user")
    except Exception as e:
        print(f"❌ Error starting Master Orchestrator: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_master_orchestrator()
"""
    
    with open("start_orchestrator.py", "w") as f:
        f.write(startup_script)
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_orchestrator.py", 0o755)
    
    logger.info("✅ Created start_orchestrator.py")
    return True

def main():
    """Main installation process."""
    logger.info("🚀 Starting Master Orchestrator Installation...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Dependency installation failed")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("❌ Directory creation failed") 
        sys.exit(1)
    
    # Check infrastructure
    infrastructure_status = check_infrastructure()
    
    # Create startup script
    if not create_startup_script():
        logger.error("❌ Startup script creation failed")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        logger.warning("⚠️  Some tests failed, but installation continues")
    
    # Final summary
    logger.info("🎉 Master Orchestrator installation completed!")
    logger.info("")
    logger.info("📋 Installation Summary:")
    logger.info("✅ Dependencies installed")
    logger.info("✅ Directories created")
    logger.info("✅ Startup script created")
    
    logger.info("")
    logger.info("🏃 Next steps:")
    logger.info("1. Review the configuration in config.yaml")
    logger.info("2. Start infrastructure services (Redis, MongoDB, Kafka) if needed")
    logger.info("3. Run the Master Orchestrator:")
    logger.info("   python start_orchestrator.py")
    logger.info("   OR")
    logger.info("   python master_orchestrator_api.py")
    logger.info("")
    logger.info("🌐 API will be available at: http://localhost:8001")
    logger.info("📚 API docs will be at: http://localhost:8001/docs")
    
    # Infrastructure warnings
    if not infrastructure_status["redis"]:
        logger.info("")
        logger.info("💡 To install Redis:")
        logger.info("   - Windows: Download from https://redis.io/download")
        logger.info("   - macOS: brew install redis")
        logger.info("   - Linux: sudo apt-get install redis-server")
    
    if not infrastructure_status["mongodb"]:
        logger.info("")
        logger.info("💡 To install MongoDB:")
        logger.info("   - Follow instructions at https://docs.mongodb.com/manual/installation/")

if __name__ == "__main__":
    main() 