#!/usr/bin/env python3
"""
Start Master Orchestrator Service

Simple script to start the Master Orchestrator API service.
"""

import uvicorn
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from master_orchestrator_api import app

if __name__ == "__main__":
    print("🚀 Starting Master Orchestrator API...")
    print("📍 Service URL: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    ) 