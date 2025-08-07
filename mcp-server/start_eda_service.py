#!/usr/bin/env python3
"""
Simple EDA Agent Service Startup Script
"""

import uvicorn
from eda_agent import app

if __name__ == "__main__":
    print("Starting EDA Agent Service...")
    print("Service will be available at: http://localhost:8001")
    print("Health check: http://localhost:8001/health")
    print("Press Ctrl+C to stop the service")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )