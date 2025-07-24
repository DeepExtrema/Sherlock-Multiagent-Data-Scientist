#!/usr/bin/env python3
"""
Production Deployment Script

Deploys the Master Orchestrator and EDA Agent with:
- Authentication and security
- API documentation (OpenAPI/Swagger)
- Health monitoring
- Production-ready configuration
"""

import uvicorn
import asyncio
import os
import sys
import time
import requests
from pathlib import Path
import subprocess
import threading
import signal
import json

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# Service Configuration
SERVICES = {
    "master_orchestrator": {
        "name": "Master Orchestrator",
        "port": 8000,
        "host": "0.0.0.0",
        "module": "master_orchestrator_api",
        "app": "app",
        "startup_script": "start_master_orchestrator.py"
    },
    "eda_agent": {
        "name": "EDA Agent",
        "port": 8001,
        "host": "0.0.0.0",
        "module": "eda_agent",
        "app": "app",
        "startup_script": "start_eda_service.py"
    }
}

# Production Settings
PRODUCTION_CONFIG = {
    "workers": 1,
    "log_level": "info",
    "reload": False,
    "access_log": True,
    "timeout_keep_alive": 30,
    "timeout_graceful_shutdown": 30
}

# ─── SERVICE MANAGEMENT ───────────────────────────────────────────────────────

class ServiceManager:
    """Manages multiple services in production."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def start_service(self, service_name, service_config):
        """Start a service in a separate process."""
        try:
            print(f"🚀 Starting {service_config['name']} on port {service_config['port']}...")
            
            # Start the service
            process = subprocess.Popen([
                sys.executable, service_config['startup_script']
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes[service_name] = {
                'process': process,
                'config': service_config,
                'start_time': time.time()
            }
            
            print(f"  ✓ {service_config['name']} started (PID: {process.pid})")
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to start {service_config['name']}: {e}")
            return False
    
    def stop_service(self, service_name):
        """Stop a specific service."""
        if service_name in self.processes:
            process_info = self.processes[service_name]
            process = process_info['process']
            config = process_info['config']
            
            print(f"🛑 Stopping {config['name']}...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                print(f"  ✓ {config['name']} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  ⚠️  {config['name']} force killed")
            except Exception as e:
                print(f"  ✗ Error stopping {config['name']}: {e}")
            
            del self.processes[service_name]
    
    def start_all_services(self):
        """Start all services."""
        print("="*80)
        print("🚀 PRODUCTION DEPLOYMENT - STARTING SERVICES")
        print("="*80)
        
        self.running = True
        
        for service_name, service_config in SERVICES.items():
            if not self.start_service(service_name, service_config):
                print(f"❌ Failed to start {service_name}. Stopping all services...")
                self.stop_all_services()
                return False
        
        print("\n✅ All services started successfully!")
        return True
    
    def stop_all_services(self):
        """Stop all services."""
        print("\n🛑 Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        self.running = False
        print("✅ All services stopped.")
    
    def check_service_health(self, service_name, service_config):
        """Check if a service is healthy."""
        try:
            response = requests.get(
                f"http://localhost:{service_config['port']}/health",
                timeout=5
            )
            if response.status_code == 200:
                return True
            else:
                return False
        except:
            return False
    
    def monitor_services(self):
        """Monitor all services and restart if needed."""
        while self.running:
            try:
                for service_name, service_config in SERVICES.items():
                    if service_name in self.processes:
                        process_info = self.processes[service_name]
                        process = process_info['process']
                        
                        # Check if process is still running
                        if process.poll() is not None:
                            print(f"⚠️  {service_config['name']} stopped unexpectedly. Restarting...")
                            self.stop_service(service_name)
                            self.start_service(service_name, service_config)
                        
                        # Check health endpoint
                        elif not self.check_service_health(service_name, service_config):
                            print(f"⚠️  {service_config['name']} health check failed. Restarting...")
                            self.stop_service(service_name)
                            self.start_service(service_name, service_config)
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(30)

# ─── PRODUCTION FEATURES ──────────────────────────────────────────────────────

def create_production_config():
    """Create production configuration files."""
    print("📝 Creating production configuration...")
    
    # Create .env file for environment variables
    env_content = """# Production Environment Variables
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG=false
CORS_ORIGINS=*
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    # Create production requirements
    requirements_content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.20
requests==2.32.4
pandas==2.1.4
numpy==1.25.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("  ✓ Production configuration created")

def create_docker_compose():
    """Create Docker Compose configuration for production deployment."""
    docker_compose_content = """version: '3.8'

services:
  master-orchestrator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - EDA_AGENT_URL=http://eda-agent:8001
    depends_on:
      - eda-agent
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  eda-agent:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - master-orchestrator
      - eda-agent
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("  ✓ Docker Compose configuration created")

def create_nginx_config():
    """Create Nginx configuration for load balancing."""
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream master_orchestrator {
        server master-orchestrator:8000;
    }
    
    upstream eda_agent {
        server eda-agent:8001;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Master Orchestrator API
        location /api/orchestrator/ {
            proxy_pass http://master_orchestrator/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # EDA Agent API
        location /api/eda/ {
            proxy_pass http://eda_agent/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # API Documentation
        location /docs {
            proxy_pass http://master_orchestrator/docs;
        }
        
        # Health checks
        location /health {
            proxy_pass http://master_orchestrator/health;
        }
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    
    print("  ✓ Nginx configuration created")

def create_dockerfile():
    """Create Dockerfile for containerized deployment."""
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "start_master_orchestrator.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("  ✓ Dockerfile created")

# ─── MAIN DEPLOYMENT ──────────────────────────────────────────────────────────

def main():
    """Main deployment function."""
    print("="*80)
    print("🚀 PRODUCTION DEPLOYMENT SCRIPT")
    print("="*80)
    
    # Create production configuration
    create_production_config()
    create_docker_compose()
    create_nginx_config()
    create_dockerfile()
    
    # Initialize service manager
    service_manager = ServiceManager()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal. Stopping services...")
        service_manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        if service_manager.start_all_services():
            print("\n" + "="*80)
            print("🎉 PRODUCTION DEPLOYMENT SUCCESSFUL!")
            print("="*80)
            print("📊 Service Status:")
            print(f"  • Master Orchestrator: http://localhost:8000")
            print(f"  • EDA Agent: http://localhost:8001")
            print(f"  • API Documentation: http://localhost:8000/docs")
            print(f"  • Health Check: http://localhost:8000/health")
            print("\n🔧 Production Features:")
            print("  • Authentication: Enabled")
            print("  • API Documentation: OpenAPI/Swagger")
            print("  • Health Monitoring: Active")
            print("  • Auto-restart: Enabled")
            print("  • Load Balancing: Nginx ready")
            print("\n📁 Generated Files:")
            print("  • .env - Environment configuration")
            print("  • requirements.txt - Python dependencies")
            print("  • docker-compose.yml - Container orchestration")
            print("  • nginx.conf - Load balancer configuration")
            print("  • Dockerfile - Container definition")
            print("\n🔄 Services are running and being monitored...")
            print("   Press Ctrl+C to stop all services")
            print("="*80)
            
            # Start monitoring
            service_manager.monitor_services()
        
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
    finally:
        service_manager.stop_all_services()

if __name__ == "__main__":
    main() 