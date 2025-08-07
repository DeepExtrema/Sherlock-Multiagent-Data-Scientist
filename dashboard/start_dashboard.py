#!/usr/bin/env python3
"""
Deepline Dashboard Startup Script
Starts all services and provides interactive testing capabilities.
"""

import asyncio
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import httpx
import json

class DashboardStartup:
    def __init__(self):
        self.processes = []
        self.services = {
            'dashboard_backend': {
                'command': ['python', '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'],
                'cwd': 'backend',
                'url': 'http://localhost:8000',
                'health_endpoint': '/',
                'name': 'Dashboard Backend'
            },
            'eda_agent': {
                'command': ['python', '-m', 'uvicorn', 'eda_agent_simple:app', '--host', '0.0.0.0', '--port', '8001'],
                'cwd': '../mcp-server',
                'url': 'http://localhost:8001',
                'health_endpoint': '/health',
                'name': 'EDA Agent'
            },
            'ml_agent': {
                'command': ['python', '-m', 'uvicorn', 'ml_agent:app', '--host', '0.0.0.0', '--port', '8002'],
                'cwd': '../mcp-server',
                'url': 'http://localhost:8002',
                'health_endpoint': '/health',
                'name': 'ML Agent'
            }
        }
        
    def start_service(self, service_name: str):
        """Start a service in a new process."""
        service = self.services[service_name]
        print(f"🚀 Starting {service['name']}...")
        
        try:
            process = subprocess.Popen(
                service['command'],
                cwd=service['cwd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append((service_name, process))
            print(f"✅ {service['name']} started (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {e}")
            return None
    
    def wait_for_service(self, service_name: str, timeout=30):
        """Wait for a service to be ready."""
        service = self.services[service_name]
        print(f"⏳ Waiting for {service['name']} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"{service['url']}{service['health_endpoint']}", timeout=2)
                if response.status_code == 200:
                    print(f"✅ {service['name']} is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        print(f"⚠️ {service['name']} may not be ready (timeout)")
        return False
    
    def start_all_services(self):
        """Start all services."""
        print("🎯 Starting Deepline Dashboard Services")
        print("=" * 50)
        
        # Start services
        for service_name in self.services.keys():
            self.start_service(service_name)
            time.sleep(2)  # Give each service time to start
        
        # Wait for services to be ready
        print("\n⏳ Waiting for services to be ready...")
        for service_name in self.services.keys():
            self.wait_for_service(service_name)
        
        print("\n🎉 All services started!")
        return True
    
    def test_services(self):
        """Test all services."""
        print("\n🧪 Testing Services")
        print("=" * 30)
        
        for service_name, service in self.services.items():
            try:
                response = httpx.get(f"{service['url']}{service['health_endpoint']}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service['name']}: Working")
                else:
                    print(f"⚠️ {service['name']}: Status {response.status_code}")
            except Exception as e:
                print(f"❌ {service['name']}: {e}")
    
    def create_test_workflow(self):
        """Create a test workflow."""
        print("\n🔄 Creating Test Workflow")
        print("=" * 30)
        
        try:
            workflow_data = {
                "name": "Interactive Test Workflow",
                "description": "Workflow created during interactive testing",
                "steps": [
                    {
                        "name": "data_analysis",
                        "agent": "eda",
                        "parameters": {"dataset": "iris"}
                    },
                    {
                        "name": "model_training",
                        "agent": "ml",
                        "parameters": {"algorithm": "random_forest"}
                    }
                ]
            }
            
            response = httpx.post(
                "http://localhost:8000/runs",
                json=workflow_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Workflow created: {result.get('workflow_id', 'Unknown')}")
                return result
            else:
                print(f"⚠️ Workflow creation failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Workflow creation error: {e}")
            return None
    
    def show_dashboard_info(self):
        """Show dashboard information."""
        print("\n📊 Dashboard Information")
        print("=" * 30)
        print("🌐 Dashboard Backend: http://localhost:8000")
        print("📈 EDA Agent: http://localhost:8001")
        print("🤖 ML Agent: http://localhost:8002")
        print("🎨 Frontend: http://localhost:3000 (if started)")
        
        print("\n📋 Available Endpoints:")
        print("  Dashboard Backend:")
        print("    - GET  /           - Health check")
        print("    - GET  /runs       - List workflow runs")
        print("    - POST /runs       - Create new workflow")
        print("    - WS   /ws/events  - Real-time events")
        
        print("  EDA Agent:")
        print("    - GET  /health     - Health check")
        print("    - GET  /datasets   - List datasets")
        print("    - POST /basic_info - Dataset info")
        
        print("  ML Agent:")
        print("    - GET  /health     - Health check")
        print("    - GET  /metrics    - Prometheus metrics")
        print("    - GET  /experiments - List experiments")
    
    def interactive_menu(self):
        """Show interactive menu."""
        while True:
            print("\n" + "=" * 50)
            print("🎯 DEEPLINE DASHBOARD INTERACTIVE MENU")
            print("=" * 50)
            print("1. Test all services")
            print("2. Create test workflow")
            print("3. Show dashboard info")
            print("4. Open dashboard in browser")
            print("5. Exit")
            
            choice = input("\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                self.test_services()
            elif choice == '2':
                self.create_test_workflow()
            elif choice == '3':
                self.show_dashboard_info()
            elif choice == '4':
                try:
                    webbrowser.open('http://localhost:8000')
                    print("✅ Opened dashboard in browser")
                except Exception as e:
                    print(f"❌ Failed to open browser: {e}")
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
    
    def cleanup(self):
        """Clean up processes."""
        print("\n🧹 Cleaning up...")
        for service_name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {service_name}")
            except:
                try:
                    process.kill()
                    print(f"⚠️ Force killed {service_name}")
                except:
                    pass

def main():
    """Main function."""
    startup = DashboardStartup()
    
    try:
        # Start all services
        if startup.start_all_services():
            # Show initial info
            startup.show_dashboard_info()
            
            # Start interactive menu
            startup.interactive_menu()
        else:
            print("❌ Failed to start services")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        startup.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 