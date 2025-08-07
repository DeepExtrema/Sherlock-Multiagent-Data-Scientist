#!/usr/bin/env python3
"""
Interactive Dashboard Test
Demonstrates the working dashboard functionality.
"""

import httpx
import json
import time

def test_dashboard():
    """Test the dashboard functionality."""
    print("🎯 Testing Deepline Dashboard")
    print("=" * 40)
    
    # Test Dashboard Backend
    print("\n1. Testing Dashboard Backend...")
    try:
        response = httpx.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard Backend: Working")
            print(f"   Response: {response.text[:100]}...")
        else:
            print(f"⚠️ Dashboard Backend: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard Backend: {e}")
    
    # Test EDA Agent
    print("\n2. Testing EDA Agent...")
    try:
        response = httpx.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ EDA Agent: Working")
            print(f"   Response: {response.text[:100]}...")
        else:
            print(f"⚠️ EDA Agent: Status {response.status_code}")
    except Exception as e:
        print(f"❌ EDA Agent: {e}")
    
    # Test ML Agent
    print("\n3. Testing ML Agent...")
    try:
        response = httpx.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ ML Agent: Working")
            print(f"   Response: {response.text[:100]}...")
        else:
            print(f"⚠️ ML Agent: Status {response.status_code}")
    except Exception as e:
        print(f"❌ ML Agent: {e}")
    
    # Test Workflow Creation
    print("\n4. Testing Workflow Creation...")
    try:
        workflow_data = {
            "name": "Interactive Demo Workflow",
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
            print("✅ Workflow Creation: Success")
            print(f"   Workflow ID: {result.get('workflow_id', 'Unknown')}")
            print(f"   Run ID: {result.get('run_id', 'Unknown')}")
        else:
            print(f"⚠️ Workflow Creation: Status {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Workflow Creation: {e}")
    
    # Test Workflow Listing
    print("\n5. Testing Workflow Listing...")
    try:
        response = httpx.get("http://localhost:8000/runs", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Workflow Listing: Success")
            runs = result.get('runs', [])
            print(f"   Found {len(runs)} workflow runs")
            for run in runs[:3]:  # Show first 3 runs
                print(f"   - {run.get('name', 'Unknown')} (ID: {run.get('run_id', 'Unknown')})")
        else:
            print(f"⚠️ Workflow Listing: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Workflow Listing: {e}")
    
    print("\n🎉 Interactive Dashboard Test Complete!")
    print("\n🌐 Access your dashboard at: http://localhost:8000")
    print("📈 EDA Agent at: http://localhost:8001")
    print("🤖 ML Agent at: http://localhost:8002")

if __name__ == "__main__":
    test_dashboard() 