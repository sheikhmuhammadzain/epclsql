#!/usr/bin/env python3
"""
Test script to verify the system works without authentication
"""

import requests
import json

def test_no_auth():
    """Test that the API works without authentication."""
    
    base_url = "http://127.0.0.1:8000"
    
    print("Testing EPCL VEHS API without authentication...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint works")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test 2: Metrics
    print("\n2. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            print("✅ Metrics endpoint works")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
    
    # Test 3: SQL Query (for simple version)
    print("\n3. Testing SQL execute endpoint...")
    try:
        query_data = {
            "query": "SELECT COUNT(*) as total FROM incident"
        }
        response = requests.post(
            f"{base_url}/execute", 
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ SQL execute endpoint works")
                print(f"   Query result: {result.get('data')}")
            else:
                print(f"⚠️  SQL execute returned error: {result.get('error')}")
        else:
            print(f"❌ SQL execute failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ SQL execute error: {e}")
    
    # Test 4: Natural Language Query (for full version)
    print("\n4. Testing natural language query endpoint...")
    try:
        query_data = {
            "query": "How many incidents are there?",
            "include_explanation": True
        }
        response = requests.post(
            f"{base_url}/query", 
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Natural language query endpoint works")
                print(f"   Query result: {result.get('response', {}).get('answer', 'No answer')[:100]}...")
            else:
                print(f"⚠️  Natural language query returned error: {result.get('error')}")
        else:
            print(f"❌ Natural language query failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Natural language query error: {e}")
    
    print("\n" + "="*60)
    print("Test completed! If you see ✅ for the endpoints you're using, the system is working.")
    print("You can now use the web interface at: http://127.0.0.1:8000")

if __name__ == "__main__":
    test_no_auth()
