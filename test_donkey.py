#!/usr/bin/env python3
"""Test client for Donkey-DSL API"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_api():
    print("Testing Donkey-DSL API...\n")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Health Check:", response.json())
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the server is running: python donkey_dsl.py")
        return
    
    # Test recommendation endpoint
    test_tasks = [
        {"task": "fix login authentication bug", "context": {}},
        {"task": "refactor payment processing module", "context": {}},
        {"task": "explore new caching mechanisms", "context": {}},
        {"task": "fix production memory leak", "context": {}},
    ]
    
    print("\n--- Testing Recommendations ---")
    for test in test_tasks:
        response = requests.post(f"{BASE_URL}/donkey", json=test)
        result = response.json()
        print(f"\nTask: {test['task']}")
        print(f"Recommended DSL: {result['dsl']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Based on: {result['based_on']} similar tasks")
    
    # Test recording a new trajectory
    print("\n--- Recording New Trajectory ---")
    new_trajectory = {
        "task": "fix memory leak in user service",
        "dsl": "u0.4 f! | mem(0.9) | test(0.8)",
        "success": True,
        "time": 1500
    }
    
    response = requests.post(f"{BASE_URL}/record", json=new_trajectory)
    result = response.json()
    print(f"Recording result: {result}")
    
    # Get updated stats
    print("\n--- System Statistics ---")
    response = requests.get(f"{BASE_URL}/stats")
    stats = response.json()
    print(json.dumps(stats, indent=2))
    
    # Test the newly recorded task
    print("\n--- Testing After Learning ---")
    response = requests.post(f"{BASE_URL}/donkey", 
                           json={"task": "fix another memory leak", "context": {}})
    result = response.json()
    print(f"Task: fix another memory leak")
    print(f"Recommended DSL: {result['dsl']}")
    print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    test_api()