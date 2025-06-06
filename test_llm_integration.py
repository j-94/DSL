#!/usr/bin/env python3
"""Test LLM-Donkey integration"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_llm_execution():
    """Test single LLM execution with DSL"""
    print("\n=== Testing LLM Execution with DSL ===")
    
    tasks = [
        {
            "task": "fix authentication bug in API",
            "prompt": "Debug and fix the authentication issue where tokens expire too quickly",
            "context": {"error": "JWT token expired", "file": "auth.py"}
        },
        {
            "task": "explore new caching strategies",
            "prompt": "Research and propose innovative caching solutions for our microservices",
            "context": {"current": "Redis", "issues": "cache invalidation"}
        },
        {
            "task": "refactor database connection pool",
            "prompt": "Improve the database connection pooling implementation",
            "context": {"database": "PostgreSQL", "framework": "SQLAlchemy"}
        }
    ]
    
    for task_data in tasks:
        print(f"\nTask: {task_data['task']}")
        
        # Execute with DSL
        response = requests.post(f"{BASE_URL}/llm/execute", json=task_data)
        if response.status_code == 200:
            result = response.json()
            print(f"DSL: {result['dsl']['recommended']}")
            print(f"Confidence: {result['dsl']['confidence']}")
            print(f"Actual behavior - u: {result['behavior']['actual_u']:.1f}")
            print(f"Compliance: {result['behavior']['compliance']:.0%}")
            print(f"Response preview: {result['response'][:100]}...")
        else:
            print(f"Error: {response.text}")

def test_conversation():
    """Test multi-turn conversation with DSL adaptation"""
    print("\n\n=== Testing DSL-Guided Conversation ===")
    
    # Start conversation
    start_response = requests.post(f"{BASE_URL}/llm/conversation/start", json={
        "task": "implement user authentication system"
    })
    
    if start_response.status_code != 200:
        print(f"Failed to start conversation: {start_response.text}")
        return
    
    conv_data = start_response.json()
    conversation_id = conv_data['conversation_id']
    print(f"Started conversation: {conversation_id}")
    print(f"Initial DSL: {conv_data['initial_dsl']}")
    
    # Conversation turns
    turns = [
        "What authentication methods should we support?",
        "Let's focus on JWT tokens. How should we handle refresh tokens?",
        "Can you show me a more experimental approach using WebAuthn?",
        "Actually, let's stick with the standard JWT implementation for now."
    ]
    
    for i, user_input in enumerate(turns):
        print(f"\n--- Turn {i+1} ---")
        print(f"User: {user_input}")
        
        turn_response = requests.post(f"{BASE_URL}/llm/conversation/turn", json={
            "conversation_id": conversation_id,
            "input": user_input
        })
        
        if turn_response.status_code == 200:
            turn_data = turn_response.json()
            print(f"Current DSL: {turn_data['current_dsl']['dsl']}")
            print(f"Behavior u: {turn_data['behavior']['actual_u']:.1f}")
            print(f"Response: {turn_data['response']}")
        else:
            print(f"Error: {turn_response.text}")
    
    # End conversation
    end_response = requests.post(f"{BASE_URL}/llm/conversation/end", json={
        "conversation_id": conversation_id
    })
    
    if end_response.status_code == 200:
        end_data = end_response.json()
        print(f"\n--- Conversation Summary ---")
        print(f"Trajectory recorded: {end_data['recorded']}")
        print(f"Success estimate: {end_data['trajectory']['success']:.0%}")
        print(f"Average compliance: {end_data['trajectory']['avg_compliance']:.0%}")

def test_calibration():
    """Test calibration data retrieval"""
    print("\n\n=== Testing Calibration Data ===")
    
    response = requests.get(f"{BASE_URL}/llm/calibrate")
    if response.status_code == 200:
        calibration = response.json()['calibration']
        
        print("U parameter effects (requested → actual):")
        for requested, actual in calibration['u_effects'].items():
            print(f"  u{requested} → {actual}")
        
        print(f"\nTest compliance rate: {calibration['test_compliance']:.0%}")
        print(f"Memory compliance rate: {calibration['memory_compliance']:.0%}")

def test_comparison():
    """Compare with and without DSL"""
    print("\n\n=== Comparing With vs Without DSL ===")
    
    task_data = {
        "task": "optimize database queries",
        "prompt": "Analyze and optimize slow database queries"
    }
    
    # With DSL
    print("With DSL:")
    with_dsl = requests.post(f"{BASE_URL}/llm/execute", json={
        **task_data,
        "use_donkey": True
    })
    
    if with_dsl.status_code == 200:
        result = with_dsl.json()
        print(f"  DSL: {result['dsl']['recommended']}")
        print(f"  Approaches found: {result['behavior']['unique_approaches']}")
    
    # Without DSL
    print("\nWithout DSL:")
    without_dsl = requests.post(f"{BASE_URL}/llm/execute", json={
        **task_data,
        "use_donkey": False
    })
    
    if without_dsl.status_code == 200:
        result = without_dsl.json()
        print(f"  Standard execution (no behavior modification)")

def main():
    """Run all tests"""
    print("Testing LLM-Donkey Integration")
    print("Make sure the server is running: python donkey_dsl.py")
    
    try:
        # Check if server is running
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code != 200:
            print("Server is not responding. Please start it first.")
            return
        
        test_llm_execution()
        test_conversation()
        test_calibration()
        test_comparison()
        
        print("\n\nAll tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Make sure it's running on port 5000.")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()