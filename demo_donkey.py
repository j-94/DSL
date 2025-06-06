#!/usr/bin/env python3
"""Demo of Donkey-DSL system with LLM integration"""

from donkey_dsl import DonkeyDSL
from llm_donkey_integration import (
    PromptInjector,
    LLMBehaviorAnalyzer,
    DSLConversation,
    DSLCalibrator
)
import json

def demo():
    print("=== Donkey-DSL Demo ===\n")
    
    # Create instance
    donkey = DonkeyDSL()
    print(f"Loaded {len(donkey.trajectories)} trajectories\n")
    
    # Test recommendations
    test_tasks = [
        "fix authentication bug in API",
        "refactor database connection pool",
        "explore new ML model architectures", 
        "fix production timeout issue",
        "implement new caching layer"
    ]
    
    print("--- Testing Recommendations ---")
    for task in test_tasks:
        dsl, confidence, based_on = donkey.donkey(task)
        print(f"\nTask: {task}")
        print(f"Recommended DSL: {dsl}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Based on: {based_on} similar tasks")
    
    # Show statistics
    print("\n--- System Statistics ---")
    stats = donkey.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Demonstrate learning
    print("\n--- Recording New Trajectory ---")
    success_rate = donkey.record_trajectory(
        task="fix memory leak in cache service",
        dsl="u0.3 f! | mem(0.9) | test(0.9)",
        success=True,
        time=1400
    )
    print(f"Recorded! New success rate for this DSL: {success_rate:.2f}")
    
    # Test again with similar task
    print("\n--- Testing After Learning ---")
    dsl, confidence, based_on = donkey.donkey("fix memory leak in user service")
    print(f"Task: fix memory leak in user service")
    print(f"Recommended DSL: {dsl}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Based on: {based_on} similar tasks")
    
    # Show DSL parameter patterns
    print("\n--- DSL Pattern Analysis ---")
    patterns = {
        "Bug fixes": [t for t in donkey.trajectories if 'fix' in t.task.lower() or 'bug' in t.task.lower()],
        "Refactoring": [t for t in donkey.trajectories if 'refactor' in t.task.lower()],
        "Exploration": [t for t in donkey.trajectories if 'explore' in t.task.lower()],
        "Production": [t for t in donkey.trajectories if 'production' in t.task.lower() or 'prod' in t.task.lower()]
    }
    
    for pattern_name, trajectories in patterns.items():
        if trajectories:
            print(f"\n{pattern_name}:")
            successful = [t for t in trajectories if t.success]
            if successful:
                # Average parameters for successful trajectories
                avg_params = {"u": 0, "test": 0, "mem": 0}
                for t in successful:
                    params = donkey.parse_dsl(t.dsl)
                    avg_params["u"] += params.u
                    avg_params["test"] += params.test
                    avg_params["mem"] += params.mem
                
                for k in avg_params:
                    avg_params[k] /= len(successful)
                
                print(f"  Average u: {avg_params['u']:.2f}")
                print(f"  Average test: {avg_params['test']:.2f}")
                print(f"  Average mem: {avg_params['mem']:.2f}")
    
    # LLM Integration Demo
    print("\n\n=== LLM INTEGRATION DEMO ===")
    
    # Create integration components
    injector = PromptInjector()
    analyzer = LLMBehaviorAnalyzer()
    
    # Demonstrate prompt injection
    print("\n--- Prompt Injection Examples ---")
    
    test_prompts = [
        ("fix critical bug in payment system", {"u": 0.2, "test_prob": 1.0, "mem_thresh": 0.9}),
        ("explore novel database architectures", {"u": 4.0, "test_prob": 0.2, "mem_thresh": 0.3}),
        ("refactor user authentication", {"u": 1.0, "test_prob": 0.5, "mem_thresh": 0.5})
    ]
    
    for base_prompt, dsl_params in test_prompts:
        print(f"\nBase prompt: {base_prompt}")
        print(f"DSL params: u={dsl_params['u']}, test_prob={dsl_params['test_prob']:.0%}")
        injected = injector.inject_dsl_behavior(base_prompt, dsl_params)
        print("Injected prompt:")
        print(injected[:200] + "...")
    
    # Demonstrate behavior analysis
    print("\n\n--- Behavior Analysis ---")
    
    sample_responses = [
        ("Let me explore several unconventional approaches:\n"
         "Approach 1: We could use a novel graph-based algorithm...\n"
         "Approach 2: An experimental neural architecture might work...\n"
         "def test_solution():\n    assert validate_output() == True",
         {"u": 3.0, "test_prob": 0.8, "mem_thresh": 0.5}),
        
        ("I'll use the standard, proven approach for this:\n"
         "The conventional solution involves using established patterns.\n"
         "This traditional method has been tested extensively.",
         {"u": 0.3, "test_prob": 0.5, "mem_thresh": 0.8})
    ]
    
    for response, recommended_dsl in sample_responses:
        behavior = analyzer.measure_actual_dsl(response, recommended_dsl)
        print(f"\nResponse preview: {response[:80]}...")
        print(f"Recommended u: {recommended_dsl['u']:.1f}")
        print(f"Actual u measured: {behavior.actual_u:.1f}")
        print(f"Test detected: {behavior.actual_test}")
        print(f"Compliance score: {behavior.followed_recommendation:.0%}")
        print(f"Unique approaches: {behavior.unique_approaches}")
    
    # Demonstrate calibration
    print("\n\n--- DSL Calibration ---")
    calibrator = DSLCalibrator()
    
    # Mock LLM client for demo
    class MockLLM:
        def complete(self, prompt):
            from collections import namedtuple
            Response = namedtuple('Response', ['content'])
            return Response(content="Mock response based on prompt")
    
    print("Calibrating DSL effects (simulated)...")
    transfer_functions = calibrator.calibrate_dsl_effects(
        MockLLM(), 
        ["solve a simple problem", "implement a feature", "debug an issue"]
    )
    
    print("\nTransfer functions (requested → actual):")
    for requested, actual in transfer_functions.items():
        print(f"  u{requested} → {actual:.1f}")
    
    # Demonstrate conversation
    print("\n\n--- DSL-Guided Conversation ---")
    
    conv = DSLConversation("build a recommendation engine", donkey)
    print(f"Task: {conv.task}")
    print(f"Initial DSL: {conv.dsl['dsl']}")
    
    # Simulate conversation turns
    turns = [
        "What algorithms should we consider?",
        "Let's explore more creative approaches",
        "Actually, we need something production-ready"
    ]
    
    for i, user_input in enumerate(turns):
        print(f"\nTurn {i+1}: {user_input}")
        response, behavior = conv.turn(user_input, MockLLM())
        print(f"Current DSL: {conv.dsl['dsl']}")
        print(f"Behavior compliance: {behavior.followed_recommendation:.0%}")
    
    # Get trajectory
    trajectory = conv.get_trajectory()
    print(f"\nConversation trajectory:")
    print(f"  Success estimate: {trajectory['success']:.0%}")
    print(f"  Average compliance: {trajectory['avg_compliance']:.0%}")
    print(f"  Total turns: {trajectory['turns']}")

if __name__ == "__main__":
    demo()