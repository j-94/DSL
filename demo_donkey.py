#!/usr/bin/env python3
"""Demo of Donkey-DSL system without running server"""

from donkey_dsl import DonkeyDSL
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

if __name__ == "__main__":
    demo()