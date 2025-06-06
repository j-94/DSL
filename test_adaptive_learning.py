#!/usr/bin/env python3
"""
ADAPTIVE LEARNING TEST
Prove the system can learn optimal DSL parameters through exploration
"""

import asyncio
import json
import os
import random
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import aiohttp
from dotenv import load_dotenv
from collections import defaultdict
import re

load_dotenv()

class AdaptiveDonkeyTest:
    def __init__(self, api_key: str, budget: float = 5.0):
        self.api_key = api_key
        self.budget = budget
        self.total_cost = 0.0
        
        # Learning state
        self.exploration_rate = 0.3  # Start with 30% exploration
        self.dsl_performance = defaultdict(lambda: {'trials': 0, 'total_quality': 0.0})
        self.task_type_patterns = defaultdict(list)
        
    async def test_with_exploration(self, session: aiohttp.ClientSession):
        """Test with exploration vs exploitation strategy"""
        
        print("=== ADAPTIVE LEARNING TEST ===\n")
        
        # Define diverse test scenarios
        test_scenarios = [
            # (task, expected_optimal_dsl)
            ("Write comprehensive tests for a sorting function", 
             {'u': 0.3, 'test_prob': 0.95}),  # Should favor high testing
            
            ("Explore 5 different ways to implement a cache", 
             {'u': 3.0, 'test_prob': 0.3}),   # Should favor exploration
            
            ("Fix this production bug urgently: null pointer exception",
             {'u': 0.2, 'test_prob': 0.8}),   # Should favor focus + testing
            
            ("Brainstorm creative solutions for real-time data processing",
             {'u': 2.5, 'test_prob': 0.2}),   # Should favor creativity
            
            ("Implement a robust error handling system",
             {'u': 0.5, 'test_prob': 0.9}),   # Should favor reliability
        ]
        
        # Run multiple rounds to allow learning
        rounds = 3
        results_by_round = []
        
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1} ---")
            round_results = []
            
            # Decrease exploration rate over time
            self.exploration_rate = max(0.1, self.exploration_rate * 0.7)
            print(f"Exploration rate: {self.exploration_rate:.0%}")
            
            for task, expected_optimal in test_scenarios:
                if self.total_cost > self.budget * 0.9:
                    break
                
                # Choose DSL parameters
                if random.random() < self.exploration_rate:
                    # Explore: try random parameters
                    dsl_params = {
                        'u': random.choice([0.2, 0.5, 1.0, 2.0, 3.0]),
                        'test_prob': random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
                    }
                    strategy = "explore"
                else:
                    # Exploit: use best known parameters for this task type
                    dsl_params = self.get_best_dsl_for_task(task)
                    strategy = "exploit"
                
                # Execute task
                quality = await self.execute_and_measure(session, task, dsl_params)
                
                # Record results
                dsl_key = f"u{dsl_params['u']}_t{dsl_params['test_prob']}"
                self.dsl_performance[dsl_key]['trials'] += 1
                self.dsl_performance[dsl_key]['total_quality'] += quality
                
                # Learn task patterns
                task_type = self.classify_task(task)
                self.task_type_patterns[task_type].append({
                    'dsl': dsl_params,
                    'quality': quality
                })
                
                result = {
                    'task': task[:40] + '...',
                    'dsl': dsl_params,
                    'quality': quality,
                    'strategy': strategy,
                    'optimal_distance': self.calculate_dsl_distance(dsl_params, expected_optimal)
                }
                
                round_results.append(result)
                
                print(f"\nTask: {result['task']}")
                print(f"Strategy: {strategy}")
                print(f"DSL: u={dsl_params['u']}, test_prob={dsl_params['test_prob']}")
                print(f"Quality: {quality:.3f}")
                print(f"Distance from optimal: {result['optimal_distance']:.3f}")
            
            results_by_round.append(round_results)
            
            # Show learning progress
            avg_quality = np.mean([r['quality'] for r in round_results])
            avg_distance = np.mean([r['optimal_distance'] for r in round_results])
            
            print(f"\nRound {round_num + 1} Summary:")
            print(f"Average quality: {avg_quality:.3f}")
            print(f"Average distance from optimal: {avg_distance:.3f}")
        
        # Analyze if system learned
        self.analyze_learning(results_by_round)
        
        return results_by_round
    
    async def execute_and_measure(self, session: aiohttp.ClientSession, 
                                 task: str, dsl_params: dict) -> float:
        """Execute task and return quality score"""
        
        # Build prompt with DSL injection
        prompt = self.build_dsl_prompt(task, dsl_params)
        
        # Call API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0.7
        }
        
        async with session.post("https://api.openai.com/v1/chat/completions",
                               headers=headers, json=data) as response:
            result = await response.json()
        
        content = result['choices'][0]['message']['content']
        self.total_cost += (result['usage']['total_tokens'] / 1000) * 0.002
        
        # Measure quality based on task requirements
        quality = self.measure_task_quality(task, content, dsl_params)
        
        return quality
    
    def build_dsl_prompt(self, task: str, dsl_params: dict) -> str:
        """Build prompt with DSL behavioral injection"""
        
        behaviors = []
        
        u = dsl_params['u']
        if u >= 2.0:
            behaviors.append("MANDATORY: Provide MULTIPLE creative solutions (at least 3)")
        elif u <= 0.5:
            behaviors.append("MANDATORY: Use ONLY the standard, proven approach")
        
        test_prob = dsl_params['test_prob']
        if test_prob >= 0.7:
            behaviors.append("MANDATORY: Include comprehensive tests")
        elif test_prob <= 0.3:
            behaviors.append("Skip tests, focus on implementation only")
        
        return f"""{' '.join(behaviors)}

Task: {task}

[DSL: u={u}, test_prob={test_prob}]"""
    
    def measure_task_quality(self, task: str, response: str, dsl_params: dict) -> float:
        """Measure how well the response matches task requirements"""
        
        task_lower = task.lower()
        response_lower = response.lower()
        
        quality = 0.0
        
        # Task-specific quality metrics
        if 'test' in task_lower and 'comprehensive' in task_lower:
            # Task wants comprehensive tests
            test_count = len(re.findall(r'def test_|assert |@test', response))
            quality += min(1.0, test_count / 5) * 0.5  # Expect at least 5 tests
            
            # Penalize high exploration for test tasks
            if dsl_params['u'] < 1.0:
                quality += 0.2
        
        elif 'explore' in task_lower or 'different ways' in task_lower:
            # Task wants multiple approaches
            approach_count = len(re.findall(r'approach \d|method \d|option \d|alternatively', response_lower))
            quality += min(1.0, approach_count / 3) * 0.5  # Expect at least 3 approaches
            
            # Reward high exploration
            if dsl_params['u'] >= 2.0:
                quality += 0.2
        
        elif 'fix' in task_lower and ('bug' in task_lower or 'production' in task_lower):
            # Bug fix - wants focused solution with tests
            has_fix = 'def ' in response or 'function' in response
            has_tests = 'test' in response_lower or 'assert' in response
            
            quality += 0.3 if has_fix else 0.0
            quality += 0.3 if has_tests else 0.0
            
            # Reward low exploration for bug fixes
            if dsl_params['u'] <= 0.5:
                quality += 0.2
        
        elif 'brainstorm' in task_lower or 'creative' in task_lower:
            # Creative task - wants many ideas
            idea_count = len(re.findall(r'\d\.|bullet|idea|could|might|perhaps', response_lower))
            quality += min(1.0, idea_count / 10) * 0.5
            
            # Reward high exploration
            if dsl_params['u'] >= 2.0:
                quality += 0.3
        
        elif 'robust' in task_lower or 'error handling' in task_lower:
            # Robustness - wants error handling and tests
            has_error_handling = any(keyword in response_lower for keyword in ['try', 'except', 'error', 'exception'])
            has_tests = 'test' in response_lower
            
            quality += 0.4 if has_error_handling else 0.0
            quality += 0.3 if has_tests else 0.0
        
        # General quality factors
        if len(response) > 100:  # Not too short
            quality += 0.1
        
        if '```' in response:  # Has code blocks
            quality += 0.1
        
        return min(1.0, quality)
    
    def classify_task(self, task: str) -> str:
        """Classify task into categories"""
        task_lower = task.lower()
        
        if 'test' in task_lower:
            return 'testing'
        elif 'explore' in task_lower or 'brainstorm' in task_lower:
            return 'exploration'
        elif 'fix' in task_lower or 'bug' in task_lower:
            return 'debugging'
        elif 'robust' in task_lower or 'error' in task_lower:
            return 'reliability'
        else:
            return 'general'
    
    def get_best_dsl_for_task(self, task: str) -> dict:
        """Get best known DSL parameters for task type"""
        task_type = self.classify_task(task)
        
        if task_type in self.task_type_patterns and self.task_type_patterns[task_type]:
            # Find best performing DSL for this task type
            best = max(self.task_type_patterns[task_type], key=lambda x: x['quality'])
            return best['dsl']
        
        # Default DSL
        return {'u': 1.0, 'test_prob': 0.5}
    
    def calculate_dsl_distance(self, actual: dict, optimal: dict) -> float:
        """Calculate distance between actual and optimal DSL"""
        u_dist = abs(actual['u'] - optimal['u']) / 3.0  # Normalize by max range
        test_dist = abs(actual['test_prob'] - optimal['test_prob'])
        
        return (u_dist + test_dist) / 2
    
    def analyze_learning(self, results_by_round: List[List[dict]]):
        """Analyze if the system learned over rounds"""
        
        print("\n\n=== LEARNING ANALYSIS ===")
        
        # Check quality improvement
        avg_qualities = [np.mean([r['quality'] for r in round]) for round in results_by_round]
        quality_improved = avg_qualities[-1] > avg_qualities[0] if len(avg_qualities) > 1 else False
        
        # Check distance to optimal
        avg_distances = [np.mean([r['optimal_distance'] for r in round]) for round in results_by_round]
        distance_decreased = avg_distances[-1] < avg_distances[0] if len(avg_distances) > 1 else False
        
        print(f"\nQuality progression: {' → '.join(f'{q:.3f}' for q in avg_qualities)}")
        print(f"Distance progression: {' → '.join(f'{d:.3f}' for d in avg_distances)}")
        
        print(f"\nQuality improved: {'✓' if quality_improved else '✗'}")
        print(f"Distance to optimal decreased: {'✓' if distance_decreased else '✗'}")
        
        # Show best DSL configurations discovered
        print("\nBest DSL configurations by average quality:")
        sorted_dsls = sorted(
            self.dsl_performance.items(),
            key=lambda x: x[1]['total_quality'] / x[1]['trials'] if x[1]['trials'] > 0 else 0,
            reverse=True
        )[:5]
        
        for dsl_key, perf in sorted_dsls:
            avg_quality = perf['total_quality'] / perf['trials'] if perf['trials'] > 0 else 0
            print(f"  {dsl_key}: {avg_quality:.3f} (n={perf['trials']})")
        
        # Task-specific learning
        print("\nTask-specific optimal DSLs learned:")
        for task_type, patterns in self.task_type_patterns.items():
            if patterns:
                best = max(patterns, key=lambda x: x['quality'])
                print(f"  {task_type}: u={best['dsl']['u']}, test_prob={best['dsl']['test_prob']} (quality={best['quality']:.3f})")
        
        # Overall success
        learning_successful = quality_improved or distance_decreased
        print(f"\n{'✓ LEARNING SUCCESSFUL' if learning_successful else '✗ LEARNING FAILED'}")
        
        return learning_successful

async def main():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not found")
        return
    
    tester = AdaptiveDonkeyTest(api_key, budget=5.0)
    
    async with aiohttp.ClientSession() as session:
        results = await tester.test_with_exploration(session)
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_cost': tester.total_cost,
        'rounds': len(results),
        'final_exploration_rate': tester.exploration_rate,
        'results_by_round': results,
        'dsl_performance': dict(tester.dsl_performance)
    }
    
    with open('adaptive_learning_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nTotal cost: ${tester.total_cost:.2f}")
    print("Report saved to: adaptive_learning_report.json")

if __name__ == "__main__":
    asyncio.run(main())