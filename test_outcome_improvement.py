#!/usr/bin/env python3
"""
OUTCOME IMPROVEMENT VALIDATION
Prove the full loop: Task → DSL → Better Outcome → Learning → Convergence
"""

import asyncio
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import aiohttp
from collections import defaultdict
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TaskOutcome:
    task: str
    dsl_params: dict
    response: str
    metrics: dict
    quality_score: float
    execution_time: float
    cost: float

@dataclass 
class LearningIteration:
    iteration: int
    trajectories: List[TaskOutcome]
    avg_quality: float
    best_dsl: dict
    convergence_delta: float

class OutcomeQualityMeasurer:
    """Objective metrics for code quality"""
    
    def measure_code_quality(self, task: str, response: str) -> Dict[str, float]:
        """Calculate objective quality metrics"""
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        code = '\n'.join(code_blocks) if code_blocks else response
        
        metrics = {}
        
        # 1. Correctness indicators
        metrics['has_implementation'] = 1.0 if any(keyword in code for keyword in ['def ', 'class ', 'lambda']) else 0.0
        metrics['handles_edge_cases'] = self._check_edge_cases(code)
        metrics['has_error_handling'] = 1.0 if any(keyword in code for keyword in ['try:', 'except', 'raise']) else 0.0
        
        # 2. Test coverage
        metrics['has_tests'] = 1.0 if 'test' in code.lower() or 'assert' in code else 0.0
        metrics['test_completeness'] = self._measure_test_completeness(code)
        
        # 3. Code quality
        metrics['readability'] = self._measure_readability(code)
        metrics['maintainability'] = self._measure_maintainability(code)
        
        # 4. Performance indicators
        metrics['efficiency'] = self._check_efficiency(code)
        
        # 5. Documentation
        metrics['has_docstring'] = 1.0 if '"""' in code or "'''" in code else 0.0
        metrics['explains_approach'] = 1.0 if any(word in response.lower() for word in ['because', 'since', 'therefore']) else 0.0
        
        # Calculate overall quality score (weighted average)
        weights = {
            'has_implementation': 0.2,
            'handles_edge_cases': 0.15,
            'has_error_handling': 0.1,
            'has_tests': 0.15,
            'test_completeness': 0.1,
            'readability': 0.1,
            'maintainability': 0.05,
            'efficiency': 0.1,
            'has_docstring': 0.025,
            'explains_approach': 0.025
        }
        
        quality_score = sum(metrics.get(k, 0) * v for k, v in weights.items())
        metrics['quality_score'] = quality_score
        
        return metrics
    
    def _check_edge_cases(self, code: str) -> float:
        """Check if code handles edge cases"""
        edge_case_patterns = [
            r'if.*[<>=]=?\s*0',  # Zero checks
            r'if.*is None',       # None checks
            r'if.*len\(',         # Empty checks
            r'if not\s+\w+',      # Falsy checks
            r'[<>=]=\s*len\(',    # Bounds checks
        ]
        
        score = sum(0.2 for pattern in edge_case_patterns if re.search(pattern, code))
        return min(1.0, score)
    
    def _measure_test_completeness(self, code: str) -> float:
        """Measure how comprehensive the tests are"""
        if 'test' not in code.lower():
            return 0.0
        
        test_patterns = {
            'basic_test': r'test_.*basic|test_.*simple',
            'edge_cases': r'test_.*edge|test_.*empty|test_.*none',
            'error_cases': r'test_.*error|test_.*invalid|test_.*exception',
            'multiple_tests': r'def test_.*\n.*def test_',
        }
        
        score = sum(0.25 for pattern in test_patterns.values() if re.search(pattern, code, re.IGNORECASE))
        return score
    
    def _measure_readability(self, code: str) -> float:
        """Measure code readability"""
        lines = code.split('\n')
        
        # Good practices
        good_practices = 0
        if any('def ' in line for line in lines):  # Has functions
            good_practices += 0.25
        if any(re.match(r'^\s{4}#', line) for line in lines):  # Has comments
            good_practices += 0.25
        if any(re.match(r'^[a-z_]+\s*=', line) for line in lines):  # Has variables
            good_practices += 0.25
        
        # Bad practices
        bad_practices = 0
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines > 2:
            bad_practices += 0.25
        
        # Check for reasonable function/variable names
        if re.search(r'def [a-z_]{3,}\(', code):
            good_practices += 0.25
        
        return min(1.0, good_practices - bad_practices)
    
    def _measure_maintainability(self, code: str) -> float:
        """Measure code maintainability"""
        score = 0.0
        
        # Modular design
        function_count = len(re.findall(r'def \w+\(', code))
        if function_count > 1:
            score += 0.5
        
        # Type hints
        if re.search(r'def \w+\([^)]*:.*\)', code):
            score += 0.25
        
        # Constants
        if re.search(r'^[A-Z_]+\s*=', code, re.MULTILINE):
            score += 0.25
        
        return score
    
    def _check_efficiency(self, code: str) -> float:
        """Check for efficient patterns"""
        efficient_patterns = [
            r'\.join\(',           # String join instead of concatenation
            r'comprehension',      # List/dict comprehensions
            r'enumerate\(',        # Using enumerate
            r'with\s+.*:',        # Context managers
            r'@\w+',              # Decorators
        ]
        
        inefficient_patterns = [
            r'for.*range.*len\(', # Using range(len()) antipattern
            r'\+=.*loop',         # String concatenation in loops
        ]
        
        score = 0.5  # Base score
        score += sum(0.1 for pattern in efficient_patterns if re.search(pattern, code))
        score -= sum(0.2 for pattern in inefficient_patterns if re.search(pattern, code))
        
        return max(0.0, min(1.0, score))

class FullLoopValidator:
    """Validates the complete learning loop"""
    
    def __init__(self, api_key: str, budget_limit: float = 10.0):
        self.api_key = api_key
        self.budget_limit = budget_limit
        self.total_cost = 0.0
        self.quality_measurer = OutcomeQualityMeasurer()
        self.learning_history: List[LearningIteration] = []
        
        # Simulated donkey system
        self.trajectories = []
        self.dsl_performance = defaultdict(lambda: {'total': 0, 'quality_sum': 0.0})
        
    async def execute_task_with_dsl(self, session: aiohttp.ClientSession, 
                                   task: str, dsl_params: dict) -> TaskOutcome:
        """Execute a task with specific DSL parameters"""
        
        # Inject DSL behavior
        prompt = self._inject_dsl_behavior(task, dsl_params)
        
        # Call OpenAI API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        start_time = time.time()
        
        async with session.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=data) as response:
            result = await response.json()
        
        execution_time = time.time() - start_time
        
        # Extract response
        response_text = result['choices'][0]['message']['content']
        cost = (result['usage']['total_tokens'] / 1000) * 0.002
        self.total_cost += cost
        
        # Measure quality
        metrics = self.quality_measurer.measure_code_quality(task, response_text)
        
        return TaskOutcome(
            task=task,
            dsl_params=dsl_params,
            response=response_text,
            metrics=metrics,
            quality_score=metrics['quality_score'],
            execution_time=execution_time,
            cost=cost
        )
    
    def _inject_dsl_behavior(self, task: str, dsl_params: dict) -> str:
        """Convert DSL parameters to behavioral instructions"""
        
        instructions = []
        
        # Exploration vs Focus
        u = dsl_params.get('u', 1.0)
        if u > 2.0:
            instructions.append("IMPORTANT: Provide MULTIPLE different approaches (at least 3). Be creative and explore unusual solutions.")
        elif u > 1.0:
            instructions.append("Consider multiple approaches and explain trade-offs.")
        elif u < 0.5:
            instructions.append("IMPORTANT: Use ONLY the most standard, proven approach. No alternatives.")
        else:
            instructions.append("Provide a solid, standard solution.")
        
        # Testing requirement
        test_prob = dsl_params.get('test_prob', 0.5)
        if test_prob > 0.8:
            instructions.append("MANDATORY: Include comprehensive tests with edge cases and error handling tests.")
        elif test_prob > 0.5:
            instructions.append("Include basic tests for the main functionality.")
        elif test_prob < 0.2:
            instructions.append("Focus on implementation only, tests are not required.")
        
        # Documentation/Memory
        mem_thresh = dsl_params.get('mem_thresh', 0.5)
        if mem_thresh > 0.7:
            instructions.append("Be concise. Only explain critical design decisions.")
        else:
            instructions.append("Provide detailed explanations and documentation.")
        
        prompt = f"""{' '.join(instructions)}

Task: {task}

Requirements:
1. Provide working code
2. Handle edge cases appropriately
3. Follow the behavioral directives above

[Active DSL: u={u:.1f}, test_prob={test_prob:.1f}, mem_thresh={mem_thresh:.1f}]"""
        
        return prompt
    
    def recommend_dsl(self, task: str) -> dict:
        """Simulate donkey recommendation based on learned trajectories"""
        
        # Default DSL for unknown tasks
        default_dsl = {'u': 1.0, 'test_prob': 0.5, 'mem_thresh': 0.5}
        
        if not self.trajectories:
            return default_dsl
        
        # Find similar tasks
        task_lower = task.lower()
        similar_trajectories = []
        
        for traj in self.trajectories:
            similarity = self._calculate_similarity(task_lower, traj.task.lower())
            if similarity > 0.3:
                similar_trajectories.append((traj, similarity))
        
        if not similar_trajectories:
            return default_dsl
        
        # Weight by similarity and quality
        weighted_params = defaultdict(float)
        total_weight = 0.0
        
        for traj, similarity in similar_trajectories:
            weight = similarity * traj.quality_score
            total_weight += weight
            
            for param, value in traj.dsl_params.items():
                weighted_params[param] += value * weight
        
        # Calculate weighted average
        recommended_dsl = {}
        for param in ['u', 'test_prob', 'mem_thresh']:
            recommended_dsl[param] = weighted_params[param] / total_weight if total_weight > 0 else default_dsl[param]
        
        return recommended_dsl
    
    def _calculate_similarity(self, task1: str, task2: str) -> float:
        """Simple word overlap similarity"""
        words1 = set(task1.split())
        words2 = set(task2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def record_trajectory(self, outcome: TaskOutcome):
        """Record outcome as trajectory for learning"""
        self.trajectories.append(outcome)
        
        # Update DSL performance stats
        dsl_key = f"u{outcome.dsl_params['u']:.1f}_t{outcome.dsl_params['test_prob']:.1f}"
        self.dsl_performance[dsl_key]['total'] += 1
        self.dsl_performance[dsl_key]['quality_sum'] += outcome.quality_score
    
    async def run_learning_loop(self, session: aiohttp.ClientSession, 
                               test_tasks: List[str], iterations: int = 5):
        """Run the full learning loop multiple times"""
        
        print("\n=== FULL LEARNING LOOP VALIDATION ===\n")
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            iteration_outcomes = []
            
            for task in test_tasks:
                if self.total_cost > self.budget_limit * 0.9:
                    print("Approaching budget limit, stopping...")
                    break
                
                # Get DSL recommendation
                recommended_dsl = self.recommend_dsl(task)
                print(f"\nTask: {task[:50]}...")
                print(f"Recommended DSL: u={recommended_dsl['u']:.1f}, test_prob={recommended_dsl['test_prob']:.1f}")
                
                # Execute with recommended DSL
                outcome = await self.execute_task_with_dsl(session, task, recommended_dsl)
                iteration_outcomes.append(outcome)
                
                # Record trajectory
                self.record_trajectory(outcome)
                
                print(f"Quality Score: {outcome.quality_score:.3f}")
                print(f"Key Metrics: tests={outcome.metrics['has_tests']}, " +
                      f"edge_cases={outcome.metrics['handles_edge_cases']:.1f}, " +
                      f"readable={outcome.metrics['readability']:.1f}")
            
            # Calculate iteration statistics
            if iteration_outcomes:
                avg_quality = np.mean([o.quality_score for o in iteration_outcomes])
                
                # Find best performing DSL
                best_dsl_key = max(self.dsl_performance.items(), 
                                  key=lambda x: x[1]['quality_sum'] / x[1]['total'] if x[1]['total'] > 0 else 0)[0]
                
                # Calculate convergence (how much recommendations are changing)
                if self.learning_history:
                    prev_iter = self.learning_history[-1]
                    convergence_delta = abs(avg_quality - prev_iter.avg_quality)
                else:
                    convergence_delta = 1.0
                
                learning_iter = LearningIteration(
                    iteration=iteration,
                    trajectories=iteration_outcomes,
                    avg_quality=avg_quality,
                    best_dsl={'key': best_dsl_key},
                    convergence_delta=convergence_delta
                )
                
                self.learning_history.append(learning_iter)
                
                print(f"\nIteration Summary:")
                print(f"Average Quality: {avg_quality:.3f}")
                print(f"Best DSL so far: {best_dsl_key}")
                print(f"Convergence Delta: {convergence_delta:.3f}")
    
    async def validate_improvement(self):
        """Validate that the system improves over time"""
        
        print("\n=== IMPROVEMENT VALIDATION ===\n")
        
        if len(self.learning_history) < 2:
            print("Insufficient data for improvement validation")
            return False
        
        # Check if quality improves over iterations
        qualities = [iter.avg_quality for iter in self.learning_history]
        
        # Linear regression to check trend
        x = np.arange(len(qualities))
        slope = np.polyfit(x, qualities, 1)[0]
        
        print(f"Quality trend over {len(qualities)} iterations:")
        for i, q in enumerate(qualities):
            print(f"  Iteration {i+1}: {q:.3f}")
        
        print(f"\nTrend slope: {slope:.4f}")
        improving = slope > 0.01  # Positive trend
        
        # Check convergence
        deltas = [iter.convergence_delta for iter in self.learning_history[1:]]
        converging = all(d < 0.1 for d in deltas[-2:]) if len(deltas) >= 2 else False
        
        print(f"System is {'improving' if improving else 'not improving'}")
        print(f"System is {'converging' if converging else 'still exploring'}")
        
        # Check if system found optimal patterns
        task_patterns = self._analyze_learned_patterns()
        
        return improving and len(task_patterns) > 0
    
    def _analyze_learned_patterns(self) -> dict:
        """Analyze what patterns the system learned"""
        
        patterns = {}
        
        # Group trajectories by task type
        task_types = {
            'implementation': ['implement', 'create', 'write', 'build'],
            'debugging': ['fix', 'debug', 'error', 'bug'],
            'optimization': ['optimize', 'improve', 'faster', 'efficient'],
            'refactoring': ['refactor', 'clean', 'readable', 'maintain']
        }
        
        for task_type, keywords in task_types.items():
            relevant_trajectories = [
                t for t in self.trajectories 
                if any(kw in t.task.lower() for kw in keywords)
            ]
            
            if relevant_trajectories:
                # Find best performing DSL for this task type
                best_trajectory = max(relevant_trajectories, key=lambda t: t.quality_score)
                avg_quality = np.mean([t.quality_score for t in relevant_trajectories])
                
                patterns[task_type] = {
                    'best_dsl': best_trajectory.dsl_params,
                    'avg_quality': avg_quality,
                    'sample_count': len(relevant_trajectories)
                }
        
        print("\n=== LEARNED PATTERNS ===")
        for task_type, pattern in patterns.items():
            print(f"\n{task_type}:")
            print(f"  Best DSL: u={pattern['best_dsl']['u']:.1f}, " +
                  f"test_prob={pattern['best_dsl']['test_prob']:.1f}")
            print(f"  Avg Quality: {pattern['avg_quality']:.3f}")
            print(f"  Samples: {pattern['sample_count']}")
        
        return patterns

async def main():
    """Run the full validation suite"""
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not found")
        return
    
    validator = FullLoopValidator(api_key, budget_limit=10.0)
    
    # Test tasks covering different types
    test_tasks = [
        "Write a function to find the nth Fibonacci number",
        "Implement binary search for a sorted array",
        "Create a function to validate email addresses",
        "Fix this bug: function returns None for empty input instead of empty list",
        "Optimize this code: nested loops checking for duplicates",
        "Refactor this function to be more readable: one-line lambda with multiple operations",
        "Build a simple LRU cache with get and put methods",
        "Debug this recursive function that causes stack overflow"
    ]
    
    async with aiohttp.ClientSession() as session:
        # Run learning loop
        await validator.run_learning_loop(session, test_tasks, iterations=3)
        
        # Validate improvement
        improved = await validator.validate_improvement()
        
        # Generate final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cost": float(validator.total_cost),
            "iterations": len(validator.learning_history),
            "total_trajectories": len(validator.trajectories),
            "improvement_validated": bool(improved),
            "learning_history": [
                {
                    "iteration": h.iteration,
                    "avg_quality": float(h.avg_quality),
                    "convergence_delta": float(h.convergence_delta)
                }
                for h in validator.learning_history
            ],
            "final_dsl_performance": {
                k: {
                    "total": v["total"],
                    "quality_sum": float(v["quality_sum"]),
                    "avg_quality": float(v["quality_sum"] / v["total"]) if v["total"] > 0 else 0
                }
                for k, v in validator.dsl_performance.items()
            }
        }
        
        with open('outcome_improvement_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== FINAL VALIDATION ===")
        print(f"Total Cost: ${validator.total_cost:.2f}")
        print(f"Improvement Validated: {improved}")
        print(f"Report saved to: outcome_improvement_report.json")

if __name__ == "__main__":
    asyncio.run(main())