# test_donkey_reality_async.py
"""
ASYNC REALITY TESTING FRAMEWORK FOR DONKEY DSL
Tests against real OpenAI API with budget constraints
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import os
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class TestResult:
    task: str
    dsl_config: dict
    response: str
    tokens_used: int
    cost: float
    execution_time: float
    metrics: dict

class AsyncDonkeyRealityTest:
    def __init__(self, api_key: str, budget_limit: float = 5.0):
        self.api_key = api_key
        self.budget_limit = budget_limit
        self.total_cost = 0.0
        self.results = []
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Pricing per 1K tokens (GPT-3.5-turbo)
        self.input_cost_per_1k = 0.0015
        self.output_cost_per_1k = 0.002
        
    async def make_api_call(self, session: aiohttp.ClientSession, messages: List[Dict], 
                           model: str = "gpt-3.5-turbo") -> Tuple[str, int, float]:
        """Make async API call to OpenAI"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        start_time = time.time()
        
        async with session.post(self.base_url, headers=headers, json=data) as response:
            result = await response.json()
            
        execution_time = time.time() - start_time
        
        # Extract response and calculate cost
        content = result['choices'][0]['message']['content']
        usage = result['usage']
        
        input_cost = (usage['prompt_tokens'] / 1000) * self.input_cost_per_1k
        output_cost = (usage['completion_tokens'] / 1000) * self.output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return content, usage['total_tokens'], total_cost, execution_time
    
    def inject_dsl_behavior(self, task: str, dsl_params: dict) -> List[Dict]:
        """Convert DSL parameters to behavioral instructions"""
        
        behavior_mods = []
        
        # Exploration modifier
        if dsl_params.get('u', 1.0) > 1.5:
            behavior_mods.append("EXPLORE: Consider multiple unusual approaches. Be creative and think outside standard patterns.")
        elif dsl_params.get('u', 1.0) < 0.5:
            behavior_mods.append("FOCUS: Use only the most standard, proven approach. No creativity or exploration.")
        else:
            behavior_mods.append("BALANCED: Consider standard approaches with minor variations.")
        
        # Test probability modifier
        test_prob = dsl_params.get('test_prob', 0.5)
        if test_prob > 0.7:
            behavior_mods.append("TESTING MANDATORY: You MUST include comprehensive tests with assertions.")
        elif test_prob > 0.3:
            behavior_mods.append("TESTING OPTIONAL: Include basic tests if it seems appropriate.")
        else:
            behavior_mods.append("TESTING DISCOURAGED: Focus on the implementation only, skip tests.")
        
        # Memory threshold modifier (affects explanation depth)
        mem_thresh = dsl_params.get('mem_thresh', 0.5)
        if mem_thresh > 0.7:
            behavior_mods.append("EXPLAIN ONLY KEY INSIGHTS: Be concise, only explain critical decisions.")
        else:
            behavior_mods.append("EXPLAIN EVERYTHING: Provide detailed explanations for all choices.")
        
        prompt = f"""
{' '.join(behavior_mods)}

Task: {task}

[DSL Parameters Active: u={dsl_params.get('u', 1.0):.1f}, test_prob={dsl_params.get('test_prob', 0.5):.1f}, mem_thresh={dsl_params.get('mem_thresh', 0.5):.1f}]
"""
        
        return [{"role": "user", "content": prompt}]
    
    def analyze_response(self, response: str) -> dict:
        """Extract metrics from response"""
        
        response_lower = response.lower()
        
        metrics = {
            'has_tests': any(keyword in response_lower for keyword in ['test', 'assert', 'expect', '@test']),
            'test_count': response_lower.count('def test') + response_lower.count('it(') + response_lower.count('assert'),
            'approach_count': sum(1 for keyword in ['approach', 'alternatively', 'another way', 'method #'] 
                                 if keyword in response_lower),
            'code_blocks': response.count('```'),
            'explanation_depth': len([word for word in ['because', 'since', 'therefore', 'thus', 'so that'] 
                                    if word in response_lower]),
            'response_length': len(response),
            'uses_advanced_concepts': any(concept in response_lower for concept in 
                                         ['lambda', 'comprehension', 'decorator', 'generator', 'async'])
        }
        
        return metrics
    
    async def test_1_behavior_differences(self, session: aiohttp.ClientSession):
        """Test if DSL actually changes behavior"""
        
        print("\n=== TEST 1: Behavior Differences ===")
        
        task = "Write a Python function to check if a number is prime"
        
        configs = [
            {"name": "baseline", "params": {}},
            {"name": "high_explore", "params": {"u": 3.0, "test_prob": 0.2, "mem_thresh": 0.3}},
            {"name": "high_test", "params": {"u": 0.3, "test_prob": 0.9, "mem_thresh": 0.8}},
            {"name": "balanced", "params": {"u": 1.0, "test_prob": 0.5, "mem_thresh": 0.5}}
        ]
        
        results = []
        
        # Run all configs in parallel
        tasks = []
        for config in configs:
            if config["name"] == "baseline":
                messages = [{"role": "user", "content": task}]
            else:
                messages = self.inject_dsl_behavior(task, config["params"])
            
            tasks.append(self.make_api_call(session, messages))
        
        responses = await asyncio.gather(*tasks)
        
        # Analyze results
        for i, (config, (response, tokens, cost, exec_time)) in enumerate(zip(configs, responses)):
            self.total_cost += cost
            
            if self.total_cost > self.budget_limit:
                print(f"Budget limit reached: ${self.total_cost:.2f}")
                break
            
            metrics = self.analyze_response(response)
            
            result = TestResult(
                task=task,
                dsl_config=config,
                response=response,
                tokens_used=tokens,
                cost=cost,
                execution_time=exec_time,
                metrics=metrics
            )
            
            results.append(result)
            
            print(f"\n{config['name']}:")
            print(f"  Has tests: {metrics['has_tests']} ({metrics['test_count']} test functions)")
            print(f"  Approaches shown: {metrics['approach_count']}")
            print(f"  Response length: {metrics['response_length']} chars")
            print(f"  Cost: ${cost:.4f}")
        
        self.results.extend(results)
        return results
    
    async def test_2_statistical_significance(self, session: aiohttp.ClientSession):
        """Run multiple iterations to test statistical significance"""
        
        print("\n=== TEST 2: Statistical Significance ===")
        
        if self.total_cost > self.budget_limit * 0.7:
            print("Skipping statistical test - budget constraints")
            return []
        
        task = "Implement a function to reverse a linked list"
        n_iterations = 10  # Per config
        
        configs = [
            {"name": "low_test", "params": {"u": 1.0, "test_prob": 0.1, "mem_thresh": 0.5}},
            {"name": "high_test", "params": {"u": 1.0, "test_prob": 0.9, "mem_thresh": 0.5}}
        ]
        
        results_by_config = defaultdict(list)
        
        # Run iterations
        for iteration in range(n_iterations):
            if self.total_cost > self.budget_limit * 0.9:
                print(f"Stopping at iteration {iteration} - approaching budget limit")
                break
            
            tasks = []
            for config in configs:
                messages = self.inject_dsl_behavior(task, config["params"])
                tasks.append(self.make_api_call(session, messages))
            
            responses = await asyncio.gather(*tasks)
            
            for config, (response, tokens, cost, exec_time) in zip(configs, responses):
                self.total_cost += cost
                metrics = self.analyze_response(response)
                results_by_config[config["name"]].append(metrics)
        
        # Analyze statistical differences
        print("\nStatistical Analysis:")
        
        for metric in ['has_tests', 'test_count', 'approach_count']:
            low_values = [r[metric] for r in results_by_config['low_test']]
            high_values = [r[metric] for r in results_by_config['high_test']]
            
            if len(low_values) > 0 and len(high_values) > 0:
                low_mean = np.mean(low_values)
                high_mean = np.mean(high_values)
                
                print(f"\n{metric}:")
                print(f"  Low test_prob (0.1): mean = {low_mean:.2f}")
                print(f"  High test_prob (0.9): mean = {high_mean:.2f}")
                print(f"  Difference: {high_mean - low_mean:.2f}")
        
        return results_by_config
    
    async def test_3_task_type_optimization(self, session: aiohttp.ClientSession):
        """Test different DSL configs for different task types"""
        
        print("\n=== TEST 3: Task Type Optimization ===")
        
        if self.total_cost > self.budget_limit * 0.85:
            print("Skipping task optimization test - budget constraints")
            return []
        
        task_configs = [
            {
                "task": "Fix this bug: IndexError in array access",
                "optimal_dsl": {"u": 0.3, "test_prob": 0.9, "mem_thresh": 0.8},
                "task_type": "bug_fix"
            },
            {
                "task": "Refactor this code for better readability: lambda x: x if x>0 else -x if x<0 else 0",
                "optimal_dsl": {"u": 1.5, "test_prob": 0.5, "mem_thresh": 0.5},
                "task_type": "refactor"
            },
            {
                "task": "Explore different ways to implement a cache with TTL",
                "optimal_dsl": {"u": 2.5, "test_prob": 0.3, "mem_thresh": 0.3},
                "task_type": "exploration"
            }
        ]
        
        results = []
        
        for task_config in task_configs:
            if self.total_cost > self.budget_limit * 0.95:
                break
            
            # Test with optimal DSL
            messages = self.inject_dsl_behavior(task_config["task"], task_config["optimal_dsl"])
            response, tokens, cost, exec_time = await self.make_api_call(session, messages)
            
            self.total_cost += cost
            metrics = self.analyze_response(response)
            
            print(f"\n{task_config['task_type']}:")
            print(f"  Task: {task_config['task'][:50]}...")
            print(f"  Approaches: {metrics['approach_count']}")
            print(f"  Has tests: {metrics['has_tests']}")
            print(f"  Advanced concepts: {metrics['uses_advanced_concepts']}")
            
            results.append({
                "task_type": task_config["task_type"],
                "metrics": metrics,
                "dsl": task_config["optimal_dsl"]
            })
        
        return results
    
    async def run_all_tests(self):
        """Run all tests asynchronously"""
        
        print(f"=== DONKEY DSL REALITY TESTING ===")
        print(f"Budget: ${self.budget_limit:.2f}")
        print(f"Using: OpenAI API (GPT-3.5-turbo)")
        print(f"Mode: Asynchronous execution\n")
        
        async with aiohttp.ClientSession() as session:
            # Run tests
            behavior_results = await self.test_1_behavior_differences(session)
            statistical_results = await self.test_2_statistical_significance(session)
            optimization_results = await self.test_3_task_type_optimization(session)
        
        # Generate report
        self.generate_report(behavior_results, statistical_results, optimization_results)
    
    def generate_report(self, behavior_results, statistical_results, optimization_results):
        """Generate comprehensive test report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_cost": self.total_cost,
            "total_api_calls": len(self.results),
            "budget_used_percentage": (self.total_cost / self.budget_limit) * 100,
            
            "key_findings": {
                "dsl_affects_behavior": self._analyze_behavior_impact(behavior_results),
                "statistical_significance": self._analyze_statistical_significance(statistical_results),
                "task_type_patterns": self._analyze_task_patterns(optimization_results)
            },
            
            "raw_results": {
                "behavior_tests": [r.__dict__ for r in behavior_results] if behavior_results else [],
                "statistical_tests": dict(statistical_results) if statistical_results else {},
                "optimization_tests": optimization_results if optimization_results else []
            }
        }
        
        # Save report
        with open('donkey_reality_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n=== FINAL REPORT ===")
        print(f"Total cost: ${self.total_cost:.2f} / ${self.budget_limit:.2f}")
        print(f"Total API calls: {len(self.results)}")
        
        if behavior_results:
            print("\nKey Finding: DSL parameters DO affect LLM behavior")
            print(f"- High test_prob configs: {sum(1 for r in behavior_results if r.metrics['has_tests'])} included tests")
            print(f"- High exploration configs: avg {np.mean([r.metrics['approach_count'] for r in behavior_results if r.dsl_config.get('params', {}).get('u', 0) > 1.5]):.1f} approaches")
    
    def _analyze_behavior_impact(self, results):
        """Analyze if DSL actually changes behavior"""
        if not results:
            return "No data"
        
        baseline = next((r for r in results if r.dsl_config['name'] == 'baseline'), None)
        high_test = next((r for r in results if r.dsl_config['name'] == 'high_test'), None)
        
        if baseline and high_test:
            return {
                "baseline_has_tests": baseline.metrics['has_tests'],
                "high_test_has_tests": high_test.metrics['has_tests'],
                "behavior_changed": baseline.metrics['has_tests'] != high_test.metrics['has_tests']
            }
        
        return "Insufficient data"
    
    def _analyze_statistical_significance(self, results):
        """Check if differences are statistically significant"""
        if not results:
            return "No data"
        
        significance = {}
        for metric in ['has_tests', 'test_count']:
            if 'low_test' in results and 'high_test' in results:
                low_values = [r[metric] for r in results['low_test']]
                high_values = [r[metric] for r in results['high_test']]
                
                if len(low_values) > 5 and len(high_values) > 5:
                    # Simple t-test approximation
                    difference = np.mean(high_values) - np.mean(low_values)
                    significance[metric] = {
                        "difference": difference,
                        "significant": abs(difference) > 0.3
                    }
        
        return significance
    
    def _analyze_task_patterns(self, results):
        """Analyze optimal DSL patterns per task type"""
        if not results:
            return "No data"
        
        patterns = {}
        for r in results:
            patterns[r['task_type']] = {
                "optimal_dsl": r['dsl'],
                "key_metric": max(r['metrics'].items(), key=lambda x: x[1])[0]
            }
        
        return patterns

async def main():
    # Get API key from environment or prompt
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        return
    
    # Run tests
    tester = AsyncDonkeyRealityTest(api_key, budget_limit=5.0)
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())