#!/usr/bin/env python3
"""
LLM-Donkey Integration v1
Transforms DSL recommendations into actual LLM behavior modifications
"""
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from collections import defaultdict
import numpy as np

@dataclass
class LLMResponse:
    content: str
    metadata: Dict[str, Any]
    
@dataclass
class BehaviorMeasurement:
    actual_u: float
    actual_test: bool
    actual_mem: float
    info_gained: float
    followed_recommendation: float
    unique_approaches: int

class PromptInjector:
    """Converts DSL parameters into behavioral instructions for LLM"""
    
    def inject_dsl_behavior(self, base_prompt: str, dsl_params: Dict[str, float]) -> str:
        """Inject DSL-based behavioral modifiers into prompt"""
        behavior_mods = []
        
        # Exploration modifier based on uncertainty parameter
        u_value = dsl_params.get('u', 1.0)
        if u_value > 1.5:
            behavior_mods.append("EXPLORE: Consider unusual approaches. Think outside standard patterns. Be creative and experimental.")
        elif u_value > 0.8:
            behavior_mods.append("BALANCED: Mix proven methods with some exploration. Consider alternatives.")
        elif u_value < 0.5:
            behavior_mods.append("FOCUS: Stick to proven methods. No experimentation. Use battle-tested solutions.")
        
        # Test probability modifier
        test_prob = dsl_params.get('test_prob', 0.5)
        if random.random() < test_prob:
            behavior_mods.append("MANDATORY: Write and run comprehensive tests for this solution. Include edge cases.")
        
        # Memory threshold modifier
        mem_thresh = dsl_params.get('mem_thresh', 0.5)
        behavior_mods.append(f"MEMORY: Only remember insights with information gain > {mem_thresh:.1f}. Be selective.")
        
        # Simulation modifier
        if dsl_params.get('sim', False):
            behavior_mods.append("SIMULATE: First describe what would happen, then execute. Think step-by-step.")
        
        # Mutation modifier
        if dsl_params.get('mutate', 0) > 0:
            behavior_mods.append(f"ADAPTIVE: Be ready to change approach if initial attempts fail. Mutation rate: {dsl_params['mutate']:.0%}")
        
        # Format injection
        injection = f"""
[DSL BEHAVIORAL DIRECTIVES]
{' '.join(behavior_mods)}

[SESSION PARAMETERS: u={u_value:.1f}, test_prob={test_prob:.0%}, mem={mem_thresh:.1f}]

{base_prompt}
"""
        return injection

class LLMBehaviorAnalyzer:
    """Measures actual LLM behavior against DSL recommendations"""
    
    def __init__(self):
        self.exploration_keywords = {
            'high': ['alternatively', 'unconventional', 'novel', 'experimental', 'creative', 'unusual'],
            'low': ['standard', 'typical', 'conventional', 'proven', 'established', 'traditional']
        }
        self.test_patterns = [
            r'test_\w+', r'assert\s+', r'@test', r'def test', r'describe\(', r'it\(',
            r'expect\(', r'\.test\.', r'testing', r'unittest', r'pytest'
        ]
        
    def measure_actual_dsl(self, response: str, recommended_dsl: Dict[str, float]) -> BehaviorMeasurement:
        """Analyze response to determine actual behavior vs recommended"""
        
        actual_u = self.measure_exploration(response)
        actual_test = self.detect_testing(response)
        actual_mem = self.measure_selectivity(response)
        info_gained = self.measure_novel_concepts(response)
        unique_approaches = self.count_unique_approaches(response)
        
        # Calculate compliance score
        compliance_scores = []
        
        # U parameter compliance
        u_diff = abs(actual_u - recommended_dsl.get('u', 1.0))
        u_compliance = max(0, 1 - u_diff / 5.0)  # Normalize to 0-1
        compliance_scores.append(u_compliance)
        
        # Test compliance
        test_recommended = random.random() < recommended_dsl.get('test_prob', 0.5)
        test_compliance = 1.0 if actual_test == test_recommended else 0.0
        compliance_scores.append(test_compliance)
        
        followed_recommendation = np.mean(compliance_scores)
        
        return BehaviorMeasurement(
            actual_u=actual_u,
            actual_test=actual_test,
            actual_mem=actual_mem,
            info_gained=info_gained,
            followed_recommendation=followed_recommendation,
            unique_approaches=unique_approaches
        )
    
    def measure_exploration(self, response: str) -> float:
        """Measure exploration level (u parameter) from response"""
        response_lower = response.lower()
        
        high_exploration_count = sum(1 for kw in self.exploration_keywords['high'] 
                                   if kw in response_lower)
        low_exploration_count = sum(1 for kw in self.exploration_keywords['low'] 
                                  if kw in response_lower)
        
        # Count number of different approaches mentioned
        approach_markers = ['approach', 'method', 'solution', 'alternative', 'option']
        approach_count = sum(1 for marker in approach_markers if marker in response_lower)
        
        # Calculate exploration score (0-5 scale matching u parameter)
        if high_exploration_count > low_exploration_count:
            base_score = 2.0 + (high_exploration_count * 0.5)
        else:
            base_score = 1.0 - (low_exploration_count * 0.2)
        
        # Adjust based on approach diversity
        exploration_score = base_score + (approach_count * 0.3)
        
        return min(5.0, max(0.1, exploration_score))
    
    def detect_testing(self, response: str) -> bool:
        """Detect if response includes testing"""
        for pattern in self.test_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False
    
    def measure_selectivity(self, response: str) -> float:
        """Measure memory selectivity (what portion of content is marked as important)"""
        # Look for emphasis markers
        emphasis_patterns = [
            r'important:', r'key insight:', r'remember:', r'note:',
            r'!important', r'\*\*[^*]+\*\*', r'IMPORTANT', r'KEY'
        ]
        
        total_sentences = len([s for s in response.split('.') if s.strip()])
        emphasized_count = sum(1 for pattern in emphasis_patterns 
                             if re.search(pattern, response))
        
        if total_sentences == 0:
            return 0.5
        
        selectivity = 1.0 - (emphasized_count / total_sentences)
        return max(0.1, min(1.0, selectivity))
    
    def measure_novel_concepts(self, response: str) -> float:
        """Estimate information gain from response"""
        # Simple heuristic: unique technical terms and concepts
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', response)
        unique_terms = len(set(technical_terms))
        
        # Normalize to 0-1 scale
        info_gain = min(1.0, unique_terms / 20.0)
        return info_gain
    
    def count_unique_approaches(self, response: str) -> int:
        """Count number of distinct approaches or solutions mentioned"""
        approach_markers = [
            r'approach \d+:', r'solution \d+:', r'method \d+:',
            r'alternatively', r'another way', r'different approach'
        ]
        
        count = 0
        for marker in approach_markers:
            count += len(re.findall(marker, response, re.IGNORECASE))
        
        # Also count numbered lists
        numbered_items = re.findall(r'^\d+\.', response, re.MULTILINE)
        count += len(numbered_items) // 2  # Assume half are approaches
        
        return max(1, count)

class DSLCalibrator:
    """Learns how DSL parameters actually affect LLM behavior"""
    
    def __init__(self):
        self.calibration_data = defaultdict(list)
        self.transfer_functions = {}
    
    def calibrate_dsl_effects(self, llm_client, test_prompts: List[str]):
        """Run calibration to learn DSL → behavior mapping"""
        u_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        injector = PromptInjector()
        analyzer = LLMBehaviorAnalyzer()
        
        for u in u_values:
            responses = []
            for prompt in test_prompts:
                dsl_params = {'u': u, 'test_prob': 0.5, 'mem_thresh': 0.5}
                injected_prompt = injector.inject_dsl_behavior(prompt, dsl_params)
                
                # Simulate LLM response (in real implementation, call actual LLM)
                response = self._simulate_llm_response(injected_prompt, u)
                
                behavior = analyzer.measure_actual_dsl(response, dsl_params)
                responses.append(behavior.actual_u)
            
            # Store mapping
            avg_actual_u = np.mean(responses)
            self.transfer_functions[u] = avg_actual_u
            self.calibration_data[u] = responses
        
        return self.transfer_functions
    
    def _simulate_llm_response(self, prompt: str, u_value: float) -> str:
        """Simulate LLM response based on u value (for testing)"""
        if u_value > 2.0:
            return """Let me explore several unconventional approaches to this problem:
            
            Approach 1: We could use a novel graph-based algorithm...
            Approach 2: An experimental neural architecture might work...
            Approach 3: Consider this unusual mathematical transformation...
            
            Each of these creative solutions offers unique advantages."""
        elif u_value < 0.5:
            return """I'll use the standard, proven approach for this:
            
            The conventional solution involves using established patterns.
            This traditional method has been tested extensively.
            We'll stick to typical best practices throughout."""
        else:
            return """Here's a balanced solution:
            
            We'll primarily use proven methods, but with some optimization.
            The standard approach works well, with minor enhancements.
            This combines reliability with modest improvements."""

class DSLConversation:
    """Manages multi-turn conversations with evolving DSL"""
    
    def __init__(self, task: str, donkey_client):
        self.task = task
        self.donkey = donkey_client
        dsl_str, confidence, based_on = donkey_client.donkey(task)
        self.dsl = {
            'dsl': dsl_str,
            'confidence': confidence,
            'params': self._parse_dsl_to_params(dsl_str, donkey_client)
        }
        self.turns = []
        self.injector = PromptInjector()
        self.analyzer = LLMBehaviorAnalyzer()
    
    def _parse_dsl_to_params(self, dsl_str: str, donkey_client) -> Dict[str, float]:
        """Parse DSL string to parameter dictionary"""
        params = donkey_client.parse_dsl(dsl_str)
        return {
            'u': params.u,
            'test_prob': params.test,
            'mem_thresh': params.mem,
            'sim': params.sim,
            'mutate': params.mutate
        }
    
    def turn(self, user_input: str, llm_client) -> Tuple[str, BehaviorMeasurement]:
        """Process one conversation turn with DSL injection"""
        # Inject current DSL
        prompt = self.injector.inject_dsl_behavior(user_input, self.dsl['params'])
        
        # Get LLM response (simulated for now)
        response = self._simulate_llm_turn(prompt, len(self.turns))
        
        # Measure behavior
        behavior = self.analyzer.measure_actual_dsl(response, self.dsl['params'])
        
        # Record turn
        self.turns.append({
            'user_input': user_input,
            'dsl': self.dsl,
            'response': response,
            'behavior': behavior
        })
        
        # Maybe mutate DSL based on conversation flow
        if self.dsl['params'].get('mutate', 0) > random.random():
            self.dsl = self._mutate_dsl(self.dsl, behavior)
        
        return response, behavior
    
    def _simulate_llm_turn(self, prompt: str, turn_number: int) -> str:
        """Simulate LLM response for testing"""
        return f"Turn {turn_number + 1} response to: {prompt[:50]}..."
    
    def _mutate_dsl(self, current_dsl: Dict, behavior: BehaviorMeasurement) -> Dict:
        """Mutate DSL based on observed behavior"""
        new_dsl = current_dsl.copy()
        new_params = current_dsl['params'].copy()
        
        # If exploration is too low, increase u
        if behavior.actual_u < current_dsl['params']['u'] * 0.7:
            new_params['u'] = min(5.0, new_params['u'] * 1.2)
        
        # If no testing when recommended, increase test_prob
        if not behavior.actual_test and new_params['test_prob'] > 0.5:
            new_params['test_prob'] = min(1.0, new_params['test_prob'] * 1.1)
        
        new_dsl['params'] = new_params
        return new_dsl
    
    def get_trajectory(self) -> Dict:
        """Get conversation trajectory for donkey learning"""
        success = self._estimate_success()
        return {
            'task': self.task,
            'dsl': self.dsl['dsl'],
            'success': success,
            'turns': len(self.turns),
            'avg_compliance': np.mean([t['behavior'].followed_recommendation 
                                      for t in self.turns])
        }
    
    def _estimate_success(self) -> float:
        """Estimate conversation success (would come from user feedback in practice)"""
        # Simple heuristic: high compliance + reasonable exploration
        avg_compliance = np.mean([t['behavior'].followed_recommendation 
                                 for t in self.turns])
        avg_exploration = np.mean([t['behavior'].actual_u for t in self.turns])
        
        return avg_compliance * 0.7 + min(avg_exploration / 3.0, 1.0) * 0.3

# Integration utilities
def create_llm_donkey_pipeline(donkey_client, llm_client):
    """Create integrated LLM-Donkey pipeline"""
    
    injector = PromptInjector()
    analyzer = LLMBehaviorAnalyzer()
    
    def execute_with_dsl(task: str, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Execute task with DSL-modified behavior"""
        
        # Get DSL recommendation
        dsl_str, confidence, based_on = donkey_client.donkey(task, context)
        params = donkey_client.parse_dsl(dsl_str)
        dsl_rec = {
            'dsl': dsl_str,
            'confidence': confidence,
            'params': {
                'u': params.u,
                'test_prob': params.test,
                'mem_thresh': params.mem,
                'sim': params.sim,
                'mutate': params.mutate
            }
        }
        
        # Inject behavior
        modified_prompt = injector.inject_dsl_behavior(prompt, dsl_rec['params'])
        
        # Execute (simulated)
        response = llm_client.complete(modified_prompt)
        
        # Measure behavior
        behavior = analyzer.measure_actual_dsl(response.content, dsl_rec['params'])
        
        # Record trajectory
        success_estimate = behavior.followed_recommendation * 0.8 + 0.2
        donkey_client.record(task, dsl_rec['dsl'], success_estimate)
        
        return {
            'response': response.content,
            'dsl': dsl_rec,
            'behavior': behavior,
            'metadata': {
                'compliance': behavior.followed_recommendation,
                'exploration': behavior.actual_u,
                'tested': behavior.actual_test
            }
        }
    
    return execute_with_dsl