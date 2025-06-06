#!/usr/bin/env python3
"""
DONKEY-DSL ORCHESTRATOR v1
Trajectory-based DSL recommendation system
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify
from collections import defaultdict
import re
from llm_donkey_integration import (
    PromptInjector, 
    LLMBehaviorAnalyzer, 
    DSLConversation,
    BehaviorMeasurement
)

@dataclass
class DSLParams:
    u: float  # exploration parameter (0.1-5)
    f: str    # flag: '!', '?', or ''
    mem: float  # memory threshold (0-1)
    test: float  # test probability (0-1)
    sim: bool  # simulation flag
    mutate: float  # mutation rate (0-1)

@dataclass
class Trajectory:
    task: str
    dsl: str
    success: bool
    time: int
    timestamp: Optional[str] = None
    context: Optional[Dict] = None

class DonkeyDSL:
    def __init__(self, data_path: str = "data/trajectories.json"):
        self.data_path = data_path
        self.trajectories: List[Trajectory] = []
        self.task_embeddings: Dict[str, Dict] = {}
        self.dsl_stats: Dict[str, Dict] = defaultdict(lambda: {"success": 0, "total": 0})
        self.app = Flask(__name__)
        # LLM integration components
        self.prompt_injector = PromptInjector()
        self.behavior_analyzer = LLMBehaviorAnalyzer()
        self.active_conversations = {}  # Store ongoing conversations
        self.setup_routes()
        self.load_trajectories()
        
    def load_trajectories(self):
        """Load trajectories from JSON file"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.trajectories = [Trajectory(**t) for t in data]
                    self.index_trajectories()
                    print(f"Loaded {len(self.trajectories)} trajectories")
            except Exception as e:
                print(f"Error loading trajectories: {e}")
                self.trajectories = []
        else:
            print(f"No trajectories file found at {self.data_path}")
            os.makedirs(os.path.dirname(self.data_path) or '.', exist_ok=True)
            self.save_trajectories()
    
    def save_trajectories(self):
        """Save trajectories to JSON file"""
        with open(self.data_path, 'w') as f:
            json.dump([asdict(t) for t in self.trajectories], f, indent=2)
    
    def index_trajectories(self):
        """Index trajectories by task keywords and update stats"""
        for traj in self.trajectories:
            # Simple keyword extraction
            keywords = self.extract_keywords(traj.task)
            self.task_embeddings[traj.task] = {"keywords": keywords}
            
            # Update DSL stats
            self.dsl_stats[traj.dsl]["total"] += 1
            if traj.success:
                self.dsl_stats[traj.dsl]["success"] += 1
    
    def extract_keywords(self, task: str) -> List[str]:
        """Extract keywords from task description"""
        # Simple keyword extraction: split and filter
        words = re.findall(r'\w+', task.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def similarity(self, task1: str, task2: str) -> float:
        """Calculate similarity between two tasks based on keyword overlap"""
        keywords1 = set(self.extract_keywords(task1))
        keywords2 = set(self.extract_keywords(task2))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union)
    
    def parse_dsl(self, dsl: str) -> DSLParams:
        """Parse DSL string into parameters"""
        params = DSLParams(u=1.0, f='', mem=0.5, test=0.5, sim=False, mutate=0.5)
        
        # Extract u parameter
        u_match = re.search(r'u([\d.]+)', dsl)
        if u_match:
            params.u = float(u_match.group(1))
        
        # Extract f flag
        f_match = re.search(r'f([!?])', dsl)
        if f_match:
            params.f = f_match.group(1)
        
        # Extract mem parameter
        mem_match = re.search(r'mem\(([\d.]+)\)', dsl)
        if mem_match:
            params.mem = float(mem_match.group(1))
        
        # Extract test parameter
        test_match = re.search(r'test\(([\d.]+)\)', dsl)
        if test_match:
            params.test = float(test_match.group(1))
        
        # Check for sim
        if 'sim()' in dsl:
            params.sim = True
        
        # Extract mutate parameter
        mutate_match = re.search(r'mutate\(([\d.]+)\)', dsl)
        if mutate_match:
            params.mutate = float(mutate_match.group(1))
        
        return params
    
    def format_dsl(self, params: DSLParams) -> str:
        """Format DSL parameters into string"""
        parts = [f"u{params.u:.1f}"]
        
        if params.f:
            parts.append(f"f{params.f}")
        
        parts.append(f"mem({params.mem:.1f})")
        parts.append(f"test({params.test:.1f})")
        
        if params.sim:
            parts.append("sim()")
        
        if params.mutate != 0.5:  # Only include if not default
            parts.append(f"mutate({params.mutate:.1f})")
        
        return " | ".join(parts)
    
    def find_similar_tasks(self, task: str, k: int = 5) -> List[Tuple[Trajectory, float]]:
        """Find k most similar tasks from trajectories"""
        similarities = []
        
        for traj in self.trajectories:
            sim = self.similarity(task, traj.task)
            if sim > 0:
                similarities.append((traj, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def donkey(self, task: str, context: Optional[Dict] = None) -> Tuple[str, float, int]:
        """Main recommendation algorithm"""
        similar = self.find_similar_tasks(task, k=5)
        
        if not similar:
            # Default DSL for unknown tasks
            return self.get_default_dsl(task), 0.1, 0
        
        # Calculate weighted average of DSL parameters
        total_weight = 0
        weighted_params = DSLParams(u=0, f='', mem=0, test=0, sim=False, mutate=0)
        flag_counts = defaultdict(float)
        
        for traj, sim in similar:
            # Weight by success rate and similarity
            weight = (1 if traj.success else 0.3) * sim
            total_weight += weight
            
            params = self.parse_dsl(traj.dsl)
            weighted_params.u += params.u * weight
            weighted_params.mem += params.mem * weight
            weighted_params.test += params.test * weight
            weighted_params.mutate += params.mutate * weight
            
            if params.f:
                flag_counts[params.f] += weight
            if params.sim:
                flag_counts['sim'] += weight
        
        if total_weight > 0:
            weighted_params.u /= total_weight
            weighted_params.mem /= total_weight
            weighted_params.test /= total_weight
            weighted_params.mutate /= total_weight
            
            # Choose most weighted flag
            if flag_counts:
                weighted_params.f = max(flag_counts.items(), key=lambda x: x[1])[0]
                if weighted_params.f == 'sim':
                    weighted_params.f = ''
                    weighted_params.sim = True
        
        # Apply task-specific patterns
        weighted_params = self.apply_patterns(task, weighted_params)
        
        # Calculate confidence
        confidence = min(len(similar) / 10, 0.9)
        
        return self.format_dsl(weighted_params), confidence, len(similar)
    
    def apply_patterns(self, task: str, params: DSLParams) -> DSLParams:
        """Apply known patterns based on task type"""
        task_lower = task.lower()
        
        if 'fix' in task_lower or 'bug' in task_lower:
            params.test = max(params.test, 0.8)
            params.u = min(params.u, 0.5)
        elif 'refactor' in task_lower:
            params.u = 0.5
            params.test = 0.5
            params.mem = 0.5
        elif 'explore' in task_lower:
            params.u = max(params.u, 2.0)
            params.test = min(params.test, 0.3)
        elif 'prod' in task_lower or 'production' in task_lower:
            params.test = 1.0
            params.sim = True
        
        return params
    
    def get_default_dsl(self, task: str) -> str:
        """Get default DSL for unknown task types"""
        params = DSLParams(u=1.0, f='?', mem=0.5, test=0.5, sim=False, mutate=0.5)
        params = self.apply_patterns(task, params)
        return self.format_dsl(params)
    
    def record_trajectory(self, task: str, dsl: str, success: bool, time: int):
        """Record a new trajectory and update stats"""
        traj = Trajectory(
            task=task,
            dsl=dsl,
            success=success,
            time=time,
            timestamp=datetime.now().isoformat()
        )
        
        self.trajectories.append(traj)
        self.dsl_stats[dsl]["total"] += 1
        if success:
            self.dsl_stats[dsl]["success"] += 1
        
        # Re-index if needed (every 10 trajectories)
        if len(self.trajectories) % 10 == 0:
            self.index_trajectories()
        
        # Save to disk
        self.save_trajectories()
        
        # Calculate new success rate
        stats = self.dsl_stats[dsl]
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        return success_rate
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        total = len(self.trajectories)
        successful = sum(1 for t in self.trajectories if t.success)
        avg_success = successful / total if total > 0 else 0
        
        # Find best DSL configs by task type
        task_types = ['bug', 'refactor', 'explore', 'prod']
        best = {}
        
        for task_type in task_types:
            type_trajectories = [t for t in self.trajectories 
                               if task_type in t.task.lower()]
            if type_trajectories:
                # Group by DSL and find most successful
                dsl_success = defaultdict(lambda: {"success": 0, "total": 0})
                for t in type_trajectories:
                    dsl_success[t.dsl]["total"] += 1
                    if t.success:
                        dsl_success[t.dsl]["success"] += 1
                
                # Find DSL with highest success rate
                best_dsl = max(dsl_success.items(), 
                             key=lambda x: x[1]["success"] / x[1]["total"])
                best[task_type] = best_dsl[0]
        
        return {
            "total": total,
            "avg_success": round(avg_success, 2),
            "best": best
        }
    
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/donkey', methods=['POST'])
        def recommend():
            data = request.json
            task = data.get('task', '')
            context = data.get('context', {})
            
            if not task:
                return jsonify({"error": "Task is required"}), 400
            
            dsl, confidence, based_on = self.donkey(task, context)
            
            return jsonify({
                "dsl": dsl,
                "confidence": round(confidence, 2),
                "based_on": based_on
            })
        
        @self.app.route('/record', methods=['POST'])
        def record():
            data = request.json
            required = ['task', 'dsl', 'success', 'time']
            
            if not all(key in data for key in required):
                return jsonify({"error": f"Required fields: {required}"}), 400
            
            new_success_rate = self.record_trajectory(
                task=data['task'],
                dsl=data['dsl'],
                success=data['success'],
                time=data['time']
            )
            
            return jsonify({
                "recorded": True,
                "new_success_rate": round(new_success_rate, 2)
            })
        
        @self.app.route('/stats', methods=['GET'])
        def stats():
            return jsonify(self.get_stats())
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "trajectories": len(self.trajectories)})
        
        @self.app.route('/llm/execute', methods=['POST'])
        def llm_execute():
            """Execute task with LLM using DSL behavior modification"""
            data = request.json
            task = data.get('task', '')
            prompt = data.get('prompt', task)
            context = data.get('context', {})
            use_donkey = data.get('use_donkey', True)
            
            if not task:
                return jsonify({"error": "Task is required"}), 400
            
            if use_donkey:
                # Get DSL recommendation
                dsl, confidence, based_on = self.donkey(task, context)
                params = self.parse_dsl(dsl)
                params_dict = {
                    'u': params.u,
                    'test_prob': params.test,
                    'mem_thresh': params.mem,
                    'sim': params.sim,
                    'mutate': params.mutate
                }
                
                # Inject DSL behavior
                modified_prompt = self.prompt_injector.inject_dsl_behavior(prompt, params_dict)
                
                # Simulate LLM response (in production, call actual LLM)
                response = self._simulate_llm_response(modified_prompt, params.u)
                
                # Measure behavior
                behavior = self.behavior_analyzer.measure_actual_dsl(response, params_dict)
                
                # Record trajectory
                success = behavior.followed_recommendation > 0.7
                self.record_trajectory(task, dsl, success, 1000)
                
                return jsonify({
                    "response": response,
                    "dsl": {
                        "recommended": dsl,
                        "confidence": confidence,
                        "params": params_dict
                    },
                    "behavior": {
                        "actual_u": behavior.actual_u,
                        "actual_test": behavior.actual_test,
                        "compliance": behavior.followed_recommendation,
                        "unique_approaches": behavior.unique_approaches
                    }
                })
            else:
                # Direct execution without DSL
                response = self._simulate_llm_response(prompt, 1.0)
                return jsonify({
                    "response": response,
                    "dsl": None
                })
        
        @self.app.route('/llm/conversation/start', methods=['POST'])
        def start_conversation():
            """Start a new DSL-guided conversation"""
            data = request.json
            task = data.get('task', '')
            conversation_id = data.get('conversation_id', None)
            
            if not task:
                return jsonify({"error": "Task is required"}), 400
            
            if not conversation_id:
                import uuid
                conversation_id = str(uuid.uuid4())
            
            # Create new conversation
            conv = DSLConversation(task, self)
            self.active_conversations[conversation_id] = conv
            
            return jsonify({
                "conversation_id": conversation_id,
                "task": task,
                "initial_dsl": conv.dsl
            })
        
        @self.app.route('/llm/conversation/turn', methods=['POST'])
        def conversation_turn():
            """Process a conversation turn with DSL adaptation"""
            data = request.json
            conversation_id = data.get('conversation_id')
            user_input = data.get('input', '')
            
            if not conversation_id or conversation_id not in self.active_conversations:
                return jsonify({"error": "Invalid conversation_id"}), 400
            
            conv = self.active_conversations[conversation_id]
            
            # Process turn (simulated LLM)
            response, behavior = conv.turn(user_input, None)
            
            return jsonify({
                "response": response,
                "turn": len(conv.turns),
                "current_dsl": conv.dsl,
                "behavior": {
                    "actual_u": behavior.actual_u,
                    "compliance": behavior.followed_recommendation
                }
            })
        
        @self.app.route('/llm/conversation/end', methods=['POST'])
        def end_conversation():
            """End conversation and record trajectory"""
            data = request.json
            conversation_id = data.get('conversation_id')
            
            if not conversation_id or conversation_id not in self.active_conversations:
                return jsonify({"error": "Invalid conversation_id"}), 400
            
            conv = self.active_conversations[conversation_id]
            trajectory = conv.get_trajectory()
            
            # Record to donkey
            self.record_trajectory(
                trajectory['task'],
                trajectory['dsl'],
                trajectory['success'] > 0.7,
                len(conv.turns) * 1000
            )
            
            # Clean up
            del self.active_conversations[conversation_id]
            
            return jsonify({
                "trajectory": trajectory,
                "recorded": True
            })
        
        @self.app.route('/llm/calibrate', methods=['GET'])
        def get_calibration():
            """Get current DSL effect calibration data"""
            # In production, this would return actual calibration data
            return jsonify({
                "calibration": {
                    "u_effects": {
                        "0.1": 0.15,
                        "0.5": 0.6,
                        "1.0": 1.1,
                        "2.0": 1.8,
                        "5.0": 3.5
                    },
                    "test_compliance": 0.75,
                    "memory_compliance": 0.82
                }
            })
        
    def _simulate_llm_response(self, prompt: str, u_value: float) -> str:
        """Simulate LLM response based on DSL parameters"""
        if u_value > 2.0:
            return f"""Based on the exploration directive, I'll consider multiple approaches:

Approach 1: We could use an innovative algorithm that...
Approach 2: An experimental technique involves...
Approach 3: A creative solution would be to...

Given the high exploration parameter (u={u_value:.1f}), I recommend trying the most novel approach first."""
        elif u_value < 0.5:
            return f"""Following the focus directive, I'll use the standard approach:

The proven solution is to implement a traditional pattern that has been tested extensively. 
This conventional method ensures reliability and maintainability.

As specified by the low exploration parameter (u={u_value:.1f}), I'm avoiding experimental solutions."""
        else:
            return f"""I'll provide a balanced solution:

The standard approach works well here, with some optimizations for better performance.
This combines reliability with modest improvements.

The moderate exploration parameter (u={u_value:.1f}) allows for some innovation within proven patterns."""

def main():
    # Create donkey instance
    donkey = DonkeyDSL()
    
    # Run Flask app
    print("Starting Donkey-DSL Orchestrator on port 5000...")
    donkey.app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()