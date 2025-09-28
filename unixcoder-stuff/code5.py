# PROBLEM: Two Sum
# Given an array of integers nums and an integer target, 
# return indices of the two numbers such that they add up to target.

REFERENCE_SOLUTIONS = {
    "brute_force": '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
    
    "hash_map": '''
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
''',
    
    "two_pointers": '''
def two_sum(nums, target):
    # Create list of (value, index) pairs
    indexed_nums = [(num, i) for i, num in enumerate(nums)]
    indexed_nums.sort()  # Sort by value
    
    left, right = 0, len(indexed_nums) - 1
    
    while left < right:
        current_sum = indexed_nums[left][0] + indexed_nums[right][0]
        if current_sum == target:
            return [indexed_nums[left][1], indexed_nums[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []
''',
    
    "hash_set_approach": '''
def two_sum(nums, target):
    num_set = set(nums)
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_set and complement != num:
            # Find complement's index
            for j in range(i + 1, len(nums)):
                if nums[j] == complement:
                    return [i, j]
    return []
'''
}

USER_CODE_TIMELINE = [
    # Initial exploration phase (0-2 minutes)
    {
        'timestamp': 0,  # 0 seconds
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
''',
        'keystroke_data': {
            'typing_speed_wpm': 45,
            'pause_before_ms': 0,
            'thinking_pause_after_ms': 8500,  # 8.5 second pause after comment
            'backspace_count': 2,
            'total_chars_typed': 65
        }
    },
    
    {
        'timestamp': 15000,  # 15 seconds - starts brute force
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
    for i in range(len(nums)):
        for j in range(len(nums)):
''',
        'keystroke_data': {
            'typing_speed_wpm': 52,
            'pause_before_ms': 3200,  # 3.2 second pause before typing
            'thinking_pause_after_ms': 2100,
            'backspace_count': 0,
            'total_chars_typed': 45
        }
    },
    
    {
        'timestamp': 28000,  # 28 seconds - realizes j should start from i+1
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
''',
        'keystroke_data': {
            'typing_speed_wpm': 48,
            'pause_before_ms': 1800,
            'thinking_pause_after_ms': 1500,
            'backspace_count': 8,  # Had to backspace and fix j range
            'total_chars_typed': 67
        }
    },
    
    {
        'timestamp': 45000,  # 45 seconds - completes brute force
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
        'keystroke_data': {
            'typing_speed_wpm': 55,
            'pause_before_ms': 800,
            'thinking_pause_after_ms': 12000,  # Long pause - thinking about optimization
            'backspace_count': 1,
            'total_chars_typed': 35
        }
    },
    
    # HINT INTERACTION 1: "Can you think of a more efficient approach?"
    {
        'timestamp': 65000,  # 1:05 - after hint, starts thinking about hash map
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
        'keystroke_data': {
            'typing_speed_wpm': 42,  # Slower due to thinking
            'pause_before_ms': 15500,  # Long pause after hint
            'thinking_pause_after_ms': 4200,
            'backspace_count': 3,
            'total_chars_typed': 58
        }
    },
    
    {
        'timestamp': 85000,  # 1:25 - starts implementing hash map but confused
        'code': '''
def two_sum(nums, target):
    # thinking about the problem
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i in range(len(nums)):
        seen[nums[i]] = i
        if target - nums[i] in seen:
            return [i, seen[target - nums[i]]]
    return []
''',
        'keystroke_data': {
            'typing_speed_wpm': 38,  # Even slower, struggling
            'pause_before_ms': 6800,
            'thinking_pause_after_ms': 8500,  # Long pause - realizes bug
            'backspace_count': 12,  # Multiple corrections
            'total_chars_typed': 89
        }
    },
    
    # HINT INTERACTION 2: "Think about when you should check if the complement exists"
    {
        'timestamp': 110000,  # 1:50 - after second hint, fixes the logic
        'code': '''
def two_sum(nums, target):
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
''',
        'keystroke_data': {
            'typing_speed_wpm': 51,  # Faster - more confident
            'pause_before_ms': 18200,  # Long pause processing hint
            'thinking_pause_after_ms': 3500,
            'backspace_count': 25,  # Major rewrite
            'total_chars_typed': 156
        }
    },
    
    {
        'timestamp': 125000,  # 2:05 - adds some error checking
        'code': '''
def two_sum(nums, target):
    if not nums or len(nums) < 2:
        return []
    
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
''',
        'keystroke_data': {
            'typing_speed_wpm': 58,  # Confident now
            'pause_before_ms': 2100,
            'thinking_pause_after_ms': 5500,
            'backspace_count': 2,
            'total_chars_typed': 64
        }
    },
    
    # Optimization phase - starts thinking about edge cases
    {
        'timestamp': 140000,  # 2:20 - considers two pointers approach
        'code': '''
def two_sum(nums, target):
    if not nums or len(nums) < 2:
        return []
    
    # Hash map approach - O(n) time, O(n) space
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
    
    # Alternative: two pointers approach?
    # indexed_nums = [(num, i) for i, num in enumerate(nums)]
    # indexed_nums.sort()
''',
        'keystroke_data': {
            'typing_speed_wpm': 62,
            'pause_before_ms': 4800,
            'thinking_pause_after_ms': 7200,
            'backspace_count': 5,
            'total_chars_typed': 145
        }
    },
    
    {
        'timestamp': 165000,  # 2:45 - decides to stick with hash map, adds comments
        'code': '''
def two_sum(nums, target):
    """
    Find two numbers that add up to target and return their indices.
    
    Approach: Hash map for O(n) solution
    Time: O(n), Space: O(n)
    """
    if not nums or len(nums) < 2:
        return []
    
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []  # No solution found
''',
        'keystroke_data': {
            'typing_speed_wpm': 65,  # Very confident
            'pause_before_ms': 1200,
            'thinking_pause_after_ms': 2000,
            'backspace_count': 8,
            'total_chars_typed': 187
        }
    }
]

# Hint interactions with timestamps
HINT_INTERACTIONS = [
    {
        'hint_id': 'hint_1',
        'timestamp': 57000,  # Given at 57 seconds
        'hint_given': 'Can you think of a more efficient approach? What\'s the time complexity of your current solution?',
        'hint_type': 'optimization_nudge',
        'code_before': '''
def two_sum(nums, target):
    # thinking about the problem
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
        'code_after': '''
def two_sum(nums, target):
    # thinking about the problem
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
        'integration_time_ms': 8000,  # 8 seconds to start responding
        'effectiveness_score': 0.7  # Good hint integration
    },
    
    {
        'hint_id': 'hint_2', 
        'timestamp': 95000,  # Given at 1:35
        'hint_given': 'Think about when you should check if the complement exists. Should you check before or after adding to the dictionary?',
        'hint_type': 'implementation_guidance',
        'code_before': '''
def two_sum(nums, target):
    # thinking about the problem
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i in range(len(nums)):
        seen[nums[i]] = i
        if target - nums[i] in seen:
            return [i, seen[target - nums[i]]]
    return []
''',
        'code_after': '''
def two_sum(nums, target):
    # maybe use a dictionary to store seen numbers?
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
''',
        'integration_time_ms': 15000,  # 15 seconds to fully implement
        'effectiveness_score': 0.9  # Excellent hint integration
    }
]

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import ast
import re

# =================== MISSING COGNITIVE MODEL ANALYZERS ===================

class WorkingMemoryAnalyzer:
    """Analyzes working memory load from code complexity"""
    
    def __init__(self):
        self.complexity_weights = {
            'variable_tracking': 0.3,
            'nested_structures': 0.25,
            'function_calls': 0.2,
            'conditional_branches': 0.15,
            'loop_complexity': 0.1
        }
    
    def analyze_working_memory_load(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Calculate working memory load based on code complexity"""
        
        try:
            tree = ast.parse(code)
        except:
            return 1.0  # Maximum load for unparseable code
        
        load_components = {}
        
        # 1. Variable tracking complexity
        load_components['variable_tracking'] = self._calculate_variable_tracking_load(tree)
        
        # 2. Nested structure complexity
        load_components['nested_structures'] = self._calculate_nesting_load(tree)
        
        # 3. Function call complexity
        load_components['function_calls'] = self._calculate_function_call_load(tree)
        
        # 4. Conditional branch complexity
        load_components['conditional_branches'] = self._calculate_conditional_load(tree)
        
        # 5. Loop complexity
        load_components['loop_complexity'] = self._calculate_loop_complexity_load(tree)
        
        # Weighted sum
        total_load = sum(
            load_components[component] * self.complexity_weights[component]
            for component in load_components
        )
        
        return min(total_load, 1.0)
    
    def _calculate_variable_tracking_load(self, tree: ast.AST) -> float:
        """Calculate load from tracking variables"""
        
        variables = set()
        modifications = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.add(node.id)
                if isinstance(node.ctx, ast.Store):
                    modifications += 1
        
        # More variables and modifications = higher load
        base_load = min(len(variables) / 15.0, 1.0)  # 15+ variables = max load
        modification_load = min(modifications / 20.0, 0.5)  # 20+ modifications = 0.5 additional load
        
        return base_load + modification_load
    
    def _calculate_nesting_load(self, tree: ast.AST) -> float:
        """Calculate load from nested structures"""
        
        max_depth = 0
        current_depth = 0
        
        def calculate_depth(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.FunctionDef, ast.ClassDef)):
                    calculate_depth(child, depth + 1)
                else:
                    calculate_depth(child, depth)
        
        calculate_depth(tree)
        
        # Exponential penalty for deep nesting
        return min(max_depth / 5.0, 1.0) ** 0.5
    
    def _calculate_function_call_load(self, tree: ast.AST) -> float:
        """Calculate load from function calls"""
        
        call_count = 0
        unique_functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_count += 1
                if isinstance(node.func, ast.Name):
                    unique_functions.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    unique_functions.add(node.func.attr)
        
        # Load based on number of calls and unique functions
        call_load = min(call_count / 10.0, 0.7)
        function_diversity_load = min(len(unique_functions) / 8.0, 0.3)
        
        return call_load + function_diversity_load
    
    def _calculate_conditional_load(self, tree: ast.AST) -> float:
        """Calculate load from conditional complexity"""
        
        conditionals = 0
        complex_conditions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                conditionals += 1
                # Check for complex conditions (multiple operators)
                if self._is_complex_condition(node.test):
                    complex_conditions += 1
        
        base_load = min(conditionals / 8.0, 0.6)
        complexity_load = min(complex_conditions / 4.0, 0.4)
        
        return base_load + complexity_load
    
    def _calculate_loop_complexity_load(self, tree: ast.AST) -> float:
        """Calculate load from loop complexity"""
        
        loops = 0
        nested_loops = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loops += 1
                # Check for nested loops
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        nested_loops += 1
        
        base_load = min(loops / 5.0, 0.6)
        nesting_load = min(nested_loops / 3.0, 0.4)
        
        return base_load + nesting_load
    
    def _is_complex_condition(self, condition_node) -> bool:
        """Check if condition is complex (multiple operators)"""
        
        operators = 0
        for node in ast.walk(condition_node):
            if isinstance(node, (ast.BoolOp, ast.Compare)):
                operators += 1
        
        return operators > 1


class AttentionDistributionAnalyzer:
    """Analyzes how attention is distributed across different cognitive tasks"""
    
    def __init__(self):
        self.attention_categories = {
            'algorithmic_design': [
                'recursion', 'dynamic_programming', 'divide_conquer', 'backtracking',
                'graph_dfs', 'graph_bfs', 'topological_sort'
            ],
            'data_structure_focus': [
                'hash_map', 'hash_set', 'heap_usage', 'stack_usage', 'queue_usage',
                'trie_usage', 'segment_tree', 'union_find'
            ],
            'optimization_thinking': [
                'space_optimization', 'time_optimization', 'binary_search',
                'cache_optimization', 'lazy_propagation'
            ],
            'pattern_application': [
                'two_pointers', 'sliding_window', 'monotonic_stack', 'prefix_suffix',
                'fast_slow_pointers'
            ],
            'implementation_details': [
                'array_manipulation', 'string_operations', 'bit_manipulation',
                'conditional_logic', 'iteration'
            ],
            'error_handling': [
                'edge_case_handling', 'input_validation', 'defensive_programming'
            ]
        }
    
    def analyze_attention_distribution(self, pattern_scores: Dict[str, float], 
                                     code_history: List = None) -> Dict[str, float]:
        """Analyze how attention is distributed across cognitive categories"""
        
        attention_dist = {}
        total_attention = sum(pattern_scores.values()) if pattern_scores else 1
        
        # Calculate attention for each category
        for category, patterns in self.attention_categories.items():
            category_attention = sum(pattern_scores.get(pattern, 0) for pattern in patterns)
            attention_dist[category] = category_attention / total_attention if total_attention > 0 else 0
        
        # Add meta-attention metrics
        attention_dist['focus_intensity'] = max(attention_dist.values()) if attention_dist else 0
        attention_dist['attention_scatter'] = len([v for v in attention_dist.values() if v > 0.1])
        attention_dist['cognitive_balance'] = self._calculate_cognitive_balance(attention_dist)
        
        # Historical attention analysis
        if code_history:
            attention_dist['attention_consistency'] = self._analyze_attention_consistency(code_history)
        
        return attention_dist
    
    def _calculate_cognitive_balance(self, attention_dist: Dict[str, float]) -> float:
        """Calculate how balanced attention is across categories"""
        
        # Remove meta-metrics for balance calculation
        core_categories = {k: v for k, v in attention_dist.items() 
                          if k not in ['focus_intensity', 'attention_scatter', 'cognitive_balance']}
        
        if not core_categories:
            return 0.0
        
        values = list(core_categories.values())
        mean_attention = sum(values) / len(values)
        
        # Calculate coefficient of variation (std/mean)
        if mean_attention == 0:
            return 1.0  # Perfect balance when no attention
        
        variance = sum((v - mean_attention) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_attention
        
        # Convert to balance score (lower CV = higher balance)
        balance_score = max(0, 1.0 - cv)
        return balance_score
    
    def _analyze_attention_consistency(self, code_history: List) -> float:
        """Analyze consistency of attention over time"""
        
        if len(code_history) < 2:
            return 0.5  # Neutral for insufficient history
        
        # This would analyze attention patterns over time
        # Simplified implementation
        consistency_score = 0.7  # Placeholder
        return consistency_score


class MentalModelTracker:
    """Tracks the development and clarity of mental models"""
    
    def __init__(self):
        self.model_indicators = {
            'conceptual_clarity': self._assess_conceptual_clarity,
            'implementation_coherence': self._assess_implementation_coherence,
            'problem_understanding': self._assess_problem_understanding,
            'solution_confidence': self._assess_solution_confidence
        }
    
    def track_mental_model_development(self, code: str, pattern_scores: Dict[str, float], 
                                     history: List = None) -> Dict[str, float]:
        """Track mental model clarity and development"""
        
        model_metrics = {}
        
        for indicator_name, assessment_func in self.model_indicators.items():
            model_metrics[indicator_name] = assessment_func(code, pattern_scores, history)
        
        # Overall mental model clarity
        model_metrics['overall_clarity'] = sum(model_metrics.values()) / len(model_metrics)
        
        # Model stability (if history available)
        if history and len(history) >= 2:
            model_metrics['model_stability'] = self._assess_model_stability(history)
        
        return model_metrics
    
    def _assess_conceptual_clarity(self, code: str, pattern_scores: Dict[str, float], 
                                  history: List) -> float:
        """Assess clarity of conceptual understanding"""
        
        # Strong pattern signals indicate clear conceptual understanding
        max_pattern_strength = max(pattern_scores.values()) if pattern_scores else 0
        
        # Multiple weak patterns might indicate confusion
        weak_patterns = len([score for score in pattern_scores.values() if 0.1 < score < 0.4])
        confusion_penalty = min(weak_patterns * 0.1, 0.3)
        
        clarity_score = max_pattern_strength - confusion_penalty
        return max(0, min(clarity_score, 1.0))
    
    def _assess_implementation_coherence(self, code: str, pattern_scores: Dict[str, float], 
                                       history: List) -> float:
        """Assess coherence between concept and implementation"""
        
        try:
            tree = ast.parse(code)
            syntax_coherence = 1.0
        except:
            syntax_coherence = 0.0
        
        # Check if implementation matches intended pattern
        dominant_pattern = max(pattern_scores, key=pattern_scores.get) if pattern_scores else None
        
        if dominant_pattern and pattern_scores[dominant_pattern] > 0.5:
            pattern_coherence = pattern_scores[dominant_pattern]
        else:
            pattern_coherence = 0.5
        
        return (syntax_coherence + pattern_coherence) / 2
    
    def _assess_problem_understanding(self, code: str, pattern_scores: Dict[str, float], 
                                    history: List) -> float:
        """Assess depth of problem understanding"""
        
        understanding_indicators = []
        
        # Edge case handling indicates good problem understanding
        edge_case_score = pattern_scores.get('conditional_logic', 0)
        understanding_indicators.append(edge_case_score)
        
        # Optimization awareness indicates deeper understanding
        optimization_score = max(
            pattern_scores.get('space_optimization', 0),
            pattern_scores.get('time_optimization', 0)
        )
        understanding_indicators.append(optimization_score)
        
        # Appropriate algorithm choice indicates understanding
        algorithm_appropriateness = self._assess_algorithm_appropriateness(pattern_scores)
        understanding_indicators.append(algorithm_appropriateness)
        
        return sum(understanding_indicators) / len(understanding_indicators)
    
    def _assess_solution_confidence(self, code: str, pattern_scores: Dict[str, float], 
                                  history: List) -> float:
        """Assess confidence in current solution approach"""
        
        # Strong single pattern indicates confidence
        max_score = max(pattern_scores.values()) if pattern_scores else 0
        
        # Too many competing patterns indicates uncertainty
        competing_patterns = len([score for score in pattern_scores.values() if score > 0.3])
        
        if competing_patterns <= 2:
            confidence_score = max_score
        else:
            # Penalty for too many competing approaches
            confidence_score = max_score * (2.0 / competing_patterns)
        
        return min(confidence_score, 1.0)
    
    def _assess_algorithm_appropriateness(self, pattern_scores: Dict[str, float]) -> float:
        """Assess if chosen algorithm is appropriate"""
        
        # This would need problem context to be fully accurate
        # For now, return a heuristic based on pattern strength
        max_score = max(pattern_scores.values()) if pattern_scores else 0
        return min(max_score * 1.2, 1.0)  # Slight boost for having any strong pattern
    
    def _assess_model_stability(self, history: List) -> float:
        """Assess stability of mental model over time"""
        
        if len(history) < 2:
            return 0.5
        
        # Simplified stability assessment
        # In reality, would compare mental model metrics over time
        stability_score = 0.6  # Placeholder
        return stability_score


class StressPatternAnalyzer:
    """Analyzes patterns indicating cognitive stress"""
    
    def __init__(self):
        self.stress_indicators = {
            'code_quality_degradation': self._detect_quality_degradation,
            'pattern_instability': self._detect_pattern_instability,
            'syntax_errors': self._detect_syntax_error_patterns,
            'overthinking_signals': self._detect_overthinking,
            'regression_patterns': self._detect_regression_patterns
        }
    
    def analyze_stress_patterns(self, code: str, pattern_scores: Dict[str, float],
                              history: List = None, keystroke_data: Dict = None) -> Dict[str, float]:
        """Analyze various stress indicators"""
        
        stress_metrics = {}
        
        # Analyze each stress indicator
        for indicator_name, detector_func in self.stress_indicators.items():
            stress_metrics[indicator_name] = detector_func(code, pattern_scores, history, keystroke_data)
        
        # Keystroke-based stress indicators
        if keystroke_data:
            stress_metrics.update(self._analyze_keystroke_stress(keystroke_data))
        
        # Overall stress level
        stress_metrics['overall_stress_level'] = sum(stress_metrics.values()) / len(stress_metrics)
        
        return stress_metrics
    
    def _detect_quality_degradation(self, code: str, pattern_scores: Dict[str, float],
                                   history: List, keystroke_data: Dict) -> float:
        """Detect degradation in code quality"""
        
        if not history or len(history) < 2:
            return 0.0
        
        # Compare current quality to recent history
        current_quality = max(pattern_scores.values()) if pattern_scores else 0
        
        # Get recent quality metrics (simplified)
        recent_qualities = [0.6, 0.7, 0.5]  # Placeholder - would be calculated from history
        
        if len(recent_qualities) >= 2:
            avg_recent_quality = sum(recent_qualities) / len(recent_qualities)
            if current_quality < avg_recent_quality - 0.2:
                return min((avg_recent_quality - current_quality) * 2, 1.0)
        
        return 0.0
    
    def _detect_pattern_instability(self, code: str, pattern_scores: Dict[str, float],
                                   history: List, keystroke_data: Dict) -> float:
        """Detect instability in approach patterns"""
        
        if not history or len(history) < 3:
            return 0.0
        
        # Look for frequent switching between approaches
        # This would analyze the primary patterns over recent history
        pattern_switches = 2  # Placeholder - would count actual switches
        
        if pattern_switches > 3:
            return min(pattern_switches / 5.0, 1.0)
        
        return 0.0
    
    def _detect_syntax_error_patterns(self, code: str, pattern_scores: Dict[str, float],
                                     history: List, keystroke_data: Dict) -> float:
        """Detect syntax errors indicating stress"""
        
        try:
            ast.parse(code)
            return 0.0  # No syntax errors
        except SyntaxError:
            return 0.7  # Syntax errors indicate stress
        except:
            return 0.5  # Other parsing issues
    
    def _detect_overthinking(self, code: str, pattern_scores: Dict[str, float],
                           history: List, keystroke_data: Dict) -> float:
        """Detect signs of overthinking"""
        
        # Too many weak patterns might indicate overthinking
        weak_patterns = len([score for score in pattern_scores.values() if 0.1 < score < 0.4])
        
        if weak_patterns > 4:
            return min(weak_patterns / 6.0, 1.0)
        
        return 0.0
    
    def _detect_regression_patterns(self, code: str, pattern_scores: Dict[str, float],
                                   history: List, keystroke_data: Dict) -> float:
        """Detect regression in solution sophistication"""
        
        if not history or len(history) < 2:
            return 0.0
        
        # This would compare sophistication over time
        # Placeholder implementation
        return 0.0
    
    def _analyze_keystroke_stress(self, keystroke_data: Dict) -> Dict[str, float]:
        """Analyze stress indicators from keystroke data"""
        
        stress_indicators = {}
        
        # High backspace count indicates uncertainty/stress
        if keystroke_data.get('backspace_count', 0) > 10:
            stress_indicators['correction_stress'] = min(keystroke_data['backspace_count'] / 20.0, 1.0)
        else:
            stress_indicators['correction_stress'] = 0.0
        
        # Very long thinking pauses might indicate stress
        thinking_pause = keystroke_data.get('thinking_pause_after_ms', 0)
        if thinking_pause > 15000:  # 15+ seconds
            stress_indicators['pause_stress'] = min((thinking_pause - 15000) / 30000.0, 1.0)
        else:
            stress_indicators['pause_stress'] = 0.0
        
        # Decreasing typing speed might indicate uncertainty
        typing_speed = keystroke_data.get('typing_speed_wpm', 50)
        if typing_speed < 30:
            stress_indicators['speed_stress'] = (30 - typing_speed) / 30.0
        else:
            stress_indicators['speed_stress'] = 0.0
        
        return stress_indicators


class LearningCurveAnalyzer:
    """Analyzes learning patterns and improvement rates"""
    
    def __init__(self):
        self.learning_dimensions = [
            'pattern_recognition_speed',
            'implementation_quality',
            'optimization_awareness',
            'error_correction_ability',
            'conceptual_understanding'
        ]
    
    def analyze_learning_progression(self, timeline_data: List) -> Dict[str, float]:
        """Analyze learning curve from timeline data"""
        
        if len(timeline_data) < 3:
            return {'insufficient_data': True}
        
        learning_metrics = {}
        
        # Analyze improvement in each dimension
        for dimension in self.learning_dimensions:
            improvement_rate = self._calculate_improvement_rate(timeline_data, dimension)
            learning_metrics[f'{dimension}_improvement'] = improvement_rate
        
        # Overall learning velocity
        learning_metrics['overall_learning_velocity'] = self._calculate_overall_velocity(learning_metrics)
        
        # Learning acceleration (is improvement rate increasing?)
        learning_metrics['learning_acceleration'] = self._calculate_learning_acceleration(timeline_data)
        
        # Learning consistency (steady vs erratic improvement)
        learning_metrics['learning_consistency'] = self._calculate_learning_consistency(timeline_data)
        
        # Plateau detection
        learning_metrics['plateau_detection'] = self._detect_learning_plateau(timeline_data)
        
        return learning_metrics
    
    def _calculate_improvement_rate(self, timeline_data: List, dimension: str) -> float:
        """Calculate improvement rate for specific dimension"""
        
        # This would extract the specific dimension values from timeline
        # For now, using placeholder values
        values = [0.3, 0.4, 0.6, 0.7, 0.8]  # Placeholder progression
        
        if len(values) < 2:
            return 0.0
        
        # Calculate linear improvement rate
        improvement_rate = (values[-1] - values[0]) / len(values)
        return max(0, improvement_rate)
    
    def _calculate_overall_velocity(self, learning_metrics: Dict[str, float]) -> float:
        """Calculate overall learning velocity"""
        
        improvement_rates = [v for k, v in learning_metrics.items() if k.endswith('_improvement')]
        
        if not improvement_rates:
            return 0.0
        
        return sum(improvement_rates) / len(improvement_rates)
    
    def _calculate_learning_acceleration(self, timeline_data: List) -> float:
        """Calculate if learning is accelerating"""
        
        if len(timeline_data) < 4:
            return 0.0
        
        # Compare early vs late improvement rates
        mid_point = len(timeline_data) // 2
        early_data = timeline_data[:mid_point]
        late_data = timeline_data[mid_point:]
        
        # This would calculate actual acceleration
        # Placeholder implementation
        acceleration = 0.1  # Slight positive acceleration
        return acceleration
    
    def _calculate_learning_consistency(self, timeline_data: List) -> float:
        """Calculate consistency of learning progression"""
        
        # This would analyze variance in improvement rates
        # Placeholder implementation
        consistency = 0.7  # Fairly consistent
        return consistency
    
    def _detect_learning_plateau(self, timeline_data: List) -> float:
        """Detect if learning has plateaued"""
        
        if len(timeline_data) < 4:
            return 0.0
        
        # Look for flat regions in recent timeline
        # Placeholder implementation
        plateau_score = 0.2  # Slight plateau detected
        return plateau_score

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class CognitiveState:
    timestamp: int
    working_memory_load: float
    attention_distribution: Dict[str, float]
    processing_confidence: float
    cognitive_fluency: float
    mental_model_clarity: float
    stress_indicators: Dict[str, float]

class CognitiveLoadType(Enum):
    INTRINSIC = "intrinsic"  # Problem complexity
    EXTRANEOUS = "extraneous"  # Poor problem representation
    GERMANE = "germane"  # Schema construction/learning

class AdvancedCognitiveLoadTracker:
    def __init__(self, detector):
        self.detector = detector
        self.cognitive_models = {
            'working_memory': WorkingMemoryAnalyzer(),
            'attention': AttentionDistributionAnalyzer(),
            'mental_model': MentalModelTracker(),
            'stress_response': StressPatternAnalyzer(),
            'learning_curve': LearningCurveAnalyzer()
        }

    def analyze_cognitive_state_at_moment(self, timestamp: int, code: str, history: List[CognitiveState]) -> CognitiveState:
        """Analyze cognitive state at a specific moment"""
        
        # Get pattern scores for this moment
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # Calculate cognitive dimensions
        working_memory_load = self.calculate_working_memory_load(code, pattern_scores)
        attention_distribution = self.analyze_attention_distribution(pattern_scores, history)
        
        # FIX: Add 'code' parameter to this method call
        processing_confidence = self.measure_processing_confidence(pattern_scores, history, code)
        
        cognitive_fluency = self.assess_cognitive_fluency(code, history)
        mental_model_clarity = self.evaluate_mental_model_clarity(pattern_scores, history, code)
        stress_indicators = self.detect_stress_indicators(code, pattern_scores, history)
        
        return CognitiveState(
            timestamp=timestamp,
            working_memory_load=working_memory_load,
            attention_distribution=attention_distribution,
            processing_confidence=processing_confidence,
            cognitive_fluency=cognitive_fluency,
            mental_model_clarity=mental_model_clarity,
            stress_indicators=stress_indicators
        )
    
    def measure_processing_confidence(self, pattern_scores: Dict[str, float], history: List[CognitiveState], code: str) -> float:
        """Measure confidence in current processing approach"""
        
        confidence_indicators = {}
        
        # 1. PATTERN STRENGTH CONFIDENCE
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0
        confidence_indicators['pattern_strength'] = max_pattern_score
        
        # 2. CONSISTENCY CONFIDENCE (are they staying with an approach?)
        if len(history) >= 2:
            prev_scores = history[-1].attention_distribution if history else {}
            current_primary = max(pattern_scores, key=pattern_scores.get) if pattern_scores else None
            
            # Check if they're consistent with previous approach
            consistency_score = 0.0
            if current_primary and len(history) > 0:
                # Look at recent history for consistency
                recent_approaches = []
                for state in history[-3:]:  # Last 3 states
                    if hasattr(state, 'primary_pattern'):
                        recent_approaches.append(state.primary_pattern)
                
                if current_primary in recent_approaches:
                    consistency_score = recent_approaches.count(current_primary) / len(recent_approaches)
            
            confidence_indicators['consistency'] = consistency_score
        else:
            confidence_indicators['consistency'] = 0.5  # Neutral for early stages
        
        # 3. PROGRESS CONFIDENCE (are they making forward progress?)
        if len(history) >= 2:
            recent_complexity = [state.working_memory_load for state in history[-3:]]
            if len(recent_complexity) >= 2:
                complexity_trend = np.polyfit(range(len(recent_complexity)), recent_complexity, 1)[0]
                # Negative trend = reducing complexity = increasing confidence
                progress_confidence = max(0, 1.0 + complexity_trend)  # Invert trend
            else:
                progress_confidence = 0.5
            
            confidence_indicators['progress'] = progress_confidence
        else:
            confidence_indicators['progress'] = 0.5
        
        # 4. IMPLEMENTATION READINESS CONFIDENCE
        implementation_signals = [
            self.detector.detect_conditional_logic(code) > 0.3,  # Now 'code' is properly passed
            any(score > 0.5 for score in pattern_scores.values()),
            len([s for s in pattern_scores.values() if s > 0.3]) <= 2  # Focused approach
        ]
        
        readiness_confidence = sum(implementation_signals) / len(implementation_signals)
        confidence_indicators['implementation_readiness'] = readiness_confidence
        
        # Weighted average
        overall_confidence = (
            confidence_indicators['pattern_strength'] * 0.3 +
            confidence_indicators['consistency'] * 0.25 +
            confidence_indicators['progress'] * 0.25 +
            confidence_indicators['implementation_readiness'] * 0.2
        )
        
        return overall_confidence

    
    def measure_processing_confidence(self, pattern_scores: Dict[str, float], history: List[CognitiveState], code: str) -> float:
        """Measure confidence in current processing approach"""
        
        confidence_indicators = {}
        
        # 1. PATTERN STRENGTH CONFIDENCE
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 0
        confidence_indicators['pattern_strength'] = max_pattern_score
        
        # 2. CONSISTENCY CONFIDENCE (are they staying with an approach?)
        if len(history) >= 2:
            prev_scores = history[-1].attention_distribution if history else {}
            current_primary = max(pattern_scores, key=pattern_scores.get) if pattern_scores else None
            
            # Check if they're consistent with previous approach
            consistency_score = 0.0
            if current_primary and len(history) > 0:
                # Look at recent history for consistency
                recent_approaches = []
                for state in history[-3:]:  # Last 3 states
                    if hasattr(state, 'primary_pattern'):
                        recent_approaches.append(state.primary_pattern)
                
                if current_primary in recent_approaches:
                    consistency_score = recent_approaches.count(current_primary) / len(recent_approaches)
            
            confidence_indicators['consistency'] = consistency_score
        else:
            confidence_indicators['consistency'] = 0.5  # Neutral for early stages
        
        # 3. PROGRESS CONFIDENCE (are they making forward progress?)
        if len(history) >= 2:
            recent_complexity = [state.working_memory_load for state in history[-3:]]
            if len(recent_complexity) >= 2:
                complexity_trend = np.polyfit(range(len(recent_complexity)), recent_complexity, 1)[0]
                # Negative trend = reducing complexity = increasing confidence
                progress_confidence = max(0, 1.0 + complexity_trend)  # Invert trend
            else:
                progress_confidence = 0.5
            
            confidence_indicators['progress'] = progress_confidence
        else:
            confidence_indicators['progress'] = 0.5
        
        # 4. IMPLEMENTATION READINESS CONFIDENCE
        implementation_signals = [
            self.detector.detect_conditional_logic(code) > 0.3,  # Fixed: now code is available
            any(score > 0.5 for score in pattern_scores.values()),
            len([s for s in pattern_scores.values() if s > 0.3]) <= 2  # Focused approach
        ]
        
        readiness_confidence = sum(implementation_signals) / len(implementation_signals)
        confidence_indicators['implementation_readiness'] = readiness_confidence
        
        # Weighted average
        overall_confidence = (
            confidence_indicators['pattern_strength'] * 0.3 +
            confidence_indicators['consistency'] * 0.25 +
            confidence_indicators['progress'] * 0.25 +
            confidence_indicators['implementation_readiness'] * 0.2
        )
        
        return overall_confidence
    
    def evaluate_mental_model_clarity(self, pattern_scores: Dict[str, float], history: List[CognitiveState], code: str) -> float:
        """Evaluate mental model clarity"""
        
        # Use the mental model tracker
        if hasattr(self, 'cognitive_models') and 'mental_model' in self.cognitive_models:
            mental_model_metrics = self.cognitive_models['mental_model'].track_mental_model_development(
                code, pattern_scores, history
            )
            return mental_model_metrics.get('overall_clarity', 0.5)
        
        # Fallback implementation
        clarity_score = 0.0
        
        # Strong pattern signals indicate clear understanding
        max_pattern_strength = max(pattern_scores.values()) if pattern_scores else 0
        clarity_score += max_pattern_strength * 0.4
        
        # Multiple weak patterns might indicate confusion
        weak_patterns = len([score for score in pattern_scores.values() if 0.1 < score < 0.4])
        confusion_penalty = min(weak_patterns * 0.1, 0.3)
        clarity_score -= confusion_penalty
        
        # Code structure clarity
        try:
            tree = ast.parse(code)
            # Well-structured code indicates clear mental model
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            if functions > 0:
                clarity_score += 0.2
            
            # Comments indicate thinking process
            comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
            if comment_lines > 0:
                clarity_score += 0.1
        except:
            clarity_score -= 0.2  # Syntax errors indicate unclear thinking
        
        return max(0, min(clarity_score, 1.0))
    
    def analyze_attention_distribution(self, pattern_scores: Dict[str, float], history: List[CognitiveState]) -> Dict[str, float]:
        """Analyze how attention is distributed across different cognitive tasks"""
        
        # Use the attention analyzer
        if hasattr(self, 'cognitive_models') and 'attention' in self.cognitive_models:
            return self.cognitive_models['attention'].analyze_attention_distribution(pattern_scores, history)
        
        # Fallback implementation
        attention_categories = {
            'algorithmic_thinking': ['recursion', 'dynamic_programming', 'divide_conquer', 'backtracking'],
            'data_structure_management': ['hash_map', 'heap_usage', 'stack_usage', 'queue_usage', 'trie_usage'],
            'optimization_focus': ['space_optimization', 'time_optimization', 'binary_search'],
            'pattern_recognition': ['two_pointers', 'sliding_window', 'monotonic_stack'],
            'system_design': ['graph_algorithms', 'tree_traversal', 'union_find'],
            'implementation_details': ['array_manipulation', 'string_operations', 'bit_manipulation']
        }
        
        attention_distribution = {}
        
        for category, patterns in attention_categories.items():
            category_attention = sum(pattern_scores.get(pattern, 0) for pattern in patterns)
            total_attention = sum(pattern_scores.values()) if pattern_scores else 1
            
            attention_distribution[category] = category_attention / total_attention if total_attention > 0 else 0
        
        # Analyze attention focus vs. dispersion
        attention_distribution['focus_intensity'] = max(attention_distribution.values()) if attention_distribution else 0
        attention_distribution['attention_dispersion'] = len([v for v in attention_distribution.values() if v > 0.1])
        
        return attention_distribution

    def detect_early_termination_patterns(self, code: str) -> float:
        """Detect early termination optimization patterns"""
        
        termination_score = 0.0
        
        # Look for early return patterns
        early_returns = len(re.findall(r'^\s*if.*:\s*return', code, re.MULTILINE))
        if early_returns > 0:
            termination_score += min(early_returns * 0.2, 0.4)
        
        # Look for break statements in loops
        break_statements = len(re.findall(r'\bbreak\b', code))
        if break_statements > 0:
            termination_score += min(break_statements * 0.1, 0.3)
        
        # Look for continue statements
        continue_statements = len(re.findall(r'\bcontinue\b', code))
        if continue_statements > 0:
            termination_score += min(continue_statements * 0.1, 0.2)
        
        # Look for early termination conditions
        early_termination_patterns = [
            r'if.*len\(.*\)\s*==\s*0.*return',
            r'if.*not.*return',
            r'if.*is\s+None.*return'
        ]
        
        for pattern in early_termination_patterns:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            if matches > 0:
                termination_score += 0.1
        
        return min(termination_score, 1.0)

    def analyze_redundancy_elimination(self, code: str) -> float:
        """Analyze redundancy elimination patterns"""
        
        redundancy_score = 0.0
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        # Look for variable reuse
        variable_assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        variable_assignments[var_name] = variable_assignments.get(var_name, 0) + 1
        
        # Reusing variables instead of creating new ones
        reused_vars = sum(1 for count in variable_assignments.values() if count > 1)
        if reused_vars > 0:
            redundancy_score += min(reused_vars * 0.1, 0.3)
        
        # Look for function calls that avoid redundancy
        function_calls = len([node for node in ast.walk(tree) if isinstance(node, ast.Call)])
        if function_calls > 0:
            redundancy_score += min(function_calls * 0.05, 0.2)
        
        # Look for efficient data structure usage
        efficient_operations = [
            'dict.get', 'set.add', 'set.remove', 'list.pop', 'collections.defaultdict'
        ]
        
        for operation in efficient_operations:
            if operation in code:
                redundancy_score += 0.1
        
        return min(redundancy_score, 1.0)

    def measure_solution_extensibility(self, code: str) -> float:
        """Measure how extensible the solution is"""
        
        extensibility_score = 0.0
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        # Functions make code more extensible
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            extensibility_score += 0.3
            
            # Parameterized functions are more extensible
            total_params = sum(len(func.args.args) for func in functions)
            if total_params > 0:
                extensibility_score += min(total_params * 0.1, 0.2)
        
        # Constants and configuration variables make code extensible
        constants = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants += 1
        
        if constants > 0:
            extensibility_score += min(constants * 0.1, 0.2)
        
        # Generic patterns are more extensible
        generic_patterns = ['recursion', 'dynamic_programming', 'template_method']
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        for pattern in generic_patterns:
            if pattern_scores.get(pattern, 0) > 0.3:
                extensibility_score += 0.1
        
        # Comments explaining approach make it extensible
        meaningful_comments = len([line for line in code.split('\n') 
                                 if line.strip().startswith('#') and len(line.strip()) > 10])
        if meaningful_comments > 0:
            extensibility_score += min(meaningful_comments * 0.05, 0.15)
        
        return min(extensibility_score, 1.0)


    def assess_control_flow_cognitive_load(self, code: str) -> float:
        """Assess cognitive load from control flow complexity"""
        
        try:
            tree = ast.parse(code)
        except:
            return 1.0
        
        control_flow_load = 0.0
        
        # Count different control flow constructs
        if_statements = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
        loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
        try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
        
        # Basic control flow load
        control_flow_load += min(if_statements / 5.0, 0.4)
        control_flow_load += min(loops / 3.0, 0.3)
        control_flow_load += min(try_blocks / 2.0, 0.2)
        
        # Nested control flow (much higher load)
        nested_complexity = self._calculate_nested_control_flow_complexity(tree)
        control_flow_load += nested_complexity
        
        # Complex conditions
        complex_conditions = self._count_complex_conditions(tree)
        control_flow_load += min(complex_conditions / 3.0, 0.1)
        
        return min(control_flow_load, 1.0)
    
    def _calculate_nested_control_flow_complexity(self, tree: ast.AST) -> float:
        """Calculate complexity from nested control flow"""
        
        def get_control_nesting_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                    child_depth = get_control_nesting_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_control_nesting_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        max_nesting = get_control_nesting_depth(tree)
        
        # Exponential penalty for deep nesting
        if max_nesting <= 2:
            return 0.0
        elif max_nesting <= 3:
            return 0.1
        elif max_nesting <= 4:
            return 0.3
        else:
            return 0.5
    
    def _count_complex_conditions(self, tree: ast.AST) -> int:
        """Count complex conditional expressions"""
        
        complex_conditions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if condition has multiple boolean operators
                bool_ops = len([n for n in ast.walk(node.test) if isinstance(n, ast.BoolOp)])
                comparisons = len([n for n in ast.walk(node.test) if isinstance(n, ast.Compare)])
                
                if bool_ops > 0 or comparisons > 1:
                    complex_conditions += 1
        
        return complex_conditions
    
    def classify_trajectory_pattern(self, values: List[float]) -> Dict:
        """Classify the pattern of value changes over time"""
        
        if len(values) < 3:
            return {'pattern': 'INSUFFICIENT_DATA'}
        
        # Calculate first and second derivatives (rate of change and acceleration)
        first_diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        second_diffs = [first_diffs[i+1] - first_diffs[i] for i in range(len(first_diffs)-1)]
        
        avg_first_diff = sum(first_diffs) / len(first_diffs)
        avg_second_diff = sum(second_diffs) / len(second_diffs) if second_diffs else 0
        
        # Classify pattern based on trends
        if avg_first_diff > 0.05:
            if avg_second_diff > 0.02:
                return {
                    'pattern': 'ACCELERATING_GROWTH',
                    'description': 'Rapidly improving with increasing rate',
                    'interpretation': 'Excellent learning acceleration'
                }
            elif avg_second_diff < -0.02:
                return {
                    'pattern': 'DECELERATING_GROWTH', 
                    'description': 'Improving but rate is slowing',
                    'interpretation': 'Good improvement but may be plateauing'
                }
            else:
                return {
                    'pattern': 'LINEAR_GROWTH',
                    'description': 'Steady consistent improvement',
                    'interpretation': 'Reliable, predictable progress'
                }
        elif avg_first_diff < -0.05:
            return {
                'pattern': 'DECLINING',
                'description': 'Performance declining over time',
                'interpretation': 'Concerning - may indicate fatigue or confusion'
            }
        else:
            variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
            if variance < 0.01:
                return {
                    'pattern': 'STABLE',
                    'description': 'Consistent performance level',
                    'interpretation': 'Steady, predictable performance'
                }
            else:
                return {
                    'pattern': 'FLUCTUATING',
                    'description': 'Inconsistent performance with ups and downs',
                    'interpretation': 'Variable performance - may need more time to stabilize'
                }
    
    def identify_constraint_moments(self, timeline: List) -> List[Dict]:
        """Identify moments where constraints (time, complexity) affected performance"""
        
        constraint_moments = []
        
        for i, point in enumerate(timeline[1:], 1):  # Start from second point
            prev_point = timeline[i-1]
            
            # Detect sudden complexity increases
            complexity_jump = point.get('complexity', 0) - prev_point.get('complexity', 0)
            if complexity_jump > 0.3:
                constraint_moments.append({
                    'timestamp': point.get('timestamp', i),
                    'type': 'COMPLEXITY_CONSTRAINT',
                    'severity': min(complexity_jump, 1.0),
                    'description': 'Sudden increase in problem complexity'
                })
            
            # Detect performance drops under pressure
            if hasattr(point, 'working_memory_load') and hasattr(prev_point, 'working_memory_load'):
                load_increase = point.working_memory_load - prev_point.working_memory_load
                fluency_decrease = prev_point.cognitive_fluency - point.cognitive_fluency
                
                if load_increase > 0.2 and fluency_decrease > 0.1:
                    constraint_moments.append({
                        'timestamp': point.timestamp,
                        'type': 'COGNITIVE_OVERLOAD',
                        'severity': min(load_increase + fluency_decrease, 1.0),
                        'description': 'Cognitive overload affecting performance'
                    })
        
        return constraint_moments
    
    def analyze_constraint_adaptation(self, constraint_moments: List[Dict]) -> Dict:
        """Analyze how well they adapt to constraints"""
        
        if not constraint_moments:
            return {
                'constraint_experience': 'NO_CONSTRAINTS_DETECTED',
                'adaptation_ability': 'UNKNOWN'
            }
        
        adaptation_analysis = {}
        
        # Analyze recovery from constraints
        recovery_scores = []
        
        for constraint in constraint_moments:
            severity = constraint['severity']
            constraint_type = constraint['type']
            
            # Simulate recovery analysis (in real implementation, would track subsequent performance)
            if severity < 0.4:
                recovery_score = 0.8  # Good recovery from mild constraints
            elif severity < 0.7:
                recovery_score = 0.6  # Moderate recovery
            else:
                recovery_score = 0.3  # Poor recovery from severe constraints
            
            recovery_scores.append(recovery_score)
        
        avg_recovery = sum(recovery_scores) / len(recovery_scores)
        
        # Adaptation assessment
        if avg_recovery > 0.7:
            adaptation_analysis['adaptation_ability'] = {
                'level': 'EXCELLENT',
                'description': 'Adapts well to constraints and pressure',
                'job_implication': 'Can handle tight deadlines and complex requirements'
            }
        elif avg_recovery > 0.5:
            adaptation_analysis['adaptation_ability'] = {
                'level': 'GOOD',
                'description': 'Handles constraints reasonably well',
                'job_implication': 'Suitable for standard pressure situations'
            }
        else:
            adaptation_analysis['adaptation_ability'] = {
                'level': 'POOR',
                'description': 'Struggles under constraints',
                'job_implication': 'May need support during high-pressure periods'
            }
        
        adaptation_analysis['constraint_count'] = len(constraint_moments)
        adaptation_analysis['average_recovery_score'] = avg_recovery
        
        return adaptation_analysis
    
    def identify_major_approach_changes(self, code_timeline: List) -> List[Dict]:
        """Identify major changes in problem-solving approach"""
        
        approach_changes = []
        
        if len(code_timeline) < 2:
            return approach_changes
        
        # Track primary approaches over time
        primary_approaches = []
        
        for timestamp, code in code_timeline:
            pattern_scores = self.detector.get_all_pattern_scores(code)
            if pattern_scores:
                primary_pattern = max(pattern_scores, key=pattern_scores.get)
                primary_score = pattern_scores[primary_pattern]
                primary_approaches.append((timestamp, primary_pattern, primary_score))
        
        # Detect significant approach changes
        for i in range(1, len(primary_approaches)):
            prev_timestamp, prev_pattern, prev_score = primary_approaches[i-1]
            curr_timestamp, curr_pattern, curr_score = primary_approaches[i]
            
            # Major change if pattern changes and new pattern has strong signal
            if (prev_pattern != curr_pattern and 
                curr_score > 0.4 and 
                abs(curr_score - prev_score) > 0.3):
                
                approach_changes.append({
                    'change_id': f'change_{i}',
                    'timestamp': curr_timestamp,
                    'from_approach': prev_pattern,
                    'to_approach': curr_pattern,
                    'code_before': code_timeline[i-1][1] if i-1 < len(code_timeline) else '',
                    'code_after': code_timeline[i][1] if i < len(code_timeline) else '',
                    'time_to_pivot': curr_timestamp - prev_timestamp,
                    'new_approach_strength': curr_score,
                    'change_magnitude': abs(curr_score - prev_score)
                })
        
        return approach_changes
    
    def calculate_quality_retention(self, pre_change_quality: Dict, post_change_quality: Dict) -> float:
        """Calculate how much quality was retained through a change"""
        
        if not pre_change_quality or not post_change_quality:
            return 0.5  # Neutral if can't assess
        
        # Compare quality metrics
        quality_comparisons = []
        
        common_metrics = set(pre_change_quality.keys()) & set(post_change_quality.keys())
        
        for metric in common_metrics:
            pre_value = pre_change_quality[metric]
            post_value = post_change_quality[metric]
            
            if pre_value > 0:
                retention_ratio = post_value / pre_value
                quality_comparisons.append(min(retention_ratio, 1.0))
            else:
                # If pre-value was 0, any post-value is improvement
                quality_comparisons.append(1.0)
        
        if not quality_comparisons:
            return 0.5
        
        avg_retention = sum(quality_comparisons) / len(quality_comparisons)
        return avg_retention
    
    def extract_numeric_score(self, analysis_result: Dict) -> float:
        """Extract a numeric score from analysis results"""
        
        if isinstance(analysis_result, (int, float)):
            return float(analysis_result)
        
        if not isinstance(analysis_result, dict):
            return 0.5  # Default neutral score
        
        # Look for common score indicators
        score_keys = [
            'overall_score', 'total_score', 'score', 'rating', 
            'level', 'quality', 'performance', 'effectiveness'
        ]
        
        for key in score_keys:
            if key in analysis_result:
                value = analysis_result[key]
                if isinstance(value, (int, float)):
                    return min(max(float(value), 0.0), 1.0)  # Clamp to [0,1]
        
        # Try to extract from tier classifications
        tier_mappings = {
            'EXCEPTIONAL': 0.95,
            'EXCELLENT': 0.9,
            'HIGH': 0.8,
            'STRONG': 0.75,
            'GOOD': 0.7,
            'SOLID': 0.65,
            'AVERAGE': 0.5,
            'MODERATE': 0.5,
            'DEVELOPING': 0.4,
            'WEAK': 0.3,
            'LOW': 0.2,
            'POOR': 0.15,
            'CONCERNING': 0.1
        }
        
        # Check all values for tier keywords
        for value in analysis_result.values():
            if isinstance(value, str):
                value_upper = value.upper()
                for tier, score in tier_mappings.items():
                    if tier in value_upper:
                        return score
        
        # If no clear score found, try to average numeric values
        numeric_values = []
        for value in analysis_result.values():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                nested_score = self.extract_numeric_score(value)
                numeric_values.append(nested_score)
        
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        
        return 0.5  # Default neutral score
    
    def convert_to_letter_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'A-'
        elif score >= 0.75:
            return 'B+'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.65:
            return 'B-'
        elif score >= 0.6:
            return 'C+'
        elif score >= 0.55:
            return 'C'
        elif score >= 0.5:
            return 'C-'
        elif score >= 0.45:
            return 'D+'
        elif score >= 0.4:
            return 'D'
        else:
            return 'F'
    
    def classify_thinking_tier(self, score: float) -> Dict:
        """Classify thinking ability into tiers"""
        
        if score >= 0.85:
            return {
                'tier': 'EXCEPTIONAL',
                'percentile': '95th+',
                'description': 'Top-tier thinking abilities',
                'job_level': 'Senior/Staff Engineer',
                'team_impact': 'Can mentor others and drive technical decisions'
            }
        elif score >= 0.7:
            return {
                'tier': 'STRONG',
                'percentile': '75-95th',
                'description': 'Above-average thinking abilities',
                'job_level': 'Mid to Senior Engineer',
                'team_impact': 'Solid contributor with growth potential'
            }
        elif score >= 0.5:
            return {
                'tier': 'AVERAGE',
                'percentile': '25-75th',
                'description': 'Typical thinking abilities',
                'job_level': 'Junior to Mid Engineer',
                'team_impact': 'Can contribute with guidance'
            }
        elif score >= 0.3:
            return {
                'tier': 'DEVELOPING',
                'percentile': '10-25th',
                'description': 'Below-average thinking abilities',
                'job_level': 'Junior Engineer with mentoring',
                'team_impact': 'Needs significant support'
            }
        else:
            return {
                'tier': 'CONCERNING',
                'percentile': 'Bottom 10th',
                'description': 'Significant thinking challenges',
                'job_level': 'Not recommended for current role',
                'team_impact': 'Would require extensive training'
            }

    def calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth in AST"""
        
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    def calculate_function_complexity_load(self, tree: ast.AST) -> float:
        """Calculate cognitive load from function complexity"""
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 0.2  # Low load for no functions
        
        total_complexity = 0.0
        
        for func in functions:
            func_complexity = 0.0
            
            # Parameter complexity
            param_count = len(func.args.args)
            func_complexity += min(param_count / 5.0, 0.3)
            
            # Body complexity
            body_size = len(list(ast.walk(func)))
            func_complexity += min(body_size / 30.0, 0.4)
            
            # Control flow complexity
            control_nodes = len([node for node in ast.walk(func) 
                               if isinstance(node, (ast.If, ast.For, ast.While))])
            func_complexity += min(control_nodes / 5.0, 0.3)
            
            total_complexity += func_complexity
        
        return total_complexity / len(functions)
    
    def analyze_scope_complexity(self, tree: ast.AST) -> float:
        """Analyze scope complexity"""
        
        scope_complexity = 0.0
        
        # Function scopes
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        scope_complexity += min(functions / 5.0, 0.4)
        
        # Class scopes
        classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        scope_complexity += min(classes / 2.0, 0.3)
        
        # Nested scopes
        nested_depth = self.calculate_max_nesting_depth(tree)
        scope_complexity += min(nested_depth / 4.0, 0.3)
        
        return scope_complexity
    
    def detect_abstraction_level_changes(self, tree: ast.AST) -> float:
        """Detect abstraction level changes"""
        
        abstraction_changes = 0.0
        
        # Low-level constructs
        low_level = len([node for node in ast.walk(tree) 
                        if isinstance(node, (ast.For, ast.While, ast.Index))])
        
        # High-level constructs
        high_level = len([node for node in ast.walk(tree) 
                         if isinstance(node, (ast.ListComp, ast.DictComp, ast.GeneratorExp))])
        
        # Mixed levels indicate abstraction jumps
        if low_level > 0 and high_level > 0:
            abstraction_changes = min((low_level + high_level) / 10.0, 0.5)
        
        return abstraction_changes
    
    def estimate_variable_type_complexity(self, var_name: str, tree: ast.AST) -> float:
        """Estimate variable type complexity"""
        
        # Simple heuristic based on variable name and usage
        complexity = 0.1  # Base complexity
        
        # Check variable usage patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == var_name:
                parent = getattr(node, 'parent', None)
                
                # Subscript usage (arrays, dicts)
                if isinstance(parent, ast.Subscript):
                    complexity += 0.2
                
                # Attribute access (objects)
                if isinstance(parent, ast.Attribute):
                    complexity += 0.3
                
                # Function calls
                if isinstance(parent, ast.Call):
                    complexity += 0.1
        
        return min(complexity, 1.0)
    
    def assess_code_structure_quality(self, code: str) -> float:
        """Assess overall code structure quality"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        quality_score = 0.0
        
        # Function organization
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            avg_func_size = sum(len(list(ast.walk(func))) for func in functions) / len(functions)
            
            if 5 <= avg_func_size <= 25:
                quality_score += 0.3
            elif avg_func_size <= 40:
                quality_score += 0.2
            else:
                quality_score += 0.1
        
        # Variable naming quality
        variables = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        
        meaningful_names = sum(1 for var in variables if len(var) >= 3 and var.islower())
        if variables:
            naming_quality = meaningful_names / len(variables)
            quality_score += naming_quality * 0.3
        
        # Complexity management
        max_nesting = self.calculate_max_nesting_depth(tree)
        if max_nesting <= 3:
            quality_score += 0.2
        elif max_nesting <= 5:
            quality_score += 0.1
        
        # Comments presence
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        code_lines = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            if 0.1 <= comment_ratio <= 0.3:
                quality_score += 0.2
            elif comment_ratio > 0:
                quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def measure_code_coherence(self, code: str) -> float:
        """Measure code coherence"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        coherence_score = 0.0
        
        # Consistent naming patterns
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.append(node.id)
        
        if variables:
            # Check naming consistency (snake_case, length, etc.)
            snake_case_vars = sum(1 for var in variables if '_' in var or var.islower())
            consistency = snake_case_vars / len(variables)
            coherence_score += consistency * 0.3
        
        # Logical organization
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if len(functions) > 1:
            # Functions suggest good organization
            coherence_score += 0.3
        
        # Import organization
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            coherence_score += 0.2
        
        # Error handling
        try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        if try_blocks:
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def extract_pattern_from_hint(self, hint: str) -> str:
        """Extract target pattern from hint text"""
        
        hint_lower = hint.lower()
        
        # Map hint keywords to patterns
        hint_pattern_map = {
            'hash': 'hash_map',
            'dictionary': 'hash_map',
            'map': 'hash_map',
            'efficient': 'optimization',
            'faster': 'time_optimization',
            'optimize': 'optimization',
            'two pointer': 'two_pointers',
            'sliding window': 'sliding_window',
            'binary search': 'binary_search',
            'dynamic programming': 'dynamic_programming',
            'recursion': 'recursion',
            'memoization': 'memoization',
            'complement': 'hash_map',
            'seen': 'hash_map'
        }
        
        for keyword, pattern in hint_pattern_map.items():
            if keyword in hint_lower:
                return pattern
        
        return 'general_improvement'
    
    def classify_integration_type(self, direct_improvement: float, ripple_effects: List) -> Dict:
        """Classify how they integrate hints"""
        
        if direct_improvement > 0.6 and len(ripple_effects) >= 2:
            return {
                'type': 'SYNTHESIZER',
                'description': 'Integrates hint and makes broader improvements',
                'employability': 'EXCEPTIONAL - Will maximize value from mentoring',
                'team_impact': 'Multiplies the value of senior developer guidance',
                'promotion_velocity': 'Fast track - learns exponentially from feedback'
            }
        elif direct_improvement > 0.6 and len(ripple_effects) < 1:
            return {
                'type': 'IMPLEMENTER',
                'description': 'Follows guidance accurately but narrowly',
                'employability': 'GOOD - Reliable execution of directions',
                'team_impact': 'Solid contributor, won\'t cause problems',
                'promotion_velocity': 'Steady - needs explicit guidance for growth'
            }
        elif direct_improvement < 0.3:
            return {
                'type': 'RESISTANT',
                'description': 'Doesn\'t fully integrate guidance',
                'employability': 'CONCERNING - May be difficult to mentor',
                'team_impact': 'Could clash with senior developers',
                'development_needs': 'Work on receptiveness to feedback'
            }
        else:
            return {
                'type': 'GRADUAL_LEARNER',
                'description': 'Slowly but steadily integrates guidance',
                'employability': 'AVERAGE - Standard learning curve',
                'team_impact': 'Will improve with patient mentoring'
            }
    
    def measure_hint_complexity(self, hint: str) -> float:
        """Measure complexity of hint given"""
        
        complexity_score = 0.1  # Base complexity
        
        # Length-based complexity
        word_count = len(hint.split())
        complexity_score += min(word_count / 20.0, 0.3)
        
        # Technical terms
        technical_terms = ['algorithm', 'complexity', 'optimization', 'data structure', 'efficient']
        tech_term_count = sum(1 for term in technical_terms if term.lower() in hint.lower())
        complexity_score += tech_term_count * 0.2
        
        # Question complexity
        question_marks = hint.count('?')
        complexity_score += min(question_marks * 0.1, 0.2)
        
        # Specificity (more specific = more complex)
        specific_terms = ['hash map', 'binary search', 'two pointers', 'dynamic programming']
        specific_count = sum(1 for term in specific_terms if term.lower() in hint.lower())
        complexity_score += specific_count * 0.3
        
        return min(complexity_score, 1.0)
    
    def measure_total_code_improvement(self, code_before: str, code_after: str) -> float:
        """Measure total improvement in code quality"""
        
        # Get pattern scores for both versions
        scores_before = self.detector.get_all_pattern_scores(code_before)
        scores_after = self.detector.get_all_pattern_scores(code_after)
        
        # Calculate overall improvement
        total_before = sum(scores_before.values()) if scores_before else 0
        total_after = sum(scores_after.values()) if scores_after else 0
        
        improvement = total_after - total_before
        
        # Additional quality metrics
        structural_improvement = 0.0
        
        try:
            tree_before = ast.parse(code_before)
            tree_after = ast.parse(code_after)
            
            # Compare complexity
            complexity_before = self.calculate_max_nesting_depth(tree_before)
            complexity_after = self.calculate_max_nesting_depth(tree_after)
            
            if complexity_after < complexity_before:
                structural_improvement += 0.2
            
            # Compare function organization
            funcs_before = len([n for n in ast.walk(tree_before) if isinstance(n, ast.FunctionDef)])
            funcs_after = len([n for n in ast.walk(tree_after) if isinstance(n, ast.FunctionDef)])
            
            if funcs_after > funcs_before:
                structural_improvement += 0.1
        
        except:
            pass
        
        total_improvement = improvement + structural_improvement
        return max(0, total_improvement)
    
    def integrate_hint_cognitive_impact(self, cognitive_timeline: List[CognitiveState], 
                                      hint_interactions: List[Dict]) -> List[CognitiveState]:
        """Integrate hint interactions into cognitive timeline"""
        
        # Create a copy of the timeline to modify
        enhanced_timeline = cognitive_timeline.copy()
        
        for hint in hint_interactions:
            hint_timestamp = hint['timestamp']
            
            # Find the cognitive state closest to this hint
            closest_state_idx = None
            min_time_diff = float('inf')
            
            for i, state in enumerate(enhanced_timeline):
                time_diff = abs(state.timestamp - hint_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_state_idx = i
            
            if closest_state_idx is not None:
                # Modify cognitive state to reflect hint impact
                state = enhanced_timeline[closest_state_idx]
                
                # Hints typically increase cognitive load temporarily
                hint_load_impact = min(hint['effectiveness_score'] * 0.2, 0.3)
                state.working_memory_load = min(state.working_memory_load + hint_load_impact, 1.0)
                
                # But can improve confidence if well integrated
                if hint['effectiveness_score'] > 0.7:
                    state.processing_confidence = min(state.processing_confidence + 0.1, 1.0)
        
        return enhanced_timeline
    
    def analyze_stress_resilience(self, cognitive_timeline: List[CognitiveState]) -> Dict:
        """Analyze stress resilience patterns"""
        
        if len(cognitive_timeline) < 3:
            return {'insufficient_data': True}
        
        stress_analysis = {}
        
        # Extract stress levels over time
        stress_levels = []
        for state in cognitive_timeline:
            avg_stress = sum(state.stress_indicators.values()) / len(state.stress_indicators) if state.stress_indicators else 0
            stress_levels.append(avg_stress)
        
        # Stress trajectory
        if len(stress_levels) >= 3:
            stress_trend = np.polyfit(range(len(stress_levels)), stress_levels, 1)[0]
            
            if stress_trend < -0.05:
                stress_analysis['stress_trajectory'] = {
                    'type': 'IMPROVING',
                    'description': 'Stress decreasing over time',
                    'resilience': 'HIGH - Adapts well to pressure',
                    'job_performance': 'Will handle increasing responsibility well'
                }
            elif stress_trend > 0.05:
                stress_analysis['stress_trajectory'] = {
                    'type': 'ACCUMULATING',
                    'description': 'Stress increasing over time',
                    'resilience': 'CONCERNING - May struggle under sustained pressure',
                    'job_performance': 'May need support during crunch periods'
                }
            else:
                stress_analysis['stress_trajectory'] = {
                    'type': 'STABLE',
                    'description': 'Consistent stress management',
                    'resilience': 'GOOD - Steady under pressure',
                    'job_performance': 'Reliable under normal conditions'
                }
        
        # Peak stress handling
        max_stress = max(stress_levels) if stress_levels else 0
        if max_stress > 0.7:
            # Find recovery pattern
            max_stress_idx = stress_levels.index(max_stress)
            if max_stress_idx < len(stress_levels) - 1:
                recovery_trend = stress_levels[max_stress_idx:]
                if len(recovery_trend) >= 2 and recovery_trend[-1] < recovery_trend[0]:
                    stress_analysis['peak_stress_recovery'] = 'GOOD - Recovers from high stress'
                else:
                    stress_analysis['peak_stress_recovery'] = 'POOR - Struggles to recover'
        
        # Overall resilience assessment
        avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 0
        stress_variance = np.var(stress_levels) if len(stress_levels) > 1 else 0
        
        if avg_stress < 0.3 and stress_variance < 0.1:
            stress_analysis['overall_resilience'] = 'EXCELLENT - Low stress, high stability'
        elif avg_stress < 0.5 and stress_variance < 0.2:
            stress_analysis['overall_resilience'] = 'GOOD - Moderate stress, stable'
        else:
            stress_analysis['overall_resilience'] = 'NEEDS_SUPPORT - High stress or instability'
        
        return stress_analysis
    
    def analyze_learning_progression(self, cognitive_timeline: List[CognitiveState]) -> Dict:
        """Analyze learning progression patterns"""
        
        if len(cognitive_timeline) < 3:
            return {'insufficient_data': True}
        
        learning_analysis = {}
        
        # Track cognitive fluency progression
        fluency_progression = [state.cognitive_fluency for state in cognitive_timeline]
        
        if len(fluency_progression) >= 3:
            fluency_trend = np.polyfit(range(len(fluency_progression)), fluency_progression, 1)[0]
            
            if fluency_trend > 0.05:
                learning_analysis['fluency_development'] = {
                    'type': 'IMPROVING',
                    'rate': fluency_trend,
                    'learning_velocity': 'HIGH - Getting more fluent over time'
                }
            elif fluency_trend < -0.05:
                learning_analysis['fluency_development'] = {
                    'type': 'DECLINING',
                    'rate': fluency_trend,
                    'concern': 'Fluency decreasing - possible fatigue or confusion'
                }
            else:
                learning_analysis['fluency_development'] = {
                    'type': 'STABLE',
                    'rate': fluency_trend,
                    'assessment': 'Consistent cognitive fluency'
                }
        
        # Mental model clarity progression
        clarity_progression = [state.mental_model_clarity for state in cognitive_timeline]
        
        if len(clarity_progression) >= 3:
            clarity_trend = np.polyfit(range(len(clarity_progression)), clarity_progression, 1)[0]
            learning_analysis['mental_model_development'] = {
                'trend': clarity_trend,
                'final_clarity': clarity_progression[-1],
                'assessment': 'GOOD' if clarity_trend > 0 else 'NEEDS_IMPROVEMENT'
            }
        
        # Overall learning assessment
        final_fluency = fluency_progression[-1] if fluency_progression else 0
        fluency_improvement = fluency_progression[-1] - fluency_progression[0] if len(fluency_progression) >= 2 else 0
        
        if final_fluency > 0.7 and fluency_improvement > 0.2:
            learning_analysis['overall_learning'] = 'EXCELLENT - High final fluency with strong improvement'
        elif final_fluency > 0.5 and fluency_improvement > 0:
            learning_analysis['overall_learning'] = 'GOOD - Solid fluency with positive improvement'
        elif fluency_improvement > 0:
            learning_analysis['overall_learning'] = 'DEVELOPING - Showing improvement'
        else:
            learning_analysis['overall_learning'] = 'CONCERNING - Limited learning progression'
        
        return learning_analysis
    
    def analyze_cognitive_efficiency(self, cognitive_timeline: List[CognitiveState]) -> Dict:
        """Analyze cognitive efficiency patterns"""
        
        if len(cognitive_timeline) < 2:
            return {'insufficient_data': True}
        
        efficiency_analysis = {}
        
        # Calculate efficiency metrics
        working_memory_loads = [state.working_memory_load for state in cognitive_timeline]
        processing_confidence = [state.processing_confidence for state in cognitive_timeline]
        cognitive_fluency = [state.cognitive_fluency for state in cognitive_timeline]
        
        # Efficiency = high output (fluency, confidence) with low input (memory load)
        efficiency_scores = []
        for i in range(len(cognitive_timeline)):
            if working_memory_loads[i] > 0:
                efficiency = (processing_confidence[i] + cognitive_fluency[i]) / (2 * working_memory_loads[i])
            else:
                efficiency = processing_confidence[i] + cognitive_fluency[i]
            efficiency_scores.append(efficiency)
        
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        # Efficiency trajectory
        if len(efficiency_scores) >= 3:
            efficiency_trend = np.polyfit(range(len(efficiency_scores)), efficiency_scores, 1)[0]
            
            if efficiency_trend > 0.05:
                efficiency_analysis['efficiency_trajectory'] = 'IMPROVING - Getting more efficient over time'
            elif efficiency_trend < -0.05:
                efficiency_analysis['efficiency_trajectory'] = 'DECLINING - Efficiency decreasing'
            else:
                efficiency_analysis['efficiency_trajectory'] = 'STABLE - Consistent efficiency'
        
        # Overall efficiency classification
        if avg_efficiency > 1.5:
            efficiency_analysis['efficiency_level'] = 'HIGH - Excellent cognitive efficiency'
        elif avg_efficiency > 1.0:
            efficiency_analysis['efficiency_level'] = 'GOOD - Above average efficiency'
        elif avg_efficiency > 0.7:
            efficiency_analysis['efficiency_level'] = 'AVERAGE - Typical efficiency'
        else:
            efficiency_analysis['efficiency_level'] = 'LOW - Below average efficiency'
        
        efficiency_analysis['average_efficiency_score'] = avg_efficiency
        
        return efficiency_analysis
    
    def analyze_mental_model_development(self, cognitive_timeline: List[CognitiveState]) -> Dict:
        """Analyze mental model development patterns"""
        
        if len(cognitive_timeline) < 2:
            return {'insufficient_data': True}
        
        model_analysis = {}
        
        # Track mental model clarity over time
        clarity_progression = [state.mental_model_clarity for state in cognitive_timeline]
        
        # Development phases
        if len(clarity_progression) >= 4:
            early_phase = clarity_progression[:len(clarity_progression)//3]
            middle_phase = clarity_progression[len(clarity_progression)//3:2*len(clarity_progression)//3]
            late_phase = clarity_progression[2*len(clarity_progression)//3:]
            
            early_avg = sum(early_phase) / len(early_phase)
            middle_avg = sum(middle_phase) / len(middle_phase)
            late_avg = sum(late_phase) / len(late_phase)
            
            model_analysis['development_phases'] = {
                'early_clarity': early_avg,
                'middle_clarity': middle_avg,
                'late_clarity': late_avg,
                'total_improvement': late_avg - early_avg
            }
            
            # Development pattern
            if late_avg > middle_avg > early_avg:
                model_analysis['development_pattern'] = 'PROGRESSIVE - Steady mental model improvement'
            elif late_avg > early_avg:
                model_analysis['development_pattern'] = 'OVERALL_POSITIVE - Net improvement with fluctuations'
            else:
                model_analysis['development_pattern'] = 'STAGNANT - Limited mental model development'
        
        # Final model quality
        final_clarity = clarity_progression[-1]
        if final_clarity > 0.8:
            model_analysis['final_model_quality'] = 'EXCELLENT - Clear, well-developed mental model'
        elif final_clarity > 0.6:
            model_analysis['final_model_quality'] = 'GOOD - Solid mental model'
        elif final_clarity > 0.4:
            model_analysis['final_model_quality'] = 'DEVELOPING - Basic mental model'
        else:
            model_analysis['final_model_quality'] = 'POOR - Unclear mental model'
        
        return model_analysis
    
    def generate_overall_cognitive_profile(self, analysis_results: Dict) -> Dict:
        """Generate overall cognitive profile"""
        
        profile = {}
        
        # Extract key metrics
        cognitive_load = analysis_results.get('cognitive_load', {})
        stress_resilience = analysis_results.get('stress_resilience', {})
        learning_progression = analysis_results.get('learning_progression', {})
        cognitive_efficiency = analysis_results.get('cognitive_efficiency', {})
        
        # Overall cognitive strength
        strengths = []
        concerns = []
        
        # Analyze cognitive load management
        load_trajectory = cognitive_load.get('load_trajectory', {})
        if load_trajectory.get('type') == 'IMPROVING':
            strengths.append('Excellent cognitive load management - improves under pressure')
        elif load_trajectory.get('type') == 'ACCUMULATING':
            concerns.append('Cognitive load accumulation - may struggle with sustained complexity')
        
        # Analyze stress resilience
        resilience_level = stress_resilience.get('overall_resilience', '')
        if 'EXCELLENT' in resilience_level:
            strengths.append('Exceptional stress resilience')
        elif 'NEEDS_SUPPORT' in resilience_level:
            concerns.append('Stress management needs improvement')
        
        # Analyze learning
        learning_assessment = learning_progression.get('overall_learning', '')
        if 'EXCELLENT' in learning_assessment:
            strengths.append('Rapid learning and adaptation capability')
        elif 'CONCERNING' in learning_assessment:
            concerns.append('Limited learning progression during interview')
        
        # Analyze efficiency
        efficiency_level = cognitive_efficiency.get('efficiency_level', '')
        if 'HIGH' in efficiency_level:
            strengths.append('High cognitive efficiency')
        elif 'LOW' in efficiency_level:
            concerns.append('Low cognitive efficiency')
        
        profile['cognitive_strengths'] = strengths
        profile['cognitive_concerns'] = concerns
        
        # Overall assessment
        if len(strengths) >= 3 and len(concerns) <= 1:
            profile['overall_cognitive_assessment'] = 'EXCEPTIONAL - Strong across all cognitive dimensions'
            profile['hiring_recommendation'] = 'STRONG_HIRE - Excellent cognitive capabilities'
        elif len(strengths) >= 2 and len(concerns) <= 2:
            profile['overall_cognitive_assessment'] = 'STRONG - Good cognitive capabilities with minor areas for improvement'
            profile['hiring_recommendation'] = 'HIRE - Solid cognitive foundation'
        elif len(strengths) >= 1 and len(concerns) <= 3:
            profile['overall_cognitive_assessment'] = 'DEVELOPING - Some cognitive strengths but needs development'
            profile['hiring_recommendation'] = 'WEAK_HIRE - Potential but needs support'
        else:
            profile['overall_cognitive_assessment'] = 'CONCERNING - Multiple cognitive challenges identified'
            profile['hiring_recommendation'] = 'NO_HIRE - Cognitive capabilities below threshold'
        
        return profile

    def track_cognitive_evolution(self, code_snapshots_over_time, hint_interactions=None):
        """Comprehensive cognitive load analysis across time"""
        
        cognitive_timeline = []
        
        for timestamp, code in code_snapshots_over_time:
            cognitive_state = self.analyze_cognitive_state_at_moment(timestamp, code, cognitive_timeline)
            cognitive_timeline.append(cognitive_state)
        
        # Integrate hint interactions if available
        if hint_interactions:
            cognitive_timeline = self.integrate_hint_cognitive_impact(cognitive_timeline, hint_interactions)
        
        return self.generate_comprehensive_cognitive_analysis(cognitive_timeline)
    
    def analyze_cognitive_state_at_moment(self, timestamp: int, code: str, history: List[CognitiveState]) -> CognitiveState:
        """Analyze cognitive state at a specific moment"""
        
        # Get pattern scores for this moment
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # Calculate cognitive dimensions
        working_memory_load = self.calculate_working_memory_load(code, pattern_scores)
        attention_distribution = self.analyze_attention_distribution(pattern_scores, history)
        processing_confidence = self.measure_processing_confidence(pattern_scores, history,code)
        cognitive_fluency = self.assess_cognitive_fluency(code, history)
        mental_model_clarity = self.evaluate_mental_model_clarity(pattern_scores, history,code)
        stress_indicators = self.detect_stress_indicators(code, pattern_scores, history)
        
        return CognitiveState(
            timestamp=timestamp,
            working_memory_load=working_memory_load,
            attention_distribution=attention_distribution,
            processing_confidence=processing_confidence,
            cognitive_fluency=cognitive_fluency,
            mental_model_clarity=mental_model_clarity,
            stress_indicators=stress_indicators
        )
    
    def calculate_working_memory_load(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Calculate cognitive working memory load"""
        
        load_factors = {}
        
        # 1. SIMULTANEOUS PATTERN COMPLEXITY
        active_patterns = [k for k, v in pattern_scores.items() if v > 0.3]
        pattern_interaction_complexity = self.calculate_pattern_interaction_load(active_patterns)
        load_factors['pattern_complexity'] = pattern_interaction_complexity
        
        # 2. CODE STRUCTURE COMPLEXITY
        structure_complexity = self.analyze_structural_cognitive_load(code)
        load_factors['structural_load'] = structure_complexity
        
        # 3. VARIABLE TRACKING LOAD
        variable_tracking_load = self.calculate_variable_tracking_complexity(code)
        load_factors['variable_tracking'] = variable_tracking_load
        
        # 4. CONTROL FLOW COMPLEXITY
        control_flow_load = self.assess_control_flow_cognitive_load(code)
        load_factors['control_flow'] = control_flow_load
        
        # Weighted combination
        total_load = (
            load_factors['pattern_complexity'] * 0.3 +
            load_factors['structural_load'] * 0.25 +
            load_factors['variable_tracking'] * 0.25 +
            load_factors['control_flow'] * 0.2
        )
        
        return min(total_load, 1.0)
    
    def calculate_pattern_interaction_load(self, active_patterns: List[str]) -> float:
        """Calculate cognitive load from pattern interactions"""
        
        if len(active_patterns) <= 1:
            return 0.2  # Low load for single pattern
        
        # Pattern compatibility matrix - some patterns work well together, others don't
        pattern_synergies = {
            ('two_pointers', 'sliding_window'): 0.8,  # Synergistic
            ('hash_map', 'array_manipulation'): 0.9,  # Complementary
            ('recursion', 'memoization'): 0.9,  # Natural combination
            ('union_find', 'graph_dfs'): 0.7,  # Somewhat compatible
            ('binary_search', 'dynamic_programming'): 0.4,  # Competing approaches
            ('backtracking', 'greedy_choice'): 0.3,  # Conflicting strategies
        }
        
        total_interaction_load = 0.0
        interaction_count = 0
        
        for i, pattern1 in enumerate(active_patterns):
            for pattern2 in active_patterns[i+1:]:
                # Check both orderings
                synergy = pattern_synergies.get((pattern1, pattern2), 
                         pattern_synergies.get((pattern2, pattern1), 0.5))  # Default neutral
                
                # Higher synergy = lower cognitive load
                interaction_load = 1.0 - synergy
                total_interaction_load += interaction_load
                interaction_count += 1
        
        if interaction_count == 0:
            return 0.2
        
        avg_interaction_load = total_interaction_load / interaction_count
        
        # Scale by number of patterns (more patterns = exponentially harder)
        pattern_count_multiplier = min(len(active_patterns) ** 1.5 / 10, 1.0)
        
        return avg_interaction_load * pattern_count_multiplier
    
    def analyze_structural_cognitive_load(self, code: str) -> float:
        """Analyze cognitive load from code structure"""
        
        try:
            tree = ast.parse(code)
        except:
            return 1.0  # Maximum load for unparseable code
        
        load_factors = {}
        
        # 1. NESTING DEPTH LOAD
        max_nesting = self.calculate_max_nesting_depth(tree)
        nesting_load = min(max_nesting / 5.0, 1.0)  # Scale: 5+ levels = max load
        load_factors['nesting'] = nesting_load
        
        # 2. FUNCTION COMPLEXITY LOAD
        function_complexity = self.calculate_function_complexity_load(tree)
        load_factors['function_complexity'] = function_complexity
        
        # 3. VARIABLE SCOPE COMPLEXITY
        scope_complexity = self.analyze_scope_complexity(tree)
        load_factors['scope_complexity'] = scope_complexity
        
        # 4. ABSTRACTION LEVEL JUMPS
        abstraction_jumps = self.detect_abstraction_level_changes(tree)
        load_factors['abstraction_jumps'] = abstraction_jumps
        
        return sum(load_factors.values()) / len(load_factors)
    
    def calculate_variable_tracking_complexity(self, code: str) -> float:
        """Calculate cognitive load from tracking variables"""
        
        try:
            tree = ast.parse(code)
        except:
            return 1.0
        
        # Track variable lifecycle complexity
        variable_states = {}
        load_accumulator = 0.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                
                if var_name not in variable_states:
                    variable_states[var_name] = {
                        'first_use': True,
                        'modification_count': 0,
                        'scope_changes': 0,
                        'type_complexity': self.estimate_variable_type_complexity(var_name, tree)
                    }
                else:
                    variable_states[var_name]['first_use'] = False
                
                if isinstance(node.ctx, ast.Store):
                    variable_states[var_name]['modification_count'] += 1
        
        # Calculate load based on variable complexity
        for var_name, state in variable_states.items():
            var_load = (
                state['modification_count'] * 0.1 +  # More modifications = harder to track
                state['type_complexity'] * 0.3 +    # Complex types = higher load
                (1.0 if state['modification_count'] > 5 else 0.0) * 0.5  # Penalty for excessive mutations
            )
            load_accumulator += var_load
        
        # Normalize by number of variables
        if len(variable_states) == 0:
            return 0.0
        
        avg_variable_load = load_accumulator / len(variable_states)
        
        # Scale by total number of variables (more variables = harder to track)
        variable_count_factor = min(len(variable_states) / 15.0, 1.0)  # 15+ vars = max load
        
        return avg_variable_load * variable_count_factor
    
    def analyze_attention_distribution(self, pattern_scores: Dict[str, float], history: List[CognitiveState]) -> Dict[str, float]:
        """Analyze how attention is distributed across different cognitive tasks"""
        
        # Categorize patterns by cognitive attention type
        attention_categories = {
            'algorithmic_thinking': ['recursion', 'dynamic_programming', 'divide_conquer', 'backtracking'],
            'data_structure_management': ['hash_map', 'heap_usage', 'stack_usage', 'queue_usage', 'trie_usage'],
            'optimization_focus': ['space_optimization', 'time_optimization', 'binary_search'],
            'pattern_recognition': ['two_pointers', 'sliding_window', 'monotonic_stack'],
            'system_design': ['graph_algorithms', 'tree_traversal', 'union_find'],
            'implementation_details': ['array_manipulation', 'string_operations', 'bit_manipulation']
        }
        
        attention_distribution = {}
        
        for category, patterns in attention_categories.items():
            category_attention = sum(pattern_scores.get(pattern, 0) for pattern in patterns)
            total_attention = sum(pattern_scores.values()) if pattern_scores else 1
            
            attention_distribution[category] = category_attention / total_attention if total_attention > 0 else 0
        
        # Analyze attention focus vs. dispersion
        attention_distribution['focus_intensity'] = max(attention_distribution.values()) if attention_distribution else 0
        attention_distribution['attention_dispersion'] = len([v for v in attention_distribution.values() if v > 0.1])
        
        return attention_distribution
    
    
    def assess_cognitive_fluency(self, code: str, history: List[CognitiveState]) -> float:
        """Assess how fluidly they're thinking (vs. struggling)"""
        
        fluency_indicators = {}
        
        # 1. CODE QUALITY FLUENCY
        try:
            tree = ast.parse(code)
            syntax_fluency = 1.0  # Valid syntax
        except:
            syntax_fluency = 0.0  # Syntax errors indicate cognitive struggle
        
        fluency_indicators['syntax'] = syntax_fluency
        
        # 2. CONCEPTUAL FLUENCY (clear mental models)
        pattern_scores = self.detector.get_all_pattern_scores(code)
        conceptual_clarity = max(pattern_scores.values()) if pattern_scores else 0
        fluency_indicators['conceptual'] = conceptual_clarity
        
        # 3. IMPLEMENTATION FLUENCY (code structure quality)
        if syntax_fluency > 0:
            structure_quality = self.assess_code_structure_quality(code)
            fluency_indicators['implementation'] = structure_quality
        else:
            fluency_indicators['implementation'] = 0.0
        
        # 4. PROGRESSION FLUENCY (smooth development vs. erratic)
        if len(history) >= 3:
            recent_loads = [state.working_memory_load for state in history[-3:]]
            load_variance = np.var(recent_loads)
            
            # Lower variance = smoother progression = higher fluency
            progression_fluency = max(0, 1.0 - load_variance * 2)
            fluency_indicators['progression'] = progression_fluency
        else:
            fluency_indicators['progression'] = 0.5
        
        return sum(fluency_indicators.values()) / len(fluency_indicators)
    
    def detect_stress_indicators(self, code: str, pattern_scores: Dict[str, float], history: List[CognitiveState]) -> Dict[str, float]:
        """Detect cognitive stress patterns"""
        
        stress_indicators = {}
        
        # 1. COGNITIVE OVERLOAD STRESS
        working_memory_load = self.calculate_working_memory_load(code, pattern_scores)
        if working_memory_load > 0.8:
            stress_indicators['cognitive_overload'] = working_memory_load
        else:
            stress_indicators['cognitive_overload'] = 0.0
        
        # 2. APPROACH INSTABILITY STRESS (too much switching)
        if len(history) >= 4:
            recent_primary_patterns = []
            for state in history[-4:]:
                if hasattr(state, 'attention_distribution'):
                    primary = max(state.attention_distribution, key=state.attention_distribution.get)
                    recent_primary_patterns.append(primary)
            
            unique_approaches = len(set(recent_primary_patterns))
            if unique_approaches > 2:  # Too much switching
                stress_indicators['approach_instability'] = min(unique_approaches / 4.0, 1.0)
            else:
                stress_indicators['approach_instability'] = 0.0
        else:
            stress_indicators['approach_instability'] = 0.0
        
        # 3. REGRESSION STRESS (going backwards in quality)
        if len(history) >= 2:
            current_fluency = self.assess_cognitive_fluency(code, history)
            prev_fluency = history[-1].cognitive_fluency
            
            if current_fluency < prev_fluency - 0.2:  # Significant regression
                stress_indicators['regression_stress'] = prev_fluency - current_fluency
            else:
                stress_indicators['regression_stress'] = 0.0
        else:
            stress_indicators['regression_stress'] = 0.0
        
        # 4. COMPLEXITY ACCUMULATION STRESS
        if len(history) >= 3:
            recent_complexity = [state.working_memory_load for state in history[-3:]]
            complexity_trend = np.polyfit(range(len(recent_complexity)), recent_complexity, 1)[0]
            
            if complexity_trend > 0.1:  # Increasing complexity
                stress_indicators['complexity_accumulation'] = min(complexity_trend * 5, 1.0)
            else:
                stress_indicators['complexity_accumulation'] = 0.0
        else:
            stress_indicators['complexity_accumulation'] = 0.0
        
        return stress_indicators
    
    def generate_comprehensive_cognitive_analysis(self, cognitive_timeline: List[CognitiveState]) -> Dict:
        """Generate comprehensive analysis of cognitive patterns"""
        
        analysis = {}
        
        # 1. COGNITIVE LOAD PATTERNS
        load_analysis = self.analyze_cognitive_load_patterns(cognitive_timeline)
        analysis['cognitive_load'] = load_analysis
        
        # 2. STRESS RESILIENCE PATTERNS
        stress_analysis = self.analyze_stress_resilience(cognitive_timeline)
        analysis['stress_resilience'] = stress_analysis
        
        # 3. LEARNING CURVE ANALYSIS
        learning_analysis = self.analyze_learning_progression(cognitive_timeline)
        analysis['learning_progression'] = learning_analysis
        
        # 4. COGNITIVE EFFICIENCY ANALYSIS
        efficiency_analysis = self.analyze_cognitive_efficiency(cognitive_timeline)
        analysis['cognitive_efficiency'] = efficiency_analysis
        
        # 5. MENTAL MODEL DEVELOPMENT
        mental_model_analysis = self.analyze_mental_model_development(cognitive_timeline)
        analysis['mental_model_development'] = mental_model_analysis
        
        # 6. OVERALL COGNITIVE PROFILE
        overall_profile = self.generate_overall_cognitive_profile(analysis)
        analysis['overall_profile'] = overall_profile
        
        return analysis
    
    def analyze_cognitive_load_patterns(self, timeline: List[CognitiveState]) -> Dict:
        """Analyze patterns in cognitive load over time"""
        
        loads = [state.working_memory_load for state in timeline]
        
        patterns = {}
        
        # Load trajectory
        if len(loads) >= 3:
            load_trend = np.polyfit(range(len(loads)), loads, 1)[0]
            
            if load_trend < -0.05:
                patterns['load_trajectory'] = {
                    'type': 'IMPROVING',
                    'description': 'Cognitive load decreasing over time',
                    'implication': 'Learning and organizing knowledge effectively',
                    'job_performance': 'Will handle increasing complexity well over time'
                }
            elif load_trend > 0.05:
                patterns['load_trajectory'] = {
                    'type': 'ACCUMULATING',
                    'description': 'Cognitive load increasing over time',
                    'implication': 'May be struggling with complexity management',
                    'job_performance': 'May need support with complex long-term projects'
                }
            else:
                patterns['load_trajectory'] = {
                    'type': 'STABLE',
                    'description': 'Consistent cognitive load management',
                    'implication': 'Steady, predictable cognitive processing',
                    'job_performance': 'Reliable performance under consistent conditions'
                }
        
        # Load variance (stability)
        load_variance = np.var(loads)
        if load_variance < 0.1:
            patterns['load_stability'] = 'HIGH - Very consistent cognitive processing'
        elif load_variance < 0.25:
            patterns['load_stability'] = 'MODERATE - Some variation in processing'
        else:
            patterns['load_stability'] = 'LOW - Erratic cognitive processing'
        
        # Peak load analysis
        max_load = max(loads)
        if max_load > 0.8:
            patterns['peak_load_handling'] = {
                'max_load': max_load,
                'assessment': 'Reached high cognitive load',
                'recovery': self.analyze_load_recovery(timeline, loads.index(max_load))
            }
        
        return patterns

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ast
import re

import ast
import re
from typing import Dict, List, Tuple, Set, Optional
import keyword

# =================== MISSING QUALITY ANALYZERS ===================

class StructuralSophisticationAnalyzer:
    """Analyzes sophistication of code structures and patterns"""
    
    def analyze_pattern_combination_sophistication(self, active_patterns: List[str]) -> float:
        """Analyze sophistication of pattern combinations"""
        
        if len(active_patterns) <= 1:
            return 0.2  # Low sophistication for single or no patterns
        
        # Define sophisticated pattern combinations
        sophisticated_combinations = {
            # Highly sophisticated combinations (score: 0.9-1.0)
            ('dynamic_programming', 'memoization'): 1.0,
            ('binary_search', 'divide_conquer'): 0.95,
            ('graph_dfs', 'union_find'): 0.9,
            ('segment_tree', 'lazy_propagation'): 0.9,
            
            # Advanced combinations (score: 0.7-0.8)
            ('two_pointers', 'sliding_window'): 0.8,
            ('hash_map', 'prefix_suffix'): 0.75,
            ('heap_usage', 'greedy_choice'): 0.75,
            ('backtracking', 'pruning'): 0.7,
            ('trie_usage', 'string_operations'): 0.7,
            
            # Good combinations (score: 0.5-0.6)
            ('hash_map', 'array_manipulation'): 0.6,
            ('stack_usage', 'monotonic_stack'): 0.6,
            ('recursion', 'divide_conquer'): 0.55,
            ('binary_search', 'array_manipulation'): 0.5,
            
            # Basic combinations (score: 0.3-0.4)
            ('iteration', 'conditional_logic'): 0.4,
            ('array_manipulation', 'conditional_logic'): 0.35,
            ('hash_map', 'iteration'): 0.3,
            
            # Conflicting combinations (score: 0.1-0.2)
            ('recursion', 'iteration'): 0.2,  # Mixing paradigms
            ('space_optimization', 'memoization'): 0.15,  # Contradictory goals
            ('time_optimization', 'space_optimization'): 0.1  # Trade-off conflict
        }
        
        combination_scores = []
        patterns_set = set(active_patterns)
        
        # Check all possible pattern pairs in the active patterns
        for i, pattern1 in enumerate(active_patterns):
            for pattern2 in active_patterns[i+1:]:
                # Check both orderings of the pattern pair
                combo_score = (
                    sophisticated_combinations.get((pattern1, pattern2)) or
                    sophisticated_combinations.get((pattern2, pattern1))
                )
                
                if combo_score is not None:
                    combination_scores.append(combo_score)
        
        # Check for multi-pattern sophisticated combinations
        multi_pattern_combinations = {
            # Triadic combinations
            frozenset(['dynamic_programming', 'memoization', 'recursion']): 1.0,
            frozenset(['graph_dfs', 'graph_bfs', 'union_find']): 0.95,
            frozenset(['binary_search', 'two_pointers', 'array_manipulation']): 0.8,
            frozenset(['hash_map', 'sliding_window', 'two_pointers']): 0.75,
            frozenset(['backtracking', 'recursion', 'pruning']): 0.7,
            
            # Advanced algorithmic patterns
            frozenset(['divide_conquer', 'recursion', 'memoization']): 0.85,
            frozenset(['greedy_choice', 'heap_usage', 'optimization']): 0.8,
            frozenset(['trie_usage', 'string_operations', 'recursion']): 0.75,
        }
        
        # Check for multi-pattern combinations
        for combo_set, score in multi_pattern_combinations.items():
            if combo_set.issubset(patterns_set):
                combination_scores.append(score)
        
        # If no specific combinations found, use heuristic scoring
        if not combination_scores:
            # Base score on number of patterns and their general sophistication
            pattern_sophistication_scores = {
                # Expert level patterns
                'segment_tree': 1.0,
                'fenwick_tree': 0.95,
                'union_find': 0.9,
                'dynamic_programming': 0.85,
                
                # Advanced patterns
                'binary_search': 0.8,
                'heap_usage': 0.75,
                'trie_usage': 0.7,
                'graph_dfs': 0.7,
                'backtracking': 0.65,
                
                # Intermediate patterns
                'two_pointers': 0.6,
                'sliding_window': 0.6,
                'hash_map': 0.55,
                'recursion': 0.5,
                
                # Basic patterns
                'array_manipulation': 0.3,
                'iteration': 0.2,
                'conditional_logic': 0.15
            }
            
            avg_sophistication = sum(pattern_sophistication_scores.get(p, 0.3) 
                                   for p in active_patterns) / len(active_patterns)
            
            # Bonus for having multiple sophisticated patterns
            pattern_count_bonus = min(len(active_patterns) / 5.0, 0.3)
            
            combination_scores.append(avg_sophistication + pattern_count_bonus)
        
        # Calculate final combination sophistication score
        if combination_scores:
            max_score = max(combination_scores)
            avg_score = sum(combination_scores) / len(combination_scores)
            
            # Weight towards the best combination but consider overall quality
            final_score = (max_score * 0.7) + (avg_score * 0.3)
        else:
            final_score = 0.1  # Very low if no meaningful combinations
        
        # Apply pattern diversity bonus
        unique_categories = self._categorize_patterns_by_type(active_patterns)
        if len(unique_categories) >= 3:
            final_score += 0.1  # Bonus for cross-category sophistication
        
        return min(final_score, 1.0)
    
    def _categorize_patterns_by_type(self, patterns: List[str]) -> Set[str]:
        """Categorize patterns by their type for diversity analysis"""
        
        pattern_categories = {
            'algorithmic': {
                'dynamic_programming', 'recursion', 'divide_conquer', 'backtracking',
                'greedy_choice', 'binary_search'
            },
            'data_structure': {
                'hash_map', 'hash_set', 'heap_usage', 'stack_usage', 'queue_usage',
                'trie_usage', 'segment_tree', 'fenwick_tree', 'union_find'
            },
            'optimization': {
                'two_pointers', 'sliding_window', 'monotonic_stack', 'space_optimization',
                'time_optimization', 'cache_optimization', 'lazy_propagation'
            },
            'graph': {
                'graph_dfs', 'graph_bfs', 'topological_sort', 'shortest_path',
                'minimum_spanning_tree'
            },
            'string': {
                'string_operations', 'string_matching', 'pattern_matching', 'text_processing'
            },
            'mathematical': {
                'number_theory', 'combinatorics', 'probability', 'bit_manipulation'
            },
            'implementation': {
                'array_manipulation', 'iteration', 'conditional_logic', 'sorting'
            }
        }
        
        categories_found = set()
        
        for pattern in patterns:
            for category, category_patterns in pattern_categories.items():
                if pattern in category_patterns:
                    categories_found.add(category)
                    break
        
        return categories_found

# Also, make sure the method is called correctly in the analyze_structural_sophistication method:


    def __init__(self):
        self.sophistication_levels = {
            'basic': {
                'patterns': ['iteration', 'conditional_logic', 'array_manipulation'],
                'weight': 0.1
            },
            'intermediate': {
                'patterns': ['hash_map', 'two_pointers', 'sliding_window', 'stack_usage'],
                'weight': 0.3
            },
            'advanced': {
                'patterns': ['dynamic_programming', 'binary_search', 'graph_dfs', 'heap_usage'],
                'weight': 0.6
            },
            'expert': {
                'patterns': ['segment_tree', 'union_find', 'string_matching', 'number_theory'],
                'weight': 1.0
            }
        }
    
    def analyze_structural_sophistication(self, code: str, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """Analyze sophistication of structures used"""
        
        sophistication_metrics = {}
        
        # Calculate weighted sophistication by level
        for level, config in self.sophistication_levels.items():
            level_score = sum(pattern_scores.get(pattern, 0) for pattern in config['patterns'])
            weighted_score = level_score * config['weight']
            sophistication_metrics[f'{level}_sophistication'] = min(weighted_score, 1.0)
        
        # Overall sophistication
        sophistication_metrics['overall_sophistication'] = sum(sophistication_metrics.values()) / 4
        
        # Pattern combination sophistication
        active_patterns = [pattern for pattern, score in pattern_scores.items() if score > 0.3]
        sophistication_metrics['combination_sophistication'] = self.analyze_pattern_combination_sophistication(active_patterns)
        
        # Implementation elegance
        sophistication_metrics['implementation_elegance'] = self._assess_implementation_elegance(code)
        
        return sophistication_metrics
    
    def _analyze_pattern_combinations(self, pattern_scores: Dict[str, float]) -> float:
        """Analyze sophistication of pattern combinations"""
        
        active_patterns = [pattern for pattern, score in pattern_scores.items() if score > 0.3]
        
        # Sophisticated combinations
        sophisticated_combos = [
            ('dynamic_programming', 'memoization'),
            ('graph_dfs', 'union_find'),
            ('binary_search', 'two_pointers'),
            ('sliding_window', 'hash_map'),
            ('heap_usage', 'greedy_choice')
        ]
        
        combination_score = 0.0
        for combo in sophisticated_combos:
            if all(pattern in active_patterns for pattern in combo):
                combination_score += 0.3
        
        return min(combination_score, 1.0)
    
    def _assess_implementation_elegance(self, code: str) -> float:
        """Assess elegance of implementation"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        elegance_factors = []
        
        # Code conciseness (lines vs functionality)
        line_count = len(code.strip().split('\n'))
        functionality_score = self._estimate_functionality_complexity(tree)
        
        if line_count > 0:
            conciseness = min(functionality_score / line_count, 1.0)
            elegance_factors.append(conciseness)
        
        # Function decomposition quality
        decomposition_score = self._assess_function_decomposition(tree)
        elegance_factors.append(decomposition_score)
        
        # Variable naming quality
        naming_score = self._assess_variable_naming(tree)
        elegance_factors.append(naming_score)
        
        return sum(elegance_factors) / len(elegance_factors) if elegance_factors else 0.0
    
    def _estimate_functionality_complexity(self, tree: ast.AST) -> float:
        """Estimate complexity of functionality implemented"""
        
        complexity_score = 0.0
        
        # Count different types of constructs
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                complexity_score += 0.2
            elif isinstance(node, ast.If):
                complexity_score += 0.1
            elif isinstance(node, ast.FunctionDef):
                complexity_score += 0.3
            elif isinstance(node, ast.Call):
                complexity_score += 0.05
        
        return min(complexity_score, 2.0)
    
    def _assess_function_decomposition(self, tree: ast.AST) -> float:
        """Assess quality of function decomposition"""
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 0.5  # Neutral for no functions
        
        # Analyze function sizes and responsibilities
        avg_function_size = sum(len(list(ast.walk(func))) for func in functions) / len(functions)
        
        # Optimal function size (not too big, not too small)
        if 10 <= avg_function_size <= 30:
            size_score = 1.0
        elif 5 <= avg_function_size <= 50:
            size_score = 0.7
        else:
            size_score = 0.3
        
        return size_score
    
    def _assess_variable_naming(self, tree: ast.AST) -> float:
        """Assess quality of variable naming"""
        
        variable_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not keyword.iskeyword(node.id):
                variable_names.add(node.id)
        
        if not variable_names:
            return 0.5
        
        naming_quality = 0.0
        
        for name in variable_names:
            # Good naming characteristics
            if len(name) >= 3 and name.islower() and '_' in name:
                naming_quality += 1.0
            elif len(name) >= 2 and name.islower():
                naming_quality += 0.7
            elif len(name) == 1 and name in 'ijkxyz':  # Acceptable single letters
                naming_quality += 0.5
            else:
                naming_quality += 0.2
        
        return naming_quality / len(variable_names)


class ArchitecturalThinkingAnalyzer:
    """Analyzes evidence of architectural and system-level thinking"""
    
    def __init__(self):
        self.architectural_indicators = {
            'separation_of_concerns': self._assess_separation_of_concerns,
            'modularity': self._assess_modularity,
            'scalability_awareness': self._assess_scalability_awareness,
            'extensibility': self._assess_extensibility,
            'abstraction_levels': self._assess_abstraction_levels,
            'interface_design': self._assess_interface_design,
            'dependency_management': self._assess_dependency_management
        }
    
    def analyze_architectural_thinking(self, code: str, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """Analyze architectural thinking patterns"""
        
        architectural_metrics = {}
        
        for indicator_name, assessment_func in self.architectural_indicators.items():
            architectural_metrics[indicator_name] = assessment_func(code, pattern_scores)
        
        # Overall architectural thinking score
        architectural_metrics['overall_architectural_thinking'] = (
            sum(architectural_metrics.values()) / len(architectural_metrics)
        )
        
        # System design readiness
        architectural_metrics['system_design_readiness'] = self._assess_system_design_readiness(
            architectural_metrics
        )
        
        return architectural_metrics
    
    def _assess_separation_of_concerns(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess separation of concerns in code structure"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if len(functions) <= 1:
            return 0.3  # Low score for monolithic code
        
        # Analyze function responsibilities
        separation_score = 0.0
        
        for func in functions:
            # Simple heuristic: functions with clear, single purpose tend to be smaller
            func_complexity = len(list(ast.walk(func)))
            
            if func_complexity < 20:  # Small, focused function
                separation_score += 0.8
            elif func_complexity < 40:  # Medium function
                separation_score += 0.5
            else:  # Large function - poor separation
                separation_score += 0.2
        
        return separation_score / len(functions) if functions else 0.0
    
    def _assess_modularity(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess modular design patterns"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        modularity_indicators = []
        
        # Function definitions indicate modularity
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            modularity_indicators.append(min(len(functions) / 5.0, 1.0))
        
        # Class definitions indicate OOP modularity
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if classes:
            modularity_indicators.append(0.8)
        
        # Import statements indicate use of external modules
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            modularity_indicators.append(min(len(imports) / 3.0, 0.6))
        
        if not modularity_indicators:
            return 0.2  # Low modularity for monolithic code
        
        return sum(modularity_indicators) / len(modularity_indicators)
    
    def _assess_scalability_awareness(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess awareness of scalability concerns"""
        
        scalability_patterns = [
            'space_optimization', 'time_optimization', 'binary_search', 
            'hash_map', 'heap_usage', 'dynamic_programming'
        ]
        
        scalability_score = sum(pattern_scores.get(pattern, 0) for pattern in scalability_patterns)
        normalized_score = scalability_score / len(scalability_patterns)
        
        # Additional scalability indicators in code
        scalability_indicators = self._detect_scalability_code_patterns(code)
        
        return (normalized_score + scalability_indicators) / 2
    
    def _assess_extensibility(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess code extensibility"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        extensibility_factors = []
        
        # Parameterized functions are more extensible
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            avg_params = sum(len(func.args.args) for func in functions) / len(functions)
            param_score = min(avg_params / 4.0, 1.0)  # 4+ params = good extensibility
            extensibility_factors.append(param_score)
        
        # Configuration constants/variables
        constants = self._count_configuration_constants(tree)
        if constants > 0:
            extensibility_factors.append(min(constants / 3.0, 0.8))
        
        # Generic/reusable patterns
        generic_patterns = ['recursion', 'dynamic_programming', 'divide_conquer']
        generic_score = sum(pattern_scores.get(pattern, 0) for pattern in generic_patterns)
        extensibility_factors.append(generic_score / len(generic_patterns))
        
        return sum(extensibility_factors) / len(extensibility_factors) if extensibility_factors else 0.0
    
    def _assess_abstraction_levels(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess appropriate use of abstraction levels"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        abstraction_score = 0.0
        
        # Higher-level constructs indicate good abstraction
        list_comprehensions = len([node for node in ast.walk(tree) if isinstance(node, ast.ListComp)])
        if list_comprehensions > 0:
            abstraction_score += 0.3
        
        # Function calls to built-in functions
        builtin_calls = self._count_builtin_function_calls(tree)
        abstraction_score += min(builtin_calls / 5.0, 0.4)
        
        # Use of libraries/modules
        import_score = min(len([node for node in ast.walk(tree) 
                               if isinstance(node, (ast.Import, ast.ImportFrom))]) / 2.0, 0.3)
        abstraction_score += import_score
        
        return min(abstraction_score, 1.0)
    
    def _assess_interface_design(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess quality of interface design"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 0.0
        
        interface_quality = 0.0
        
        for func in functions:
            # Good interfaces have clear names
            name_quality = 1.0 if len(func.name) > 3 and '_' in func.name else 0.5
            
            # Good interfaces have reasonable parameter counts
            param_count = len(func.args.args)
            param_quality = 1.0 if 1 <= param_count <= 4 else 0.5
            
            # Good interfaces have return statements
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(func))
            return_quality = 1.0 if has_return else 0.3
            
            func_interface_quality = (name_quality + param_quality + return_quality) / 3
            interface_quality += func_interface_quality
        
        return interface_quality / len(functions)
    
    def _assess_dependency_management(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Assess dependency management patterns"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        # Count imports and how they're used
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        if not imports:
            return 0.5  # Neutral for no dependencies
        
        # Prefer specific imports over wildcard imports
        specific_imports = len([imp for imp in imports 
                              if isinstance(imp, ast.ImportFrom) and 
                              not any(alias.name == '*' for alias in imp.names)])
        
        dependency_score = specific_imports / len(imports) if imports else 0.0
        return dependency_score
    
    def _detect_scalability_code_patterns(self, code: str) -> float:
        """Detect code patterns that indicate scalability awareness"""
        
        scalability_indicators = 0.0
        
        # Early termination patterns
        if re.search(r'\bbreak\b|\breturn\b.*if\b', code):
            scalability_indicators += 0.2
        
        # Complexity comments (O(n), O(log n), etc.)
        if re.search(r'O\([^)]+\)', code):
            scalability_indicators += 0.3
        
        # Memory-efficient patterns
        if re.search(r'\bin-place\b|\bmemory\b|\bspace\b', code, re.IGNORECASE):
            scalability_indicators += 0.2
        
        return min(scalability_indicators, 0.7)
    
    def _count_configuration_constants(self, tree: ast.AST) -> int:
        """Count configuration constants in code"""
        
        constants = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants += 1
        
        return constants
    
    def _count_builtin_function_calls(self, tree: ast.AST) -> int:
        """Count calls to built-in functions"""
        
        builtins = {'len', 'max', 'min', 'sum', 'sorted', 'enumerate', 'zip', 'map', 'filter'}
        builtin_calls = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in builtins:
                    builtin_calls += 1
        
        return builtin_calls
    
    def _assess_system_design_readiness(self, architectural_metrics: Dict[str, float]) -> float:
        """Assess readiness for system design based on architectural thinking"""
        
        key_indicators = [
            'separation_of_concerns',
            'modularity', 
            'scalability_awareness',
            'interface_design'
        ]
        
        readiness_score = sum(architectural_metrics.get(indicator, 0) for indicator in key_indicators)
        return readiness_score / len(key_indicators)


class OptimizationAwarenessAnalyzer:
    """Analyzes awareness and application of optimization techniques"""
    
    def __init__(self):
        self.optimization_categories = {
            'algorithmic': ['binary_search', 'dynamic_programming', 'hash_map', 'two_pointers'],
            'space': ['space_optimization', 'inplace_modifications', 'constant_space'],
            'time': ['time_optimization', 'early_termination', 'cache_optimization'],
            'data_structure': ['heap_usage', 'trie_usage', 'segment_tree', 'union_find']
        }
    
    def analyze_optimization_awareness(self, code: str, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """Analyze optimization awareness across different categories"""
        
        optimization_metrics = {}
        
        # Analyze each optimization category
        for category, patterns in self.optimization_categories.items():
            category_score = sum(pattern_scores.get(pattern, 0) for pattern in patterns)
            optimization_metrics[f'{category}_optimization'] = category_score / len(patterns)
        
        # Code-level optimization indicators
        optimization_metrics['code_level_optimization'] = self._analyze_code_optimization_patterns(code)
        
        # Complexity awareness
        optimization_metrics['complexity_awareness'] = self._assess_complexity_awareness(code)
        
        # Premature optimization detection (negative indicator)
        optimization_metrics['premature_optimization_risk'] = self._detect_premature_optimization(code)
        
        # Overall optimization consciousness
        optimization_metrics['overall_optimization_consciousness'] = self._calculate_overall_consciousness(
            optimization_metrics
        )
        
        return optimization_metrics
    
    def _analyze_code_optimization_patterns(self, code: str) -> float:
        """Analyze optimization patterns in code structure"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        optimization_score = 0.0
        
        # Early termination patterns
        early_terminations = len([node for node in ast.walk(tree) 
                                if isinstance(node, (ast.Break, ast.Continue, ast.Return))
                                and self._is_in_loop_or_condition(node, tree)])
        
        if early_terminations > 0:
            optimization_score += 0.3
        
        # Efficient loop patterns
        efficient_loops = self._count_efficient_loop_patterns(tree)
        optimization_score += min(efficient_loops * 0.2, 0.4)
        
        # Memory-efficient operations
        memory_efficient_ops = self._count_memory_efficient_operations(tree)
        optimization_score += min(memory_efficient_ops * 0.1, 0.3)
        
        return min(optimization_score, 1.0)
    
    def _assess_complexity_awareness(self, code: str) -> float:
        """Assess awareness of algorithmic complexity"""
        
        complexity_indicators = 0.0
        
        # Complexity comments
        complexity_comments = len(re.findall(r'O\([^)]+\)', code))
        if complexity_comments > 0:
            complexity_indicators += 0.4
        
        # Performance-related comments
        perf_comments = len(re.findall(r'(?i)(time|space|complexity|efficient|optimize)', code))
        if perf_comments > 0:
            complexity_indicators += 0.3
        
        # Algorithm choice comments
        algo_comments = len(re.findall(r'(?i)(algorithm|approach|method)', code))
        if algo_comments > 0:
            complexity_indicators += 0.2
        
        return min(complexity_indicators, 0.9)
    
    def _detect_premature_optimization(self, code: str) -> float:
        """Detect signs of premature optimization"""
        
        premature_indicators = 0.0
        
        # Overly complex one-liners
        complex_oneliners = len(re.findall(r'.*\[.*for.*in.*if.*\].*', code))
        if complex_oneliners > 2:
            premature_indicators += 0.3
        
        # Excessive bit manipulation for simple operations
        bit_ops = len(re.findall(r'[&|^~]|<<|>>', code))
        if bit_ops > 3:
            premature_indicators += 0.2
        
        # Overly complex variable names
        complex_names = len(re.findall(r'\b\w{15,}\b', code))
        if complex_names > 0:
            premature_indicators += 0.1
        
        return min(premature_indicators, 0.6)
    
    def _is_in_loop_or_condition(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is inside a loop or conditional"""
        
        # Simplified implementation
        return True  # Placeholder
    
    def _count_efficient_loop_patterns(self, tree: ast.AST) -> int:
        """Count efficient loop patterns"""
        
        efficient_patterns = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for enumerate usage
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'enumerate':
                        efficient_patterns += 1
                
                # Check for range with step
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == 'range' and len(node.iter.args) >= 3:
                        efficient_patterns += 1
        
        return efficient_patterns
    
    def _count_memory_efficient_operations(self, tree: ast.AST) -> int:
        """Count memory-efficient operations"""
        
        efficient_ops = 0
        
        # Generator expressions
        efficient_ops += len([node for node in ast.walk(tree) if isinstance(node, ast.GeneratorExp)])
        
        # In-place operations
        efficient_ops += len([node for node in ast.walk(tree) 
                            if isinstance(node, ast.AugAssign)])
        
        return efficient_ops
    
    def _calculate_overall_consciousness(self, optimization_metrics: Dict[str, float]) -> float:
        """Calculate overall optimization consciousness"""
        
        key_metrics = [
            'algorithmic_optimization',
            'space_optimization', 
            'time_optimization',
            'complexity_awareness'
        ]
        
        consciousness_score = sum(optimization_metrics.get(metric, 0) for metric in key_metrics)
        
        # Penalty for premature optimization
        premature_penalty = optimization_metrics.get('premature_optimization_risk', 0)
        consciousness_score -= premature_penalty
        
        return max(0, consciousness_score / len(key_metrics))


class CodeOrganizationAnalyzer:
    """Analyzes code organization and structure quality"""
    
    def __init__(self):
        self.organization_aspects = {
            'structure': self._analyze_code_structure,
            'naming': self._analyze_naming_conventions,
            'formatting': self._analyze_code_formatting,
            'comments': self._analyze_comment_quality,
            'logical_flow': self._analyze_logical_flow,
            'readability': self._analyze_readability
        }
    
    def analyze_code_organization(self, code: str) -> Dict[str, float]:
        """Analyze various aspects of code organization"""
        
        organization_metrics = {}
        
        for aspect_name, analysis_func in self.organization_aspects.items():
            organization_metrics[aspect_name] = analysis_func(code)
        
        # Overall organization score
        organization_metrics['overall_organization'] = (
            sum(organization_metrics.values()) / len(organization_metrics)
        )
        
        # Professional code quality
        organization_metrics['professional_quality'] = self._assess_professional_quality(
            organization_metrics
        )
        
        return organization_metrics
    
    def _analyze_code_structure(self, code: str) -> float:
        """Analyze overall code structure"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.1
        
        structure_score = 0.0
        
        # Function organization
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if functions:
            # Prefer multiple small functions over one large function
            avg_function_size = sum(len(list(ast.walk(func))) for func in functions) / len(functions)
            
            if 10 <= avg_function_size <= 25:
                structure_score += 0.8
            elif 5 <= avg_function_size <= 40:
                structure_score += 0.6
            else:
                structure_score += 0.3
        else:
            structure_score += 0.4  # Neutral for no functions
        
        # Import organization
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if imports:
            structure_score += 0.2
        
        return min(structure_score, 1.0)
    
    def _analyze_naming_conventions(self, code: str) -> float:
        """Analyze naming convention quality"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        naming_score = 0.0
        total_names = 0
        
        # Analyze variable names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not keyword.iskeyword(node.id):
                total_names += 1
                name = node.id
                
                # Good naming patterns
                if len(name) >= 3 and name.islower():
                    if '_' in name or name.isalpha():
                        naming_score += 1.0
                    else:
                        naming_score += 0.7
                elif len(name) == 1 and name in 'ijklmnxyz':  # Common single letters
                    naming_score += 0.6
                elif len(name) == 2 and name.islower():
                    naming_score += 0.5
                else:
                    naming_score += 0.2
        
        # Analyze function names
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in functions:
            total_names += 1
            name = func.name
            
            if len(name) >= 3 and name.islower() and ('_' in name or name.isalpha()):
                naming_score += 1.0
            else:
                naming_score += 0.4
        
        return naming_score / total_names if total_names > 0 else 0.5
    
    def _analyze_code_formatting(self, code: str) -> float:
        """Analyze code formatting quality"""
        
        formatting_score = 0.0
        
        lines = code.split('\n')
        
        # Consistent indentation
        indentations = []
        for line in lines:
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if indentations:
            # Check for consistent indentation (multiples of 4)
            consistent_indent = all(indent % 4 == 0 for indent in indentations)
            if consistent_indent:
                formatting_score += 0.3
            else:
                formatting_score += 0.1
        
        # Reasonable line length
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines == 0:
            formatting_score += 0.2
        elif long_lines <= len(lines) * 0.1:  # Less than 10% long lines
            formatting_score += 0.1
        
        # Blank lines for separation
        blank_lines = sum(1 for line in lines if not line.strip())
        if blank_lines > 0:
            formatting_score += 0.2
        
        # Proper spacing around operators
        good_spacing = len(re.findall(r'\w\s*[+\-*/=]\s*\w', code))
        total_operators = len(re.findall(r'[+\-*/=]', code))
        
        if total_operators > 0:
            spacing_ratio = good_spacing / total_operators
            formatting_score += spacing_ratio * 0.3
        
        return min(formatting_score, 1.0)
    
    def _analyze_comment_quality(self, code: str) -> float:
        """Analyze quality and usefulness of comments"""
        
        comment_score = 0.0
        
        # Extract comments
        comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
        docstrings = re.findall(r'""".*?"""', code, re.DOTALL)
        
        total_lines = len([line for line in code.split('\n') if line.strip()])
        code_lines = total_lines - len(comment_lines)
        
        if code_lines == 0:
            return 0.0
        
        comment_ratio = len(comment_lines) / code_lines
        
        # Good comment ratio (not too few, not too many)
        if 0.1 <= comment_ratio <= 0.3:
            comment_score += 0.4
        elif 0.05 <= comment_ratio <= 0.5:
            comment_score += 0.2
        
        # Quality of comments
        meaningful_comments = 0
        for comment in comment_lines:
            comment_text = comment.strip('#').strip()
            if len(comment_text) > 10 and not comment_text.lower().startswith('todo'):
                meaningful_comments += 1
        
        if comment_lines:
            meaningful_ratio = meaningful_comments / len(comment_lines)
            comment_score += meaningful_ratio * 0.4
        
        # Docstrings
        if docstrings:
            comment_score += 0.2
        
        return min(comment_score, 1.0)
    
    def _analyze_logical_flow(self, code: str) -> float:
        """Analyze logical flow and structure"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        flow_score = 0.0
        
        # Check for early returns
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
            
            if len(returns) > 1:
                # Multiple returns can indicate good early termination
                flow_score += 0.3
            elif len(returns) == 1:
                flow_score += 0.2
        
        # Check for guard clauses
        guard_clauses = self._count_guard_clauses(tree)
        if guard_clauses > 0:
            flow_score += 0.2
        
        # Check for proper error handling
        try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
        if try_blocks > 0:
            flow_score += 0.3
        
        # Avoid deeply nested code
        max_nesting = self._calculate_max_nesting_depth(tree)
        if max_nesting <= 3:
            flow_score += 0.2
        elif max_nesting <= 5:
            flow_score += 0.1
        
        return min(flow_score, 1.0)
    
    def _analyze_readability(self, code: str) -> float:
        """Analyze overall code readability"""
        
        readability_score = 0.0
        
        # Simple readability metrics
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            return 0.0
        
        # Average line length
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if 20 <= avg_line_length <= 60:
            readability_score += 0.3
        elif 10 <= avg_line_length <= 80:
            readability_score += 0.2
        
        # Complexity per line
        total_complexity = sum(self._calculate_line_complexity(line) for line in lines)
        avg_complexity = total_complexity / len(lines)
        
        if avg_complexity <= 2:
            readability_score += 0.3
        elif avg_complexity <= 4:
            readability_score += 0.2
        
        # Use of descriptive names
        descriptive_names = len(re.findall(r'\b[a-z_]{4,}\b', code))
        total_identifiers = len(re.findall(r'\b[a-zA-Z_]\w*\b', code))
        
        if total_identifiers > 0:
            descriptive_ratio = descriptive_names / total_identifiers
            readability_score += descriptive_ratio * 0.4
        
        return min(readability_score, 1.0)
    
    def _count_guard_clauses(self, tree: ast.AST) -> int:
        """Count guard clauses (early validation)"""
        
        guard_count = 0
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            # Look for early returns with conditions
            for node in func.body[:3]:  # Check first few statements
                if isinstance(node, ast.If):
                    # Check if if-block contains return
                    if any(isinstance(child, ast.Return) for child in ast.walk(node)):
                        guard_count += 1
        
        return guard_count
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    def _calculate_line_complexity(self, line: str) -> int:
        """Calculate complexity of a single line"""
        
        complexity = 0
        
        # Operators add complexity
        complexity += len(re.findall(r'[+\-*/=<>!&|]', line))
        
        # Function calls add complexity
        complexity += len(re.findall(r'\w+\(', line))
        
        # Brackets/indexing add complexity
        complexity += len(re.findall(r'[\[\]()]', line))
        
        return complexity
    
    def _assess_professional_quality(self, organization_metrics: Dict[str, float]) -> float:
        """Assess if code meets professional quality standards"""
        
        key_aspects = ['structure', 'naming', 'formatting', 'logical_flow']
        
        professional_score = sum(organization_metrics.get(aspect, 0) for aspect in key_aspects)
        return professional_score / len(key_aspects)


class ErrorHandlingAnalyzer:
    """Analyzes error handling and defensive programming practices"""
    
    def __init__(self):
        self.error_handling_aspects = {
            'explicit_error_handling': self._analyze_explicit_error_handling,
            'input_validation': self._analyze_input_validation,
            'edge_case_handling': self._analyze_edge_case_handling,
            'defensive_programming': self._analyze_defensive_programming,
            'graceful_degradation': self._analyze_graceful_degradation
        }
    
    def analyze_error_handling_maturity(self, code: str) -> Dict[str, float]:
        """Analyze error handling maturity"""
        
        error_handling_metrics = {}
        
        for aspect_name, analysis_func in self.error_handling_aspects.items():
            error_handling_metrics[aspect_name] = analysis_func(code)
        
        # Overall error handling maturity
        error_handling_metrics['overall_error_handling_maturity'] = (
            sum(error_handling_metrics.values()) / len(error_handling_metrics)
        )
        
        # Production readiness based on error handling
        error_handling_metrics['production_readiness'] = self._assess_production_readiness(
            error_handling_metrics
        )
        
        return error_handling_metrics
    
    def _analyze_explicit_error_handling(self, code: str) -> float:
        """Analyze explicit error handling constructs"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        error_handling_score = 0.0
        
        # Try-except blocks
        try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        if try_blocks:
            error_handling_score += 0.5
            
            # Quality of exception handling
            for try_block in try_blocks:
                # Specific exception types (better than bare except)
                specific_exceptions = sum(1 for handler in try_block.handlers 
                                        if handler.type is not None)
                if specific_exceptions > 0:
                    error_handling_score += 0.2
                
                # Finally blocks
                if try_block.finalbody:
                    error_handling_score += 0.1
        
        # Assertions
        assertions = len([node for node in ast.walk(tree) if isinstance(node, ast.Assert)])
        if assertions > 0:
            error_handling_score += min(assertions * 0.1, 0.3)
        
        return min(error_handling_score, 1.0)
    
    def _analyze_input_validation(self, code: str) -> float:
        """Analyze input validation patterns"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        validation_score = 0.0
        
        # Check for input validation patterns
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            if func.args.args:  # Function has parameters
                # Look for validation in function body
                for node in func.body[:3]:  # Check first few statements
                    if isinstance(node, ast.If):
                        # Simple heuristic: if statements near function start might be validation
                        validation_score += 0.2
                    elif isinstance(node, ast.Assert):
                        validation_score += 0.3
        
        # Type checking patterns
        type_checks = len(re.findall(r'isinstance\(|type\(', code))
        if type_checks > 0:
            validation_score += min(type_checks * 0.2, 0.4)
        
        # None checks
        none_checks = len(re.findall(r'is\s+None|is\s+not\s+None|==\s*None|!=\s*None', code))
        if none_checks > 0:
            validation_score += min(none_checks * 0.1, 0.3)
        
        return min(validation_score, 1.0)
    
    def _analyze_edge_case_handling(self, code: str) -> float:
        """Analyze edge case handling"""
        
        edge_case_score = 0.0
        
        # Empty input handling
        empty_checks = len(re.findall(r'if\s+not\s+\w+|len\(\w+\)\s*==\s*0', code))
        if empty_checks > 0:
            edge_case_score += 0.3
        
        # Boundary condition checks
        boundary_checks = len(re.findall(r'<=|>=|<\s*len|>\s*len', code))
        if boundary_checks > 0:
            edge_case_score += 0.2
        
        # Zero division checks
        zero_checks = len(re.findall(r'!=\s*0|==\s*0', code))
        if zero_checks > 0:
            edge_case_score += 0.2
        
        # Early return patterns (often used for edge cases)
        early_returns = len(re.findall(r'^\s*if.*:\s*return', code, re.MULTILINE))
        if early_returns > 0:
            edge_case_score += min(early_returns * 0.1, 0.3)
        
        return min(edge_case_score, 1.0)
    
    def _analyze_defensive_programming(self, code: str) -> float:
        """Analyze defensive programming practices"""
        
        defensive_score = 0.0
        
        # Defensive checks
        defensive_patterns = [
            r'if\s+\w+\s+is\s+not\s+None',
            r'hasattr\(',
            r'getattr\(',
            r'\.get\(',  # Safe dictionary access
            r'try\s*:.*except.*:'
        ]
        
        for pattern in defensive_patterns:
            matches = len(re.findall(pattern, code, re.DOTALL))
            if matches > 0:
                defensive_score += min(matches * 0.15, 0.3)
        
        # Bounds checking
        bounds_checks = len(re.findall(r'0\s*<=.*<\s*len|range\(.*len\(', code))
        if bounds_checks > 0:
            defensive_score += 0.2
        
        return min(defensive_score, 1.0)
    
    def _analyze_graceful_degradation(self, code: str) -> float:
        """Analyze graceful degradation patterns"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        degradation_score = 0.0
        
        # Default values
        default_values = 0
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            for default in func.args.defaults:
                default_values += 1
        
        if default_values > 0:
            degradation_score += 0.3
        
        # Fallback patterns
        fallback_patterns = len(re.findall(r'else\s*:|except.*:\s*.*=', code, re.DOTALL))
        if fallback_patterns > 0:
            degradation_score += 0.3
        
        # Alternative implementations
        alternative_patterns = len(re.findall(r'try\s*:.*except.*:.*', code, re.DOTALL))
        if alternative_patterns > 0:
            degradation_score += 0.4
        
        return min(degradation_score, 1.0)
    
    def _assess_production_readiness(self, error_handling_metrics: Dict[str, float]) -> float:
        """Assess production readiness based on error handling"""
        
        critical_aspects = [
            'explicit_error_handling',
            'input_validation',
            'edge_case_handling'
        ]
        
        readiness_score = sum(error_handling_metrics.get(aspect, 0) for aspect in critical_aspects)
        return readiness_score / len(critical_aspects)


class AbstractionLevelAnalyzer:
    """Analyzes appropriate use of abstraction levels"""
    
    def analyze_abstraction_level(self, code: str) -> Dict[str, float]:
        """Analyze abstraction level usage"""
        
        try:
            tree = ast.parse(code)
        except:
            return {'parse_error': 1.0}
        
        abstraction_metrics = {}
        
        # Function abstraction
        abstraction_metrics['function_abstraction'] = self._analyze_function_abstraction(tree)
        
        # Data abstraction
        abstraction_metrics['data_abstraction'] = self._analyze_data_abstraction(tree)
        
        # Language feature usage
        abstraction_metrics['language_feature_usage'] = self._analyze_language_features(tree, code)
        
        # Appropriate abstraction level
        abstraction_metrics['abstraction_appropriateness'] = self._assess_abstraction_appropriateness(tree)
        
        return abstraction_metrics
    
    def _analyze_function_abstraction(self, tree: ast.AST) -> float:
        """Analyze function-level abstraction"""
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 0.3  # Low score for no functions
        
        abstraction_score = 0.0
        
        for func in functions:
            # Function size (smaller functions = better abstraction)
            func_size = len(list(ast.walk(func)))
            
            if func_size <= 20:
                abstraction_score += 0.8
            elif func_size <= 40:
                abstraction_score += 0.5
            else:
                abstraction_score += 0.2
            
            # Parameter abstraction
            param_count = len(func.args.args)
            if 1 <= param_count <= 4:
                abstraction_score += 0.2
        
        return abstraction_score / len(functions)
    
    def _analyze_data_abstraction(self, tree: ast.AST) -> float:
        """Analyze data abstraction patterns"""
        
        abstraction_score = 0.0
        
        # Use of data structures
        data_structures = {
            'dict': 0.3,
            'list': 0.2,
            'set': 0.4,
            'tuple': 0.3
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in data_structures:
                    abstraction_score += data_structures[node.func.id]
        
        # List comprehensions (higher abstraction)
        list_comps = len([node for node in ast.walk(tree) if isinstance(node, ast.ListComp)])
        abstraction_score += min(list_comps * 0.2, 0.4)
        
        return min(abstraction_score, 1.0)
    
    def _analyze_language_features(self, tree: ast.AST, code: str) -> float:
        """Analyze usage of high-level language features"""
        
        feature_score = 0.0
        
        # Built-in function usage
        builtins = ['map', 'filter', 'zip', 'enumerate', 'sorted', 'max', 'min', 'sum']
        builtin_usage = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in builtins:
                    builtin_usage += 1
        
        feature_score += min(builtin_usage * 0.1, 0.4)
        
        # Context managers
        with_statements = len([node for node in ast.walk(tree) if isinstance(node, ast.With)])
        if with_statements > 0:
            feature_score += 0.3
        
        # Decorators
        decorators = sum(len(node.decorator_list) for node in ast.walk(tree) 
                        if hasattr(node, 'decorator_list'))
        if decorators > 0:
            feature_score += 0.2
        
        return min(feature_score, 1.0)
    
    def _assess_abstraction_appropriateness(self, tree: ast.AST) -> float:
        """Assess if abstraction level is appropriate"""
        
        # This is a complex assessment that would need problem context
        # For now, provide a heuristic based on code complexity vs abstraction used
        
        total_nodes = len(list(ast.walk(tree)))
        functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        
        if total_nodes == 0:
            return 0.0
        
        # Ratio of functions to total complexity
        abstraction_ratio = functions / (total_nodes / 10)  # Normalize by complexity
        
        # Appropriate abstraction: not too little, not too much
        if 0.1 <= abstraction_ratio <= 0.5:
            return 0.8
        elif 0.05 <= abstraction_ratio <= 0.8:
            return 0.6
        else:
            return 0.3


class MaintainabilityAnalyzer:
    """Analyzes code maintainability factors"""
    
    def analyze_maintainability(self, code: str) -> Dict[str, float]:
        """Analyze code maintainability"""
        
        maintainability_metrics = {}
        
        # Code readability
        maintainability_metrics['readability'] = self._analyze_readability_for_maintenance(code)
        
        # Modularity
        maintainability_metrics['modularity'] = self._analyze_modularity_for_maintenance(code)
        
        # Testability
        maintainability_metrics['testability'] = self._analyze_testability(code)
        
        # Documentation quality
        maintainability_metrics['documentation'] = self._analyze_documentation_quality(code)
        
        # Code complexity
        maintainability_metrics['complexity_management'] = self._analyze_complexity_for_maintenance(code)
        
        # Overall maintainability
        maintainability_metrics['overall_maintainability'] = (
            sum(maintainability_metrics.values()) / len(maintainability_metrics)
        )
        
        return maintainability_metrics
    
    def _analyze_readability_for_maintenance(self, code: str) -> float:
        """Analyze readability from maintenance perspective"""
        
        readability_score = 0.0
        
        # Clear variable names
        clear_names = len(re.findall(r'\b[a-z_]{4,}\b', code))
        total_names = len(re.findall(r'\b[a-zA-Z_]\w*\b', code))
        
        if total_names > 0:
            name_clarity = clear_names / total_names
            readability_score += name_clarity * 0.4
        
        # Comments explaining why, not what
        meaningful_comments = len(re.findall(r'#.*(?:why|because|reason|purpose)', code, re.IGNORECASE))
        total_comments = len(re.findall(r'#.*', code))
        
        if total_comments > 0:
            comment_quality = meaningful_comments / total_comments
            readability_score += comment_quality * 0.3
        
        # Consistent formatting
        lines = code.split('\n')
        consistent_indent = self._check_consistent_indentation(lines)
        if consistent_indent:
            readability_score += 0.3
        
        return min(readability_score, 1.0)
    
    def _analyze_modularity_for_maintenance(self, code: str) -> float:
        """Analyze modularity from maintenance perspective"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        modularity_score = 0.0
        
        # Function count and size
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if functions:
            avg_function_size = sum(len(list(ast.walk(func))) for func in functions) / len(functions)
            
            # Prefer smaller, focused functions
            if avg_function_size <= 15:
                modularity_score += 0.5
            elif avg_function_size <= 30:
                modularity_score += 0.3
            else:
                modularity_score += 0.1
        
        # Single responsibility principle
        if functions:
            # Heuristic: functions with clear, specific names
            specific_names = sum(1 for func in functions if len(func.name) > 5 and '_' in func.name)
            srp_score = specific_names / len(functions)
            modularity_score += srp_score * 0.3
        
        # Low coupling indicators
        global_vars = len([node for node in ast.walk(tree) 
                          if isinstance(node, ast.Global)])
        if global_vars == 0:
            modularity_score += 0.2
        
        return min(modularity_score, 1.0)
    
    def _analyze_testability(self, code: str) -> float:
        """Analyze how testable the code is"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        testability_score = 0.0
        
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if functions:
            # Pure functions (no side effects) are more testable
            pure_functions = 0
            
            for func in functions:
                # Heuristic: functions that return values and don't modify globals
                has_return = any(isinstance(node, ast.Return) for node in ast.walk(func))
                modifies_global = any(isinstance(node, ast.Global) for node in ast.walk(func))
                
                if has_return and not modifies_global:
                    pure_functions += 1
            
            purity_ratio = pure_functions / len(functions)
            testability_score += purity_ratio * 0.5
        
        # Dependency injection patterns
        functions_with_params = sum(1 for func in functions if func.args.args)
        if functions:
            param_ratio = functions_with_params / len(functions)
            testability_score += param_ratio * 0.3
        
        # Avoid hard-coded values
        literals = len([node for node in ast.walk(tree) 
                       if isinstance(node, ast.Constant) and isinstance(node.value, (int, str))
                       and node.value not in (0, 1, "", None)])
        
        if literals <= 3:
            testability_score += 0.2
        
        return min(testability_score, 1.0)
    
    def _analyze_documentation_quality(self, code: str) -> float:
        """Analyze documentation quality"""
        
        doc_score = 0.0
        
        # Docstrings
        docstrings = len(re.findall(r'""".*?"""', code, re.DOTALL))
        if docstrings > 0:
            doc_score += 0.4
        
        # Meaningful comments
        comments = re.findall(r'#.*', code)
        meaningful_comments = [c for c in comments 
                             if len(c.strip('#').strip()) > 10 
                             and not c.strip().lower().startswith('#todo')]
        
        if comments:
            meaningful_ratio = len(meaningful_comments) / len(comments)
            doc_score += meaningful_ratio * 0.4
        
        # Type hints (if any)
        type_hints = len(re.findall(r':\s*\w+|def\s+\w+\(.*\)\s*->', code))
        if type_hints > 0:
            doc_score += 0.2
        
        return min(doc_score, 1.0)
    
    def _analyze_complexity_for_maintenance(self, code: str) -> float:
        """Analyze complexity from maintenance perspective"""
        
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        complexity_score = 1.0  # Start high, subtract for complexity
        
        # Cyclomatic complexity
        complexity_nodes = len([node for node in ast.walk(tree) 
                               if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))])
        
        # Penalize high complexity
        if complexity_nodes > 10:
            complexity_score -= 0.5
        elif complexity_nodes > 5:
            complexity_score -= 0.3
        
        # Nesting depth
        max_depth = self._calculate_max_nesting_depth(tree)
        if max_depth > 4:
            complexity_score -= 0.3
        elif max_depth > 3:
            complexity_score -= 0.2
        
        return max(complexity_score, 0.0)
    
    def _check_consistent_indentation(self, lines: List[str]) -> bool:
        """Check for consistent indentation"""
        
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indents.append(indent)
        
        if not indents:
            return True
        
        # Check if all indents are multiples of the same number
        min_indent = min(indent for indent in indents if indent > 0) if any(indent > 0 for indent in indents) else 4
        
        return all(indent % min_indent == 0 for indent in indents)
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.Try, ast.With)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)


class PerformanceConsciousnessAnalyzer:
    """Analyzes consciousness of performance implications"""
    
    def analyze_performance_consciousness(self, code: str, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """Analyze performance consciousness"""
        
        performance_metrics = {}
        
        # Algorithm efficiency awareness
        performance_metrics['algorithm_efficiency'] = self._analyze_algorithm_efficiency(pattern_scores)
        
        # Data structure efficiency
        performance_metrics['data_structure_efficiency'] = self._analyze_data_structure_choices(pattern_scores)
        
        # Memory efficiency
        performance_metrics['memory_efficiency'] = self._analyze_memory_efficiency(code, pattern_scores)
        
        # Time complexity awareness
        performance_metrics['time_complexity_awareness'] = self._analyze_time_complexity_awareness(code)
        
        # Performance comments/annotations
        performance_metrics['performance_documentation'] = self._analyze_performance_documentation(code)
        
        # Overall performance consciousness
        performance_metrics['overall_performance_consciousness'] = (
            sum(performance_metrics.values()) / len(performance_metrics)
        )
        
        return performance_metrics
    
    def _analyze_algorithm_efficiency(self, pattern_scores: Dict[str, float]) -> float:
        """Analyze choice of efficient algorithms"""
        
        efficient_algorithms = {
            'binary_search': 0.8,
            'hash_map': 0.7,
            'two_pointers': 0.6,
            'sliding_window': 0.6,
            'dynamic_programming': 0.8,
            'heap_usage': 0.7
        }
        
        inefficient_algorithms = {
            'nested_loops': -0.3,
            'linear_search': -0.2
        }
        
        efficiency_score = 0.0
        
        # Add points for efficient algorithms
        for algo, score in efficient_algorithms.items():
            if pattern_scores.get(algo, 0) > 0.3:
                efficiency_score += score * pattern_scores[algo]
        
        # Subtract points for inefficient algorithms
        for algo, penalty in inefficient_algorithms.items():
            if pattern_scores.get(algo, 0) > 0.3:
                efficiency_score += penalty * pattern_scores[algo]
        
        return max(0, min(efficiency_score, 1.0))
    
    def _analyze_data_structure_choices(self, pattern_scores: Dict[str, float]) -> float:
        """Analyze efficiency of data structure choices"""
        
        efficient_structures = {
            'hash_map': 0.8,  # O(1) lookup
            'hash_set': 0.7,  # O(1) membership
            'heap_usage': 0.8,  # O(log n) operations
            'trie_usage': 0.6  # Efficient for string operations
        }
        
        inefficient_structures = {
            'array_manipulation': -0.1  # Can be inefficient for frequent insertions
        }
        
        structure_score = 0.0
        
        for structure, score in efficient_structures.items():
            if pattern_scores.get(structure, 0) > 0.3:
                structure_score += score * pattern_scores[structure]
        
        for structure, penalty in inefficient_structures.items():
            if pattern_scores.get(structure, 0) > 0.5:
                structure_score += penalty * pattern_scores[structure]
        
        return max(0, min(structure_score, 1.0))
    
    def _analyze_memory_efficiency(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Analyze memory efficiency patterns"""
        
        memory_score = 0.0
        
        # Space optimization patterns
        space_optimization = pattern_scores.get('space_optimization', 0)
        memory_score += space_optimization * 0.5
        
        # In-place operations
        try:
            tree = ast.parse(code)
            inplace_ops = len([node for node in ast.walk(tree) if isinstance(node, ast.AugAssign)])
            if inplace_ops > 0:
                memory_score += min(inplace_ops * 0.1, 0.3)
        except:
            pass
        
        # Avoid unnecessary data copies
        copy_patterns = len(re.findall(r'\.copy\(\)|list\(|set\(.*\)', code))
        if copy_patterns > 2:
            memory_score -= 0.2  # Penalty for excessive copying
        
        # Generator expressions (memory efficient)
        generator_patterns = len(re.findall(r'\(.*for.*in.*\)', code))
        if generator_patterns > 0:
            memory_score += min(generator_patterns * 0.15, 0.3)
        
        return max(0, min(memory_score, 1.0))
    
    def _analyze_time_complexity_awareness(self, code: str) -> float:
        """Analyze awareness of time complexity"""
        
        complexity_score = 0.0
        
        # Complexity annotations in comments
        complexity_mentions = len(re.findall(r'O\([^)]+\)', code))
        if complexity_mentions > 0:
            complexity_score += 0.4
        
        # Performance-related comments
        perf_comments = len(re.findall(r'(?i)(time|complexity|performance|efficient|optimize|fast)', code))
        if perf_comments > 0:
            complexity_score += min(perf_comments * 0.1, 0.3)
        
        # Early termination patterns
        early_termination = len(re.findall(r'\bbreak\b|\breturn\b.*if', code))
        if early_termination > 0:
            complexity_score += min(early_termination * 0.1, 0.3)
        
        return min(complexity_score, 1.0)
    
    def _analyze_performance_documentation(self, code: str) -> float:
        """Analyze performance-related documentation"""
        
        doc_score = 0.0
        
        # Time/space complexity documentation
        complexity_docs = len(re.findall(r'(?i)(time.*complexity|space.*complexity|big.*o)', code))
        if complexity_docs > 0:
            doc_score += 0.5
        
        # Performance trade-off discussions
        tradeoff_docs = len(re.findall(r'(?i)(trade.*off|vs\.|versus|alternatively)', code))
        if tradeoff_docs > 0:
            doc_score += 0.3
        
        # Algorithm choice justification
        justification_docs = len(re.findall(r'(?i)(because|since|reason|why|chosen)', code))
        if justification_docs > 0:
            doc_score += 0.2
        
        return min(doc_score, 1.0)


# Now the AdvancedCodeEvolutionAnalyzer can be properly initialized:
class AdvancedCodeEvolutionAnalyzer:
    def __init__(self, detector):
        self.detector = detector
        self.quality_analyzers = {
            'structure': StructuralSophisticationAnalyzer(),
            'architecture': ArchitecturalThinkingAnalyzer(),
            'optimization': OptimizationAwarenessAnalyzer(),
            'organization': CodeOrganizationAnalyzer(),
            'error_handling': ErrorHandlingAnalyzer(),
            'abstraction': AbstractionLevelAnalyzer(),
            'maintainability': MaintainabilityAnalyzer(),
            'performance': PerformanceConsciousnessAnalyzer()
        }
    
    # ... rest of the previously defined methods

@dataclass
class CodeQualitySnapshot:
    timestamp: int
    structural_sophistication: float
    architectural_thinking: float
    optimization_awareness: float
    code_organization: float
    error_handling_maturity: float
    abstraction_level: float
    maintainability_score: float
    performance_consciousness: float

class ArchitecturalPattern(Enum):
    MONOLITHIC = "monolithic"
    MODULAR = "modular"
    LAYERED = "layered"
    FUNCTIONAL = "functional"
    OBJECT_ORIENTED = "object_oriented"

class AdvancedCodeEvolutionAnalyzer:
    def __init__(self, detector):
        self.detector = detector
        self.quality_analyzers = {
            'structure': StructuralSophisticationAnalyzer(),
            'architecture': ArchitecturalThinkingAnalyzer(),
            'optimization': OptimizationAwarenessAnalyzer(),
            'organization': CodeOrganizationAnalyzer(),
            'error_handling': ErrorHandlingAnalyzer(),
            'abstraction': AbstractionLevelAnalyzer(),
            'maintainability': MaintainabilityAnalyzer(),
            'performance': PerformanceConsciousnessAnalyzer()
        }
    def classify_trajectory_pattern(self, values: List[float]) -> Dict:
        """Classify the pattern of value changes over time"""
        
        if len(values) < 3:
            return {'pattern': 'INSUFFICIENT_DATA'}
        
        # Calculate first and second derivatives (rate of change and acceleration)
        first_diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        second_diffs = [first_diffs[i+1] - first_diffs[i] for i in range(len(first_diffs)-1)]
        
        avg_first_diff = sum(first_diffs) / len(first_diffs)
        avg_second_diff = sum(second_diffs) / len(second_diffs) if second_diffs else 0
        
        # Classify pattern based on trends
        if avg_first_diff > 0.05:
            if avg_second_diff > 0.02:
                return {
                    'pattern': 'ACCELERATING_GROWTH',
                    'description': 'Rapidly improving with increasing rate',
                    'interpretation': 'Excellent learning acceleration'
                }
            elif avg_second_diff < -0.02:
                return {
                    'pattern': 'DECELERATING_GROWTH', 
                    'description': 'Improving but rate is slowing',
                    'interpretation': 'Good improvement but may be plateauing'
                }
            else:
                return {
                    'pattern': 'LINEAR_GROWTH',
                    'description': 'Steady consistent improvement',
                    'interpretation': 'Reliable, predictable progress'
                }
        elif avg_first_diff < -0.05:
            return {
                'pattern': 'DECLINING',
                'description': 'Performance declining over time',
                'interpretation': 'Concerning - may indicate fatigue or confusion'
            }
        else:
            variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
            if variance < 0.01:
                return {
                    'pattern': 'STABLE',
                    'description': 'Consistent performance level',
                    'interpretation': 'Steady, predictable performance'
                }
            else:
                return {
                    'pattern': 'FLUCTUATING',
                    'description': 'Inconsistent performance with ups and downs',
                    'interpretation': 'Variable performance - may need more time to stabilize'
                }
    
    def identify_breakthrough_moments(self, timeline: List) -> Dict:
        """Identify breakthrough moments in code evolution"""
        
        breakthrough_moments = []
        
        if len(timeline) < 2:
            return {'breakthrough_moments': breakthrough_moments}
        
        # Look for significant jumps in quality
        for i in range(1, len(timeline)):
            prev_snapshot = timeline[i-1]
            curr_snapshot = timeline[i]
            
            # Calculate quality improvement
            prev_quality = prev_snapshot.overall_quality if hasattr(prev_snapshot, 'overall_quality') else 0
            curr_quality = curr_snapshot.overall_quality if hasattr(curr_snapshot, 'overall_quality') else 0
            
            quality_jump = curr_quality - prev_quality
            
            # Significant breakthrough if quality jumps by > 0.3
            if quality_jump > 0.3:
                breakthrough_moments.append({
                    'timestamp': curr_snapshot.timestamp if hasattr(curr_snapshot, 'timestamp') else i,
                    'type': 'QUALITY_BREAKTHROUGH',
                    'improvement': quality_jump,
                    'description': f'Significant quality improvement: +{quality_jump:.2f}'
                })
            
            # Look for sophistication breakthroughs
            prev_soph = prev_snapshot.structural_sophistication if hasattr(prev_snapshot, 'structural_sophistication') else 0
            curr_soph = curr_snapshot.structural_sophistication if hasattr(curr_snapshot, 'structural_sophistication') else 0
            
            soph_jump = curr_soph - prev_soph
            
            if soph_jump > 0.4:
                breakthrough_moments.append({
                    'timestamp': curr_snapshot.timestamp if hasattr(curr_snapshot, 'timestamp') else i,
                    'type': 'ALGORITHMIC_BREAKTHROUGH', 
                    'improvement': soph_jump,
                    'description': f'Major algorithmic improvement: +{soph_jump:.2f}'
                })
        
        return {
            'breakthrough_moments': breakthrough_moments,
            'breakthrough_count': len(breakthrough_moments),
            'breakthrough_frequency': 'HIGH' if len(breakthrough_moments) >= 2 else 'MODERATE' if len(breakthrough_moments) == 1 else 'LOW'
        }
    
    def analyze_regression_patterns(self, timeline: List) -> Dict:
        """Analyze regression patterns in code evolution"""
        
        regression_events = []
        
        if len(timeline) < 2:
            return {'regression_events': regression_events}
        
        # Look for quality drops
        for i in range(1, len(timeline)):
            prev_snapshot = timeline[i-1]
            curr_snapshot = timeline[i]
            
            prev_quality = prev_snapshot.overall_quality if hasattr(prev_snapshot, 'overall_quality') else 0
            curr_quality = curr_snapshot.overall_quality if hasattr(curr_snapshot, 'overall_quality') else 0
            
            quality_drop = prev_quality - curr_quality
            
            # Significant regression if quality drops by > 0.2
            if quality_drop > 0.2:
                regression_events.append({
                    'timestamp': curr_snapshot.timestamp if hasattr(curr_snapshot, 'timestamp') else i,
                    'type': 'QUALITY_REGRESSION',
                    'severity': quality_drop,
                    'description': f'Quality regression: -{quality_drop:.2f}'
                })
        
        # Analyze regression recovery
        recovery_analysis = self._analyze_regression_recovery(regression_events, timeline)
        
        return {
            'regression_events': regression_events,
            'regression_count': len(regression_events),
            'regression_severity': 'HIGH' if any(r['severity'] > 0.4 for r in regression_events) else 'MODERATE' if regression_events else 'LOW',
            'recovery_analysis': recovery_analysis
        }
    
    def _analyze_regression_recovery(self, regression_events: List[Dict], timeline: List) -> Dict:
        """Analyze how well they recover from regressions"""
        
        if not regression_events:
            return {'no_regressions': True}
        
        recovery_scores = []
        
        for regression in regression_events:
            regression_time = regression['timestamp']
            
            # Look for recovery in subsequent snapshots
            recovery_found = False
            for snapshot in timeline:
                if hasattr(snapshot, 'timestamp') and snapshot.timestamp > regression_time:
                    # Check if quality recovered
                    current_quality = snapshot.overall_quality if hasattr(snapshot, 'overall_quality') else 0
                    if current_quality > 0.6:  # Good recovery threshold
                        recovery_scores.append(1.0)
                        recovery_found = True
                        break
            
            if not recovery_found:
                recovery_scores.append(0.0)
        
        avg_recovery = sum(recovery_scores) / len(recovery_scores) if recovery_scores else 0
        
        return {
            'average_recovery_score': avg_recovery,
            'recovery_ability': 'EXCELLENT' if avg_recovery > 0.8 else 'GOOD' if avg_recovery > 0.5 else 'POOR'
        }

    def analyze_structural_sophistication(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Analyze sophistication of code structures used"""
        
        # Use the structure analyzer instead of calling method on self
        return self.quality_analyzers['structure'].analyze_structural_sophistication(code, pattern_scores)
    
    def analyze_architectural_thinking(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Analyze evidence of architectural thinking"""
        
        return self.quality_analyzers['architecture'].analyze_architectural_thinking(code, pattern_scores)
    
    def analyze_optimization_awareness(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Analyze awareness of optimization opportunities"""
        
        return self.quality_analyzers['optimization'].analyze_optimization_awareness(code, pattern_scores)
    
    def analyze_code_organization(self, code: str) -> float:
        """Analyze how well code is organized"""
        
        return self.quality_analyzers['organization'].analyze_code_organization(code)
    
    def analyze_error_handling_maturity(self, code: str) -> float:
        """Analyze sophistication of error handling"""
        
        return self.quality_analyzers['error_handling'].analyze_error_handling_maturity(code)
    
    def analyze_abstraction_level(self, code: str) -> float:
        """Analyze appropriate use of abstraction levels"""
        
        abstraction_results = self.quality_analyzers['abstraction'].analyze_abstraction_level(code)
        
        # Return overall abstraction score
        if 'parse_error' in abstraction_results:
            return 0.0
        
        return sum(abstraction_results.values()) / len(abstraction_results)
    
    def analyze_maintainability(self, code: str) -> float:
        """Analyze maintainability factors"""
        
        maintainability_results = self.quality_analyzers['maintainability'].analyze_maintainability(code)
        return maintainability_results.get('overall_maintainability', 0.0)
    
    def analyze_performance_consciousness(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Analyze consciousness of performance implications"""
        
        performance_results = self.quality_analyzers['performance'].analyze_performance_consciousness(code, pattern_scores)
        return performance_results.get('overall_performance_consciousness', 0.0)

    
    def analyze_code_evolution_sophistication(self, code_versions_with_timestamps: List[Tuple[int, str]]) -> Dict:
        """Comprehensive analysis of how code sophistication evolves"""
        
        evolution_timeline = []
        
        for timestamp, code in code_versions_with_timestamps:
            quality_snapshot = self.create_quality_snapshot(timestamp, code)
            evolution_timeline.append(quality_snapshot)
        
        return self.analyze_evolution_patterns(evolution_timeline)
    
    def create_quality_snapshot(self, timestamp: int, code: str) -> CodeQualitySnapshot:
        """Create comprehensive quality snapshot at specific moment"""
        
        # Get basic pattern scores
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # Analyze each quality dimension using the quality analyzers
        structural_sophistication = self.analyze_structural_sophistication(code, pattern_scores)
        architectural_thinking = self.analyze_architectural_thinking(code, pattern_scores)
        optimization_awareness = self.analyze_optimization_awareness(code, pattern_scores)
        code_organization = self.analyze_code_organization(code)
        error_handling_maturity = self.analyze_error_handling_maturity(code)
        abstraction_level = self.analyze_abstraction_level(code)
        maintainability_score = self.analyze_maintainability(code)
        performance_consciousness = self.analyze_performance_consciousness(code, pattern_scores)
        
        # Extract numeric values if results are dictionaries
        if isinstance(structural_sophistication, dict):
            structural_sophistication = structural_sophistication.get('overall_sophistication', 0.0)
        if isinstance(architectural_thinking, dict):
            architectural_thinking = architectural_thinking.get('overall_architectural_thinking', 0.0)
        if isinstance(optimization_awareness, dict):
            optimization_awareness = optimization_awareness.get('overall_optimization_consciousness', 0.0)
        if isinstance(code_organization, dict):
            code_organization = code_organization.get('overall_organization', 0.0)
        if isinstance(error_handling_maturity, dict):
            error_handling_maturity = error_handling_maturity.get('overall_error_handling_maturity', 0.0)
        
        return CodeQualitySnapshot(
            timestamp=timestamp,
            structural_sophistication=structural_sophistication,
            architectural_thinking=architectural_thinking,
            optimization_awareness=optimization_awareness,
            code_organization=code_organization,
            error_handling_maturity=error_handling_maturity,
            abstraction_level=abstraction_level,
            maintainability_score=maintainability_score,
            performance_consciousness=performance_consciousness
        )

    
    
    
    def analyze_evolution_patterns(self, timeline: List[CodeQualitySnapshot]) -> Dict:
        """Analyze patterns in code evolution"""
        
        evolution_analysis = {}
        
        # 1. QUALITY TRAJECTORY ANALYSIS
        trajectory_analysis = self.analyze_quality_trajectories(timeline)
        evolution_analysis['quality_trajectories'] = trajectory_analysis
        
        # 2. LEARNING VELOCITY ANALYSIS
        learning_velocity = self.analyze_learning_velocity(timeline)
        evolution_analysis['learning_velocity'] = learning_velocity
        
        # 3. SOPHISTICATION GROWTH PATTERNS
        sophistication_growth = self.analyze_sophistication_growth_patterns(timeline)
        evolution_analysis['sophistication_growth'] = sophistication_growth
        
        # 4. ARCHITECTURAL MATURITY DEVELOPMENT
        architectural_development = self.analyze_architectural_development(timeline)
        evolution_analysis['architectural_development'] = architectural_development
        
        # 5. OPTIMIZATION CONSCIOUSNESS EVOLUTION
        optimization_evolution = self.analyze_optimization_consciousness_evolution(timeline)
        evolution_analysis['optimization_evolution'] = optimization_evolution
        
        # 6. CODE QUALITY STABILITY
        quality_stability = self.analyze_quality_stability(timeline)
        evolution_analysis['quality_stability'] = quality_stability
        
        # 7. BREAKTHROUGH MOMENTS
        breakthrough_analysis = self.identify_breakthrough_moments(timeline)
        evolution_analysis['breakthrough_moments'] = breakthrough_analysis
        
        # 8. REGRESSION PATTERNS
        regression_analysis = self.analyze_regression_patterns(timeline)
        evolution_analysis['regression_patterns'] = regression_analysis
        
        return evolution_analysis
    
    def analyze_quality_trajectories(self, timeline: List[CodeQualitySnapshot]) -> Dict:
        """Analyze trajectories of different quality dimensions"""
        
        trajectories = {}
        
        quality_dimensions = [
            'structural_sophistication',
            'architectural_thinking', 
            'optimization_awareness',
            'code_organization',
            'error_handling_maturity',
            'abstraction_level',
            'maintainability_score',
            'performance_consciousness'
        ]
        
        for dimension in quality_dimensions:
            values = [getattr(snapshot, dimension) for snapshot in timeline]
            
            if len(values) >= 3:
                # Calculate trend
                trend = np.polyfit(range(len(values)), values, 1)[0]
                
                # Calculate consistency
                variance = np.var(values)
                
                # Analyze trajectory pattern
                trajectory_pattern = self.classify_trajectory_pattern(values)
                
                trajectories[dimension] = {
                    'trend': trend,
                    'variance': variance,
                    'pattern': trajectory_pattern,
                    'final_value': values[-1],
                    'improvement': values[-1] - values[0],
                    'peak_value': max(values),
                    'consistency': 1.0 - variance  # Higher variance = lower consistency
                }
        
        return trajectories
    
    def analyze_learning_velocity(self, timeline: List[CodeQualitySnapshot]) -> Dict:
        """Analyze how quickly they learn and improve"""
        
        if len(timeline) < 3:
            return {'insufficient_data': True}
        
        # Calculate improvement rates for different dimensions
        improvement_rates = {}
        
        dimensions = ['structural_sophistication', 'architectural_thinking', 'optimization_awareness']
        
        for dimension in dimensions:
            values = [getattr(snapshot, dimension) for snapshot in timeline]
            
            # Calculate rate of improvement
            improvements = []
            for i in range(1, len(values)):
                improvement = values[i] - values[i-1]
                improvements.append(improvement)
            
            avg_improvement_rate = sum(improvements) / len(improvements) if improvements else 0
            improvement_rates[dimension] = avg_improvement_rate
        
        # Overall learning velocity
        overall_velocity = sum(improvement_rates.values()) / len(improvement_rates)
        
        # Learning acceleration (is learning speed increasing?)
        if len(timeline) >= 4:
            early_improvements = improvement_rates  # Simplified for this example
            # In reality, you'd calculate improvement rates for early vs late periods
            learning_acceleration = 0  # Placeholder
        else:
            learning_acceleration = 0
        
        return {
            'overall_velocity': overall_velocity,
            'dimension_velocities': improvement_rates,
            'learning_acceleration': learning_acceleration,
            'velocity_classification': self.classify_learning_velocity(overall_velocity)
        }
    
    def classify_learning_velocity(self, velocity: float) -> Dict:
        """Classify learning velocity"""
        
        if velocity > 0.1:
            return {
                'classification': 'RAPID_LEARNER',
                'description': 'Shows rapid improvement during interview',
                'job_implication': 'Will quickly master new technologies and domains',
                'team_value': 'High mentoring ROI, can tackle new challenges',
                'career_trajectory': 'Fast promotion potential'
            }
        elif velocity > 0.05:
            return {
                'classification': 'STEADY_LEARNER',
                'description': 'Shows consistent improvement',
                'job_implication': 'Reliable improvement over time',
                'team_value': 'Steady contributor, good for long-term projects'
            }
        elif velocity > 0:
            return {
                'classification': 'SLOW_LEARNER',
                'description': 'Shows gradual improvement',
                'job_implication': 'Needs time and support to develop',
                'team_value': 'Requires patient mentoring'
            }
        else:
            return {
                'classification': 'STAGNANT',
                'description': 'No clear improvement during interview',
                'job_implication': 'May struggle with learning new concepts',
                'concern': 'Limited growth potential'
            }

def run_comprehensive_test():
    """Run the cognitive load and code evolution analysis"""
    
   # from cognitive_load_tracker import AdvancedCognitiveLoadTracker
   # from code_evolution_analyzer import AdvancedCodeEvolutionAnalyzer
    from compcheck import CompleteEnhancedDSAPatternDetector
    # Initialize analyzers
    detector = CompleteEnhancedDSAPatternDetector()  # Your main detector
    cognitive_tracker = AdvancedCognitiveLoadTracker(detector)
    evolution_analyzer = AdvancedCodeEvolutionAnalyzer(detector)
    
    # Test Case 1: Two Sum Problem
    print("=== ANALYZING TWO SUM PROBLEM ===")
    
    # Prepare code timeline for cognitive analysis
    code_snapshots = [(entry['timestamp'], entry['code']) for entry in USER_CODE_TIMELINE]
    
    # Run cognitive load analysis
    cognitive_results = cognitive_tracker.track_cognitive_evolution(
        code_snapshots, 
        HINT_INTERACTIONS
    )
    
    print("Cognitive Load Analysis Results:")
    print(f"Overall Cognitive Pattern: {cognitive_results['overall_profile']}")
    print(f"Stress Resilience: {cognitive_results['stress_resilience']}")
    print(f"Learning Progression: {cognitive_results['learning_progression']}")
    
    # Run code evolution analysis
    evolution_results = evolution_analyzer.analyze_code_evolution_sophistication(code_snapshots)
    
    print("\nCode Evolution Analysis Results:")
    print(f"Learning Velocity: {evolution_results['learning_velocity']}")
    print(f"Quality Trajectories: {evolution_results['quality_trajectories']}")
    print(f"Breakthrough Moments: {evolution_results['breakthrough_moments']}")
    
    # Compare against reference solutions
    final_user_code = USER_CODE_TIMELINE[-1]['code']
    
    print("\n=== APPROACH SIMILARITY ANALYSIS ===")
    for approach_name, reference_code in REFERENCE_SOLUTIONS.items():
        user_scores = detector.get_all_pattern_scores(final_user_code)
        ref_scores = detector.get_all_pattern_scores(reference_code)
        
        # Calculate similarity
        similarity = calculate_approach_similarity(user_scores, ref_scores)
        print(f"Similarity to {approach_name}: {similarity:.2f}")
    
    # Test Case 2: LIS Problem (more complex)
    print("\n\n=== ANALYZING LIS PROBLEM ===")
    
    lis_snapshots = [(entry['timestamp'], entry['code']) for entry in LIS_USER_TIMELINE]
    
    lis_cognitive_results = cognitive_tracker.track_cognitive_evolution(lis_snapshots)
    lis_evolution_results = evolution_analyzer.analyze_code_evolution_sophistication(lis_snapshots)
    
    print("LIS Cognitive Analysis:")
    print(f"Approach Evolution: {lis_cognitive_results.get('approach_evolution', {})}")
    print(f"Paradigm Shifts: {lis_evolution_results.get('breakthrough_moments', {})}")
    
    return {
        'two_sum': {
            'cognitive': cognitive_results,
            'evolution': evolution_results
        },
        'lis': {
            'cognitive': lis_cognitive_results,
            'evolution': lis_evolution_results
        }
    }

def calculate_approach_similarity(user_scores, ref_scores):
    """Calculate similarity between user's approach and reference solution"""
    
    # Weighted similarity based on pattern importance
    important_patterns = [
        'hash_map', 'two_pointers', 'dynamic_programming', 'recursion', 
        'memoization', 'binary_search', 'array_manipulation'
    ]
    
    similarity = 0.0
    total_weight = 0.0
    
    for pattern in important_patterns:
        user_score = user_scores.get(pattern, 0)
        ref_score = ref_scores.get(pattern, 0)
        
        # Calculate pattern similarity (1 - absolute difference)
        pattern_similarity = 1.0 - abs(user_score - ref_score)
        
        # Weight by pattern importance and strength
        weight = max(user_score, ref_score)
        
        similarity += pattern_similarity * weight
        total_weight += weight
    
    return similarity / total_weight if total_weight > 0 else 0.0

# Example usage with keystroke analysis
def analyze_keystroke_patterns(timeline):
    """Analyze typing patterns for cognitive insights"""
    
    keystroke_insights = {}
    
    # Extract keystroke data
    typing_speeds = [entry['keystroke_data']['typing_speed_wpm'] for entry in timeline]
    thinking_pauses = [entry['keystroke_data']['thinking_pause_after_ms'] for entry in timeline]
    backspace_counts = [entry['keystroke_data']['backspace_count'] for entry in timeline]
    
    # Analyze patterns
    keystroke_insights['typing_speed_trend'] = np.polyfit(range(len(typing_speeds)), typing_speeds, 1)[0]
    keystroke_insights['avg_thinking_pause'] = sum(thinking_pauses) / len(thinking_pauses)
    keystroke_insights['total_corrections'] = sum(backspace_counts)
    keystroke_insights['correction_frequency'] = keystroke_insights['total_corrections'] / len(timeline)
    
    # Confidence indicators
    if keystroke_insights['typing_speed_trend'] > 5:  # Increasing speed
        keystroke_insights['confidence_trend'] = 'INCREASING - Getting more confident'
    elif keystroke_insights['typing_speed_trend'] < -5:  # Decreasing speed
        keystroke_insights['confidence_trend'] = 'DECREASING - Becoming more uncertain'
    else:
        keystroke_insights['confidence_trend'] = 'STABLE - Consistent confidence level'
    
    return keystroke_insights

if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Analyze keystroke patterns
    keystroke_analysis = analyze_keystroke_patterns(USER_CODE_TIMELINE)
    print(f"\nKeystroke Analysis: {keystroke_analysis}")