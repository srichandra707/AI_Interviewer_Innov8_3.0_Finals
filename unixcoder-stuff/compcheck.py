import ast
import re
from typing import Dict, List, Any, Set, Optional, Tuple
import textwrap

import ast
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque, defaultdict, Counter
from typing import Dict, List, Any
import time, threading, numpy as np
import ast, re

_real_ast_parse = ast.parse  # keep original reference

def safe_ast_parse(code, *args, **kwargs):
    """
    A wrapper around ast.parse that handles:
    - incomplete code blocks
    - multi-line expressions like lists, dicts, tuples
    - syntax errors in some lines
    """
    try:
        return _real_ast_parse(code, *args, **kwargs)
    except SyntaxError:
        # 1. Wrap in a dummy function (to handle partial indented blocks)
        wrapped = "def __dummy__():\n" + "\n".join("    " + line for line in code.splitlines())
        try:
            return _real_ast_parse(wrapped, *args, **kwargs)
        except SyntaxError:
            # 2. Try to fix incomplete multi-line expressions
            fixed_lines = []
            buffer = []
            parens = 0
            for line in code.splitlines():
                buffer.append(line)
                # Count brackets and parens
                parens += line.count('(') + line.count('[') + line.count('{')
                parens -= line.count(')') + line.count(']') + line.count('}')
                if parens <= 0:
                    # attempt to parse current buffer
                    try:
                        _real_ast_parse("\n".join(buffer), *args, **kwargs)
                        fixed_lines.extend(buffer)
                    except SyntaxError:
                        # add a pass to incomplete line
                        fixed_lines.append(buffer[-1] + "  # incomplete")
                    buffer = []
            # 3. Parse whatever we collected
            module = ast.Module(body=[], type_ignores=[])
            for line in fixed_lines:
                try:
                    tree = _real_ast_parse(line, *args, **kwargs)
                    module.body.extend(tree.body)
                except SyntaxError:
                    continue
            return module

# Monkey-patch ast.parse globally
ast.parse = safe_ast_parse


class ASTAnalyzer:
    """Helper class for AST-based pattern detection"""
    
    def __init__(self):
        self.function_calls = []
        self.variable_names = set()
        self.loop_structures = []
        self.conditional_structures = []
        self.assignments = []
        self.imports = []
    
    def analyze_tree(self, tree: ast.AST):
        """Comprehensive AST analysis"""
        self.function_calls = []
        self.variable_names = set()
        self.loop_structures = []
        self.conditional_structures = []
        self.assignments = []
        self.imports = []
        
        for node in ast.walk(tree):
            self._process_node(node)
    
    def _process_node(self, node):
        """Process individual AST nodes"""
        if isinstance(node, ast.Call):
            self._process_function_call(node)
        elif isinstance(node, ast.Name):
            self.variable_names.add(node.id)
        elif isinstance(node, (ast.For, ast.While)):
            self._process_loop(node)
        elif isinstance(node, ast.If):
            self._process_conditional(node)
        elif isinstance(node, ast.Assign):
            self._process_assignment(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._process_import(node)
    
    def _process_function_call(self, node: ast.Call):
        """Extract function call information"""
        call_info = {
            'node': node,
            'function_name': None,
            'attribute_chain': [],
            'args': len(node.args),
            'keywords': [kw.arg for kw in node.keywords]
        }
        
        if isinstance(node.func, ast.Name):
            call_info['function_name'] = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_info['attribute_chain'] = self._get_attribute_chain(node.func)
        
        self.function_calls.append(call_info)
    
    def _get_attribute_chain(self, node: ast.Attribute) -> List[str]:
        """Get full attribute chain like obj.method1.method2"""
        chain = [node.attr]
        current = node.value
        
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            chain.append(current.id)
        
        return list(reversed(chain))
    
    def _process_loop(self, node):
        """Analyze loop structures"""
        loop_info = {
            'type': type(node).__name__,
            'node': node,
            'nested_loops': self._count_nested_loops(node),
            'has_break': self._has_break_continue(node, ast.Break),
            'has_continue': self._has_break_continue(node, ast.Continue)
        }
        
        if isinstance(node, ast.For):
            loop_info['iterator'] = self._get_iterator_info(node)
        elif isinstance(node, ast.While):
            loop_info['condition'] = self._analyze_condition(node.test)
        
        self.loop_structures.append(loop_info)
    
    def _count_nested_loops(self, node) -> int:
        """Count nested loops within a loop"""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)) and child != node:
                count += 1
        return count
    
    def _has_break_continue(self, node, stmt_type) -> bool:
        """Check if loop contains break/continue statements"""
        for child in ast.walk(node):
            if isinstance(child, stmt_type):
                return True
        return False
    
    def _get_iterator_info(self, for_node: ast.For) -> Dict:
        """Analyze for loop iterator"""
        info = {'target': None, 'iter_type': None, 'range_info': None}
        
        if isinstance(for_node.target, ast.Name):
            info['target'] = for_node.target.id
        
        if isinstance(for_node.iter, ast.Call):
            if isinstance(for_node.iter.func, ast.Name):
                if for_node.iter.func.id == 'range':
                    info['iter_type'] = 'range'
                    info['range_info'] = self._analyze_range_call(for_node.iter)
                elif for_node.iter.func.id == 'enumerate':
                    info['iter_type'] = 'enumerate'
        
        return info
    
    def _analyze_range_call(self, range_call: ast.Call) -> Dict:
        """Analyze range() function parameters"""
        args = range_call.args
        info = {'start': None, 'stop': None, 'step': None}
        
        if len(args) == 1:
            info['stop'] = self._extract_value(args[0])
        elif len(args) == 2:
            info['start'] = self._extract_value(args[0])
            info['stop'] = self._extract_value(args[1])
        elif len(args) == 3:
            info['start'] = self._extract_value(args[0])
            info['stop'] = self._extract_value(args[1])
            info['step'] = self._extract_value(args[2])
        
        return info
    
    def _extract_value(self, node) -> Any:
        """Extract value from AST node"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return f"var:{node.id}"
        elif isinstance(node, ast.BinOp):
            return f"expr:{ast.unparse(node)}"
        return None
    
    def _analyze_condition(self, condition_node) -> Dict:
        """Analyze conditional expressions"""
        info = {'type': type(condition_node).__name__, 'operators': []}
        
        for node in ast.walk(condition_node):
            if isinstance(node, ast.Compare):
                info['operators'].extend([type(op).__name__ for op in node.ops])
            elif isinstance(node, (ast.And, ast.Or)):
                info['operators'].append(type(node).__name__)
        
        return info

    def _process_conditional(self, node: ast.If):
        """Analyze if/elif/else conditionals"""
        cond_info = {
            'node': node,
            'condition': self._analyze_condition(node.test),
            'has_else': node.orelse != [],
            'nested_ifs': sum(isinstance(n, ast.If) for n in ast.walk(node))
        }
        self.conditional_structures.append(cond_info)

    def _process_import(self, node):
        """Analyze import statements"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports.append({'type': 'import', 'module': alias.name, 'asname': alias.asname})
        elif isinstance(node, ast.ImportFrom):
            self.imports.append({
                'type': 'from',
                'module': node.module,
                'level': node.level,
                'names': [(alias.name, alias.asname) for alias in node.names]
            })

    def _process_assignment(self, node: ast.Assign):
        """Analyze assignment statements"""
        targets = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.append(t.id)
            elif isinstance(t, ast.Tuple):
                targets.extend([e.id for e in t.elts if isinstance(e, ast.Name)])
        self.assignments.append({
            'targets': targets,
            'value': self._extract_value(node.value)
        })

class EnhancedDSAPatternDetector:
    """Enhanced detector with comprehensive AST + Regex analysis"""
    
    def __init__(self):
        self.ast_analyzer = ASTAnalyzer()
      #  ast_tree = ast.parse(code)
      #  score = 0.0
       # self.ast_analyzer.analyze_tree(ast_tree)

    
    # =================== ENHANCED DETECTION FUNCTIONS ===================
    
    def detect_recursion(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced recursion detection with AST analysis"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST-based analysis
        function_definitions = {}
        recursive_calls = set()
        
        # Collect function definitions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                function_definitions[node.name] = node
        
        # Check for recursive calls within each function
        for func_name, func_node in function_definitions.items():
            for call_info in self.ast_analyzer.function_calls:
                if (call_info['function_name'] == func_name and 
                    self._is_call_within_function(call_info['node'], func_node)):
                    recursive_calls.add(func_name)
                    score += 0.8
        
        # Regex backup for edge cases
        if re.search(r'def\s+(\w+).*:\s*.*\1\s*\(', code, re.DOTALL):
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_two_pointers(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced two pointers detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST Analysis: Look for two pointer patterns
        pointer_vars = self._identify_pointer_variables()
        if len(pointer_vars) >= 2:
            score += 0.4
        
        # Check for convergence pattern in while loops
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'While':
                condition = loop_info['condition']
                if 'Lt' in condition['operators'] or 'Le' in condition['operators']:
                    score += 0.3
        
        # Check for pointer movement patterns
        movement_score = self._detect_pointer_movement_patterns(ast_tree)
        score += movement_score
        
        # Regex patterns as backup
        if re.search(r'\b(left|start).*\b(right|end)', code):
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_binary_search(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced binary search detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST Analysis: Look for binary search structure
        mid_calculations = self._find_mid_calculations(ast_tree)
        if mid_calculations:
            score += 0.5
        
        # Check for while loop with comparison
        binary_search_loops = self._identify_binary_search_loops()
        if binary_search_loops:
            score += 0.4
        
        # Check for built-in binary search
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['bisect_left', 'bisect_right', 'bisect']:
                score += 0.8
        
        # Import analysis
        if any('bisect' in imp for imp in self.ast_analyzer.imports):
            score += 0.3
        
        return min(score, 1.0)
    
    def detect_dynamic_programming(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced DP detection combining memoization and tabulation"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for memoization patterns
        memo_score = self._detect_memoization_ast(ast_tree)
        score += memo_score * 0.5
        
        # Check for tabulation patterns
        tabulation_score = self._detect_tabulation_ast(ast_tree)
        score += tabulation_score * 0.5
        
        # Check for DP array initialization
        dp_arrays = self._find_dp_array_initialization(ast_tree)
        if dp_arrays:
            score += 0.3
        
        return min(score, 1.0)
    
    def detect_graph_dfs(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced DFS detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for recursive DFS pattern
        if self._has_recursive_neighbor_traversal(ast_tree):
            score += 0.6
        
        # Check for iterative DFS with stack
        if self._has_stack_based_traversal(ast_tree):
            score += 0.5
        
        # Check for visited set usage
        if self._has_visited_set_pattern(ast_tree):
            score += 0.3
        
        # Function naming patterns
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] and 'dfs' in call_info['function_name'].lower():
                score += 0.2
        
        return min(score, 1.0)
    
    def detect_heap_usage(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced heap detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for heapq imports
        heapq_imported = any('heapq' in imp for imp in self.ast_analyzer.imports)
        if heapq_imported:
            score += 0.4
        
        # Check for heap operations
        heap_operations = ['heappush', 'heappop', 'heapify', 'nlargest', 'nsmallest']
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in heap_operations:
                score += 0.3
            elif (call_info['attribute_chain'] and 
                  len(call_info['attribute_chain']) >= 2 and
                  call_info['attribute_chain'][0] == 'heapq'):
                score += 0.3
        
        return min(score, 1.0)
    
    def detect_sliding_window(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced sliding window detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for window size variables
        window_vars = [var for var in self.ast_analyzer.variable_names 
                      if 'window' in var.lower() or var in ['k', 'size']]
        if window_vars:
            score += 0.3
        
        # Check for sliding mechanism in loops
        sliding_pattern = self._detect_sliding_mechanism(ast_tree)
        score += sliding_pattern
        
        # Check for window expansion/contraction
        expansion_pattern = self._detect_window_expansion(ast_tree)
        score += expansion_pattern
        
        return min(score, 1.0)
    
    # =================== HELPER METHODS FOR AST ANALYSIS ===================
    
    def _is_call_within_function(self, call_node: ast.Call, func_node: ast.FunctionDef) -> bool:
        """Check if a function call is within a specific function definition"""
        for node in ast.walk(func_node):
            if node is call_node:
                return True
        return False
    
    def _identify_pointer_variables(self) -> List[str]:
        """Identify variables that could be pointers based on naming"""
        pointer_patterns = ['left', 'right', 'start', 'end', 'low', 'high', 'begin', 'finish']
        identified = []
        
        for var in self.ast_analyzer.variable_names:
            if any(pattern in var.lower() for pattern in pointer_patterns):
                identified.append(var)
        
        return identified
    
    def _detect_pointer_movement_patterns(self, ast_tree: ast.AST) -> float:
        """Detect pointer increment/decrement patterns"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    var_name = node.target.id
                    if (var_name in ['left', 'start', 'i'] and 
                        isinstance(node.op, ast.Add)):
                        score += 0.2
                    elif (var_name in ['right', 'end', 'j'] and 
                          isinstance(node.op, ast.Sub)):
                        score += 0.2
        
        return min(score, 0.4)
    
    def _find_mid_calculations(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find mid-point calculations typical in binary search"""
        mid_calculations = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and 
                        target.id == 'mid' and
                        isinstance(node.value, ast.BinOp)):
                        # Check if it's a division by 2
                        if (isinstance(node.value.op, ast.FloorDiv) and
                            isinstance(node.value.right, ast.Constant) and
                            node.value.right.value == 2):
                            mid_calculations.append(node)
        
        return mid_calculations
    
    def _identify_binary_search_loops(self) -> List[Dict]:
        """Identify while loops that match binary search pattern"""
        binary_loops = []
        
        for loop_info in self.ast_analyzer.loop_structures:
            if (loop_info['type'] == 'While' and
                'Lt' in loop_info['condition']['operators']):
                binary_loops.append(loop_info)
        
        return binary_loops
    
    def _detect_memoization_ast(self, ast_tree: ast.AST) -> float:
        """Detect memoization using AST analysis"""
        score = 0.0
        
        # Check for decorator usage
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Name) and 
                        decorator.id in ['cache', 'lru_cache']):
                        score += 0.8
                    elif (isinstance(decorator, ast.Call) and
                          isinstance(decorator.func, ast.Name) and
                          decorator.func.id in ['cache', 'lru_cache']):
                        score += 0.8
        
        # Check for manual memoization
        memo_checks = self._find_memo_checks(ast_tree)
        if memo_checks:
            score += 0.5
        
        return min(score, 1.0)
    
    def _find_memo_checks(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find manual memoization check patterns"""
        memo_checks = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.If):
                # Look for "if key in memo" patterns
                if isinstance(node.test, ast.Compare):
                    if (len(node.test.ops) == 1 and
                        isinstance(node.test.ops[0], ast.In)):
                        # Check if comparing with memo/cache variable
                        for comparator in node.test.comparators:
                            if (isinstance(comparator, ast.Name) and
                                comparator.id in ['memo', 'cache', 'dp']):
                                memo_checks.append(node)
        
        return memo_checks
    
    def _detect_tabulation_ast(self, ast_tree: ast.AST) -> float:
        """Detect tabulation (bottom-up DP) using AST"""
        score = 0.0
        
        # Look for DP array assignments within nested loops
        nested_dp_updates = self._find_nested_dp_updates(ast_tree)
        if nested_dp_updates:
            score += 0.6
        
        # Look for base case initialization
        base_case_init = self._find_base_case_initialization(ast_tree)
        if base_case_init:
            score += 0.3
        
        return min(score, 1.0)
    
    def _find_dp_array_initialization(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find DP array initialization patterns"""
        dp_inits = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name) and 
                        target.id in ['dp', 'table', 'memo']):
                        # Check if initializing with list
                        if isinstance(node.value, (ast.List, ast.ListComp)):
                            dp_inits.append(node)
        
        return dp_inits
    
    def _has_recursive_neighbor_traversal(self, ast_tree: ast.AST) -> bool:
        """Check for recursive neighbor traversal pattern"""
        # Look for patterns like: for neighbor in graph[node]: dfs(neighbor)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.For):
                # Check if iterating over neighbors/adjacency
                if isinstance(node.iter, ast.Subscript):
                    for call_node in ast.walk(node):
                        if (isinstance(call_node, ast.Call) and
                            isinstance(call_node.func, ast.Name) and
                            'dfs' in call_node.func.id.lower()):
                            return True
        return False
    
    def _has_stack_based_traversal(self, ast_tree: ast.AST) -> bool:
        """Check for stack-based DFS traversal"""
        has_stack = False
        has_append_pop = False
        
        # Check for stack variable
        for var in self.ast_analyzer.variable_names:
            if 'stack' in var.lower():
                has_stack = True
                break
        
        # Check for stack.append and stack.pop pattern
        for call_info in self.ast_analyzer.function_calls:
            if (call_info['attribute_chain'] and 
                len(call_info['attribute_chain']) >= 2):
                obj, method = call_info['attribute_chain'][0], call_info['attribute_chain'][1]
                if 'stack' in obj.lower() and method in ['append', 'pop']:
                    has_append_pop = True
        
        return has_stack and has_append_pop
    
    def _has_visited_set_pattern(self, ast_tree: ast.AST) -> bool:
        """Check for visited set usage pattern"""
        has_visited = any('visited' in var.lower() for var in self.ast_analyzer.variable_names)
        
        # Check for visited.add() calls
        has_add_operation = False
        for call_info in self.ast_analyzer.function_calls:
            if (call_info['attribute_chain'] and 
                len(call_info['attribute_chain']) >= 2):
                obj, method = call_info['attribute_chain'][0], call_info['attribute_chain'][1]
                if 'visited' in obj.lower() and method == 'add':
                    has_add_operation = True
        
        return has_visited and has_add_operation
    
    def _detect_sliding_mechanism(self, ast_tree: ast.AST) -> float:
        """Detect sliding window mechanism in loops"""
        score = 0.0
        
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                # Check if loop modifies window bounds
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.AugAssign):
                        if isinstance(node.target, ast.Name):
                            var_name = node.target.id
                            if any(window_var in var_name.lower() 
                                  for window_var in ['window', 'sum', 'count']):
                                score += 0.3
        
        return min(score, 0.4)
    
    def _detect_window_expansion(self, ast_tree: ast.AST) -> float:
        """Detect window expansion/contraction pattern"""
        score = 0.0
        
        # Look for while loops that expand/contract windows
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'While':
                # Check for window size conditions
                condition = loop_info['condition']
                if 'Gt' in condition['operators'] or 'Lt' in condition['operators']:
                    score += 0.3
        
        return min(score, 0.3)
    
    def _find_nested_dp_updates(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find DP updates within nested loops"""
        nested_updates = []
        
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['nested_loops'] > 0:
                # Look for DP array updates within this loop
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (isinstance(target, ast.Subscript) and
                                isinstance(target.value, ast.Name) and
                                target.value.id in ['dp', 'table']):
                                nested_updates.append(node)
        
        return nested_updates
    
    def _find_base_case_initialization(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find base case initialization for DP"""
        base_cases = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        # Check for dp[0] = ... or dp[i][0] = ... patterns
                        if (isinstance(target.slice, ast.Constant) and
                            target.slice.value == 0):
                            base_cases.append(node)
        
        return base_cases

    def get_all_pattern_scores(self, code: str) -> Dict[str, float]:
        """Get comprehensive pattern scores using both AST and regex"""
        try:
            ast_tree = ast.parse(code)
        except:
            ast_tree = None
        
        scores = {
            'recursion': self.detect_recursion(code, ast_tree),
            'two_pointers': self.detect_two_pointers(code, ast_tree),
            'binary_search': self.detect_binary_search(code, ast_tree),
            'dynamic_programming': self.detect_dynamic_programming(code, ast_tree),
            'graph_dfs': self.detect_graph_dfs(code, ast_tree),
            'heap_usage': self.detect_heap_usage(code, ast_tree),
            'sliding_window': self.detect_sliding_window(code, ast_tree),
            # Add all other enhanced detection functions...
        }
        
        return scores


class CompleteEnhancedDSAPatternDetector(EnhancedDSAPatternDetector):
    """Complete implementation with all detection functions enhanced"""
    
    def __init__(self):
        super().__init__()
        #self.ast_analyzer = ast_analyzer
        self.detection_functions = {
            # Basic Patterns
            "recursion": self.detect_recursion,
            "iteration": self.detect_iteration,
            "nested_loops": self.detect_nested_loops,
            "conditional_logic": self.detect_conditional_logic,
            
            # Dynamic Programming
            "memoization": self.detect_memoization,
            "top_down_dp": self.detect_top_down_dp,
            "bottom_up_dp": self.detect_bottom_up_dp,
            "dp_optimization": self.detect_dp_optimization,
            
            # Search & Divide Patterns
            "binary_search": self.detect_binary_search,
            "ternary_search": self.detect_ternary_search,
            "linear_search": self.detect_linear_search,
            "divide_conquer": self.detect_divide_conquer,
            
            # Two Pointer & Window Patterns
            "two_pointers": self.detect_two_pointers,
            "sliding_window": self.detect_sliding_window,
            "fast_slow_pointers": self.detect_fast_slow_pointers,
            "prefix_suffix": self.detect_prefix_suffix,
            
            # Data Structure Usage
            "stack_usage": self.detect_stack_usage,
            "queue_usage": self.detect_queue_usage,
            "deque_usage": self.detect_deque_usage,
            "heap_usage": self.detect_heap_usage,
            "priority_queue": self.detect_priority_queue,
            "hash_map": self.detect_hash_map,
            "hash_set": self.detect_hash_set,
            "array_manipulation": self.detect_array_manipulation,
            "string_operations": self.detect_string_operations,
            
            # Graph Algorithms
            "graph_dfs": self.detect_graph_dfs,
            "graph_bfs": self.detect_graph_bfs,
            "topological_sort": self.detect_topological_sort,
            "shortest_path": self.detect_shortest_path,
            "union_find": self.detect_union_find,
            "minimum_spanning_tree": self.detect_minimum_spanning_tree,
            "graph_coloring": self.detect_graph_coloring,
            
            # Tree Algorithms
            "tree_traversal": self.detect_tree_traversal,
            "binary_tree_operations": self.detect_binary_tree_operations,
            "trie_usage": self.detect_trie_usage,
            "segment_tree": self.detect_segment_tree,
            "fenwick_tree": self.detect_fenwick_tree,
            "tree_dp": self.detect_tree_dp,
            
            # Advanced Data Structures
            "monotonic_stack": self.detect_monotonic_stack,
            "monotonic_queue": self.detect_monotonic_queue,
            "coordinate_compression": self.detect_coordinate_compression,
            "sparse_table": self.detect_sparse_table,
            "disjoint_set_union": self.detect_disjoint_set_union,
            
            # Mathematical Patterns
            "number_theory": self.detect_number_theory,
            "combinatorics": self.detect_combinatorics,
            "probability": self.detect_probability,
            "geometry": self.detect_geometry,
            "bit_manipulation": self.detect_bit_manipulation,
            "modular_arithmetic": self.detect_modular_arithmetic,
            "matrix_operations": self.detect_matrix_operations,
            
            # Sorting & Selection
            "custom_sorting": self.detect_custom_sorting,
            "bucket_sort": self.detect_bucket_sort,
            "counting_sort": self.detect_counting_sort,
            "quickselect": self.detect_quickselect,
            "merge_operations": self.detect_merge_operations,
            "partitioning": self.detect_partitioning,
            
            # Greedy Patterns
            "greedy_choice": self.detect_greedy_choice,
            "interval_scheduling": self.detect_interval_scheduling,
            "activity_selection": self.detect_activity_selection,
            "huffman_encoding": self.detect_huffman_encoding,
            "fractional_knapsack": self.detect_fractional_knapsack,
            
            # String Algorithms
            "string_matching": self.detect_string_matching,
            "rolling_hash": self.detect_rolling_hash,
            "suffix_array": self.detect_suffix_array,
            "lcs_patterns": self.detect_lcs_patterns,
            "palindrome_detection": self.detect_palindrome_detection,
            "anagram_detection": self.detect_anagram_detection,
            
            # Advanced Patterns
            "backtracking": self.detect_backtracking,
            "branch_bound": self.detect_branch_bound,
            "meet_in_middle": self.detect_meet_in_middle,
            "sqrt_decomposition": self.detect_sqrt_decomposition,
            "heavy_light_decomposition": self.detect_heavy_light_decomposition,
            "centroid_decomposition": self.detect_centroid_decomposition,
            
            # Optimization Patterns
            "space_optimization": self.detect_space_optimization,
            "time_optimization": self.detect_time_optimization,
            "cache_optimization": self.detect_cache_optimization,
            "lazy_propagation": self.detect_lazy_propagation,
            "persistent_structures": self.detect_persistent_structures,
        }

    # =================== BASIC PATTERNS (ENHANCED) ===================
    
    def detect_iteration(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced iteration detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST analysis for loops
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                score += 0.4
                # Check for range-based iteration
                if (loop_info.get('iterator', {}).get('iter_type') == 'range'):
                    score += 0.2
            elif loop_info['type'] == 'While':
                score += 0.3
        
        # Check for list comprehensions
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                score += 0.2
        
        # Regex backup
        if re.search(r'\bfor\b.*\bin\b', code):
            score += 0.1
        
        return min(score, 1.0)
    
    def detect_nested_loops(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced nested loops detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST analysis for actual nesting
        max_nesting = 0
        for loop_info in self.ast_analyzer.loop_structures:
            nesting = loop_info['nested_loops']
            max_nesting = max(max_nesting, nesting)
            
            if nesting >= 1:
                score += 0.5 * nesting  # More nesting = higher score
        
        # Check for specific nesting patterns
        if max_nesting >= 2:
            score += 0.3  # Triple nested or more
        
        return min(score, 1.0)
    
    def detect_conditional_logic(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced conditional logic detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # AST analysis for conditionals
        complex_conditions = 0
        simple_conditions = 0
        
        for cond_info in self.ast_analyzer.conditional_structures:
            operators = cond_info.get('operators', [])
            if len(operators) > 2:  # Complex condition
                complex_conditions += 1
                score += 0.2
            else:
                simple_conditions += 1
                score += 0.1
        
        # Check for nested conditionals
        nested_if_count = self._count_nested_ifs(ast_tree)
        score += nested_if_count * 0.1
        
        return min(score, 1.0)

    # =================== DYNAMIC PROGRAMMING (ENHANCED) ===================
    
    def detect_memoization(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced memoization detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for @lru_cache decorator
        decorator_score = self._detect_memoization_decorators(ast_tree)
        score += decorator_score
        
        # Check for manual memoization
        manual_memo_score = self._detect_manual_memoization(ast_tree)
        score += manual_memo_score
        
        # Check for memo parameter in function definitions
        memo_param_score = self._detect_memo_parameters(ast_tree)
        score += memo_param_score
        
        # Regex backup
        if re.search(r'@(lru_cache|cache)', code):
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_top_down_dp(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced top-down DP detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Must have both recursion and memoization
        recursion_score = self.detect_recursion(code, ast_tree)
        memo_score = self.detect_memoization(code, ast_tree)
        
        if recursion_score > 0.5 and memo_score > 0.5:
            score += 0.8
        elif recursion_score > 0.3 and memo_score > 0.3:
            score += 0.5
        
        # Check for typical top-down patterns
        if self._has_recursive_subproblem_pattern(ast_tree):
            score += 0.3
        
        return min(score, 1.0)
    
    def detect_bottom_up_dp(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced bottom-up DP detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for DP table initialization
        dp_tables = self._find_dp_table_initialization(ast_tree)
        if dp_tables:
            score += 0.4
        
        # Check for iterative filling (nested loops with DP updates)
        iterative_filling = self._detect_iterative_dp_filling(ast_tree)
        score += iterative_filling
        
        # Check for base case initialization
        base_cases = self._find_dp_base_cases(ast_tree)
        if base_cases:
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_dp_optimization(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced DP optimization detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for space optimization patterns
        space_opt_score = self._detect_dp_space_optimization(ast_tree)
        score += space_opt_score
        
        # Check for rolling array patterns
        rolling_array_score = self._detect_rolling_arrays(ast_tree)
        score += rolling_array_score
        
        return min(score, 1.0)

    # =================== SEARCH & DIVIDE PATTERNS (ENHANCED) ===================
    
    def detect_ternary_search(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced ternary search detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for ternary division calculations
        ternary_calcs = self._find_ternary_calculations(ast_tree)
        if ternary_calcs:
            score += 0.6
        
        # Check for three-way comparisons
        three_way_comps = self._find_three_way_comparisons(ast_tree)
        if three_way_comps:
            score += 0.4
        
        # Regex backup
        if re.search(r'mid1.*mid2|//.*3', code):
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_linear_search(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced linear search detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for simple iteration with comparison
        linear_patterns = self._detect_linear_search_patterns(ast_tree)
        score += linear_patterns
        
        # Check for early return in loops
        early_return_score = self._detect_early_return_in_loops(ast_tree)
        score += early_return_score
        
        return min(score, 1.0)
    
    def detect_divide_conquer(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced divide and conquer detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Must have recursion
        recursion_score = self.detect_recursion(code, ast_tree)
        if recursion_score > 0.5:
            score += 0.3
        
        # Check for problem division patterns
        division_patterns = self._detect_problem_division(ast_tree)
        score += division_patterns
        
        # Check for merge/combine patterns
        merge_patterns = self._detect_merge_combine_patterns(ast_tree)
        score += merge_patterns
        
        return min(score, 1.0)

    # =================== POINTER & WINDOW PATTERNS (ENHANCED) ===================
    
    def detect_fast_slow_pointers(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced fast/slow pointers detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for fast/slow variable naming
        fast_slow_vars = self._find_fast_slow_variables()
        if fast_slow_vars:
            score += 0.5
        
        # Check for different movement speeds
        speed_patterns = self._detect_different_movement_speeds(ast_tree)
        score += speed_patterns
        
        # Check for cycle detection patterns
        cycle_detection = self._detect_cycle_detection_pattern(ast_tree)
        score += cycle_detection
        
        return min(score, 1.0)
    
    def detect_prefix_suffix(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced prefix/suffix detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for prefix/suffix variable naming
        prefix_suffix_vars = self._find_prefix_suffix_variables()
        if prefix_suffix_vars:
            score += 0.4
        
        # Check for cumulative calculation patterns
        cumulative_patterns = self._detect_cumulative_patterns(ast_tree)
        score += cumulative_patterns
        
        # Check for preprocessing patterns
        preprocessing_score = self._detect_preprocessing_patterns(ast_tree)
        score += preprocessing_score
        
        return min(score, 1.0)

    # =================== DATA STRUCTURE USAGE (ENHANCED) ===================
    
    def detect_stack_usage(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced stack usage detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for stack variable declaration
        stack_vars = self._find_stack_variables()
        if stack_vars:
            score += 0.3
        
        # Check for LIFO operations
        lifo_operations = self._detect_lifo_operations(ast_tree)
        score += lifo_operations
        
        # Check for list used as stack
        list_as_stack = self._detect_list_as_stack(ast_tree)
        score += list_as_stack
        
        return min(score, 1.0)
    
    def detect_queue_usage(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced queue usage detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for queue imports
        queue_imports = self._detect_queue_imports()
        score += queue_imports
        
        # Check for FIFO operations
        fifo_operations = self._detect_fifo_operations(ast_tree)
        score += fifo_operations
        
        # Check for deque usage as queue
        deque_as_queue = self._detect_deque_as_queue(ast_tree)
        score += deque_as_queue
        
        return min(score, 1.0)
    
    def detect_deque_usage(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced deque usage detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for deque imports
        deque_imports = self._detect_deque_imports()
        score += deque_imports
        
        # Check for deque-specific operations
        deque_operations = self._detect_deque_operations(ast_tree)
        score += deque_operations
        
        # Check for double-ended operations
        double_ended_ops = self._detect_double_ended_operations(ast_tree)
        score += double_ended_ops
        
        return min(score, 1.0)
    
    def detect_priority_queue(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced priority queue detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for PriorityQueue imports
        pq_imports = self._detect_priority_queue_imports()
        score += pq_imports
        
        # Check for heap used as priority queue
        heap_as_pq = self.detect_heap_usage(code, ast_tree)
        score += heap_as_pq * 0.8
        
        # Check for priority-based operations
        priority_ops = self._detect_priority_operations(ast_tree)
        score += priority_ops
        
        return min(score, 1.0)
    
    def detect_hash_map(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced hash map detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for dictionary initialization
        dict_init = self._detect_dictionary_initialization(ast_tree)
        score += dict_init
        
        # Check for key-value operations
        kv_operations = self._detect_key_value_operations(ast_tree)
        score += kv_operations
        
        # Check for dictionary methods
        dict_methods = self._detect_dictionary_methods(ast_tree)
        score += dict_methods
        
        return min(score, 1.0)
    
    def detect_hash_set(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced hash set detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for set initialization
        set_init = self._detect_set_initialization(ast_tree)
        score += set_init
        
        # Check for set operations
        set_operations = self._detect_set_operations(ast_tree)
        score += set_operations
        
        # Check for membership testing
        membership_testing = self._detect_membership_testing(ast_tree)
        score += membership_testing
        
        return min(score, 1.0)
    
    def detect_array_manipulation(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced array manipulation detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for array/list operations
        array_operations = self._detect_array_operations(ast_tree)
        score += array_operations
        
        # Check for slicing operations
        slicing_operations = self._detect_slicing_operations(ast_tree)
        score += slicing_operations
        
        # Check for in-place modifications
        inplace_modifications = self._detect_inplace_modifications(ast_tree)
        score += inplace_modifications
        
        return min(score, 1.0)
    
    def detect_string_operations(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced string operations detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for string methods
        string_methods = self._detect_string_methods(ast_tree)
        score += string_methods
        
        # Check for string formatting
        string_formatting = self._detect_string_formatting(ast_tree)
        score += string_formatting
        
        # Check for character-level operations
        char_operations = self._detect_character_operations(ast_tree)
        score += char_operations
        
        return min(score, 1.0)

    # =================== GRAPH ALGORITHMS (ENHANCED) ===================
    
    def detect_graph_bfs(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced BFS detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for queue-based traversal
        queue_traversal = self._detect_queue_based_traversal(ast_tree)
        score += queue_traversal
        
        # Check for level-by-level processing
        level_processing = self._detect_level_processing(ast_tree)
        score += level_processing
        
        # Check for BFS function naming
        bfs_naming = self._detect_bfs_function_naming()
        score += bfs_naming
        
        return min(score, 1.0)
    
    def detect_topological_sort(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced topological sort detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for indegree calculation
        indegree_calc = self._detect_indegree_calculation(ast_tree)
        score += indegree_calc
        
        # Check for Kahn's algorithm pattern
        kahns_pattern = self._detect_kahns_algorithm(ast_tree)
        score += kahns_pattern
        
        # Check for DFS-based topological sort
        dfs_topo = self._detect_dfs_topological_sort(ast_tree)
        score += dfs_topo
        
        return min(score, 1.0)
    
    def detect_shortest_path(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced shortest path detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for distance array initialization
        dist_array = self._detect_distance_array_initialization(ast_tree)
        score += dist_array
        
        # Check for relaxation operations
        relaxation = self._detect_edge_relaxation(ast_tree)
        score += relaxation
        
        # Check for specific algorithms
        dijkstra_score = self._detect_dijkstra_pattern(ast_tree)
        bellman_ford_score = self._detect_bellman_ford_pattern(ast_tree)
        floyd_warshall_score = self._detect_floyd_warshall_pattern(ast_tree)
        
        score += max(dijkstra_score, bellman_ford_score, floyd_warshall_score)
        
        return min(score, 1.0)
    
    def detect_union_find(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced Union-Find detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for parent array
        parent_array = self._detect_parent_array(ast_tree)
        score += parent_array
        
        # Check for find function with path compression
        find_function = self._detect_find_function(ast_tree)
        score += find_function
        
        # Check for union function
        union_function = self._detect_union_function(ast_tree)
        score += union_function
        
        # Check for rank/size optimization
        rank_optimization = self._detect_rank_optimization(ast_tree)
        score += rank_optimization
        
        return min(score, 1.0)
    
    def detect_minimum_spanning_tree(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced MST detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for Kruskal's algorithm
        kruskal_score = self._detect_kruskal_algorithm(ast_tree)
        score += kruskal_score
        
        # Check for Prim's algorithm
        prim_score = self._detect_prim_algorithm(ast_tree)
        score += prim_score
        
        # Check for edge sorting by weight
        edge_sorting = self._detect_edge_sorting_by_weight(ast_tree)
        score += edge_sorting
        
        return min(score, 1.0)
    
    def detect_graph_coloring(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced graph coloring detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for color assignment
        color_assignment = self._detect_color_assignment(ast_tree)
        score += color_assignment
        
        # Check for adjacency conflict checking
        conflict_checking = self._detect_adjacency_conflict_checking(ast_tree)
        score += conflict_checking
        
        # Check for backtracking with coloring
        backtrack_coloring = self._detect_backtracking_coloring(ast_tree)
        score += backtrack_coloring
        
        return min(score, 1.0)

    # =================== TREE ALGORITHMS (ENHANCED) ===================
    
    def detect_tree_traversal(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced tree traversal detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for tree node access patterns
        tree_access = self._detect_tree_node_access(ast_tree)
        score += tree_access
        
        # Check for specific traversal patterns
        inorder_score = self._detect_inorder_traversal(ast_tree)
        preorder_score = self._detect_preorder_traversal(ast_tree)
        postorder_score = self._detect_postorder_traversal(ast_tree)
        level_order_score = self._detect_level_order_traversal(ast_tree)
        
        score += max(inorder_score, preorder_score, postorder_score, level_order_score)
        
        return min(score, 1.0)
    
    def detect_binary_tree_operations(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced binary tree operations detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for TreeNode class usage
        treenode_usage = self._detect_treenode_usage(ast_tree)
        score += treenode_usage
        
        # Check for binary tree specific operations
        binary_operations = self._detect_binary_tree_operations(ast_tree)
        score += binary_operations
        
        # Check for tree construction/modification
        tree_modification = self._detect_tree_modification(ast_tree)
        score += tree_modification
        
        return min(score, 1.0)
    
    def detect_trie_usage(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced trie detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for TrieNode class definition
        trie_class = self._detect_trie_class_definition(ast_tree)
        score += trie_class
        
        # Check for character-based navigation
        char_navigation = self._detect_character_navigation(ast_tree)
        score += char_navigation
        
        # Check for trie operations (insert, search, startsWith)
        trie_operations = self._detect_trie_operations(ast_tree)
        score += trie_operations
        
        # Check for children dictionary usage
        children_dict = self._detect_children_dictionary(ast_tree)
        score += children_dict
        
        return min(score, 1.0)
    
    def detect_segment_tree(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced segment tree detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for segment tree class
        segtree_class = self._detect_segment_tree_class(ast_tree)
        score += segtree_class
        
        # Check for tree indexing patterns (2*i, 2*i+1)
        tree_indexing = self._detect_tree_indexing_patterns(ast_tree)
        score += tree_indexing
        
        # Check for range query operations
        range_queries = self._detect_range_query_operations(ast_tree)
        score += range_queries
        
        # Check for lazy propagation
        lazy_prop = self.detect_lazy_propagation(code, ast_tree)
        score += lazy_prop * 0.3
        
        return min(score, 1.0)
    
    def detect_fenwick_tree(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced Fenwick tree detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for Fenwick/BIT class
        fenwick_class = self._detect_fenwick_class(ast_tree)
        score += fenwick_class
        
        # Check for bit manipulation patterns (i & -i)
        bit_manipulation = self._detect_fenwick_bit_patterns(ast_tree)
        score += bit_manipulation
        
        # Check for prefix sum operations
        prefix_sum_ops = self._detect_prefix_sum_operations(ast_tree)
        score += prefix_sum_ops
        
        return min(score, 1.0)

    def detect_tree_dp(self, code: str, ast_tree: ast.AST = None) -> float:
        """Detect dynamic programming on trees"""
        score = 0.0
        
        # Tree + DP combination
        tree_score = self.detect_tree_traversal(code, ast_tree)
        dp_score = max(self.detect_memoization(code, ast_tree), self.detect_bottom_up_dp(code, ast_tree))
        
        if tree_score > 0.3 and dp_score > 0.3:
            score += 0.6
        
        # Tree DP keywords
        if re.search(r'tree.*dp|dp.*tree|subtree.*dp', code):
            score += 0.4
        
        return min(score, 1.0)
    # =================== ADVANCED DATA STRUCTURES (ENHANCED) ===================
    
    def detect_monotonic_stack(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced monotonic stack detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for stack with comparison-based popping
        monotonic_pattern = self._detect_monotonic_stack_pattern(ast_tree)
        score += monotonic_pattern
        
        # Check for next greater/smaller element patterns
        next_element_patterns = self._detect_next_element_patterns(ast_tree)
        score += next_element_patterns
        
        # Check for increasing/decreasing maintenance
        order_maintenance = self._detect_stack_order_maintenance(ast_tree)
        score += order_maintenance
        
        return min(score, 1.0)
    
    def detect_monotonic_queue(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced monotonic queue detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for deque with monotonic property
        monotonic_deque = self._detect_monotonic_deque_pattern(ast_tree)
        score += monotonic_deque
        
        # Check for sliding window maximum pattern
        sliding_max = self._detect_sliding_window_maximum(ast_tree)
        score += sliding_max
        
        # Check for front/back maintenance
        front_back_maintenance = self._detect_deque_maintenance(ast_tree)
        score += front_back_maintenance
        
        return min(score, 1.0)
    
    def detect_coordinate_compression(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced coordinate compression detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for coordinate sorting and mapping
        coord_mapping = self._detect_coordinate_mapping(ast_tree)
        score += coord_mapping
        
        # Check for unique coordinate extraction
        unique_extraction = self._detect_unique_coordinate_extraction(ast_tree)
        score += unique_extraction
        
        # Check for discretization patterns
        discretization = self._detect_discretization_patterns(ast_tree)
        score += discretization
        
        return min(score, 1.0)
    
    def detect_sparse_table(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced sparse table detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for 2D table for sparse table
        sparse_table_2d = self._detect_2d_sparse_table(ast_tree)
        score += sparse_table_2d
        
        # Check for logarithmic preprocessing
        log_preprocessing = self._detect_logarithmic_preprocessing(ast_tree)
        score += log_preprocessing
        
        # Check for range minimum/maximum queries
        rmq_operations = self._detect_rmq_operations(ast_tree)
        score += rmq_operations
        
        return min(score, 1.0)
    
    def detect_disjoint_set_union(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced DSU detection (enhanced union-find)"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Use enhanced union-find detection as base
        uf_score = self.detect_union_find(code, ast_tree)
        score += uf_score * 0.8
        
        # Check for additional DSU features
        dsu_features = self._detect_advanced_dsu_features(ast_tree)
        score += dsu_features
        
        return min(score, 1.0)

    # =================== MATHEMATICAL PATTERNS (ENHANCED) ===================
    
    def detect_number_theory(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced number theory detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for number theory functions
        nt_functions = self._detect_number_theory_functions(ast_tree)
        score += nt_functions
        
        # Check for prime-related operations
        prime_operations = self._detect_prime_operations(ast_tree)
        score += prime_operations
        
        # Check for GCD/LCM operations
        gcd_lcm_ops = self._detect_gcd_lcm_operations(ast_tree)
        score += gcd_lcm_ops
        
        # Check for sieve patterns
        sieve_patterns = self._detect_sieve_patterns(ast_tree)
        score += sieve_patterns
        
        return min(score, 1.0)
    
    def detect_combinatorics(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced combinatorics detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for factorial calculations
        factorial_calc = self._detect_factorial_calculations(ast_tree)
        score += factorial_calc
        
        # Check for combination/permutation calculations
        comb_perm_calc = self._detect_combination_permutation(ast_tree)
        score += comb_perm_calc
        
        # Check for Pascal's triangle
        pascal_triangle = self._detect_pascal_triangle(ast_tree)
        score += pascal_triangle
        
        return min(score, 1.0)
    
    def detect_probability(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced probability detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for random number generation
        random_usage = self._detect_random_usage(ast_tree)
        score += random_usage
        
        # Check for probability calculations
        prob_calculations = self._detect_probability_calculations(ast_tree)
        score += prob_calculations
        
        # Check for statistical operations
        stats_operations = self._detect_statistical_operations(ast_tree)
        score += stats_operations
        
        return min(score, 1.0)
    
    def detect_geometry(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced geometry detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for point/coordinate operations
        coordinate_ops = self._detect_coordinate_operations(ast_tree)
        score += coordinate_ops
        
        # Check for geometric calculations
        geometric_calc = self._detect_geometric_calculations(ast_tree)
        score += geometric_calc
        
        # Check for convex hull algorithms
        convex_hull = self._detect_convex_hull_algorithms(ast_tree)
        score += convex_hull
        
        return min(score, 1.0)
    
    def detect_bit_manipulation(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced bit manipulation detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for bitwise operations in AST
        bitwise_ops = self._detect_bitwise_operations(ast_tree)
        score += bitwise_ops
        
        # Check for bit manipulation tricks
        bit_tricks = self._detect_bit_manipulation_tricks(ast_tree)
        score += bit_tricks
        
        # Check for bit masking
        bit_masking = self._detect_bit_masking(ast_tree)
        score += bit_masking
        
        return min(score, 1.0)
    
    def detect_modular_arithmetic(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced modular arithmetic detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for modulo operations
        modulo_ops = self._detect_modulo_operations(ast_tree)
        score += modulo_ops
        
        # Check for modular exponentiation
        mod_exp = self._detect_modular_exponentiation(ast_tree)
        score += mod_exp
        
        # Check for modular inverse
        mod_inverse = self._detect_modular_inverse(ast_tree)
        score += mod_inverse
        
        return min(score, 1.0)
    
    def detect_matrix_operations(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced matrix operations detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for 2D array operations
        matrix_2d = self._detect_2d_array_operations(ast_tree)
        score += matrix_2d
        
        # Check for matrix multiplication
        matrix_mult = self._detect_matrix_multiplication(ast_tree)
        score += matrix_mult
        
        # Check for matrix transformations
        matrix_transform = self._detect_matrix_transformations(ast_tree)
        score += matrix_transform
        
        return min(score, 1.0)

    # =================== SORTING & SELECTION (ENHANCED) ===================
    
    def detect_custom_sorting(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced custom sorting detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for sort with key parameter
        sort_with_key = self._detect_sort_with_key(ast_tree)
        score += sort_with_key
        
        # Check for lambda functions in sorting
        lambda_sorting = self._detect_lambda_sorting(ast_tree)
        score += lambda_sorting
        
        # Check for custom comparator functions
        custom_comparator = self._detect_custom_comparator(ast_tree)
        score += custom_comparator
        
        return min(score, 1.0)
    
    def detect_bucket_sort(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced bucket sort detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for bucket initialization
        bucket_init = self._detect_bucket_initialization(ast_tree)
        score += bucket_init
        
        # Check for bucket distribution
        bucket_distribution = self._detect_bucket_distribution(ast_tree)
        score += bucket_distribution
        
        # Check for bucket sorting and merging
        bucket_merge = self._detect_bucket_merging(ast_tree)
        score += bucket_merge
        
        return min(score, 1.0)
    
    def detect_counting_sort(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced counting sort detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for count array
        count_array = self._detect_count_array(ast_tree)
        score += count_array
        
        # Check for frequency counting
        frequency_counting = self._detect_frequency_counting(ast_tree)
        score += frequency_counting
        
        # Check for stable sorting reconstruction
        stable_reconstruction = self._detect_stable_reconstruction(ast_tree)
        score += stable_reconstruction
        
        return min(score, 1.0)
    
    def detect_quickselect(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced quickselect detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for partitioning with pivot
        partitioning = self.detect_partitioning(code, ast_tree)
        score += partitioning * 0.6
        
        # Check for kth element selection
        kth_selection = self._detect_kth_element_selection(ast_tree)
        score += kth_selection
        
        # Check for recursive selection
        recursive_selection = self._detect_recursive_selection(ast_tree)
        score += recursive_selection
        
        return min(score, 1.0)
    
    def detect_merge_operations(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced merge operations detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for two-pointer merge
        two_pointer_merge = self._detect_two_pointer_merge(ast_tree)
        score += two_pointer_merge
        
        # Check for merge function
        merge_function = self._detect_merge_function(ast_tree)
        score += merge_function
        
        # Check for divide and conquer with merge
        divide_merge = self._detect_divide_and_merge(ast_tree)
        score += divide_merge
        
        return min(score, 1.0)
    
    def detect_partitioning(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced partitioning detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for pivot selection
        pivot_selection = self._detect_pivot_selection(ast_tree)
        score += pivot_selection
        
        # Check for in-place partitioning
        inplace_partition = self._detect_inplace_partitioning(ast_tree)
        score += inplace_partition
        
        # Check for partition function
        partition_function = self._detect_partition_function(ast_tree)
        score += partition_function
        
        return min(score, 1.0)

    # =================== GREEDY PATTERNS (ENHANCED) ===================
    
    def detect_greedy_choice(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced greedy choice detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for sorting before selection
        sort_before_select = self._detect_sort_before_selection(ast_tree)
        score += sort_before_select
        
        # Check for local optimal choices
        local_optimal = self._detect_local_optimal_choices(ast_tree)
        score += local_optimal
        
        # Check for greedy selection patterns
        greedy_selection = self._detect_greedy_selection_patterns(ast_tree)
        score += greedy_selection
        
        return min(score, 1.0)
    
    def detect_interval_scheduling(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced interval scheduling detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for interval data structures
        interval_structures = self._detect_interval_structures(ast_tree)
        score += interval_structures
        
        # Check for end time sorting
        end_time_sorting = self._detect_end_time_sorting(ast_tree)
        score += end_time_sorting
        
        # Check for overlap detection
        overlap_detection = self._detect_overlap_detection(ast_tree)
        score += overlap_detection
        
        return min(score, 1.0)
    
    def detect_activity_selection(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced activity selection detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Activity selection is similar to interval scheduling
        interval_score = self.detect_interval_scheduling(code, ast_tree)
        score += interval_score * 0.8
        
        # Check for activity-specific patterns
        activity_patterns = self._detect_activity_patterns(ast_tree)
        score += activity_patterns
        
        return min(score, 1.0)
    
    def detect_huffman_encoding(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced Huffman encoding detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for frequency-based processing
        frequency_processing = self._detect_frequency_processing(ast_tree)
        score += frequency_processing
        
        # Check for heap usage in tree building
        heap_tree_building = self._detect_heap_tree_building(ast_tree)
        score += heap_tree_building
        
        # Check for binary tree encoding
        binary_encoding = self._detect_binary_encoding(ast_tree)
        score += binary_encoding
        
        return min(score, 1.0)
    
    def detect_fractional_knapsack(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced fractional knapsack detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for value-to-weight ratio calculation
        ratio_calculation = self._detect_value_weight_ratio(ast_tree)
        score += ratio_calculation
        
        # Check for fractional taking
        fractional_taking = self._detect_fractional_taking(ast_tree)
        score += fractional_taking
        
        # Check for capacity constraints
        capacity_constraints = self._detect_capacity_constraints(ast_tree)
        score += capacity_constraints
        
        return min(score, 1.0)

    # =================== STRING ALGORITHMS (ENHANCED) ===================
    
    def detect_string_matching(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced string matching detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for pattern matching algorithms
        kmp_score = self._detect_kmp_algorithm(ast_tree)
        boyer_moore_score = self._detect_boyer_moore_algorithm(ast_tree)
        rabin_karp_score = self._detect_rabin_karp_algorithm(ast_tree)
        
        score += max(kmp_score, boyer_moore_score, rabin_karp_score)
        
        # Check for pattern and text variables
        pattern_text = self._detect_pattern_text_variables(ast_tree)
        score += pattern_text
        
        return min(score, 1.0)
    
    def detect_rolling_hash(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced rolling hash detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for hash calculation with base and modulo
        hash_calculation = self._detect_hash_calculation(ast_tree)
        score += hash_calculation
        
        # Check for rolling mechanism
        rolling_mechanism = self._detect_rolling_mechanism(ast_tree)
        score += rolling_mechanism
        
        # Check for polynomial rolling hash
        polynomial_hash = self._detect_polynomial_hash(ast_tree)
        score += polynomial_hash
        
        return min(score, 1.0)
    
    def detect_suffix_array(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced suffix array detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for suffix generation
        suffix_generation = self._detect_suffix_generation(ast_tree)
        score += suffix_generation
        
        # Check for suffix sorting
        suffix_sorting = self._detect_suffix_sorting(ast_tree)
        score += suffix_sorting
        
        # Check for LCP array
        lcp_array = self._detect_lcp_array(ast_tree)
        score += lcp_array
        
        return min(score, 1.0)
    
    def detect_lcs_patterns(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced LCS detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for 2D DP with string comparison
        string_dp = self._detect_string_comparison_dp(ast_tree)
        score += string_dp
        
        # Check for character matching in nested loops
        char_matching = self._detect_character_matching_loops(ast_tree)
        score += char_matching
        
        # Check for LCS-specific patterns
        lcs_patterns = self._detect_lcs_specific_patterns(ast_tree)
        score += lcs_patterns
        
        return min(score, 1.0)
    
    def detect_palindrome_detection(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced palindrome detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for string reversal comparison
        reversal_comparison = self._detect_string_reversal_comparison(ast_tree)
        score += reversal_comparison
        
        # Check for two-pointer palindrome check
        two_pointer_palindrome = self._detect_two_pointer_palindrome(ast_tree)
        score += two_pointer_palindrome
        
        # Check for Manacher's algorithm
        manacher_algorithm = self._detect_manacher_algorithm(ast_tree)
        score += manacher_algorithm
        
        return min(score, 1.0)
    
    def detect_anagram_detection(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced anagram detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for character frequency comparison
        freq_comparison = self._detect_character_frequency_comparison(ast_tree)
        score += freq_comparison
        
        # Check for sorting comparison
        sorting_comparison = self._detect_sorting_comparison(ast_tree)
        score += sorting_comparison
        
        # Check for Counter usage
        counter_usage = self._detect_counter_usage(ast_tree)
        score += counter_usage
        
        return min(score, 1.0)

    # =================== ADVANCED PATTERNS (ENHANCED) ===================
    
    def detect_backtracking(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced backtracking detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Must have recursion
        recursion_score = self.detect_recursion(code, ast_tree)
        if recursion_score > 0.5:
            score += 0.3
        
        # Check for state modification and restoration
        state_restoration = self._detect_state_restoration_pattern(ast_tree)
        score += state_restoration
        
        # Check for exploration with undo
        exploration_undo = self._detect_exploration_undo_pattern(ast_tree)
        score += exploration_undo
        
        # Check for pruning conditions
        pruning_conditions = self._detect_pruning_conditions(ast_tree)
        score += pruning_conditions
        
        return min(score, 1.0)
    
    def detect_branch_bound(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced branch and bound detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for bound calculation
        bound_calculation = self._detect_bound_calculation(ast_tree)
        score += bound_calculation
        
        # Check for pruning based on bounds
        bound_pruning = self._detect_bound_pruning(ast_tree)
        score += bound_pruning
        
        # Check for best solution tracking
        # Check for best solution tracking
        best_solution_tracking = self._detect_best_solution_tracking(ast_tree)
        score += best_solution_tracking
        
        return min(score, 1.0)
    
    def detect_meet_in_middle(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced meet in the middle detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for problem space division
        space_division = self._detect_problem_space_division(ast_tree)
        score += space_division
        
        # Check for two-way exploration
        two_way_exploration = self._detect_two_way_exploration(ast_tree)
        score += two_way_exploration
        
        # Check for middle meeting point
        meeting_point = self._detect_meeting_point_logic(ast_tree)
        score += meeting_point
        
        return min(score, 1.0)
    
    def detect_sqrt_decomposition(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced sqrt decomposition detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for sqrt calculation for block size
        sqrt_calculation = self._detect_sqrt_calculation(ast_tree)
        score += sqrt_calculation
        
        # Check for block-based processing
        block_processing = self._detect_block_processing(ast_tree)
        score += block_processing
        
        # Check for range query optimization
        range_optimization = self._detect_range_query_optimization(ast_tree)
        score += range_optimization
        
        return min(score, 1.0)
    
    def detect_heavy_light_decomposition(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced heavy-light decomposition detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for heavy/light edge classification
        edge_classification = self._detect_heavy_light_classification(ast_tree)
        score += edge_classification
        
        # Check for chain decomposition
        chain_decomposition = self._detect_chain_decomposition(ast_tree)
        score += chain_decomposition
        
        # Check for path query optimization
        path_query_optimization = self._detect_path_query_optimization(ast_tree)
        score += path_query_optimization
        
        return min(score, 1.0)
    
    def detect_centroid_decomposition(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced centroid decomposition detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for centroid finding
        centroid_finding = self._detect_centroid_finding(ast_tree)
        score += centroid_finding
        
        # Check for tree decomposition
        tree_decomposition = self._detect_tree_decomposition(ast_tree)
        score += tree_decomposition
        
        # Check for distance queries
        distance_queries = self._detect_distance_queries(ast_tree)
        score += distance_queries
        
        return min(score, 1.0)

    # =================== OPTIMIZATION PATTERNS (ENHANCED) ===================
    
    def detect_space_optimization(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced space optimization detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for rolling arrays
        rolling_arrays = self._detect_rolling_arrays(ast_tree)
        score += rolling_arrays
        
        # Check for in-place modifications
        inplace_modifications = self._detect_inplace_space_optimization(ast_tree)
        score += inplace_modifications
        
        # Check for constant space patterns
        constant_space = self._detect_constant_space_patterns(ast_tree)
        score += constant_space
        
        return min(score, 1.0)
    
    def detect_time_optimization(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced time optimization detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for early termination
        early_termination = self._detect_early_termination(ast_tree)
        score += early_termination
        
        # Check for algorithmic improvements
        algo_improvements = self._detect_algorithmic_improvements(ast_tree)
        score += algo_improvements
        
        # Check for precomputation
        precomputation = self._detect_precomputation_patterns(ast_tree)
        score += precomputation
        
        return min(score, 1.0)
    
    def detect_cache_optimization(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced cache optimization detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        
        # Check for memoization (caching)
        memo_score = self.detect_memoization(code, ast_tree)
        score += memo_score * 0.8
        
        # Check for cache-friendly access patterns
        cache_friendly = self._detect_cache_friendly_patterns(ast_tree)
        score += cache_friendly
        
        return min(score, 1.0)
    
    def detect_lazy_propagation(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced lazy propagation detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for lazy array/flag
        lazy_structures = self._detect_lazy_structures(ast_tree)
        score += lazy_structures
        
        # Check for delayed update patterns
        delayed_updates = self._detect_delayed_update_patterns(ast_tree)
        score += delayed_updates
        
        # Check for range update optimization
        range_update_opt = self._detect_range_update_optimization(ast_tree)
        score += range_update_opt
        
        return min(score, 1.0)
    
    def detect_persistent_structures(self, code: str, ast_tree: ast.AST = None) -> float:
        """Enhanced persistent structures detection"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Check for immutable operations
        immutable_operations = self._detect_immutable_operations(ast_tree)
        score += immutable_operations
        
        # Check for version tracking
        version_tracking = self._detect_version_tracking(ast_tree)
        score += version_tracking
        
        # Check for path copying
        path_copying = self._detect_path_copying_patterns(ast_tree)
        score += path_copying
        
        return min(score, 1.0)

    # =================== COMPREHENSIVE HELPER METHODS ===================
    
    def _count_nested_ifs(self, ast_tree: ast.AST) -> int:
        """Count nested if statements"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        nested_count = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.If):
                for child in ast.walk(node):
                    if isinstance(child, ast.If) and child != node:
                        nested_count += 1
        return nested_count
    
    def _detect_memoization_decorators(self, ast_tree: ast.AST) -> float:
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        """Detect memoization decorators"""
        score = 0.0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in ['cache', 'lru_cache']:
                            score += 0.8
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            if decorator.func.id in ['cache', 'lru_cache']:
                                score += 0.8
        return min(score, 1.0)
    
    def _detect_manual_memoization(self, ast_tree: ast.AST) -> float:
        """Detect manual memoization patterns"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        score = 0.0
        memo_vars = set()
        
        # Find memo variables
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ['memo', 'cache', 'dp']:
                        memo_vars.add(target.id)
        
        # Check for memo usage in conditionals
        if memo_vars:
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.Compare):
                        for comparator in node.test.comparators:
                            if isinstance(comparator, ast.Name) and comparator.id in memo_vars:
                                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_memo_parameters(self, ast_tree: ast.AST) -> float:
        """Detect memo parameters in function definitions"""
        if ast_tree is None:
            try:
                ast_tree = ast.parse(code)
            except:
                return 0.0
        
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        score = 0.0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.arg in ['memo', 'cache']:
                        score += 0.3
        return min(score, 1.0)
    
    def _has_recursive_subproblem_pattern(self, ast_tree: ast.AST) -> bool:
        """Check for recursive subproblem patterns typical in top-down DP"""
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Look for recursive calls with modified parameters
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp):
                            if isinstance(arg.op, (ast.Add, ast.Sub)):
                                return True
        return False
    
    def _find_dp_table_initialization(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find DP table initialization"""
        dp_tables = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ['dp', 'table']:
                        if isinstance(node.value, (ast.List, ast.ListComp)):
                            dp_tables.append(node)
        return dp_tables
    
    def _detect_iterative_dp_filling(self, ast_tree: ast.AST) -> float:
        """Detect iterative DP table filling"""
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['nested_loops'] > 0:
                # Check for DP updates in nested loops
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Subscript):
                                if isinstance(target.value, ast.Name) and target.value.id in ['dp', 'table']:
                                    score += 0.4
        
        return min(score, 1.0)
    
    def _find_dp_base_cases(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find DP base case initializations"""
        base_cases = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.slice, ast.Constant) and target.slice.value == 0:
                            base_cases.append(node)
        return base_cases
    
    def _detect_dp_space_optimization(self, ast_tree: ast.AST) -> float:
        """Detect DP space optimization techniques"""
        score = 0.0
        
        # Check for rolling variables
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(keyword in target.id for keyword in ['prev', 'curr', 'last']):
                            score += 0.3
        
        return min(score, 0.6)
    
    def _detect_rolling_arrays(self, ast_tree: ast.AST) -> float:
        """Detect rolling array patterns"""
        score = 0.0
        
        # Check for modulo indexing in arrays
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.BinOp) and isinstance(node.slice.op, ast.Mod):
                    score += 0.4
        
        # Check for prev/curr swapping
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Name) and 'prev' in node.value.id:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and 'curr' in target.id:
                            score += 0.3
        
        return min(score, 1.0)
    
    def _find_ternary_calculations(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find ternary search mid calculations"""
        ternary_calcs = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.BinOp):
                    # Look for division by 3
                    if isinstance(node.value.op, ast.FloorDiv):
                        if isinstance(node.value.right, ast.Constant) and node.value.right.value == 3:
                            ternary_calcs.append(node)
        return ternary_calcs
    
    def _find_three_way_comparisons(self, ast_tree: ast.AST) -> List[ast.AST]:
        """Find three-way comparisons typical in ternary search"""
        comparisons = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.If):
                # Look for elif chains (three-way decisions)
                if node.orelse and isinstance(node.orelse[0], ast.If):
                    comparisons.append(node)
        return comparisons
    
    def _detect_linear_search_patterns(self, ast_tree: ast.AST) -> float:
        """Detect linear search patterns"""
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                # Check for comparison within loop
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.Compare):
                        score += 0.3
                        break
        
        return min(score, 0.6)
    
    def _detect_early_return_in_loops(self, ast_tree: ast.AST) -> float:
        """Detect early return patterns in loops"""
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        for loop_info in self.ast_analyzer.loop_structures:
            for node in ast.walk(loop_info['node']):
                if isinstance(node, ast.Return):
                    score += 0.3
        
        return min(score, 0.6)
    
    def _detect_problem_division(self, ast_tree: ast.AST) -> float:
        """Detect problem division patterns in divide and conquer"""
        score = 0.0
        
        # Look for mid calculations
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'mid':
                        if isinstance(node.value, ast.BinOp):
                            score += 0.4
        
        return min(score, 0.6)
    
    def _detect_merge_combine_patterns(self, ast_tree: ast.AST) -> float:
        """Detect merge/combine patterns"""
        score = 0.0
        self.ast_analyzer.analyze_tree(ast_tree)
        
        # Look for function calls with 'merge' or 'combine' in name
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name']:
                if any(keyword in call_info['function_name'].lower() for keyword in ['merge', 'combine']):
                    score += 0.4
        
        return min(score, 0.6)
    
    def _find_fast_slow_variables(self) -> List[str]:
        """Find fast/slow pointer variables"""
        fast_slow_vars = []
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['fast', 'slow', 'turtle', 'hare']):
                fast_slow_vars.append(var)
        return fast_slow_vars
    
    def _detect_different_movement_speeds(self, ast_tree: ast.AST) -> float:
        """Detect different movement speeds for pointers"""
        score = 0.0
        
        # Look for .next.next patterns (fast pointer)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Attribute):
                if node.attr == 'next' and isinstance(node.value, ast.Attribute):
                    if node.value.attr == 'next':
                        score += 0.5
        
        return min(score, 0.8)
    
    def _detect_cycle_detection_pattern(self, ast_tree: ast.AST) -> float:
        """Detect cycle detection patterns"""
        score = 0.0
        
        # Look for fast == slow comparisons
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Compare):
                if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                    left = node.left
                    right = node.comparators[0] if node.comparators else None
                    if (isinstance(left, ast.Name) and isinstance(right, ast.Name)):
                        if (('fast' in left.id and 'slow' in right.id) or 
                            ('slow' in left.id and 'fast' in right.id)):
                            score += 0.6
        
        return min(score, 1.0)

    # Continue with remaining helper methods...
    # [Due to length constraints, I'm showing the pattern - all other helper methods follow similar structure]

    def _find_stack_variables(self) -> List[str]:
        """Find variables that could be stacks"""
        stack_vars = []
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['stack', 'stk', 'st']):
                stack_vars.append(var)
        return stack_vars
    
    def _detect_lifo_operations(self, ast_tree: ast.AST) -> float:
        """Detect Last-In-First-Out operations"""
        score = 0.0
        append_count = 0
        pop_count = 0
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'append':
                    append_count += 1
                elif method == 'pop' and call_info['args'] == 0:  # pop() without args
                    pop_count += 1
        
        if append_count > 0 and pop_count > 0:
            score += 0.6
        elif append_count > 0 or pop_count > 0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_list_as_stack(self, ast_tree: ast.AST) -> float:
        """Detect list being used as stack"""
        score = 0.0
        
        # Check for list initialization followed by append/pop
        list_vars = set()
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
                        list_vars.add(target.id)
        
        # Check if these lists are used with stack operations
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                obj, method = call_info['attribute_chain'][0], call_info['attribute_chain'][-1]
                if obj in list_vars and method in ['append', 'pop']:
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_queue_imports(self) -> float:
        """Detect queue-related imports"""
        score = 0.0
        for imp in self.ast_analyzer.imports:
            if any(keyword in imp.lower() for keyword in ['queue', 'deque']):
                score += 0.5
        return min(score, 1.0)
    
    def _detect_fifo_operations(self, ast_tree: ast.AST) -> float:
        """Detect First-In-First-Out operations"""
        score = 0.0
        append_count = 0
        popleft_count = 0
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'append':
                    append_count += 1
                elif method == 'popleft':
                    popleft_count += 1
        
        if append_count > 0 and popleft_count > 0:
            score += 0.8
        
        return min(score, 1.0)
    
    def _detect_deque_as_queue(self, ast_tree: ast.AST) -> float:
        """Detect deque being used as queue"""
        score = 0.0
        
        # Check for deque usage with FIFO operations
        deque_vars = set()
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'deque':
                    deque_vars.add('deque_instance')
        
        if deque_vars:
            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_deque_imports(self) -> float:
        """Detect deque imports"""
        score = 0.0
        for imp in self.ast_analyzer.imports:
            if 'deque' in imp.lower():
                score += 0.6
        return min(score, 1.0)
    
    def _detect_deque_operations(self, ast_tree: ast.AST) -> float:
        """Detect deque-specific operations"""
        score = 0.0
        deque_ops = ['appendleft', 'popleft', 'extendleft', 'rotate']
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in deque_ops:
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_double_ended_operations(self, ast_tree: ast.AST) -> float:
        """Detect operations on both ends of deque"""
        score = 0.0
        left_ops = 0
        right_ops = 0
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if 'left' in method:
                    left_ops += 1
                elif method in ['append', 'pop']:
                    right_ops += 1
        
        if left_ops > 0 and right_ops > 0:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_priority_queue_imports(self) -> float:
        """Detect priority queue imports"""
        score = 0.0
        for imp in self.ast_analyzer.imports:
            if 'priorityqueue' in imp.lower() or 'queue.priorityqueue' in imp.lower():
                score += 0.8
        return min(score, 1.0)
    
    def _detect_priority_operations(self, ast_tree: ast.AST) -> float:
        """Detect priority-based operations"""
        score = 0.0
        
        # Check for tuple insertion with priority
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Tuple) and len(arg.elts) >= 2:
                        # Likely (priority, item) tuple
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_dictionary_initialization(self, ast_tree: ast.AST) -> float:
        """Detect dictionary initialization patterns"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Dict):
                    score += 0.3
                elif isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_key_value_operations(self, ast_tree: ast.AST) -> float:
        """Detect key-value operations"""
        score = 0.0
        
        # Check for dictionary subscript assignments
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        score += 0.2
        
        # Check for .get() method calls
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'get':
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_dictionary_methods(self, ast_tree: ast.AST) -> float:
        """Detect dictionary method usage"""
        score = 0.0
        dict_methods = ['keys', 'values', 'items', 'get', 'setdefault', 'update']
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in dict_methods:
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_set_initialization(self, ast_tree: ast.AST) -> float:
        """Detect set initialization patterns"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Set):
                    score += 0.4
                elif isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == 'set':
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_set_operations(self, ast_tree: ast.AST) -> float:
        """Detect set operations"""
        score = 0.0
        set_methods = ['add', 'remove', 'discard', 'union', 'intersection', 'difference']
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in set_methods:
                    score += 0.25
        
        return min(score, 1.0)
    
    def _detect_membership_testing(self, ast_tree: ast.AST) -> float:
        """Detect membership testing (in/not in)"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    if isinstance(op, (ast.In, ast.NotIn)):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_array_operations(self, ast_tree: ast.AST) -> float:
        """Detect array/list operations"""
        score = 0.0
        array_methods = ['append', 'extend', 'insert', 'remove', 'pop', 'index', 'count']
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in array_methods:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _detect_slicing_operations(self, ast_tree: ast.AST) -> float:
        """Detect slicing operations"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_inplace_modifications(self, ast_tree: ast.AST) -> float:
        """Detect in-place modifications"""
        score = 0.0
        
        # Check for augmented assignments (+=, -=, etc.)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                score += 0.2
        
        # Check for sort() vs sorted()
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'sort':
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_string_methods(self, ast_tree: ast.AST) -> float:
        """Detect string method usage"""
        score = 0.0
        string_methods = ['split', 'join', 'replace', 'strip', 'lower', 'upper', 'find', 'startswith', 'endswith']
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in string_methods:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _detect_string_formatting(self, ast_tree: ast.AST) -> float:
        """Detect string formatting operations"""
        score = 0.0
        
        # Check for f-strings
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.JoinedStr):
                score += 0.3
        
        # Check for .format() method
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'format':
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_character_operations(self, ast_tree: ast.AST) -> float:
        """Detect character-level operations"""
        score = 0.0
        
        # Check for ord/chr functions
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['ord', 'chr', 'isalpha', 'isdigit']:
                score += 0.3
        
        # Check for character indexing
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                    score += 0.2
        
        return min(score, 1.0)

    # =================== GRAPH ALGORITHM HELPERS ===================
    
    def _detect_queue_based_traversal(self, ast_tree: ast.AST) -> float:
        """Detect queue-based graph traversal"""
        score = 0.0
        
        # Check for queue + graph operations
        queue_score = self._detect_fifo_operations(ast_tree)
        if queue_score > 0.3:
            # Check for neighbor iteration
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.For):
                    if isinstance(node.iter, ast.Subscript):
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_level_processing(self, ast_tree: ast.AST) -> float:
        """Detect level-by-level processing in BFS"""
        score = 0.0
        
        # Check for queue length in loop
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if len(node.iter.args) >= 1:
                            arg = node.iter.args[0]
                            if isinstance(arg, ast.Call):
                                if len(arg.attribute_chain) >= 2 and arg.attribute_chain[-1] == 'len':
                                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_bfs_function_naming(self) -> float:
        """Detect BFS-related function naming"""
        score = 0.0
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name']:
                if 'bfs' in call_info['function_name'].lower():
                    score += 0.4
        return min(score, 1.0)
    
    def _detect_indegree_calculation(self, ast_tree: ast.AST) -> float:
        """Detect indegree calculation for topological sort"""
        score = 0.0
        
        # Check for indegree variable
        for var in self.ast_analyzer.variable_names:
            if 'indegree' in var.lower() or 'in_degree' in var.lower():
                score += 0.5
        
        # Check for degree counting
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Subscript):
                    if isinstance(node.op, ast.Add):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_kahns_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Kahn's algorithm pattern"""
        score = 0.0
        
        # Check for zero indegree initialization
        indegree_score = self._detect_indegree_calculation(ast_tree)
        queue_score = self._detect_fifo_operations(ast_tree)
        
        if indegree_score > 0.3 and queue_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_dfs_topological_sort(self, ast_tree: ast.AST) -> float:
        """Detect DFS-based topological sort"""
        score = 0.0
        
        # Check for DFS + stack for result
        stack_score = self._detect_lifo_operations(ast_tree)
        if stack_score > 0.3:
            # Check for post-order addition to stack
            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_distance_array_initialization(self, ast_tree: ast.AST) -> float:
        """Detect distance array initialization"""
        score = 0.0
        
        # Check for distance/dist variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['dist', 'distance']):
                score += 0.4
        
        # Check for infinity initialization
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and elt.value == float('inf'):
                            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_edge_relaxation(self, ast_tree: ast.AST) -> float:
        """Detect edge relaxation operations"""
        score = 0.0
        
        # Check for distance update pattern: if dist[u] + weight < dist[v]
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Compare):
                    if len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Lt):
                        # Check if left side is addition
                        if isinstance(node.test.left, ast.BinOp) and isinstance(node.test.left.op, ast.Add):
                            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_dijkstra_pattern(self, ast_tree: ast.AST) -> float:
        """Detect Dijkstra's algorithm pattern"""
        score = 0.0
        
        # Check for priority queue + distance array
        heap_score = self._detect_priority_operations(ast_tree)
        dist_score = self._detect_distance_array_initialization(ast_tree)
        
        if heap_score > 0.3 and dist_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_bellman_ford_pattern(self, ast_tree: ast.AST) -> float:
        """Detect Bellman-Ford algorithm pattern"""
        score = 0.0
        
        # Check for V-1 iterations
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                range_info = loop_info.get('iterator', {}).get('range_info', {})
                if range_info and range_info.get('stop'):
                    if 'n-1' in str(range_info['stop']) or 'V-1' in str(range_info['stop']):
                        score += 0.5
        
        return min(score, 1.0)
    
    def _detect_floyd_warshall_pattern(self, ast_tree: ast.AST) -> float:
        """Detect Floyd-Warshall algorithm pattern"""
        score = 0.0
        
        # Check for triple nested loops
        nested_count = 0
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['nested_loops'] >= 2:
                nested_count += 1
        
        if nested_count > 0:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_parent_array(self, ast_tree: ast.AST) -> float:
        """Detect parent array for Union-Find"""
        score = 0.0
        
        # Check for parent variable
        for var in self.ast_analyzer.variable_names:
            if 'parent' in var.lower():
                score += 0.4
        
        # Check for self-parent initialization
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.target, ast.Subscript):
                    if isinstance(node.value, ast.Name):
                        # parent[i] = i pattern
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_find_function(self, ast_tree: ast.AST) -> float:
        """Detect find function in Union-Find"""
        score = 0.0
        
        # Check for find function definition
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'find' in node.name.lower():
                    score += 0.5
        
        # Check for path compression pattern
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.target, ast.Subscript):
                    if isinstance(node.value, ast.Call):
                        if isinstance(node.value.func, ast.Name) and 'find' in node.value.func.id:
                            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_union_function(self, ast_tree: ast.AST) -> float:
        """Detect union function in Union-Find"""
        score = 0.0
        
        # Check for union function definition
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'union' in node.name.lower():
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_rank_optimization(self, ast_tree: ast.AST) -> float:
        """Detect rank/size optimization in Union-Find"""
        score = 0.0
        
        # Check for rank or size variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['rank', 'size']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_kruskal_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Kruskal's MST algorithm"""
        score = 0.0
        
        # Check for edge sorting + Union-Find
        sort_score = self._detect_edge_sorting_by_weight(ast_tree)
        uf_score = self._detect_find_function(ast_tree)
        
        if sort_score > 0.3 and uf_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_prim_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Prim's MST algorithm"""
        score = 0.0
        
        # Check for priority queue + visited set
        heap_score = self._detect_priority_operations(ast_tree)
        visited_score = self._detect_membership_testing(ast_tree)
        
        if heap_score > 0.3 and visited_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_edge_sorting_by_weight(self, ast_tree: ast.AST) -> float:
        """Detect edge sorting by weight"""
        score = 0.0
        
        # Check for sort with key parameter
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['sort', 'sorted']:
                if 'key' in call_info['keywords']:
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_color_assignment(self, ast_tree: ast.AST) -> float:
        """Detect color assignment in graph coloring"""
        score = 0.0
        
        # Check for color variable
        for var in self.ast_analyzer.variable_names:
            if 'color' in var.lower():
                score += 0.4
        
        # Check for color array indexing
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and 'color' in target.value.id.lower():
                            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_adjacency_conflict_checking(self, ast_tree: ast.AST) -> float:
        """Detect adjacency conflict checking in coloring"""
        score = 0.0
        
        # Check for neighbor color comparison
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Compare):
                # Look for color[neighbor] == color[node] patterns
                if len(node.comparators) >= 1:
                    left = node.left
                    right = node.comparators[0]
                    if (isinstance(left, ast.Subscript) and isinstance(right, ast.Subscript)):
                        if (isinstance(left.value, ast.Name) and isinstance(right.value, ast.Name)):
                            if 'color' in left.value.id.lower() and 'color' in right.value.id.lower():
                                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_backtracking_coloring(self, ast_tree: ast.AST) -> float:
        """Detect backtracking in graph coloring"""
        score = 0.0
        
        # Check for recursive function with color assignment/unassignment
        color_score = self._detect_color_assignment(ast_tree)
        
        # Check for recursive calls
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == node.name:
                            if color_score > 0.3:
                                score += 0.5
        
        return min(score, 1.0)

    # =================== TREE ALGORITHM HELPERS ===================
    
    def _detect_tree_node_access(self, ast_tree: ast.AST) -> float:
        """Detect tree node access patterns"""
        score = 0.0
        
        # Check for .left, .right, .val access
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Attribute):
                if node.attr in ['left', 'right', 'val', 'data']:
                    score += 0.25
        
        return min(score, 1.0)
    
    def _detect_inorder_traversal(self, ast_tree: ast.AST) -> float:
        """Detect inorder traversal pattern"""
        score = 0.0
        
        # Look for left -> process -> right pattern in recursion
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'inorder' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_preorder_traversal(self, ast_tree: ast.AST) -> float:
        """Detect preorder traversal pattern"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'preorder' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_postorder_traversal(self, ast_tree: ast.AST) -> float:
        """Detect postorder traversal pattern"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'postorder' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_level_order_traversal(self, ast_tree: ast.AST) -> float:
        """Detect level order traversal pattern"""
        score = 0.0
        
        # Check for queue + tree traversal
        queue_score = self._detect_fifo_operations(ast_tree)
        tree_access_score = self._detect_tree_node_access(ast_tree)
        
        if queue_score > 0.3 and tree_access_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_treenode_usage(self, ast_tree: ast.AST) -> float:
        """Detect TreeNode class usage"""
        score = 0.0
        
        # Check for TreeNode in variable names or function calls
        for var in self.ast_analyzer.variable_names:
            if 'treenode' in var.lower() or 'node' in var.lower():
                score += 0.3
        
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] and 'treenode' in call_info['function_name'].lower():
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_binary_tree_operations(self, ast_tree: ast.AST) -> float:
        """Detect binary tree specific operations"""
        score = 0.0
        
        # Check for binary tree specific patterns
        tree_access_score = self._detect_tree_node_access(ast_tree)
        if tree_access_score > 0.3:
            score += 0.4
        
        # Check for binary tree operations like insert, delete, search
        binary_ops = ['insert', 'delete', 'search', 'find']
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name']:
                if any(op in call_info['function_name'].lower() for op in binary_ops):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_tree_modification(self, ast_tree: ast.AST) -> float:
        """Detect tree construction/modification"""
        score = 0.0
        
        # Check for node creation
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and 'node' in node.func.id.lower():
                    score += 0.3
        
        # Check for tree building patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr in ['left', 'right']:
                            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_trie_class_definition(self, ast_tree: ast.AST) -> float:
        """Detect Trie class definition"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                if 'trie' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_character_navigation(self, ast_tree: ast.AST) -> float:
        """Detect character-based navigation in trie"""
        score = 0.0
        
        # Check for character to index conversion
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'ord':
                score += 0.4
        
        # Check for character indexing
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.BinOp):
                    if isinstance(node.slice.op, ast.Sub):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_trie_operations(self, ast_tree: ast.AST) -> float:
        """Detect trie operations (insert, search, startsWith)"""
        score = 0.0
        
        trie_ops = ['insert', 'search', 'startswith', 'prefix']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(op in node.name.lower() for op in trie_ops):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_children_dictionary(self, ast_tree: ast.AST) -> float:
        """Detect children dictionary usage in trie"""
        score = 0.0
        
        # Check for children variable
        for var in self.ast_analyzer.variable_names:
            if 'children' in var.lower():
                score += 0.4
        
        # Check for dictionary operations on children
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Attribute):
                if node.attr == 'children':
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_segment_tree_class(self, ast_tree: ast.AST) -> float:
        """Detect segment tree class definition"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                if any(keyword in node.name.lower() for keyword in ['segment', 'segtree']):
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_tree_indexing_patterns(self, ast_tree: ast.AST) -> float:
        """Detect tree indexing patterns (2*i, 2*i+1)"""
        score = 0.0
        
        # Check for 2*i patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Mult):
                    if isinstance(node.left, ast.Constant) and node.left.value == 2:
                        score += 0.4
                    elif isinstance(node.right, ast.Constant) and node.right.value == 2:
                        score += 0.4
        
        # Check for 2*i+1 patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Add):
                    if isinstance(node.right, ast.Constant) and node.right.value == 1:
                        if isinstance(node.left, ast.BinOp):
                            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_range_query_operations(self, ast_tree: ast.AST) -> float:
        """Detect range query operations"""
        score = 0.0
        
        # Check for range-related function names
        range_ops = ['query', 'update', 'range_sum', 'range_min', 'range_max']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(op in node.name.lower() for op in range_ops):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_fenwick_class(self, ast_tree: ast.AST) -> float:
        """Detect Fenwick tree class"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                if any(keyword in node.name.lower() for keyword in ['fenwick', 'bit', 'binary_indexed']):
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_fenwick_bit_patterns(self, ast_tree: ast.AST) -> float:
        """Detect Fenwick tree bit manipulation patterns"""
        score = 0.0
        
        # Check for i & -i pattern
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.BitAnd):
                    if isinstance(node.right, ast.UnaryOp):
                        if isinstance(node.right.op, ast.USub):
                            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_prefix_sum_operations(self, ast_tree: ast.AST) -> float:
        """Detect prefix sum operations"""
        score = 0.0
        
        # Check for prefix sum variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['prefix', 'sum']):
                score += 0.3
        
        return min(score, 1.0)

    # =================== ADVANCED DATA STRUCTURE HELPERS ===================
    
    def _detect_monotonic_stack_pattern(self, ast_tree: ast.AST) -> float:
        """Detect monotonic stack pattern"""
        score = 0.0
        
        # Check for while loop with stack comparison and pop
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.While):
                # Look for stack comparison in condition
                for child in ast.walk(node.test):
                    if isinstance(child, ast.Compare):
                        score += 0.4
                
                # Look for stack.pop() in body
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if len(child.args) == 0:  # pop() without args
                            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_next_element_patterns(self, ast_tree: ast.AST) -> float:
        """Detect next greater/smaller element patterns"""
        score = 0.0
        
        # Check for "next" in variable names
        for var in self.ast_analyzer.variable_names:
            if 'next' in var.lower() and any(keyword in var.lower() for keyword in ['greater', 'smaller', 'larger']):
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_stack_order_maintenance(self, ast_tree: ast.AST) -> float:
        """Detect stack order maintenance"""
        score = 0.0
        
        # Check for stack with comparison-based operations
        stack_score = self._detect_lifo_operations(ast_tree)
        if stack_score > 0.3:
            # Check for comparison in while loop
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.While):
                    if isinstance(node.test, ast.BoolOp):
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_monotonic_deque_pattern(self, ast_tree: ast.AST) -> float:
        """Detect monotonic deque pattern"""
        score = 0.0
        
        # Check for deque + comparison-based operations
        deque_score = self._detect_deque_operations(ast_tree)
        if deque_score > 0.3:
            # Check for front/back comparisons
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.While):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_sliding_window_maximum(self, ast_tree: ast.AST) -> float:
        """Detect sliding window maximum pattern"""
        score = 0.0
        
        # Check for deque + window operations
        deque_score = self._detect_monotonic_deque_pattern(ast_tree)
        window_score = self._detect_sliding_mechanism(ast_tree)
        
        if deque_score > 0.3 and window_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_deque_maintenance(self, ast_tree: ast.AST) -> float:
        """Detect deque maintenance patterns"""
        score = 0.0
        
        # Check for operations on both ends
        double_ended_score = self._detect_double_ended_operations(ast_tree)
        if double_ended_score > 0.3:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_coordinate_mapping(self, ast_tree: ast.AST) -> float:
        """Detect coordinate mapping for compression"""
        score = 0.0
        
        # Check for coordinate variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['coord', 'compress', 'map']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_unique_coordinate_extraction(self, ast_tree: ast.AST) -> float:
        """Detect unique coordinate extraction"""
        score = 0.0
        
        # Check for set operations on coordinates
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'set':
                score += 0.3
        
        # Check for sorted unique values
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'sorted':
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_discretization_patterns(self, ast_tree: ast.AST) -> float:
        """Detect discretization patterns"""
        score = 0.0
        
        # Check for enumerate usage
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'enumerate':
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_2d_sparse_table(self, ast_tree: ast.AST) -> float:
        """Detect 2D sparse table"""
        score = 0.0
        
        # Check for 2D table initialization
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.ListComp):
                    # Check for nested list comprehension
                    for comp in ast.walk(node.value):
                        if isinstance(comp, ast.ListComp) and comp != node.value:
                            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_logarithmic_preprocessing(self, ast_tree: ast.AST) -> float:
        """Detect logarithmic preprocessing"""
        score = 0.0
        
        # Check for log calculations
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] and 'log' in call_info['function_name']:
                score += 0.4
        
        # Check for bit_length usage
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'bit_length':
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_rmq_operations(self, ast_tree: ast.AST) -> float:
        """Detect range minimum/maximum query operations"""
        score = 0.0
        
        # Check for RMQ function names
        rmq_ops = ['rmq', 'range_min', 'range_max', 'query_min', 'query_max']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(op in node.name.lower() for op in rmq_ops):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_advanced_dsu_features(self, ast_tree: ast.AST) -> float:
        """Detect advanced DSU features"""
        score = 0.0
        
        # Check for additional DSU optimizations
        rank_score = self._detect_rank_optimization(ast_tree)
        if rank_score > 0.3:
            score += 0.4
        
        return min(score, 1.0)

    # =================== MATHEMATICAL PATTERN HELPERS ===================
    
    def _detect_number_theory_functions(self, ast_tree: ast.AST) -> float:
        """Detect number theory functions"""
        score = 0.0
        
        nt_functions = ['gcd', 'lcm', 'prime', 'factor', 'euler', 'totient']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(func in node.name.lower() for func in nt_functions):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_prime_operations(self, ast_tree: ast.AST) -> float:
        """Detect prime-related operations"""
        score = 0.0
        
        # Check for prime variables or functions
        for var in self.ast_analyzer.variable_names:
            if 'prime' in var.lower() or 'sieve' in var.lower():
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_gcd_lcm_operations(self, ast_tree: ast.AST) -> float:
        """Detect GCD/LCM operations"""
        score = 0.0
        
        # Check for math.gcd usage
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                obj, method = call_info['attribute_chain'][0], call_info['attribute_chain'][-1]
                if obj == 'math' and method in ['gcd', 'lcm']:
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_sieve_patterns(self, ast_tree: ast.AST) -> float:
        """Detect sieve patterns"""
        score = 0.0
        
        # Check for sieve variables
        for var in self.ast_analyzer.variable_names:
            if 'sieve' in var.lower():
                score += 0.5
        
        # Check for prime marking patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.For):
                # Check for nested loop with multiplication
                for child in ast.walk(node):
                    if isinstance(child, ast.For):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_factorial_calculations(self, ast_tree: ast.AST) -> float:
        """Detect factorial calculations"""
        score = 0.0
        
        # Check for factorial function
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'factorial' in node.name.lower():
                    score += 0.5
        
        # Check for factorial variable
        for var in self.ast_analyzer.variable_names:
            if 'fact' in var.lower():
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_combination_permutation(self, ast_tree: ast.AST) -> float:
        """Detect combination/permutation calculations"""
        score = 0.0
        
        combo_terms = ['comb', 'perm', 'choose', 'ncr', 'npr']
        for var in self.ast_analyzer.variable_names:
            if any(term in var.lower() for term in combo_terms):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_pascal_triangle(self, ast_tree: ast.AST) -> float:
        """Detect Pascal's triangle"""
        score = 0.0
        
        # Check for Pascal or triangle variables
        for var in self.ast_analyzer.variable_names:
            if 'pascal' in var.lower() or 'triangle' in var.lower():
                score += 0.5
        
        # Check for C[i][j] = C[i-1][j-1] + C[i-1][j] pattern
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_random_usage(self, ast_tree: ast.AST) -> float:
        """Detect random number usage"""
        score = 0.0
        
        # Check for random imports
        for imp in self.ast_analyzer.imports:
            if 'random' in imp.lower():
                score += 0.4
        
        # Check for random function calls
        random_funcs = ['random', 'randint', 'choice', 'shuffle']
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in random_funcs:
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_probability_calculations(self, ast_tree: ast.AST) -> float:
        """Detect probability calculations"""
        score = 0.0
        
        # Check for probability variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['prob', 'probability', 'chance']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_statistical_operations(self, ast_tree: ast.AST) -> float:
        """Detect statistical operations"""
        score = 0.0
        
        stat_funcs = ['mean', 'median', 'variance', 'std', 'average']
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in stat_funcs:
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_coordinate_operations(self, ast_tree: ast.AST) -> float:
        """Detect coordinate operations"""
        score = 0.0
        
        # Check for x, y coordinates
        coord_vars = 0
        for var in self.ast_analyzer.variable_names:
            if var in ['x', 'y', 'z'] or any(keyword in var.lower() for keyword in ['coord', 'point']):
                coord_vars += 1
        
        if coord_vars >= 2:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_geometric_calculations(self, ast_tree: ast.AST) -> float:
        """Detect geometric calculations"""
        score = 0.0
        
        # Check for geometric functions
        geom_funcs = ['distance', 'angle', 'area', 'perimeter', 'cross_product', 'dot_product']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(func in node.name.lower() for func in geom_funcs):
                    score += 0.3
        
        # Check for sqrt usage (distance calculations)
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'sqrt' or (len(call_info['attribute_chain']) >= 2 and call_info['attribute_chain'][-1] == 'sqrt'):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_convex_hull_algorithms(self, ast_tree: ast.AST) -> float:
        """Detect convex hull algorithms"""
        score = 0.0
        
        # Check for convex hull function names
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() for keyword in ['convex', 'hull', 'graham', 'jarvis']):
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_bitwise_operations(self, ast_tree: ast.AST) -> float:
        """Detect bitwise operations"""
        score = 0.0
        
        # Check for bitwise operators in AST
        bitwise_ops = [ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.Invert]
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if type(node.op) in bitwise_ops:
                    score += 0.3
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.Invert):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_bit_manipulation_tricks(self, ast_tree: ast.AST) -> float:
        """Detect bit manipulation tricks"""
        score = 0.0
        
        # Check for common bit tricks
        # x & -x (isolate rightmost set bit)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                if isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub):
                    score += 0.5
        
        # x & (x-1) (clear rightmost set bit)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
                if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Sub):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_bit_masking(self, ast_tree: ast.AST) -> float:
        """Detect bit masking operations"""
        score = 0.0
        
        # Check for mask variables
        for var in self.ast_analyzer.variable_names:
            if 'mask' in var.lower():
                score += 0.4
        
        # Check for bit shifting for masks
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.LShift, ast.RShift)):
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_modulo_operations(self, ast_tree: ast.AST) -> float:
        """Detect modulo operations"""
        score = 0.0
        
        # Check for modulo operator
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                score += 0.3
        
        # Check for MOD constant
        for var in self.ast_analyzer.variable_names:
            if 'mod' in var.upper():
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_modular_exponentiation(self, ast_tree: ast.AST) -> float:
        """Detect modular exponentiation"""
        score = 0.0
        
        # Check for pow with 3 arguments (base, exp, mod)
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'pow' and call_info['args'] == 3:
                score += 0.6
        
        return min(score, 1.0)
    
    def _detect_modular_inverse(self, ast_tree: ast.AST) -> float:
        """Detect modular inverse"""
        score = 0.0
        
        # Check for inverse function
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'inverse' in node.name.lower() or 'inv' in node.name.lower():
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_2d_array_operations(self, ast_tree: ast.AST) -> float:
        """Detect 2D array operations"""
        score = 0.0
        
        # Check for matrix variable names
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['matrix', 'grid', 'board']):
                score += 0.3
        
        # Check for double indexing [i][j]
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Subscript):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_matrix_multiplication(self, ast_tree: ast.AST) -> float:
        """Detect matrix multiplication"""
        score = 0.0
        
        # Check for triple nested loops (matrix multiplication pattern)
        max_nesting = 0
        for loop_info in self.ast_analyzer.loop_structures:
            max_nesting = max(max_nesting, loop_info['nested_loops'])
        
        if max_nesting >= 2:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_matrix_transformations(self, ast_tree: ast.AST) -> float:
        """Detect matrix transformations"""
        score = 0.0
        
        # Check for transformation function names
        transform_ops = ['transpose', 'rotate', 'flip', 'transform']
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(op in node.name.lower() for op in transform_ops):
                    score += 0.4
        
        return min(score, 1.0)

    # =================== SORTING & SELECTION HELPERS ===================
    
    def _detect_sort_with_key(self, ast_tree: ast.AST) -> float:
        """Detect sort with key parameter"""
        score = 0.0
        
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['sort', 'sorted']:
                if 'key' in call_info['keywords']:
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_lambda_sorting(self, ast_tree: ast.AST) -> float:
        """Detect lambda functions in sorting"""
        score = 0.0
        
        # Check for lambda expressions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Lambda):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_custom_comparator(self, ast_tree: ast.AST) -> float:
        """Detect custom comparator functions"""
        score = 0.0
        
        # Check for compare/cmp function definitions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() for keyword in ['compare', 'cmp', 'comparator']):
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_bucket_initialization(self, ast_tree: ast.AST) -> float:
        """Detect bucket initialization"""
        score = 0.0
        
        # Check for bucket variables
        for var in self.ast_analyzer.variable_names:
            if 'bucket' in var.lower() or 'bin' in var.lower():
                score += 0.4
        
        # Check for list of lists initialization
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ListComp):
                for comp in ast.walk(node):
                    if isinstance(comp, ast.List) and comp != node:
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_bucket_distribution(self, ast_tree: ast.AST) -> float:
        """Detect bucket distribution logic"""
        score = 0.0
        
        # Check for bucket indexing with division/modulo
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.BinOp):
                    if isinstance(node.slice.op, (ast.FloorDiv, ast.Mod)):
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_bucket_merging(self, ast_tree: ast.AST) -> float:
        """Detect bucket sorting and merging"""
        score = 0.0
        
        # Check for sorting buckets individually
        bucket_score = self._detect_bucket_initialization(ast_tree)
        sort_score = self._detect_sort_with_key(ast_tree)
        
        if bucket_score > 0.3 and sort_score > 0.2:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_count_array(self, ast_tree: ast.AST) -> float:
        """Detect count array for counting sort"""
        score = 0.0
        
        # Check for count variables
        for var in self.ast_analyzer.variable_names:
            if 'count' in var.lower():
                score += 0.4
        
        # Check for array initialization with zeros
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.BinOp):
                    if isinstance(node.value.op, ast.Mult):
                        # [0] * n pattern
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_frequency_counting(self, ast_tree: ast.AST) -> float:
        """Detect frequency counting patterns"""
        score = 0.0
        
        # Check for frequency/count increment
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add):
                    if isinstance(node.target, ast.Subscript):
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_stable_reconstruction(self, ast_tree: ast.AST) -> float:
        """Detect stable sorting reconstruction"""
        score = 0.0
        
        # Check for backward iteration in reconstruction
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                range_info = loop_info.get('iterator', {}).get('range_info', {})
                if range_info and range_info.get('step'):
                    if str(range_info['step']) == '-1':
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_kth_element_selection(self, ast_tree: ast.AST) -> float:
        """Detect kth element selection"""
        score = 0.0
        
        # Check for k variable
        for var in self.ast_analyzer.variable_names:
            if var.lower() in ['k', 'kth']:
                score += 0.3
        
        # Check for quickselect function name
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'select' in node.name.lower() or 'kth' in node.name.lower():
                    score += 0.4
        
        return min(score, 1.0)
    
    def _detect_recursive_selection(self, ast_tree: ast.AST) -> float:
        """Detect recursive selection pattern"""
        score = 0.0
        
        # Check for recursion + partitioning
        partition_score = self._detect_partition_function(ast_tree)
        
        # Check for recursive calls
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == node.name:
                            if partition_score > 0.3:
                                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_two_pointer_merge(self, ast_tree: ast.AST) -> float:
        """Detect two-pointer merge pattern"""
        score = 0.0
        
        # Check for two index variables in while loop
        pointer_vars = ['i', 'j', 'left', 'right', 'p1', 'p2']
        found_pointers = 0
        
        for var in self.ast_analyzer.variable_names:
            if var in pointer_vars:
                found_pointers += 1
        
        if found_pointers >= 2:
            # Check for while loop with both pointers
            for loop_info in self.ast_analyzer.loop_structures:
                if loop_info['type'] == 'While':
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_merge_function(self, ast_tree: ast.AST) -> float:
        """Detect merge function"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'merge' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_divide_and_merge(self, ast_tree: ast.AST) -> float:
        """Detect divide and merge pattern"""
        score = 0.0
        
        merge_score = self._detect_merge_function(ast_tree)
        divide_score = self._detect_problem_division(ast_tree)
        
        if merge_score > 0.3 and divide_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_pivot_selection(self, ast_tree: ast.AST) -> float:
        """Detect pivot selection"""
        score = 0.0
        
        # Check for pivot variable
        for var in self.ast_analyzer.variable_names:
            if 'pivot' in var.lower():
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_inplace_partitioning(self, ast_tree: ast.AST) -> float:
        """Detect in-place partitioning"""
        score = 0.0
        
        # Check for swap operations
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] and 'swap' in call_info['function_name'].lower():
                score += 0.4
        
        # Check for element swapping pattern
        pivot_score = self._detect_pivot_selection(ast_tree)
        if pivot_score > 0.3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_partition_function(self, ast_tree: ast.AST) -> float:
        """Detect partition function"""
        score = 0.0
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'partition' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)

    # =================== GREEDY PATTERN HELPERS ===================
    
    def _detect_sort_before_selection(self, ast_tree: ast.AST) -> float:
        """Detect sorting before greedy selection"""
        score = 0.0
        
        # Check for sort followed by iteration
        has_sort = False
        has_iteration = False
        
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['sort', 'sorted']:
                has_sort = True
        
        if self.ast_analyzer.loop_structures:
            has_iteration = True
        
        if has_sort and has_iteration:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_local_optimal_choices(self, ast_tree: ast.AST) -> float:
        """Detect local optimal choices"""
        score = 0.0
        
        # Check for max/min selections in loops
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['max', 'min']:
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_greedy_selection_patterns(self, ast_tree: ast.AST) -> float:
        """Detect greedy selection patterns"""
        score = 0.0
        
        # Check for greedy keywords in variable names
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['best', 'optimal', 'choice', 'select']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_interval_structures(self, ast_tree: ast.AST) -> float:
        """Detect interval data structures"""
        score = 0.0
        
        # Check for interval-related variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['interval', 'start', 'end', 'begin', 'finish']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_end_time_sorting(self, ast_tree: ast.AST) -> float:
        """Detect sorting by end time"""
        score = 0.0
        
        # Check for lambda with second element access
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Lambda):
                if isinstance(node.body, ast.Subscript):
                    if isinstance(node.body.slice, ast.Constant) and node.body.slice.value == 1:
                        score += 0.5
        
        return min(score, 1.0)
    
    def _detect_overlap_detection(self, ast_tree: ast.AST) -> float:
        """Detect interval overlap detection"""
        score = 0.0
        
        # Check for overlap conditions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Compare):
                # Look for start < end type comparisons
                if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Lt, ast.Le, ast.Gt, ast.GtE)):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _detect_activity_patterns(self, ast_tree: ast.AST) -> float:
        """Detect activity-specific patterns"""
        score = 0.0
        
        # Check for activity variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['activity', 'job', 'task']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_frequency_processing(self, ast_tree: ast.AST) -> float:
        """Detect frequency processing for Huffman"""
        score = 0.0
        
        # Check for frequency variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['freq', 'frequency', 'count']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_heap_tree_building(self, ast_tree: ast.AST) -> float:
        """Detect heap-based tree building"""
        score = 0.0
        
        # Check for heap + tree combination
        heap_score = self._detect_priority_operations(ast_tree)
        tree_score = self._detect_tree_node_access(ast_tree)
        
        if heap_score > 0.3 and tree_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_binary_encoding(self, ast_tree: ast.AST) -> float:
        """Detect binary encoding patterns"""
        score = 0.0
        
        # Check for encoding variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['code', 'encoding', 'binary']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_value_weight_ratio(self, ast_tree: ast.AST) -> float:
        """Detect value-to-weight ratio calculation"""
        score = 0.0
        
        # Check for ratio variables
        for var in self.ast_analyzer.variable_names:
            if 'ratio' in var.lower():
                score += 0.4
        
        # Check for division operations (value/weight)
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_fractional_taking(self, ast_tree: ast.AST) -> float:
        """Detect fractional taking in knapsack"""
        score = 0.0
        
        # Check for fractional or partial variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['fraction', 'partial', 'remain']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_capacity_constraints(self, ast_tree: ast.AST) -> float:
        """Detect capacity constraints"""
        score = 0.0
        
        # Check for capacity/weight variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['capacity', 'weight', 'limit']):
                score += 0.3
        
        return min(score, 1.0)

    # =================== STRING ALGORITHM HELPERS ===================
    
    def _detect_kmp_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect KMP algorithm"""
        score = 0.0
        
        # Check for failure function
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() for keyword in ['failure', 'lps', 'prefix']):
                    score += 0.5
        
        # Check for KMP-specific variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['lps', 'failure', 'pi']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_boyer_moore_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Boyer-Moore algorithm"""
        score = 0.0
        
        # Check for bad character table
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['bad', 'char', 'shift']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_rabin_karp_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Rabin-Karp algorithm"""
        score = 0.0
        
        # Check for rolling hash
        hash_score = self._detect_rolling_hash(ast_tree)
        if hash_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_pattern_text_variables(self, ast_tree: ast.AST) -> float:
        """Detect pattern and text variables"""
        score = 0.0
        
        # Check for pattern/text variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['pattern', 'text', 'string', 'needle', 'haystack']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_hash_calculation(self, ast_tree: ast.AST) -> float:
        """Detect hash calculation patterns"""
        score = 0.0
        
        # Check for hash variables
        for var in self.ast_analyzer.variable_names:
            if 'hash' in var.lower():
                score += 0.4
        
        # Check for modular arithmetic
        mod_score = self._detect_modulo_operations(ast_tree)
        if mod_score > 0.3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_rolling_mechanism(self, ast_tree: ast.AST) -> float:
        """Detect rolling hash mechanism"""
        score = 0.0
        
        # Check for hash update in loop
        for loop_info in self.ast_analyzer.loop_structures:
            for node in ast.walk(loop_info['node']):
                if isinstance(node, ast.AugAssign):
                    if isinstance(node.target, ast.Name) and 'hash' in node.target.id.lower():
                        score += 0.5
        
        return min(score, 1.0)
    
    def _detect_polynomial_hash(self, ast_tree: ast.AST) -> float:
        """Detect polynomial rolling hash"""
        score = 0.0
        
        # Check for base variable
        for var in self.ast_analyzer.variable_names:
            if var.lower() in ['base', 'b']:
                score += 0.3
        
        # Check for power operations
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_suffix_generation(self, ast_tree: ast.AST) -> float:
        """Detect suffix generation"""
        score = 0.0
        
        # Check for suffix variables
        for var in self.ast_analyzer.variable_names:
            if 'suffix' in var.lower():
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_suffix_sorting(self, ast_tree: ast.AST) -> float:
        """Detect suffix sorting"""
        score = 0.0
        
        suffix_score = self._detect_suffix_generation(ast_tree)
        sort_score = self._detect_sort_with_key(ast_tree)
        
        if suffix_score > 0.3 and sort_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_lcp_array(self, ast_tree: ast.AST) -> float:
        """Detect LCP (Longest Common Prefix) array"""
        score = 0.0
        
        # Check for LCP variables
        for var in self.ast_analyzer.variable_names:
            if 'lcp' in var.lower():
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_string_comparison_dp(self, ast_tree: ast.AST) -> float:
        """Detect string comparison DP"""
        score = 0.0
        
        # Check for 2D DP with string parameters
        dp_score = self._detect_iterative_dp_filling(ast_tree)
        string_vars = 0
        
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['str', 'text', 's1', 's2']):
                string_vars += 1
        
        if dp_score > 0.3 and string_vars >= 2:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_character_matching_loops(self, ast_tree: ast.AST) -> float:
        """Detect character matching in nested loops"""
        score = 0.0
        
        # Check for nested loops with character comparison
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['nested_loops'] >= 1:
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.Compare):
                        # Check for string indexing comparison
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_lcs_specific_patterns(self, ast_tree: ast.AST) -> float:
        """Detect LCS-specific patterns"""
        score = 0.0
        
        # Check for LCS function name
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'lcs' in node.name.lower():
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_string_reversal_comparison(self, ast_tree: ast.AST) -> float:
        """Detect string reversal comparison"""
        score = 0.0
        
        # Check for string slicing with [::-1]
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    if (node.slice.step and isinstance(node.slice.step, ast.UnaryOp) 
                        and isinstance(node.slice.step.op, ast.USub)):
                        score += 0.5
        
        return min(score, 1.0)
    
    def _detect_two_pointer_palindrome(self, ast_tree: ast.AST) -> float:
        """Detect two-pointer palindrome check"""
        score = 0.0
        
        # Check for two pointers + string comparison
        pointer_score = self._detect_two_pointers(ast_tree)
        string_score = self._detect_character_matching_loops(ast_tree)
        
        if pointer_score > 0.3 and string_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_manacher_algorithm(self, ast_tree: ast.AST) -> float:
        """Detect Manacher's algorithm"""
        score = 0.0
        
        # Check for Manacher function name
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'manacher' in node.name.lower():
                    score += 0.7
        
        return min(score, 1.0)
    
    def _detect_character_frequency_comparison(self, ast_tree: ast.AST) -> float:
        """Detect character frequency comparison"""
        score = 0.0
        
        # Check for Counter usage
        counter_score = self._detect_counter_usage(ast_tree)
        if counter_score > 0.3:
            score += 0.5
        
        # Check for frequency counting
        freq_score = self._detect_frequency_counting(ast_tree)
        if freq_score > 0.3:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_sorting_comparison(self, ast_tree: ast.AST) -> float:
        """Detect sorting comparison for anagrams"""
        score = 0.0
        
        # Check for sorted function on strings
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'sorted':
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_counter_usage(self, ast_tree: ast.AST) -> float:
        """Detect Counter usage"""
        score = 0.0
        
        # Check for Counter import/usage
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'Counter':
                score += 0.6
        
        return min(score, 1.0)

    # =================== ADVANCED PATTERN HELPERS ===================
    
    def _detect_state_restoration_pattern(self, ast_tree: ast.AST) -> float:
        """Detect state restoration pattern in backtracking"""
        score = 0.0
        
        # Check for append/pop pairs (state save/restore)
        append_count = 0
        pop_count = 0
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method == 'append':
                    append_count += 1
                elif method == 'pop':
                    pop_count += 1
        
        if append_count > 0 and pop_count > 0:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_exploration_undo_pattern(self, ast_tree: ast.AST) -> float:
        """Detect exploration with undo pattern"""
        score = 0.0
        
        # Check for add/remove pairs
        add_count = 0
        remove_count = 0
        
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                method = call_info['attribute_chain'][-1]
                if method in ['add', 'append']:
                    add_count += 1
                elif method in ['remove', 'pop', 'discard']:
                    remove_count += 1
        
        if add_count > 0 and remove_count > 0:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_pruning_conditions(self, ast_tree: ast.AST) -> float:
        """Detect pruning conditions"""
        score = 0.0
        
        # Check for early returns in recursive functions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        score += 0.3
        
        return min(score, 1.0)
    
    def _detect_bound_calculation(self, ast_tree: ast.AST) -> float:
        """Detect bound calculation"""
        score = 0.0
        
        # Check for bound variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['bound', 'limit', 'best']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_bound_pruning(self, ast_tree: ast.AST) -> float:
        """Detect pruning based on bounds"""
        score = 0.0
        
        bound_score = self._detect_bound_calculation(ast_tree)
        pruning_score = self._detect_pruning_conditions(ast_tree)
        
        if bound_score > 0.3 and pruning_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_best_solution_tracking(self, ast_tree: ast.AST) -> float:
        """Detect best solution tracking"""
        score = 0.0
        
        # Check for best/optimal variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['best', 'optimal', 'maximum', 'minimum']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_problem_space_division(self, ast_tree: ast.AST) -> float:
        """Detect problem space division"""
        score = 0.0
        
        # Check for half/split variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['half', 'split', 'left_part', 'right_part']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_two_way_exploration(self, ast_tree: ast.AST) -> float:
        """Detect two-way exploration"""
        score = 0.0
        
        # Check for forward/backward variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['forward', 'backward', 'front', 'back']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_meeting_point_logic(self, ast_tree: ast.AST) -> float:
        """Detect meeting point logic"""
        score = 0.0
        
        # Check for meet/middle variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['meet', 'middle', 'intersection']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_sqrt_calculation(self, ast_tree: ast.AST) -> float:
        """Detect sqrt calculation for block size"""
        score = 0.0
        
        # Check for sqrt function calls
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == 'sqrt' or (len(call_info['attribute_chain']) >= 2 and call_info['attribute_chain'][-1] == 'sqrt'):
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_block_processing(self, ast_tree: ast.AST) -> float:
        """Detect block-based processing"""
        score = 0.0
        
        # Check for block variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['block', 'chunk', 'segment']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_range_query_optimization(self, ast_tree: ast.AST) -> float:
        """Detect range query optimization"""
        score = 0.0
        
        sqrt_score = self._detect_sqrt_calculation(ast_tree)
        block_score = self._detect_block_processing(ast_tree)
        
        if sqrt_score > 0.3 and block_score > 0.3:
            score += 0.6
        
        return min(score, 1.0)
    
    def _detect_heavy_light_classification(self, ast_tree: ast.AST) -> float:
        """Detect heavy/light edge classification"""
        score = 0.0
        
        # Check for heavy/light variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['heavy', 'light']):
                score += 0.5
        return min(score, 1.0)
    
    def _detect_chain_decomposition(self, ast_tree: ast.AST) -> float:
        """Detect chain decomposition"""
        score = 0.0
        
        # Check for chain variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['chain', 'head', 'decomp']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_path_query_optimization(self, ast_tree: ast.AST) -> float:
        """Detect path query optimization"""
        score = 0.0
        
        # Check for path variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['path', 'query', 'lca']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_centroid_finding(self, ast_tree: ast.AST) -> float:
        """Detect centroid finding"""
        score = 0.0
        
        # Check for centroid function
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'centroid' in node.name.lower():
                    score += 0.6
        
        return min(score, 1.0)
    
    def _detect_tree_decomposition(self, ast_tree: ast.AST) -> float:
        """Detect tree decomposition"""
        score = 0.0
        
        centroid_score = self._detect_centroid_finding(ast_tree)
        if centroid_score > 0.3:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_distance_queries(self, ast_tree: ast.AST) -> float:
        """Detect distance queries"""
        score = 0.0
        
        # Check for distance function
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'distance' in node.name.lower() or 'dist' in node.name.lower():
                    score += 0.4
        
        return min(score, 1.0)

    # =================== OPTIMIZATION PATTERN HELPERS ===================
    
    def _detect_inplace_space_optimization(self, ast_tree: ast.AST) -> float:
        """Detect in-place space optimization"""
        score = 0.0
        
        # Check for in-place operations
        inplace_score = self._detect_inplace_modifications(ast_tree)
        if inplace_score > 0.3:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_constant_space_patterns(self, ast_tree: ast.AST) -> float:
        """Detect constant space patterns"""
        score = 0.0
        
        # Check for O(1) space variables
        rolling_score = self._detect_rolling_arrays(ast_tree)
        if rolling_score > 0.3:
            score += 0.5
        
        return min(score, 1.0)
    
    def _detect_early_termination(self, ast_tree: ast.AST) -> float:
        """Detect early termination patterns"""
        score = 0.0
        
        # Check for break/continue/return in loops
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['has_break'] or loop_info['has_continue']:
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_algorithmic_improvements(self, ast_tree: ast.AST) -> float:
        """Detect algorithmic improvements"""
        score = 0.0
        
        # Check for optimization comments or variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['opt', 'fast', 'improved']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_precomputation_patterns(self, ast_tree: ast.AST) -> float:
        """Detect precomputation patterns"""
        score = 0.0
        
        # Check for precompute/preprocessing variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['precompute', 'preprocess', 'prepare']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_cache_friendly_patterns(self, ast_tree: ast.AST) -> float:
        """Detect cache-friendly patterns"""
        score = 0.0
        
        # Check for cache variables
        for var in self.ast_analyzer.variable_names:
            if 'cache' in var.lower():
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_lazy_structures(self, ast_tree: ast.AST) -> float:
        """Detect lazy data structures"""
        score = 0.0
        
        # Check for lazy variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['lazy', 'pending', 'delayed']):
                score += 0.5
        
        return min(score, 1.0)
    
    def _detect_delayed_update_patterns(self, ast_tree: ast.AST) -> float:
        """Detect delayed update patterns"""
        score = 0.0
        
        lazy_score = self._detect_lazy_structures(ast_tree)
        if lazy_score > 0.3:
            score += 0.4
        
        return min(score, 1.0)
    
    def _detect_range_update_optimization(self, ast_tree: ast.AST) -> float:
        """Detect range update optimization"""
        score = 0.0
        
        # Check for range update functions
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if 'range_update' in node.name.lower() or 'update_range' in node.name.lower():
                    score += 0.5
        
        return min(score, 1.0)
    
    def _detect_immutable_operations(self, ast_tree: ast.AST) -> float:
        """Detect immutable operations"""
        score = 0.0
        
        # Check for immutable/persistent variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['immutable', 'persistent', 'version']):
                score += 0.4
        
        return min(score, 1.0)
    
    def _detect_version_tracking(self, ast_tree: ast.AST) -> float:
        """Detect version tracking"""
        score = 0.0
        
        # Check for version variables
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['version', 'timestamp', 'time']):
                score += 0.3
        
        return min(score, 1.0)
    
    def _detect_path_copying_patterns(self, ast_tree: ast.AST) -> float:
        """Detect path copying patterns"""
        score = 0.0
        
        # Check for copy operations
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] in ['copy', 'deepcopy']:
                score += 0.4
        
        return min(score, 1.0)

    # =================== ADDITIONAL POINTER/WINDOW HELPERS ===================
    
    def _find_prefix_suffix_variables(self) -> List[str]:
        """Find prefix/suffix variables"""
        prefix_suffix_vars = []
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in ['prefix', 'suffix', 'pre', 'suf']):
                prefix_suffix_vars.append(var)
        return prefix_suffix_vars
    
    def _detect_cumulative_patterns(self, ast_tree: ast.AST) -> float:
        """Detect cumulative calculation patterns"""
        score = 0.0
        
        # Check for cumulative sum patterns
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add):
                    if isinstance(node.target, ast.Subscript):
                        score += 0.4
        
        return min(score, 1.0)
    
    def _detect_preprocessing_patterns(self, ast_tree: ast.AST) -> float:
        """Detect preprocessing patterns"""
        score = 0.0
        
        precompute_score = self._detect_precomputation_patterns(ast_tree)
        if precompute_score > 0.3:
            score += 0.4
        
        return min(score, 1.0)

    # =================== ADDITIONAL SPECIALIZED HELPERS ===================
    
    def _detect_sliding_mechanism(self, ast_tree: ast.AST) -> float:
        """Detect sliding window mechanism (referenced earlier)"""
        score = 0.0
        
        # Check for window size management
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'For':
                for node in ast.walk(loop_info['node']):
                    if isinstance(node, ast.AugAssign):
                        if isinstance(node.target, ast.Name):
                            var_name = node.target.id
                            if any(window_var in var_name.lower() for window_var in ['window', 'sum', 'count']):
                                score += 0.3
        
        return min(score, 0.4)
    
    def _detect_window_expansion(self, ast_tree: ast.AST) -> float:
        """Detect window expansion/contraction (referenced earlier)"""
        score = 0.0
        
        # Check for while loops with window conditions
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'While':
                condition = loop_info['condition']
                if 'Gt' in condition['operators'] or 'Lt' in condition['operators']:
                    score += 0.3
        
        return min(score, 0.3)
    
    def _get_attribute_chain(self, node: ast.Attribute) -> List[str]:
        """Get full attribute chain (referenced in ASTAnalyzer)"""
        chain = [node.attr]
        current = node.value
        
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            chain.append(current.id)
        
        return list(reversed(chain))
    
    def _extract_value(self, node) -> Any:
        """Extract value from AST node (referenced in ASTAnalyzer)"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return f"var:{node.id}"
        elif isinstance(node, ast.BinOp):
            try:
                return f"expr:{ast.unparse(node)}"
            except:
                return "expr:complex"
        return None
    
    def _analyze_condition(self, condition_node) -> Dict:
        """Analyze conditional expressions (referenced in ASTAnalyzer)"""
        info = {'type': type(condition_node).__name__, 'operators': []}
        
        for node in ast.walk(condition_node):
            if isinstance(node, ast.Compare):
                info['operators'].extend([type(op).__name__ for op in node.ops])
            elif isinstance(node, (ast.And, ast.Or)):
                info['operators'].append(type(node).__name__)
        
        return info

    # =================== UTILITY METHODS ===================
    
    def _count_function_calls_by_name(self, name: str) -> int:
        """Count function calls by name"""
        count = 0
        for call_info in self.ast_analyzer.function_calls:
            if call_info['function_name'] == name:
                count += 1
        return count
    
    def _has_variable_with_keywords(self, keywords: List[str]) -> bool:
        """Check if any variable contains keywords"""
        for var in self.ast_analyzer.variable_names:
            if any(keyword in var.lower() for keyword in keywords):
                return True
        return False
    
    def _count_nested_structures(self, ast_tree: ast.AST, structure_type) -> int:
        """Count nested structures of specific type"""
        count = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, structure_type):
                for child in ast.walk(node):
                    if isinstance(child, structure_type) and child != node:
                        count += 1
        return count
    
    def _find_assignments_to_subscripts(self, ast_tree: ast.AST, var_names: List[str]) -> int:
        """Find assignments to array/dict subscripts"""
        count = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name):
                            if target.value.id in var_names:
                                count += 1
        return count
    
    def _has_specific_import_pattern(self, pattern: str) -> bool:
        """Check for specific import patterns"""
        for imp in self.ast_analyzer.imports:
            if pattern.lower() in imp.lower():
                return True
        return False
    
    def _count_binary_operations(self, ast_tree: ast.AST, op_types: List) -> int:
        """Count binary operations of specific types"""
        count = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.BinOp):
                if type(node.op) in op_types:
                    count += 1
        return count
    
    def _find_function_definitions_with_keywords(self, ast_tree: ast.AST, keywords: List[str]) -> int:
        """Find function definitions containing keywords"""
        count = 0
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() for keyword in keywords):
                    count += 1
        return count
    
    def _has_method_calls_on_variable(self, var_name: str, methods: List[str]) -> bool:
        """Check if specific methods are called on a variable"""
        for call_info in self.ast_analyzer.function_calls:
            if len(call_info['attribute_chain']) >= 2:
                obj, method = call_info['attribute_chain'][0], call_info['attribute_chain'][-1]
                if var_name.lower() in obj.lower() and method in methods:
                    return True
        return False
    
    def _count_loops_with_condition_type(self, condition_types: List) -> int:
        """Count loops with specific condition types"""
        count = 0
        for loop_info in self.ast_analyzer.loop_structures:
            if loop_info['type'] == 'While':
                condition = loop_info['condition']
                for op_type in condition_types:
                    if op_type.__name__ in condition['operators']:
                        count += 1
                        break
        return count
    
    def _detect_pattern_in_loop_body(self, ast_tree: ast.AST, pattern_check_func) -> float:
        """Detect specific patterns within loop bodies"""
        score = 0.0
        for loop_info in self.ast_analyzer.loop_structures:
            if pattern_check_func(loop_info['node']):
                score += 0.3
        return min(score, 1.0)
    
    def _has_recursive_calls_with_modification(self, ast_tree: ast.AST) -> bool:
        """Check for recursive calls with parameter modification"""
        function_names = set()
        
        # Collect function names
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
        
        # Check for recursive calls with modified parameters
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in function_names:
                    # Check if any argument is modified (contains +, -, etc.)
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp):
                            return True
        
        return False

    ################
    
    def get_all_pattern_scores(self, code: str) -> Dict[str, float]:
        """Get comprehensive pattern scores using both AST and regex"""
        code=textwrap.dedent(code)
        try:
            ast_tree = ast.parse(code)
        except:
            ast_tree = None
        
        """print(f"Variables found: {len(self.ast_analyzer.variable_names)}")
        print(f"Function calls found: {len(self.ast_analyzer.function_calls)}")
        print(f"Variables: {list(self.ast_analyzer.variable_names)}")
        """
        
        scores = {}
        for pattern_name, detection_func in self.detection_functions.items():
            try:
                scores[pattern_name] = detection_func(code, ast_tree)
            except Exception as e:
                scores[pattern_name] = 0.0
        
        return scores

# =================== USAGE EXAMPLE ===================

def main():
    """Example usage of the complete enhanced detector"""
    detector = CompleteEnhancedDSAPatternDetector()
    
    # Example code
    sample_code = """
    def tricky(arr, target):
        l, r = 0, len(arr)-1
        while l <= r:
            m = (l+r)//2
            inner_l, inner_r = 0, len(arr)-1
            while inner_l < inner_r:
                if arr[inner_l]+arr[inner_r]==target:
                    break
                inner_l += 1
            if arr[m]==target:
                return m
            elif arr[m]<target:
                l = m+1
            else:
                r = m-1
    """
    """sample_code=textwrap.dedent(sample_code)
    ast_analyzer = ASTAnalyzer()
    ast_tree = ast.parse(sample_code)
    ast_analyzer.analyze_tree(ast_tree)

    print(f"Variables found: {len(ast_analyzer.variable_names)}")
    print(f"Function calls found: {len(ast_analyzer.function_calls)}")
    print(f"Variables: {list(ast_analyzer.variable_names)}")
    """
    
    # Get all pattern scores
    scores = detector.get_all_pattern_scores(sample_code)
    print(scores)
    # Print top detected patterns
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Top detected patterns:")
    for pattern, score in sorted_scores[:10]:
        #if score > 0.1:  # Only show significant scores
        print(f"{pattern}: {score:.3f}")

if __name__ == "__main__":
    main()


import numpy as np

features = [
    'recursion', 'iteration', 'nested_loops', 'conditional_logic', 
    'memoization', 'top_down_dp', 'bottom_up_dp', 'dp_optimization', 
    'binary_search', 'ternary_search', 'linear_search', 'divide_conquer', 
    'two_pointers', 'sliding_window', 'fast_slow_pointers', 'prefix_suffix', 
    'stack_usage', 'queue_usage', 'deque_usage', 'heap_usage', 'priority_queue', 
    'hash_map', 'hash_set', 'array_manipulation', 'string_operations', 
    'graph_dfs', 'graph_bfs', 'topological_sort', 'shortest_path', 'union_find', 
    'minimum_spanning_tree', 'graph_coloring', 'tree_traversal', 'binary_tree_operations', 
    'trie_usage', 'segment_tree', 'fenwick_tree', 'tree_dp', 'monotonic_stack', 
    'monotonic_queue', 'coordinate_compression', 'sparse_table', 'disjoint_set_union', 
    'number_theory', 'combinatorics', 'probability', 'geometry', 'bit_manipulation', 
    'modular_arithmetic', 'matrix_operations', 'custom_sorting', 'bucket_sort', 
    'counting_sort', 'quickselect', 'merge_operations', 'partitioning', 'greedy_choice', 
    'interval_scheduling', 'activity_selection', 'huffman_encoding', 'fractional_knapsack', 
    'string_matching', 'rolling_hash', 'suffix_array', 'lcs_patterns', 'palindrome_detection', 
    'anagram_detection', 'backtracking', 'branch_bound', 'meet_in_middle', 'sqrt_decomposition', 
    'heavy_light_decomposition', 'centroid_decomposition', 'space_optimization', 
    'time_optimization', 'cache_optimization', 'lazy_propagation', 'persistent_structures'
]

N = len(features)
F = np.eye(N)

# Define relatedness manually based on DSA intuition (0 = unrelated, 1 = identical)
related_pairs = {
    ('graph_bfs', 'graph_dfs'): 0.8,
    ('graph_bfs', 'topological_sort'): 0.6,
    ('binary_search', 'ternary_search'): 0.8,
    ('binary_search', 'divide_conquer'): 0.6,
    ('bottom_up_dp', 'dp_optimization'): 0.7,
    ('top_down_dp', 'memoization'): 0.9,
    ('two_pointers', 'sliding_window'): 0.7,
    ('monotonic_stack', 'monotonic_queue'): 0.6,
    ('iteration', 'nested_loops'): 0.7,
    ('array_manipulation', 'two_pointers'): 0.6,
    ('binary_tree_operations', 'tree_traversal'): 0.8,
    ('lcs_patterns', 'backtracking'): 0.5,
    ('heap_usage', 'priority_queue'): 0.9,
    ('hash_map', 'hash_set'): 0.8,
    ('graph_coloring', 'union_find'): 0.5,
    ('segment_tree', 'fenwick_tree'): 0.7,
    ('merge_operations', 'partitioning'): 0.6,
    ('counting_sort', 'bucket_sort'): 0.8,
    ('custom_sorting', 'quickselect'): 0.6,
    ('activity_selection', 'interval_scheduling'): 0.9,
    ('space_optimization', 'cache_optimization'): 0.7,
    ('time_optimization', 'dp_optimization'): 0.6
}

# Fill the matrix symmetrically
for (f1, f2), score in related_pairs.items():
    i, j = features.index(f1), features.index(f2)
    F[i, j] = score
    F[j, i] = score

def compute_structural_similarity(u: dict, v: dict, feature_matrix: np.ndarray, features: list) -> float:
    """
    Compute structural similarity between two code snippets based on their feature vectors.
    
    Args:
        u: dict, feature vector of code 1 (feature -> score)
        v: dict, feature vector of code 2 (feature -> score)
        feature_matrix: NxN numpy array, similarity between features
        features: list of feature names in the same order as feature_matrix

    Returns:
        similarity score between 0 and 1
    """
    # Convert dicts to arrays in the order of `features`
    u_vec = np.array([u.get(f, 0.0) for f in features])
    v_vec = np.array([v.get(f, 0.0) for f in features])
    
    # Compute similarity using feature similarity matrix
    sim = u_vec @ feature_matrix @ v_vec
    
    # Normalize by the magnitudes to get a cosine-like similarity in [0,1]
    norm_factor = np.sqrt(u_vec @ feature_matrix @ u_vec) * np.sqrt(v_vec @ feature_matrix @ v_vec)
    if norm_factor == 0:
        return 0.0
    return sim / norm_factor


def compare_user_with_prototypes(user_score: dict, prototype_scores: dict, feature_matrix: np.ndarray, features: list) -> dict:
    """
    Compare user code with multiple prototype approaches.

    Args:
        user_score: dict, feature vector of user code
        prototype_scores: dict, mapping prototype_name -> feature vector (dict)
        feature_matrix: NxN numpy array of feature-feature similarities
        features: list of feature names in order of feature_matrix

    Returns:
        dict mapping prototype_name -> similarity score with user code
    """
    similarity_results = {}

    for proto_name, proto_vec in prototype_scores.items():
        similarity = compute_structural_similarity(user_score, proto_vec, feature_matrix, features)
        similarity_results[proto_name] = similarity

    # Optional: sort by similarity descending
    similarity_results = dict(sorted(similarity_results.items(), key=lambda x: x[1], reverse=True))

    return similarity_results

# Problem: Find length of Longest Palindromic Subsequence

# Brute Force (Nested Loops)
BRUTE_FORCE = """
def two_sum_brute(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""

# Hash Map (O(n))
HASH_MAP = """
def two_sum_hash(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""

# Two Pointers (after sorting)
TWO_POINTERS = """
def two_sum_two_pointers(nums, target):
    nums_idx = sorted((num, i) for i, num in enumerate(nums))
    left, right = 0, len(nums) - 1
    while left < right:
        curr_sum = nums_idx[left][0] + nums_idx[right][0]
        if curr_sum == target:
            return sorted([nums_idx[left][1], nums_idx[right][1]])
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return []
"""

# Set-Based Approach (O(n))
SET_BASED = """
def two_sum_set(nums, target):
    seen = set()
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            for j in range(i):
                if nums[j] == complement:
                    return [j, i]
        seen.add(num)
    return []
"""

# Sorting + Binary Search
SORT_BINSEARCH = """
from bisect import bisect_left
def two_sum_binsearch(nums, target):
    nums_sorted = sorted((num, i) for i, num in enumerate(nums))
    n = len(nums_sorted)
    for i in range(n):
        complement = target - nums_sorted[i][0]
        j = bisect_left(nums_sorted, (complement, -1), i+1)
        if j < n and nums_sorted[j][0] == complement:
            return sorted([nums_sorted[i][1], nums_sorted[j][1]])
    return []
"""

# Dictionary of prototype solutions
solutions = {
    "Brute Force": BRUTE_FORCE,
    "Hash Map": HASH_MAP,
    "Two Pointers": TWO_POINTERS,
    "Set-Based": SET_BASED,
    "Binary Search": SORT_BINSEARCH
}

# Example user code (to be compared against the above solutions)
user_code = """
def two_sum_user(nums, target):
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [seen[target - val], idx]
        seen[val] = idx
    return []
"""

def cosine_sim(a:np.ndarray,b:np.ndarray)->float:
    if a is None or b is None: return 0.0
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denom) if denom>0 else 0.0

def similarity_scores(problem:str, code:str)->Dict[str,float]:
    emb = EMBEDDER.embed(code)
    return {a: cosine_sim(emb, PROTOTYPES[problem][a]) for a in PROTOTYPES[problem]}

# ------------------ Confidence computation ------------------
def compute_confidence(problem:str, code:str, previous_smoothed:Dict[str,float]=None)->Dict[str,float]:
    sims = similarity_scores(problem,code)
    confs = {}
    approaches = list(sims.keys())
    sorted_sims = sorted(sims.values(), reverse=True)
    top = sorted_sims[0]
    second = sorted_sims[1] if len(sorted_sims)>1 else 0.0
    margin = top - second
    for a in approaches:
        s = sims[a]
        h = structural_score(problem,a,code)
        t = previous_smoothed.get(a,0.0) if previous_smoothed else 0.0
        # weighted combination: s=0.5, h=0.3, margin=0.1, temporal=0.1
        conf = 0.5*s + 0.3*h + 0.1*margin + 0.1*t
        confs[a] = min(conf,1.0)
    return confs

prototype_scores = {}
for label, code in solutions.items():
    detector = CompleteEnhancedDSAPatternDetector()
    scores=detector.get_all_pattern_scores(code)
    prototype_scores[label]=scores
#print(prototype_scores)

detector = CompleteEnhancedDSAPatternDetector()
user_scores_code=detector.get_all_pattern_scores(user_code)

structres=compare_user_with_prototypes(user_scores_code, prototype_scores, F, features)

# ------------------ Session Store ------------------
MAX_HISTORY=50
SESSION_STORE:Dict[str,Dict[str,deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=MAX_HISTORY)))
LOCK = threading.Lock()

class SnapshotPayload(BaseModel):
    session_id:str
    problem_id:str
    snapshot_id:str
    code:str
    timestamp:float=None

# ------------------ FastAPI ------------------
app = FastAPI(title="Generalized Approach Detector", version="0.1")
from fastapi.middleware.cors import CORSMiddleware

# Allow frontend (served from file:// or localhost) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/snapshot")
def post_snapshot(payload:SnapshotPayload):
    ts = payload.timestamp or time.time()
    sess = payload.session_id
    prob = payload.problem_id
    code = payload.code
    if not code.strip():
        raise HTTPException(status_code=400,detail="Empty code")
    with LOCK:
        hist = SESSION_STORE[sess][prob]
        prev_smoothed = hist[-1]['smoothed_conf'] if hist else None
        conf = compute_confidence(prob,code,prev_smoothed)
        smoothed = {a:0.3*conf[a]+0.7*prev_smoothed.get(a,0.0) if prev_smoothed else conf[a] for a in conf}
        obs = {
            "snapshot_id": payload.snapshot_id,
            "timestamp": ts,
            "code_hash": hash(code),
            "similarities": similarity_scores(prob,code),
            "structural_scores": structres,
          #  "structural_scores": {a:structural_score(prob,a,code) for a in PROTOTYPES[prob]},
            "confidence": conf,
            "smoothed_conf": smoothed
        }
        hist.append(obs)
    return {"ok":True, "session":sess, "problem":prob, "snapshot_id":payload.snapshot_id, "confidence":conf, "smoothed_conf":smoothed}

@app.get("/api/session/{session_id}/{problem_id}/label")
def get_label(session_id:str,problem_id:str):
    with LOCK:
        hist = SESSION_STORE.get(session_id,{}).get(problem_id,[])
        if not hist: raise HTTPException(status_code=404,detail="No snapshots")
        smoothed = hist[-1]['smoothed_conf']
        sorted_items = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
        top, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items)>1 else 0.0
        margin = top_score - second_score
        confidence_level = "high" if top_score>0.75 and margin>0.1 else "medium" if top_score>0.5 else "low"
        return {"approach":top,"score":top_score,"margin":margin,"confidence_level":confidence_level,"smoothed":smoothed}

@app.get("/api/session/{session_id}/{problem_id}/history")
def get_history(session_id:str,problem_id:str,limit:int=50):
    with LOCK:
        hist = list(SESSION_STORE.get(session_id,{}).get(problem_id,[]))[-limit:]
    return {"session_id":session_id,"problem_id":problem_id,"history":hist}

@app.get("/api/approaches/{problem_id}")
def list_approaches(problem_id:str):
    return {"problem_id":problem_id,"approaches":list(PROTOTYPES.get(problem_id,{}).keys()),"use_model":USE_MODEL}

@app.get("/health")
def health(): return {"ok":True,"use_model":USE_MODEL}
