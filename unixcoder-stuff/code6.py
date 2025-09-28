import ast
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CognitiveSnapshot:
    timestamp: int
    complexity_load: float
    pattern_clarity: float
    stress_level: float
    learning_rate: float

@dataclass
class CodeQualitySnapshot:
    timestamp: int
    sophistication: float
    organization: float
    optimization_awareness: float
    overall_quality: float

class SimpleCognitiveLoadAnalyzer:
    """Simplified cognitive load analysis focusing on key indicators"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_cognitive_timeline(self, code_timeline: List[Tuple[int, str]], 
                                 hint_interactions: List[Dict] = None) -> Dict:
        """Analyze cognitive patterns over time"""
        
        snapshots = []
        for timestamp, code in code_timeline:
            snapshot = self._create_cognitive_snapshot(timestamp, code, snapshots)
            snapshots.append(snapshot)
        
        return self._analyze_cognitive_patterns(snapshots, hint_interactions)
    
    def _create_cognitive_snapshot(self, timestamp: int, code: str, 
                                 history: List[CognitiveSnapshot]) -> CognitiveSnapshot:
        """Create cognitive snapshot at a moment"""
        
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # 1. Complexity Load (how much mental effort required)
        complexity_load = self._calculate_complexity_load(code, pattern_scores)
        
        # 2. Pattern Clarity (how clear their approach is)
        pattern_clarity = self._calculate_pattern_clarity(pattern_scores)
        
        # 3. Stress Level (indicators of cognitive stress)
        stress_level = self._calculate_stress_level(code, pattern_scores, history)
        
        # 4. Learning Rate (how fast they're improving)
        learning_rate = self._calculate_learning_rate(pattern_scores, history)
        
        return CognitiveSnapshot(
            timestamp=timestamp,
            complexity_load=complexity_load,
            pattern_clarity=pattern_clarity,
            stress_level=stress_level,
            learning_rate=learning_rate
        )
    
    def _calculate_complexity_load(self, code: str, pattern_scores: Dict[str, float]) -> float:
        """Calculate mental complexity load"""
        
        load = 0.0
        
        # Code structure complexity
        try:
            tree = ast.parse(code)
            
            # Nesting complexity
            nesting_depth = self._get_max_nesting(tree)
            load += min(nesting_depth / 4.0, 0.3)
            
            # Variable tracking
            variables = len(set(node.id for node in ast.walk(tree) 
                               if isinstance(node, ast.Name)))
            load += min(variables / 15.0, 0.3)
            
            # Control flow complexity
            control_nodes = len([n for n in ast.walk(tree) 
                               if isinstance(n, (ast.If, ast.For, ast.While))])
            load += min(control_nodes / 8.0, 0.2)
            
        except:
            load += 0.5  # Syntax errors = high load
        
        # Pattern complexity
        active_patterns = len([s for s in pattern_scores.values() if s > 0.3])
        load += min(active_patterns / 5.0, 0.2)
        
        return min(load, 1.0)
    
    def _calculate_pattern_clarity(self, pattern_scores: Dict[str, float]) -> float:
        """Calculate how clear their problem-solving approach is"""
        
        if not pattern_scores:
            return 0.0
        
        # Strong single pattern = high clarity
        max_score = max(pattern_scores.values())
        
        # Too many weak patterns = confusion
        weak_patterns = len([s for s in pattern_scores.values() if 0.1 < s < 0.4])
        confusion_penalty = min(weak_patterns * 0.1, 0.3)
        
        clarity = max_score - confusion_penalty
        return max(0, min(clarity, 1.0))
    
    def _calculate_stress_level(self, code: str, pattern_scores: Dict[str, float], 
                              history: List[CognitiveSnapshot]) -> float:
        """Calculate cognitive stress indicators"""
        
        stress = 0.0
        
        # Syntax errors indicate stress
        try:
            ast.parse(code)
        except:
            stress += 0.4
        
        # Decreasing pattern clarity over time
        if len(history) >= 2:
            current_clarity = self._calculate_pattern_clarity(pattern_scores)
            prev_clarity = history[-1].pattern_clarity
            
            if current_clarity < prev_clarity - 0.2:
                stress += 0.3
        
        # High complexity with low clarity
        complexity = self._calculate_complexity_load(code, pattern_scores)
        clarity = self._calculate_pattern_clarity(pattern_scores)
        
        if complexity > 0.7 and clarity < 0.4:
            stress += 0.3
        
        return min(stress, 1.0)
    
    def _calculate_learning_rate(self, pattern_scores: Dict[str, float], 
                               history: List[CognitiveSnapshot]) -> float:
        """Calculate learning/improvement rate"""
        
        if len(history) < 2:
            return 0.5  # Neutral for insufficient history
        
        current_clarity = self._calculate_pattern_clarity(pattern_scores)
        
        # Compare to recent history
        recent_clarity = [s.pattern_clarity for s in history[-3:]]
        avg_recent = sum(recent_clarity) / len(recent_clarity)
        
        improvement = current_clarity - avg_recent
        
        # Convert to rate (positive = learning, negative = declining)
        learning_rate = 0.5 + improvement  # Center around 0.5
        return max(0, min(learning_rate, 1.0))
    
    def _get_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def get_depth(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.FunctionDef)):
                    max_depth = max(max_depth, get_depth(child, depth + 1))
                else:
                    max_depth = max(max_depth, get_depth(child, depth))
            return max_depth
        
        return get_depth(tree)
    
    def _analyze_cognitive_patterns(self, snapshots: List[CognitiveSnapshot], 
                                  hint_interactions: List[Dict]) -> Dict:
        """Analyze overall cognitive patterns"""
        
        if not snapshots:
            return {'error': 'No data to analyze'}
        
        analysis = {}
        
        # Overall trends
        complexity_trend = self._calculate_trend([s.complexity_load for s in snapshots])
        clarity_trend = self._calculate_trend([s.pattern_clarity for s in snapshots])
        stress_trend = self._calculate_trend([s.stress_level for s in snapshots])
        learning_trend = self._calculate_trend([s.learning_rate for s in snapshots])
        
        analysis['trends'] = {
            'complexity': 'IMPROVING' if complexity_trend < -0.05 else 'STABLE' if abs(complexity_trend) < 0.05 else 'CONCERNING',
            'clarity': 'IMPROVING' if clarity_trend > 0.05 else 'STABLE' if abs(clarity_trend) < 0.05 else 'DECLINING',
            'stress': 'IMPROVING' if stress_trend < -0.05 else 'STABLE' if abs(stress_trend) < 0.05 else 'INCREASING',
            'learning': 'FAST' if learning_trend > 0.05 else 'STEADY' if learning_trend > -0.05 else 'SLOW'
        }
        
        # Final state assessment
        final = snapshots[-1]
        
        analysis['final_state'] = {
            'complexity_management': 'GOOD' if final.complexity_load < 0.6 else 'CHALLENGING',
            'solution_clarity': 'HIGH' if final.pattern_clarity > 0.7 else 'MEDIUM' if final.pattern_clarity > 0.4 else 'LOW',
            'stress_level': 'LOW' if final.stress_level < 0.3 else 'MEDIUM' if final.stress_level < 0.6 else 'HIGH',
            'learning_capability': 'STRONG' if final.learning_rate > 0.6 else 'AVERAGE' if final.learning_rate > 0.4 else 'WEAK'
        }
        
        # Hint responsiveness
        if hint_interactions:
            analysis['hint_responsiveness'] = self._analyze_hint_responsiveness(snapshots, hint_interactions)
        
        # Overall cognitive assessment
        analysis['overall_assessment'] = self._generate_cognitive_assessment(analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def _analyze_hint_responsiveness(self, snapshots: List[CognitiveSnapshot], 
                                   hints: List[Dict]) -> Dict:
        """Analyze how well they respond to hints"""
        
        responsiveness = []
        
        for hint in hints:
            hint_time = hint['timestamp']
            
            # Find snapshots before and after hint
            before_idx = None
            after_idx = None
            
            for i, snapshot in enumerate(snapshots):
                if snapshot.timestamp <= hint_time:
                    before_idx = i
                elif snapshot.timestamp > hint_time and after_idx is None:
                    after_idx = i
                    break
            
            if before_idx is not None and after_idx is not None:
                before = snapshots[before_idx]
                after = snapshots[after_idx]
                
                # Did clarity improve after hint?
                clarity_improvement = after.pattern_clarity - before.pattern
                clarity_improvement = after.pattern_clarity - before.pattern_clarity
                
                # Did learning rate increase?
                learning_improvement = after.learning_rate - before.learning_rate
                
                # Calculate responsiveness score
                response_score = (clarity_improvement + learning_improvement) / 2
                responsiveness.append(response_score)
        
        if responsiveness:
            avg_responsiveness = sum(responsiveness) / len(responsiveness)
            return {
                'average_responsiveness': avg_responsiveness,
                'classification': 'EXCELLENT' if avg_responsiveness > 0.3 else 'GOOD' if avg_responsiveness > 0.1 else 'POOR'
            }
        
        return {'classification': 'NO_HINTS'}
    
    def _generate_cognitive_assessment(self, analysis: Dict) -> Dict:
        """Generate overall cognitive assessment"""
        
        trends = analysis['trends']
        final_state = analysis['final_state']
        
        # Count positive indicators
        positive_indicators = 0
        
        if trends['complexity'] in ['IMPROVING', 'STABLE']:
            positive_indicators += 1
        if trends['clarity'] == 'IMPROVING':
            positive_indicators += 1
        if trends['stress'] in ['IMPROVING', 'STABLE']:
            positive_indicators += 1
        if trends['learning'] in ['FAST', 'STEADY']:
            positive_indicators += 1
        
        if final_state['complexity_management'] == 'GOOD':
            positive_indicators += 1
        if final_state['solution_clarity'] in ['HIGH', 'MEDIUM']:
            positive_indicators += 1
        if final_state['stress_level'] in ['LOW', 'MEDIUM']:
            positive_indicators += 1
        if final_state['learning_capability'] in ['STRONG', 'AVERAGE']:
            positive_indicators += 1
        
        # Generate assessment
        if positive_indicators >= 7:
            return {
                'level': 'EXCEPTIONAL',
                'description': 'Excellent cognitive performance across all dimensions',
                'hiring_recommendation': 'STRONG_HIRE'
            }
        elif positive_indicators >= 5:
            return {
                'level': 'STRONG',
                'description': 'Good cognitive performance with minor areas for improvement',
                'hiring_recommendation': 'HIRE'
            }
        elif positive_indicators >= 3:
            return {
                'level': 'AVERAGE',
                'description': 'Adequate cognitive performance but needs development',
                'hiring_recommendation': 'WEAK_HIRE'
            }
        else:
            return {
                'level': 'CONCERNING',
                'description': 'Multiple cognitive challenges identified',
                'hiring_recommendation': 'NO_HIRE'
            }


class SimpleCodeEvolutionAnalyzer:
    """Simplified code evolution analysis focusing on key quality metrics"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_code_evolution(self, code_timeline: List[Tuple[int, str]]) -> Dict:
        """Analyze how code quality evolves over time"""
        
        snapshots = []
        for timestamp, code in code_timeline:
            snapshot = self._create_quality_snapshot(timestamp, code)
            snapshots.append(snapshot)
        
        return self._analyze_evolution_patterns(snapshots)
    
    def _create_quality_snapshot(self, timestamp: int, code: str) -> CodeQualitySnapshot:
        """Create quality snapshot at a moment"""
        
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # 1. Sophistication (algorithm/data structure complexity)
        sophistication = self._calculate_sophistication(pattern_scores)
        
        # 2. Organization (code structure and readability)
        organization = self._calculate_organization(code)
        
        # 3. Optimization Awareness (performance consciousness)
        optimization_awareness = self._calculate_optimization_awareness(pattern_scores, code)
        
        # Overall quality
        overall_quality = (sophistication + organization + optimization_awareness) / 3
        
        return CodeQualitySnapshot(
            timestamp=timestamp,
            sophistication=sophistication,
            organization=organization,
            optimization_awareness=optimization_awareness,
            overall_quality=overall_quality
        )
    
    def _calculate_sophistication(self, pattern_scores: Dict[str, float]) -> float:
        """Calculate algorithmic sophistication"""
        
        # Weight patterns by sophistication level
        sophistication_weights = {
            # Expert level
            'dynamic_programming': 1.0,
            'segment_tree': 0.95,
            'union_find': 0.9,
            
            # Advanced level
            'binary_search': 0.8,
            'heap_usage': 0.75,
            'graph_dfs': 0.7,
            'backtracking': 0.7,
            
            # Intermediate level
            'two_pointers': 0.6,
            'sliding_window': 0.6,
            'hash_map': 0.5,
            'recursion': 0.5,
            
            # Basic level
            'array_manipulation': 0.3,
            'iteration': 0.2,
            'conditional_logic': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for pattern, score in pattern_scores.items():
            if pattern in sophistication_weights and score > 0.2:
                weight = sophistication_weights[pattern]
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_organization(self, code: str) -> float:
        """Calculate code organization quality"""
        
        organization_score = 0.0
        
        try:
            tree = ast.parse(code)
            
            # Function decomposition
            functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if functions:
                avg_func_size = sum(len(list(ast.walk(f))) for f in functions) / len(functions)
                if 5 <= avg_func_size <= 25:
                    organization_score += 0.3
                elif avg_func_size <= 40:
                    organization_score += 0.2
            
            # Variable naming
            variables = set(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
            meaningful_names = sum(1 for v in variables if len(v) >= 3 and v.islower())
            if variables:
                organization_score += (meaningful_names / len(variables)) * 0.3
            
            # Comments
            lines = code.split('\n')
            comment_lines = [l for l in lines if l.strip().startswith('#')]
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            
            if code_lines:
                comment_ratio = len(comment_lines) / len(code_lines)
                if 0.1 <= comment_ratio <= 0.3:
                    organization_score += 0.2
                elif comment_ratio > 0:
                    organization_score += 0.1
            
            # Nesting depth (lower is better)
            max_nesting = self._get_max_nesting(tree)
            if max_nesting <= 3:
                organization_score += 0.2
            elif max_nesting <= 5:
                organization_score += 0.1
            
        except:
            organization_score = 0.1  # Syntax errors hurt organization
        
        return min(organization_score, 1.0)
    
    def _calculate_optimization_awareness(self, pattern_scores: Dict[str, float], code: str) -> float:
        """Calculate optimization awareness"""
        
        optimization_score = 0.0
        
        # Efficient algorithms
        efficient_patterns = ['binary_search', 'hash_map', 'two_pointers', 'dynamic_programming']
        for pattern in efficient_patterns:
            if pattern_scores.get(pattern, 0) > 0.3:
                optimization_score += 0.2
        
        # Early termination patterns
        if re.search(r'\bbreak\b|\breturn\b.*if', code):
            optimization_score += 0.1
        
        # Complexity comments
        if re.search(r'O\([^)]+\)', code):
            optimization_score += 0.1
        
        # Efficient data structure usage
        if pattern_scores.get('heap_usage', 0) > 0.3:
            optimization_score += 0.1
        
        return min(optimization_score, 1.0)
    
    def _get_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        
        def get_depth(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.FunctionDef)):
                    max_depth = max(max_depth, get_depth(child, depth + 1))
                else:
                    max_depth = max(max_depth, get_depth(child, depth))
            return max_depth
        
        return get_depth(tree)
    
    def _analyze_evolution_patterns(self, snapshots: List[CodeQualitySnapshot]) -> Dict:
        """Analyze code evolution patterns"""
        
        if not snapshots:
            return {'error': 'No data to analyze'}
        
        analysis = {}
        
        # Quality trends
        sophistication_trend = self._calculate_trend([s.sophistication for s in snapshots])
        organization_trend = self._calculate_trend([s.organization for s in snapshots])
        optimization_trend = self._calculate_trend([s.optimization_awareness for s in snapshots])
        overall_trend = self._calculate_trend([s.overall_quality for s in snapshots])
        
        analysis['trends'] = {
            'sophistication': 'IMPROVING' if sophistication_trend > 0.05 else 'STABLE' if abs(sophistication_trend) < 0.05 else 'DECLINING',
            'organization': 'IMPROVING' if organization_trend > 0.05 else 'STABLE' if abs(organization_trend) < 0.05 else 'DECLINING',
            'optimization': 'IMPROVING' if optimization_trend > 0.05 else 'STABLE' if abs(optimization_trend) < 0.05 else 'DECLINING',
            'overall': 'IMPROVING' if overall_trend > 0.05 else 'STABLE' if abs(overall_trend) < 0.05 else 'DECLINING'
        }
        
        # Final quality assessment
        final = snapshots[-1]
        
        analysis['final_quality'] = {
            'sophistication_level': 'HIGH' if final.sophistication > 0.7 else 'MEDIUM' if final.sophistication > 0.4 else 'LOW',
            'organization_level': 'HIGH' if final.organization > 0.7 else 'MEDIUM' if final.organization > 0.4 else 'LOW',
            'optimization_level': 'HIGH' if final.optimization_awareness > 0.7 else 'MEDIUM' if final.optimization_awareness > 0.4 else 'LOW',
            'overall_level': 'HIGH' if final.overall_quality > 0.7 else 'MEDIUM' if final.overall_quality > 0.4 else 'LOW'
        }
        
        # Learning velocity
        if len(snapshots) >= 3:
            early_quality = sum(s.overall_quality for s in snapshots[:len(snapshots)//3]) / (len(snapshots)//3)
            late_quality = sum(s.overall_quality for s in snapshots[-len(snapshots)//3:]) / (len(snapshots)//3)
            
            improvement = late_quality - early_quality
            
            analysis['learning_velocity'] = {
                'improvement': improvement,
                'classification': 'FAST' if improvement > 0.3 else 'MODERATE' if improvement > 0.1 else 'SLOW' if improvement > 0 else 'STAGNANT'
            }
        
        # Overall assessment
        analysis['overall_assessment'] = self._generate_evolution_assessment(analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def _generate_evolution_assessment(self, analysis: Dict) -> Dict:
        """Generate overall evolution assessment"""
        
        trends = analysis['trends']
        final_quality = analysis['final_quality']
        
        # Count positive indicators
        positive_indicators = 0
        
        # Positive trends
        improving_trends = sum(1 for trend in trends.values() if trend == 'IMPROVING')
        stable_trends = sum(1 for trend in trends.values() if trend == 'STABLE')
        
        positive_indicators += improving_trends * 2  # Improving is worth 2 points
        positive_indicators += stable_trends * 1     # Stable is worth 1 point
        
        # High final quality
        high_quality = sum(1 for level in final_quality.values() if level == 'HIGH')
        medium_quality = sum(1 for level in final_quality.values() if level == 'MEDIUM')
        
        positive_indicators += high_quality * 2      # High quality worth 2 points
        positive_indicators += medium_quality * 1    # Medium quality worth 1 point
        
        # Learning velocity bonus
        if 'learning_velocity' in analysis:
            velocity = analysis['learning_velocity']['classification']
            if velocity == 'FAST':
                positive_indicators += 3
            elif velocity == 'MODERATE':
                positive_indicators += 1
        
        # Generate assessment (max possible score is around 20)
        if positive_indicators >= 15:
            return {
                'level': 'EXCEPTIONAL',
                'description': 'Outstanding code evolution with strong improvement trends',
                'hiring_recommendation': 'STRONG_HIRE'
            }
        elif positive_indicators >= 10:
            return {
                'level': 'STRONG',
                'description': 'Good code evolution with positive development',
                'hiring_recommendation': 'HIRE'
            }
        elif positive_indicators >= 6:
            return {
                'level': 'DEVELOPING',
                'description': 'Shows potential but needs more development',
                'hiring_recommendation': 'WEAK_HIRE'
            }
        else:
            return {
                'level': 'CONCERNING',
                'description': 'Limited code evolution and improvement',
                'hiring_recommendation': 'NO_HIRE'
            }


# Simple test runner
def run_simple_analysis(code_timeline, hint_interactions=None):
    """Run simplified cognitive and code evolution analysis"""
    
    # Mock detector (replace with your actual detector)
    class MockDetector:
        def get_all_pattern_scores(self, code):
            # Simple pattern detection for demo
            patterns = {}
            if 'hash' in code.lower() or 'dict' in code.lower():
                patterns['hash_map'] = 0.8
            if 'for' in code and 'for' in code[code.find('for')+10:]:
                patterns['nested_loops'] = 0.7
            if 'binary' in code.lower():
                patterns['binary_search'] = 0.9
            if 'dp' in code.lower() or 'memo' in code.lower():
                patterns['dynamic_programming'] = 0.8
            if 'def ' in code:
                patterns['function_decomposition'] = 0.6
            return patterns
    
    detector = MockDetector()
    
    # Run analyses
    cognitive_analyzer = SimpleCognitiveLoadAnalyzer(detector)
    evolution_analyzer = SimpleCodeEvolutionAnalyzer(detector)
    
    cognitive_results = cognitive_analyzer.analyze_cognitive_timeline(code_timeline, hint_interactions)
    evolution_results = evolution_analyzer.analyze_code_evolution(code_timeline)
    
    return {
        'cognitive_analysis': cognitive_results,
        'code_evolution': evolution_results
    }


import ast
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Evidence:
    type: str  # 'code_pattern', 'trend', 'metric', 'behavior'
    description: str
    code_snippet: str = ""
    metric_value: float = 0.0
    timestamp: int = 0
    impact: str = "neutral"  # 'positive', 'negative', 'neutral'

@dataclass
class ExplainableAssessment:
    score: float
    level: str
    reasoning: str
    evidence: List[Evidence]
    recommendations: List[str]

class ExplainableCognitiveAnalyzer:
    """Cognitive analyzer with explainable outputs and evidence collection"""
    
    def __init__(self, detector):
        self.detector = detector
        self.evidence_collector = []
    
    def analyze_with_explanations(self, code_timeline: List[Tuple[int, str]], 
                                hint_interactions: List[Dict] = None) -> Dict:
        """Analyze cognitive patterns with detailed explanations"""
        
        self.evidence_collector = []  # Reset evidence
        
        # Basic analysis
        snapshots = []
        for timestamp, code in code_timeline:
            snapshot = self._create_cognitive_snapshot(timestamp, code, snapshots)
            snapshots.append(snapshot)
        
        # Generate explainable assessments
        complexity_assessment = self._explain_complexity_management(snapshots)
        clarity_assessment = self._explain_pattern_clarity(snapshots, code_timeline)
        stress_assessment = self._explain_stress_patterns(snapshots, code_timeline)
        learning_assessment = self._explain_learning_progression(snapshots)
        
        if hint_interactions:
            hint_assessment = self._explain_hint_responsiveness(snapshots, hint_interactions, code_timeline)
        else:
            hint_assessment = None
        
        # Overall cognitive profile
        overall_assessment = self._generate_explainable_cognitive_profile(
            complexity_assessment, clarity_assessment, stress_assessment, 
            learning_assessment, hint_assessment
        )
        
        return {
            'complexity_management': complexity_assessment,
            'pattern_clarity': clarity_assessment,
            'stress_handling': stress_assessment,
            'learning_progression': learning_assessment,
            'hint_responsiveness': hint_assessment,
            'overall_assessment': overall_assessment,
            'all_evidence': self.evidence_collector
        }
    
    def _create_cognitive_snapshot(self, timestamp: int, code: str, history):
        """Create snapshot while collecting evidence"""
        
        pattern_scores = self.detector.get_all_pattern_scores(code)
        
        # Collect evidence during analysis
        self._collect_pattern_evidence(timestamp, code, pattern_scores)
        
        return {
            'timestamp': timestamp,
            'code': code,
            'pattern_scores': pattern_scores,
            'complexity': self._calculate_complexity_with_evidence(timestamp, code, pattern_scores),
            'clarity': self._calculate_clarity_with_evidence(timestamp, code, pattern_scores),
            'stress': self._calculate_stress_with_evidence(timestamp, code, history)
        }
    
    def _collect_pattern_evidence(self, timestamp: int, code: str, pattern_scores: Dict[str, float]):
        """Collect evidence about patterns used"""
        
        for pattern, score in pattern_scores.items():
            if score > 0.4:  # Significant pattern usage
                evidence = self._find_pattern_evidence_in_code(pattern, code, timestamp)
                if evidence:
                    self.evidence_collector.append(evidence)
    
    def _find_pattern_evidence_in_code(self, pattern: str, code: str, timestamp: int) -> Evidence:
        """Find specific code evidence for patterns"""
        
        pattern_evidence_map = {
            'hash_map': {
                'indicators': ['dict()', '{', 'in seen', '.get('],
                'description': 'Uses hash map/dictionary for O(1) lookups'
            },
            'two_pointers': {
                'indicators': ['left', 'right', 'start', 'end', 'i.*j'],
                'description': 'Uses two-pointer technique for optimization'
            },
            'binary_search': {
                'indicators': ['left.*right', 'mid =', '// 2', 'bisect'],
                'description': 'Implements binary search for O(log n) complexity'
            },
            'dynamic_programming': {
                'indicators': ['dp[', 'memo', 'cache', '@lru_cache'],
                'description': 'Uses dynamic programming with memoization'
            },
            'nested_loops': {
                'indicators': ['for.*for', 'while.*while'],
                'description': 'Uses nested loops (O(n²) complexity)'
            }
        }
        
        if pattern not in pattern_evidence_map:
            return None
        
        config = pattern_evidence_map[pattern]
        
        # Find specific code snippet
        for indicator in config['indicators']:
            matches = re.finditer(indicator, code, re.IGNORECASE)
            for match in matches:
                # Extract surrounding context
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    if indicator.lower() in line.lower():
                        # Get 2 lines of context
                        start_line = max(0, i-1)
                        end_line = min(len(lines), i+2)
                        snippet = '\n'.join(lines[start_line:end_line])
                        
                        impact = 'positive' if pattern in ['hash_map', 'binary_search', 'dynamic_programming'] else 'negative' if pattern == 'nested_loops' else 'neutral'
                        
                        return Evidence(
                            type='code_pattern',
                            description=config['description'],
                            code_snippet=snippet,
                            timestamp=timestamp,
                            impact=impact
                        )
        
        return None
    
    def _calculate_complexity_with_evidence(self, timestamp: int, code: str, pattern_scores: Dict[str, float]) -> float:
        """Calculate complexity while collecting evidence"""
        
        complexity = 0.0
        
        try:
            tree = ast.parse(code)
            
            # Nesting complexity with evidence
            nesting_depth = self._get_max_nesting(tree)
            if nesting_depth > 3:
                complexity += min(nesting_depth / 4.0, 0.4)
                
                # Find evidence of deep nesting
                nested_lines = []
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    indent = len(line) - len(line.lstrip())
                    if indent > 12:  # Assuming 4-space indents, 3+ levels
                        nested_lines.append(f"Line {i+1}: {line.strip()}")
                
                if nested_lines:
                    self.evidence_collector.append(Evidence(
                        type='metric',
                        description=f'Deep nesting detected (max depth: {nesting_depth})',
                        code_snippet='\n'.join(nested_lines[:3]),
                        metric_value=nesting_depth,
                        timestamp=timestamp,
                        impact='negative'
                    ))
            
            # Variable tracking with evidence
            variables = set(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
            if len(variables) > 10:
                complexity += min(len(variables) / 15.0, 0.3)
                
                self.evidence_collector.append(Evidence(
                    type='metric',
                    description=f'High variable count: {len(variables)} variables to track',
                    code_snippet=f"Variables: {', '.join(sorted(variables)[:8])}{'...' if len(variables) > 8 else ''}",
                    metric_value=len(variables),
                    timestamp=timestamp,
                    impact='negative'
                ))
            
        except SyntaxError as e:
            complexity += 0.5
            self.evidence_collector.append(Evidence(
                type='code_pattern',
                description=f'Syntax error detected: {str(e)}',
                code_snippet=code.split('\n')[max(0, e.lineno-2):e.lineno+1] if hasattr(e, 'lineno') else code[:100],
                timestamp=timestamp,
                impact='negative'
            ))
        
        return min(complexity, 1.0)
    
    def _calculate_clarity_with_evidence(self, timestamp: int, code: str, pattern_scores: Dict[str, float]) -> float:
        """Calculate clarity while collecting evidence"""
        
        if not pattern_scores:
            return 0.0
        
        max_score = max(pattern_scores.values())
        dominant_pattern = max(pattern_scores, key=pattern_scores.get)
        
        # Evidence of clear approach
        if max_score > 0.7:
            self.evidence_collector.append(Evidence(
                type='code_pattern',
                description=f'Clear algorithmic approach using {dominant_pattern}',
                code_snippet=self._extract_pattern_snippet(code, dominant_pattern),
                metric_value=max_score,
                timestamp=timestamp,
                impact='positive'
            ))
        
        # Evidence of confusion (multiple weak patterns)
        weak_patterns = [p for p, s in pattern_scores.items() if 0.1 < s < 0.4]
        if len(weak_patterns) > 3:
            self.evidence_collector.append(Evidence(
                type='behavior',
                description=f'Multiple competing approaches: {", ".join(weak_patterns[:3])}',
                code_snippet="",
                timestamp=timestamp,
                impact='negative'
            ))
        
        confusion_penalty = min(len(weak_patterns) * 0.1, 0.3)
        clarity = max_score - confusion_penalty
        
        return max(0, min(clarity, 1.0))
    
    def _calculate_stress_with_evidence(self, timestamp: int, code: str, history) -> float:
        """Calculate stress while collecting evidence"""
        
        stress = 0.0
        
        # Syntax errors
        try:
            ast.parse(code)
        except Exception as e:
            stress += 0.4
            self.evidence_collector.append(Evidence(
                type='code_pattern',
                description='Syntax error indicates cognitive stress',
                code_snippet=str(e),
                timestamp=timestamp,
                impact='negative'
            ))
        
        # Regression in quality
        if len(history) >= 2:
            current_clarity = self._calculate_clarity_with_evidence(timestamp, code, self.detector.get_all_pattern_scores(code))
            prev_clarity = history[-1]['clarity']
            
            if current_clarity < prev_clarity - 0.2:
                stress += 0.3
                self.evidence_collector.append(Evidence(
                    type='trend',
                    description=f'Clarity regression: {prev_clarity:.2f} → {current_clarity:.2f}',
                    code_snippet="",
                    metric_value=current_clarity - prev_clarity,
                    timestamp=timestamp,
                    impact='negative'
                ))
        
        return min(stress, 1.0)
    
    def _explain_complexity_management(self, snapshots) -> ExplainableAssessment:
        """Generate explainable complexity management assessment"""
        
        complexity_scores = [s['complexity'] for s in snapshots]
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        complexity_trend = self._calculate_trend(complexity_scores)
        
        # Gather relevant evidence
        complexity_evidence = [e for e in self.evidence_collector 
                             if e.type == 'metric' and 'nesting' in e.description.lower() or 'variable' in e.description.lower()]
        
        # Generate assessment
        if avg_complexity < 0.4 and complexity_trend <= 0:
            level = "EXCELLENT"
            reasoning = f"Maintains low cognitive complexity (avg: {avg_complexity:.2f}) with improving trend"
            recommendations = ["Continue with current approach", "Consider mentoring others on clean code practices"]
        elif avg_complexity < 0.6:
            level = "GOOD"
            reasoning = f"Moderate complexity management (avg: {avg_complexity:.2f})"
            recommendations = ["Practice breaking down complex problems", "Focus on single responsibility principle"]
        else:
            level = "CONCERNING"
            reasoning = f"High cognitive complexity (avg: {avg_complexity:.2f}) may impact problem-solving"
            recommendations = ["Practice with simpler problems first", "Focus on one concept at a time", "Use more helper functions"]
        
        return ExplainableAssessment(
            score=1.0 - avg_complexity,
            level=level,
            reasoning=reasoning,
            evidence=complexity_evidence,
            recommendations=recommendations
        )
    
    def _explain_pattern_clarity(self, snapshots, code_timeline) -> ExplainableAssessment:
        """Generate explainable pattern clarity assessment"""
        
        clarity_scores = [s['clarity'] for s in snapshots]
        final_clarity = clarity_scores[-1]
        clarity_trend = self._calculate_trend(clarity_scores)
        
        # Gather pattern evidence
        pattern_evidence = [e for e in self.evidence_collector if e.type == 'code_pattern']
        positive_evidence = [e for e in pattern_evidence if e.impact == 'positive']
        negative_evidence = [e for e in pattern_evidence if e.impact == 'negative']
        
        # Generate assessment
        if final_clarity > 0.7 and len(positive_evidence) >= 2:
            level = "EXCELLENT"
            reasoning = f"Strong pattern recognition (final clarity: {final_clarity:.2f}) with clear algorithmic thinking"
            recommendations = ["Excellent problem-solving approach", "Ready for complex algorithmic challenges"]
        elif final_clarity > 0.5:
            level = "GOOD"
            reasoning = f"Good pattern clarity (final clarity: {final_clarity:.2f}) with room for improvement"
            recommendations = ["Practice more algorithm patterns", "Focus on recognizing optimal approaches faster"]
        else:
            level = "NEEDS_IMPROVEMENT"
            reasoning = f"Limited pattern clarity (final clarity: {final_clarity:.2f}) suggests difficulty with algorithmic thinking"
            recommendations = ["Study fundamental algorithms and data structures", "Practice pattern recognition exercises", "Start with simpler problems"]
        
        return ExplainableAssessment(
            score=final_clarity,
            level=level,
            reasoning=reasoning,
            evidence=positive_evidence + negative_evidence,
            recommendations=recommendations
        )
    
    def _explain_hint_responsiveness(self, snapshots, hint_interactions, code_timeline) -> ExplainableAssessment:
        """Explain how well they respond to hints"""
        
        responsiveness_evidence = []
        improvements = []
        
        for hint in hint_interactions:
            # Find code before and after hint
            hint_time = hint['timestamp']
            before_code = None
            after_code = None
            
            for timestamp, code in code_timeline:
                if timestamp <= hint_time:
                    before_code = code
                elif timestamp > hint_time and after_code is None:
                    after_code = code
                    break
            
            if before_code and after_code:
                # Analyze improvement
                before_patterns = self.detector.get_all_pattern_scores(before_code)
                after_patterns = self.detector.get_all_pattern_scores(after_code)
                
                improvement = max(after_patterns.values()) - max(before_patterns.values()) if before_patterns and after_patterns else 0
                improvements.append(improvement)
                
                # Create evidence
                if improvement > 0.2:
                    responsiveness_evidence.append(Evidence(
                        type='behavior',
                        description=f'Significant improvement after hint: {hint["hint_given"][:50]}...',
                        code_snippet=f"BEFORE:\n{before_code[-100:]}\n\nAFTER:\n{after_code[:100]}",
                        metric_value=improvement,
                        timestamp=hint_time,
                        impact='positive'
                    ))
                elif improvement < -0.1:
                    responsiveness_evidence.append(Evidence(
                        type='behavior',
                        description=f'Regression after hint (may indicate confusion)',
                        code_snippet=f"Hint: {hint['hint_given'][:50]}...",
                        metric_value=improvement,
                        timestamp=hint_time,
                        impact='negative'
                    ))
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        
        # Generate assessment
        if avg_improvement > 0.3:
            level = "EXCELLENT"
            reasoning = f"Highly responsive to guidance (avg improvement: {avg_improvement:.2f})"
            recommendations = ["Excellent mentoring potential", "Ready for collaborative work environments"]
        elif avg_improvement > 0.1:
            level = "GOOD"
            reasoning = f"Good responsiveness to hints (avg improvement: {avg_improvement:.2f})"
            recommendations = ["Benefits well from guidance", "Good candidate for mentoring programs"]
        else:
            level = "LIMITED"
            reasoning = f"Limited responsiveness to hints (avg improvement: {avg_improvement:.2f})"
            recommendations = ["May need more structured guidance", "Consider pairing with senior developers"]
        
        return ExplainableAssessment(
            score=max(0, min(avg_improvement + 0.5, 1.0)),
            level=level,
            reasoning=reasoning,
            evidence=responsiveness_evidence,
            recommendations=recommendations
        )
    
    def _explain_learning_progression(self, snapshots) -> ExplainableAssessment:
        """Explain learning progression patterns"""
        
        if len(snapshots) < 3:
            return ExplainableAssessment(0.5, "INSUFFICIENT_DATA", "Not enough data to assess learning", [], [])
        
        clarity_scores = [s['clarity'] for s in snapshots]
        learning_trend = self._calculate_trend(clarity_scores)
        
        # Detect learning moments
        learning_evidence = []
        for i in range(1, len(snapshots)):
            improvement = clarity_scores[i] - clarity_scores[i-1]
            if improvement > 0.2:
                learning_evidence.append(Evidence(
                    type='trend',
                    description=f'Learning breakthrough: clarity improved by {improvement:.2f}',
                    code_snippet="",
                    metric_value=improvement,
                    timestamp=snapshots[i]['timestamp'],
                    impact='positive'
                ))
        
        # Generate assessment
        if learning_trend > 0.05 and len(learning_evidence) >= 2:
            level = "FAST_LEARNER"
            reasoning = f"Strong learning progression (trend: +{learning_trend:.3f}) with multiple breakthroughs"
            recommendations = ["Excellent learning agility", "Ready for rapid skill development"]
        elif learning_trend > 0:
            level = "STEADY_LEARNER"
            reasoning = f"Positive learning trend (trend: +{learning_trend:.3f})"
            recommendations = ["Good learning potential", "Benefits from incremental challenges"]
        else:
            level = "SLOW_LEARNER"
            reasoning = f"Limited learning progression (trend: {learning_trend:.3f})"
            recommendations = ["Needs more structured learning approach", "Consider breaking down concepts further"]
        
        return ExplainableAssessment(
            score=max(0, min(learning_trend * 10 + 0.5, 1.0)),
            level=level,
            reasoning=reasoning,
            evidence=learning_evidence,
            recommendations=recommendations
        )
    
    def _generate_explainable_cognitive_profile(self, complexity_assessment, clarity_assessment, 
                                              stress_assessment, learning_assessment, 
                                              hint_assessment) -> ExplainableAssessment:
        """Generate overall explainable cognitive profile"""
        
        assessments = [complexity_assessment, clarity_assessment, stress_assessment, learning_assessment]
        if hint_assessment:
            assessments.append(hint_assessment)
        
        # Calculate overall score
        overall_score = sum(a.score for a in assessments) / len(assessments)
        
        # Collect all evidence
        all_evidence = []
        for assessment in assessments:
            all_evidence.extend(assessment.evidence)
        
        # Generate level and reasoning
        positive_assessments = sum(1 for a in assessments if a.level in ['EXCELLENT', 'GOOD', 'FAST_LEARNER', 'STEADY_LEARNER'])
        
        if positive_assessments >= len(assessments) * 0.8:
            level = "STRONG_HIRE"
            reasoning = "Demonstrates strong cognitive abilities across multiple dimensions"
        elif positive_assessments >= len(assessments) * 0.6:
            level = "HIRE"
            reasoning = "Shows good cognitive potential with some areas for development"
        elif positive_assessments >= len(assessments) * 0.4:
            level = "WEAK_HIRE"
            reasoning = "Mixed cognitive performance with significant development needs"
        else:
            level = "NO_HIRE"
            reasoning = "Concerning cognitive patterns across multiple areas"
        
        # Generate recommendations
        all_recommendations = []
        for assessment in assessments:
            all_recommendations.extend(assessment.recommendations)
        
        return ExplainableAssessment(
            score=overall_score,
            level=level,
            reasoning=reasoning,
            evidence=all_evidence,
            recommendations=list(set(all_recommendations))  # Remove duplicates
        )
    
    def _extract_pattern_snippet(self, code: str, pattern: str) -> str:
        """Extract relevant code snippet for a pattern"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern == 'hash_map' and ('dict' in line or '{' in line or 'in ' in line):
                return '\n'.join(lines[max(0, i-1):i+2])
            elif pattern == 'recursion' and 'def ' in line:
                return '\n'.join(lines[i:i+3])
        return code[:100] + "..." if len(code) > 100 else code
    
    def _get_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.FunctionDef)):
                    max_depth = max(max_depth, get_depth(child, depth + 1))
                else:
                    max_depth = max(max_depth, get_depth(child, depth))
            return max_depth
        return get_depth(tree)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time"""
        if len(values) < 2:
            return 0.0
        return np.polyfit(range(len(values)), values, 1)[0]


# Example usage with explanations
def run_explainable_analysis():
    """Demo of explainable analysis"""
    
    # Mock detector
    class MockDetector:
        def get_all_pattern_scores(self, code):
            patterns = {}
            if 'for i in range' in code and 'for j in range' in code:
                patterns['nested_loops'] = 0.8
            if 'seen' in code and '{' in code:
                patterns['hash_map'] = 0.9
            if 'def ' in code:
                patterns['function_decomposition'] = 0.6
            return patterns
    
    # Sample timeline
    timeline = [
        (0, "def two_sum(nums, target):\n    # thinking about this"),
        (30000, "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]"),
        (90000, "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []")
    ]
    
    hints = [
        {
            'timestamp': 60000,
            'hint_given': 'Can you think of a more efficient approach using a hash map?'
        }
    ]
    
    analyzer = ExplainableCognitiveAnalyzer(MockDetector())
    results = analyzer.analyze_with_explanations(timeline, hints)
    
    # Print explainable results
    print("=== EXPLAINABLE COGNITIVE ANALYSIS ===\n")
    
    for dimension, assessment in results.items():
        if isinstance(assessment, ExplainableAssessment):
            print(f"## {dimension.upper().replace('_', ' ')}")
            print(f"Level: {assessment.level}")
            print(f"Score: {assessment.score:.2f}")
            print(f"Reasoning: {assessment.reasoning}")
            
            if assessment.evidence:
                print("\nEvidence:")
                for i, evidence in enumerate(assessment.evidence[:3], 1):  # Show top 3
                    print(f"  {i}. {evidence.description}")
                    if evidence.code_snippet:
                        print(f"     Code: {evidence.code_snippet[:100]}...")
            
            if assessment.recommendations:
                print(f"\nRecommendations:")
                for rec in assessment.recommendations[:2]:  # Show top 2
                    print(f"  • {rec}")
            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_explainable_analysis()

"""
# Example usage
if __name__ == "__main__":
    # Sample timeline (timestamp, code)
    sample_timeline = [
        (0, "def two_sum(nums, target):\n    # thinking"),
        (30000, "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):"),
        (60000, "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i")
    ]
    
    results = run_simple_analysis(sample_timeline)
    
    print("=== COGNITIVE ANALYSIS ===")
    print(f"Overall Assessment: {results['cognitive_analysis']['overall_assessment']}")
    print(f"Trends: {results['cognitive_analysis']['trends']}")
    
    print("\n=== CODE EVOLUTION ===")
    print(f"Overall Assessment: {results['code_evolution']['overall_assessment']}")
    print(f"Trends: {results['code_evolution']['trends']}")

"""

import ast
import re
from typing import Dict, List, Tuple

class SimpleExplainableAnalyzer:
    """Short explainable analyzer with evidence"""
    
    def __init__(self, detector):
        self.detector = detector
    
    def analyze(self, code_timeline: List[Tuple[int, str]]) -> Dict:
        """Main analysis with explanations"""
        
        evidence = []
        scores = []
        
        for timestamp, code in code_timeline:
            # Get patterns and calculate score
            patterns = self.detector.get_all_pattern_scores(code)
            score = self._calculate_score(code, patterns)
            scores.append(score)
            
            # Collect evidence
            evidence.extend(self._collect_evidence(timestamp, code, patterns))
        
        # Generate final assessment
        final_score = scores[-1] if scores else 0
        improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        
        return self._generate_explanation(final_score, improvement, evidence)
    
    def _calculate_score(self, code: str, patterns: Dict[str, float]) -> float:
        """Calculate overall code quality score"""
        
        score = 0.0
        
        # Pattern sophistication (40% of score)
        if patterns:
            pattern_weights = {
                'hash_map': 0.8, 'binary_search': 0.9, 'dynamic_programming': 1.0,
                'two_pointers': 0.7, 'recursion': 0.6, 'nested_loops': 0.2
            }
            
            weighted_score = sum(patterns.get(p, 0) * w for p, w in pattern_weights.items())
            score += min(weighted_score, 0.4)
        
        # Code quality (30% of score)
        try:
            ast.parse(code)
            score += 0.1  # Valid syntax
            
            if 'def ' in code:
                score += 0.1  # Function usage
            if '#' in code:
                score += 0.05  # Comments
            if len(code.split('\n')) < 20:
                score += 0.05  # Reasonable length
        except:
            pass  # Syntax errors hurt score
        
        # Optimization awareness (30% of score)
        if any(patterns.get(p, 0) > 0.5 for p in ['hash_map', 'binary_search', 'two_pointers']):
            score += 0.3
        elif 'O(' in code:
            score += 0.1  # At least thinking about complexity
        
        return min(score, 1.0)
    
    def _collect_evidence(self, timestamp: int, code: str, patterns: Dict[str, float]) -> List[Dict]:
        """Collect evidence for scoring decisions"""
        
        evidence = []
        
        # Pattern evidence
        for pattern, score in patterns.items():
            if score > 0.5:
                snippet = self._find_pattern_snippet(pattern, code)
                evidence.append({
                    'type': 'positive',
                    'reason': f'Uses {pattern} algorithm (score: {score:.2f})',
                    'code': snippet,
                    'timestamp': timestamp
                })
        
        # Quality evidence
        try:
            ast.parse(code)
        except Exception as e:
            evidence.append({
                'type': 'negative',
                'reason': f'Syntax error: {str(e)[:50]}',
                'code': code[:100],
                'timestamp': timestamp
            })
        
        # Check for inefficient patterns
        if patterns.get('nested_loops', 0) > 0.5:
            evidence.append({
                'type': 'negative',
                'reason': 'Uses inefficient nested loops (O(n²) complexity)',
                'code': self._find_pattern_snippet('nested_loops', code),
                'timestamp': timestamp
            })
        
        # Check for good practices
        if 'def ' in code and len(code.split('\n')) < 15:
            evidence.append({
                'type': 'positive', 
                'reason': 'Good function decomposition with reasonable length',
                'code': code.split('\n')[0],
                'timestamp': timestamp
            })
        
        return evidence
    
    def _find_pattern_snippet(self, pattern: str, code: str) -> str:
        """Find relevant code snippet for pattern"""
        
        lines = code.split('\n')
        
        if pattern == 'hash_map':
            for i, line in enumerate(lines):
                if any(x in line for x in ['{', 'dict', 'seen']):
                    return '\n'.join(lines[max(0,i-1):i+2])
        
        elif pattern == 'nested_loops':
            for i, line in enumerate(lines):
                if 'for ' in line:
                    # Look for another for loop nearby
                    for j in range(i+1, min(len(lines), i+5)):
                        if 'for ' in lines[j]:
                            return '\n'.join(lines[i:j+1])
        
        elif pattern == 'binary_search':
            for i, line in enumerate(lines):
                if any(x in line for x in ['mid', 'left', 'right']):
                    return '\n'.join(lines[max(0,i-1):i+2])
        
        return code[:80] + "..." if len(code) > 80 else code
    
    def _generate_explanation(self, final_score: float, improvement: float, evidence: List[Dict]) -> Dict:
        """Generate final explanation"""
        
        # Determine level
        if final_score >= 0.8:
            level = "EXCELLENT"
            decision = "STRONG_HIRE"
        elif final_score >= 0.6:
            level = "GOOD" 
            decision = "HIRE"
        elif final_score >= 0.4:
            level = "AVERAGE"
            decision = "WEAK_HIRE"
        else:
            level = "POOR"
            decision = "NO_HIRE"
        
        # Generate reasoning
        positive_evidence = [e for e in evidence if e['type'] == 'positive']
        negative_evidence = [e for e in evidence if e['type'] == 'negative']
        
        reasoning_parts = []
        
        if final_score >= 0.6:
            reasoning_parts.append(f"Strong performance (score: {final_score:.2f})")
        elif final_score >= 0.4:
            reasoning_parts.append(f"Adequate performance (score: {final_score:.2f})")
        else:
            reasoning_parts.append(f"Below expectations (score: {final_score:.2f})")
        
        if improvement > 0.2:
            reasoning_parts.append(f"shows significant improvement (+{improvement:.2f})")
        elif improvement > 0:
            reasoning_parts.append(f"shows some improvement (+{improvement:.2f})")
        elif improvement < -0.1:
            reasoning_parts.append(f"concerning decline ({improvement:.2f})")
        
        if len(positive_evidence) > len(negative_evidence):
            reasoning_parts.append("with more strengths than weaknesses")
        elif len(negative_evidence) > len(positive_evidence):
            reasoning_parts.append("with notable areas for improvement")
        
        reasoning = " ".join(reasoning_parts).capitalize() + "."
        
        # Generate recommendations
        recommendations = []
        
        if any('hash_map' in e['reason'] for e in positive_evidence):
            recommendations.append("✓ Demonstrates good algorithmic thinking")
        if any('syntax error' in e['reason'].lower() for e in negative_evidence):
            recommendations.append("⚠ Focus on syntax accuracy under pressure")
        if any('nested_loops' in e['reason'] for e in negative_evidence):
            recommendations.append("⚠ Practice more efficient algorithms")
        if improvement > 0.1:
            recommendations.append("✓ Shows good learning agility")
        
        if not recommendations:
            recommendations = ["Continue practicing algorithmic problem solving"]
        
        return {
            'final_score': final_score,
            'level': level,
            'decision': decision,
            'reasoning': reasoning,
            'improvement': improvement,
            'evidence': {
                'positive': positive_evidence[:3],  # Top 3
                'negative': negative_evidence[:3]   # Top 3
            },
            'recommendations': recommendations
        }


# Mock detector for testing
class MockDetector:
    def get_all_pattern_scores(self, code):
        patterns = {}
        if 'for i in range' in code and 'for j in range' in code:
            patterns['nested_loops'] = 0.8
        if 'seen' in code or 'dict' in code:
            patterns['hash_map'] = 0.9
        if 'left' in code and 'right' in code:
            patterns['two_pointers'] = 0.7
        return patterns


# Simple test
def test_analyzer():
    timeline = [
        (0, "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i+1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]"),
        (60000, "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i")
    ]
    
    analyzer = SimpleExplainableAnalyzer(MockDetector())
    result = analyzer.analyze(timeline)
    
    print(f"DECISION: {result['decision']}")
    print(f"LEVEL: {result['level']} (Score: {result['final_score']:.2f})")
    print(f"REASONING: {result['reasoning']}")
    
    print("\nPOSITIVE EVIDENCE:")
    for evidence in result['evidence']['positive']:
        print(f"  • {evidence['reason']}")
    
    print("\nRECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"  {rec}")

if __name__ == "__main__":
    test_analyzer()