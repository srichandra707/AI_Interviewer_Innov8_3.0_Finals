import ast
from typing import Dict, List, Tuple

class ExplainableAnalyzer:
    def __init__(self, detector):
        self.detector = detector
    
    def analyze(self, code_timeline: List[Tuple[int, str]]) -> Dict:
        evidence = []
        scores = []
        
        for timestamp, code in code_timeline:
            patterns = self.detector.get_all_pattern_scores(code)
            score = self._score_code(code, patterns)
            scores.append(score)
            evidence.extend(self._get_evidence(timestamp, code, patterns))
        
        final_score = scores[-1] if scores else 0
        improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
        
        return self._explain_result(final_score, improvement, evidence)
    
    def _score_code(self, code: str, patterns: Dict[str, float]) -> float:
        score = 0.0
        
        # Good patterns
        if patterns.get('hash_map', 0) > 0.5:
            score += 0.4
        if patterns.get('binary_search', 0) > 0.5:
            score += 0.4
        if patterns.get('two_pointers', 0) > 0.5:
            score += 0.3
        
        # Bad patterns
        if patterns.get('nested_loops', 0) > 0.5:
            score -= 0.2
        
        # Code quality
        try:
            ast.parse(code)
            score += 0.2  # Valid syntax
        except:
            score -= 0.3  # Syntax errors
        
        if 'def ' in code:
            score += 0.1
        if '#' in code:
            score += 0.05
        
        return max(0, min(score, 1.0))
    
    def _get_evidence(self, timestamp: int, code: str, patterns: Dict[str, float]) -> List[Dict]:
        evidence = []
        
        if patterns.get('hash_map', 0) > 0.5:
            evidence.append({
                'type': 'positive',
                'reason': 'Uses hash map for O(1) lookup',
                'code': self._find_hash_code(code)
            })
        
        if patterns.get('nested_loops', 0) > 0.5:
            evidence.append({
                'type': 'negative', 
                'reason': 'Uses nested loops (O(n²) complexity)',
                'code': self._find_loop_code(code)
            })
        
        try:
            ast.parse(code)
        except Exception as e:
            evidence.append({
                'type': 'negative',
                'reason': f'Syntax error: {str(e)[:30]}',
                'code': code[:50]
            })
        
        return evidence
    
    def _find_hash_code(self, code: str) -> str:
        for line in code.split('\n'):
            if any(x in line for x in ['{', 'dict', 'seen']):
                return line.strip()
        return code[:50]
    
    def _find_loop_code(self, code: str) -> str:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'for ' in line:
                for j in range(i+1, min(len(lines), i+3)):
                    if 'for ' in lines[j]:
                        return f"{line.strip()}\n{lines[j].strip()}"
        return code[:50]
    
    def _explain_result(self, final_score: float, improvement: float, evidence: List[Dict]) -> Dict:
        if final_score >= 0.7:
            decision = "HIRE"
            level = "GOOD"
        elif final_score >= 0.4:
            decision = "MAYBE"
            level = "AVERAGE"
        else:
            decision = "NO_HIRE"
            level = "POOR"
        
        positive = [e for e in evidence if e['type'] == 'positive']
        negative = [e for e in evidence if e['type'] == 'negative']
        
        reasoning = f"Score: {final_score:.2f}, Improvement: {improvement:+.2f}"
        if len(positive) > len(negative):
            reasoning += " - More strengths than weaknesses"
        
        return {
            'decision': decision,
            'level': level, 
            'score': final_score,
            'reasoning': reasoning,
            'positive_evidence': positive,
            'negative_evidence': negative
        }

# Test it
class MockDetector:
    def get_all_pattern_scores(self, code):
        patterns = {}
        if 'for i' in code and 'for j' in code:
            patterns['nested_loops'] = 0.8
        if 'seen' in code:
            patterns['hash_map'] = 0.9
        return patterns

def test():
    timeline = [
        (0, "for i in range(len(nums)):\n    for j in range(i+1, len(nums)):"),
        (60, "seen = {}\nfor i, num in enumerate(nums):\n    if target - num in seen:")
    ]
    
    analyzer = ExplainableAnalyzer(MockDetector())
    result = analyzer.analyze(timeline)
    
    print(f"DECISION: {result['decision']}")
    print(f"SCORE: {result['score']:.2f}")
    print(f"REASONING: {result['reasoning']}")
    
    for evidence in result['positive_evidence']:
        print(f"✓ {evidence['reason']}: {evidence['code']}")
    
    for evidence in result['negative_evidence']:
        print(f"✗ {evidence['reason']}: {evidence['code']}")

if __name__ == "__main__":
    test()