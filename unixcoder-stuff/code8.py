from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque, defaultdict, Counter
from typing import Dict, List, Any
import time, threading, numpy as np
import ast, re

# Optional: transformers + torch for UniXcoder
USE_MODEL = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    USE_MODEL = True
except:
    USE_MODEL = False
    print("Using fallback embeddings (install transformers+torch for better results)")

# ------------------ Reference Solutions (Two Sum) ------------------
BRUTE_FORCE = """
def two_sum_brute(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
"""

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

REFERENCE_SOLUTIONS = {
    "TwoSum": {
        "Brute Force": [BRUTE_FORCE],
        "Hash Map": [HASH_MAP],
        "Two Pointers": [TWO_POINTERS],
        "Set-Based": [SET_BASED],
        "Binary Search": [SORT_BINSEARCH]
    }
}

# Structural heuristics for Two Sum approaches
STRUCTURAL_RULES = {
    "TwoSum": {
        "Brute Force": ["nested_loops"],
        "Hash Map": ["dictionary", "complement"],
        "Two Pointers": ["sorting", "two_pointers"],
        "Set-Based": ["set_operations", "complement"],
        "Binary Search": ["bisect", "sorting"]
    }
}

# ------------------ Embedding Utilities ------------------
class SimpleTokenizerFallback:
    def __init__(self, refs: Dict[str, Dict[str, List[str]]]):
        counter = Counter()
        for problem in refs.values():
            for sols in problem.values():
                for s in sols:
                    for tok in self._tokenize(s):
                        counter[tok] += 1
        top = [tok for tok, _ in counter.most_common(1024)]
        self.vocab = {tok:i for i,tok in enumerate(top)}

    def _tokenize(self, s):
        return [t for t in re.split(r'[^0-9A-Za-z_]+', s) if t]

    def embed(self, s):
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in self._tokenize(s):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1.0
        norm = np.linalg.norm(vec)
        if norm>0:
            vec = vec/norm
        return vec

class UniXcoderEmbedder:
    def __init__(self, model_name="microsoft/unixcoder-base", device=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device:
            self.model.to(device)

    def embed(self, code: str):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        if hasattr(self.model,'device'):
            device = next(self.model.parameters()).device
            inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.pooler_output.squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm>0:
            emb = emb/norm
        return emb

EMBEDDER = UniXcoderEmbedder() if USE_MODEL else SimpleTokenizerFallback(REFERENCE_SOLUTIONS)

# ------------------ Precompute prototypes ------------------
def compute_prototypes(refs: Dict[str, Dict[str, List[str]]]):
    proto = {}
    for problem, approaches in refs.items():
        proto[problem] = {}
        for approach, sols in approaches.items():
            vecs = [EMBEDDER.embed(s) for s in sols]
            mean = np.mean(np.stack(vecs), axis=0)
            n = np.linalg.norm(mean)
            if n>0:
                mean = mean/n
            proto[problem][approach] = mean
    return proto

PROTOTYPES = compute_prototypes(REFERENCE_SOLUTIONS)

# ------------------ Structural heuristics for Two Sum ------------------
def detect_nested_loops(code:str)->bool:
    """Detect nested for loops (brute force pattern)"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        return True
        return False
    except:
        return 'for' in code and code.count('for') >= 2

def detect_dictionary(code:str)->bool:
    """Detect dictionary usage"""
    return bool(re.search(r'\{\}|\bdict\(\)|\bseen\s*=\s*\{\}|\w+\s*=\s*\{\}', code))

def detect_complement(code:str)->bool:
    """Detect complement calculation pattern"""
    return bool(re.search(r'complement|target\s*-\s*\w+|\w+\s*-\s*target', code))

def detect_sorting(code:str)->bool:
    """Detect sorting operations"""
    return bool(re.search(r'\.sort\(\)|sorted\(', code))

def detect_two_pointers(code:str)->bool:
    """Detect two pointers pattern"""
    return bool(re.search(r'\bleft\b.*\bright\b|\bright\b.*\bleft\b|while.*<.*:', code))

def detect_set_operations(code:str)->bool:
    """Detect set usage"""
    return bool(re.search(r'\bset\(\)|\bseen\s*=\s*set\(\)|\w+\s*=\s*set\(\)|\.add\(', code))

def detect_binary_search(code:str)->bool:
    """Detect binary search usage"""
    return bool(re.search(r'\bbisect\b|\bbisect_left\b|\bbisect_right\b', code))

def structural_score(problem:str, approach:str, code:str)->float:
    """Compute structural similarity score based on heuristics"""
    heuristics = STRUCTURAL_RULES.get(problem, {}).get(approach, [])
    score = 0.0
    total_heuristics = len(heuristics)
    
    if total_heuristics == 0:
        return 0.0
    
    for h in heuristics:
        if h=='nested_loops' and detect_nested_loops(code): 
            score += 1.0
        elif h=='dictionary' and detect_dictionary(code): 
            score += 1.0
        elif h=='complement' and detect_complement(code): 
            score += 1.0
        elif h=='sorting' and detect_sorting(code): 
            score += 1.0
        elif h=='two_pointers' and detect_two_pointers(code): 
            score += 1.0
        elif h=='set_operations' and detect_set_operations(code): 
            score += 1.0
        elif h=='bisect' and detect_binary_search(code): 
            score += 1.0
    
    return min(score / total_heuristics, 1.0)

# ------------------ Similarity ------------------
def cosine_sim(a:np.ndarray,b:np.ndarray)->float:
    if a is None or b is None: return 0.0
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denom) if denom>0 else 0.0

def similarity_scores(problem:str, code:str)->Dict[str,float]:
    emb = EMBEDDER.embed(code)
    return {a: cosine_sim(emb, PROTOTYPES[problem][a]) for a in PROTOTYPES[problem]}

# ------------------ Confidence computation ------------------
def compute_confidence(problem:str, code:str, previous_smoothed:Dict[str,float]=None, structres:Dict[str,float]=None,)->Dict[str,float]:
    sims = similarity_scores(problem,code)
    confs = {}
    approaches = list(sims.keys())
    sorted_sims = sorted(sims.values(), reverse=True)
    top = sorted_sims[0]
    second = sorted_sims[1] if len(sorted_sims)>1 else 0.0
    margin = top - second
    
    for a in approaches:
        s = sims[a]
        h = structres[a]
        t = previous_smoothed.get(a,0.0) if previous_smoothed else 0.0
        # weighted combination: semantic=0.5, structural=0.3, margin=0.1, temporal=0.1
        conf = 0.5*s + 0.3*h + 0.1*margin + 0.1*t
        confs[a] = min(conf,1.0)
    return confs

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
app = FastAPI(title="Two Sum Approach Detector", version="0.1")
from fastapi.middleware.cors import CORSMiddleware

# Allow frontend (served from file:// or localhost) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

from compcheck import CompleteEnhancedDSAPatternDetector

prototype_scores = {}
for label, code in REFERENCE_SOLUTIONS["TwoSum"].items():
    detector = CompleteEnhancedDSAPatternDetector()
    scores=detector.get_all_pattern_scores(code[0])
    prototype_scores[label]=scores
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

@app.post("/api/snapshot")
def post_snapshot(payload:SnapshotPayload):
    """Submit a code snapshot and get approach similarity scores"""
    ts = payload.timestamp or time.time()
    sess = payload.session_id
    prob = payload.problem_id
    code = payload.code
    
    if not code.strip():
        raise HTTPException(status_code=400,detail="Empty code")
    
    # Check if problem exists
    if prob not in PROTOTYPES:
        raise HTTPException(status_code=400, detail=f"Unknown problem: {prob}")
    
    with LOCK:
        detector = CompleteEnhancedDSAPatternDetector()
        user_scores_code=detector.get_all_pattern_scores(code)
        structres=compare_user_with_prototypes(user_scores_code, prototype_scores, F, features)
        hist = SESSION_STORE[sess][prob]
        prev_smoothed = hist[-1]['smoothed_conf'] if hist else None
        conf = compute_confidence(prob,code,prev_smoothed,structres)
        smoothed = {a:0.3*conf[a]+0.7*prev_smoothed.get(a,0.0) if prev_smoothed else conf[a] for a in conf}
        
        obs = {
            "snapshot_id": payload.snapshot_id,
            "timestamp": ts,
            "code_hash": hash(code),
            "similarities": similarity_scores(prob,code),
            "structural_scores": structres,
            "confidence": conf,
            "smoothed_conf": smoothed,
            "feature-vect":user_scores_code
        }
        hist.append(obs)
    
    return {
        "ok": True, 
        "session": sess, 
        "problem": prob, 
        "snapshot_id": payload.snapshot_id, 
        "confidence": conf, 
        "smoothed_conf": smoothed,
        "similarities": obs["similarities"],
        "structural_scores": obs["structural_scores"],
        "feature-vect":obs["feature-vect"]
    }

@app.get("/api/session/{session_id}/{problem_id}/label")
def get_label(session_id:str,problem_id:str):
    """Get the most likely approach label for the current session"""
    with LOCK:
        hist = SESSION_STORE.get(session_id,{}).get(problem_id,[])
        if not hist: 
            raise HTTPException(status_code=404,detail="No snapshots")
        
        smoothed = hist[-1]['smoothed_conf']
        sorted_items = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
        top, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items)>1 else 0.0
        margin = top_score - second_score
        
        confidence_level = "high" if top_score>0.75 and margin>0.1 else "medium" if top_score>0.5 else "low"
        
        return {
            "approach": top,
            "score": top_score,
            "margin": margin,
            "confidence_level": confidence_level,
            "smoothed": smoothed
        }

@app.get("/api/session/{session_id}/{problem_id}/history")
def get_history(session_id:str,problem_id:str,limit:int=50):
    """Get session history"""
    with LOCK:
        hist = list(SESSION_STORE.get(session_id,{}).get(problem_id,[]))[-limit:]
    return {"session_id":session_id,"problem_id":problem_id,"history":hist}

@app.get("/api/approaches/{problem_id}")
def list_approaches(problem_id:str):
    """List available approaches for a problem"""
    if problem_id not in PROTOTYPES:
        raise HTTPException(status_code=404, detail=f"Unknown problem: {problem_id}")
    
    return {
        "problem_id": problem_id,
        "approaches": list(PROTOTYPES[problem_id].keys()),
        "use_model": USE_MODEL
    }

@app.get("/api/problems")
def list_problems():
    """List all available problems"""
    return {
        "problems": list(PROTOTYPES.keys()),
        "use_model": USE_MODEL
    }

@app.post("/api/compare")
def compare_code(code: str, problem_id: str = "TwoSum"):
    """Compare code against all approaches without storing in session"""
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty code")
    
    if problem_id not in PROTOTYPES:
        raise HTTPException(status_code=400, detail=f"Unknown problem: {problem_id}")
    
    similarities = similarity_scores(problem_id, code)
    structural = {a: structural_score(problem_id, a, code) for a in PROTOTYPES[problem_id]}
    confidence = compute_confidence(problem_id, code)
    
    return {
        "problem_id": problem_id,
        "similarities": similarities,
        "structural_scores": structural,
        "confidence": confidence,
        "most_likely": max(confidence.items(), key=lambda x: x[1])
    }

@app.get("/health")
def health(): 
    return {"ok":True,"use_model":USE_MODEL, "problems": list(PROTOTYPES.keys())}

# Example usage endpoint
@app.get("/api/example")
def get_example():
    """Get example user code for testing"""
    user_code = """
def two_sum_user(nums, target):
    seen = {}
    for idx, val in enumerate(nums):
        if target - val in seen:
            return [seen[target - val], idx]
        seen[val] = idx
    return []
"""
    return {"example_code": user_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)