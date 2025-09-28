"""
approach_detector_server.py

Ready-to-run FastAPI server that:
- Accepts code snapshots (POST /api/snapshot)
- Computes embedding (real UniXcoder if transformers+torch installed; otherwise a simple token-frequency fallback)
- Compares against precomputed approach prototypes
- Stores a short time-series of observations per session
- Returns smoothed detection label (GET /api/session/{session_id}/label)
- Returns recent history (GET /api/session/{session_id}/history)

Run:
    pip install fastapi uvicorn
    # OPTIONAL (better quality): pip install transformers torch
    uvicorn approach_detector_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import deque, Counter, defaultdict
import time, os, threading
import numpy as np
from typing import Dict, Any, List

# Try to import transformers and torch for real embeddings; otherwise fallback.
USE_MODEL = False
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    # Prefer to enable model if available
    USE_MODEL = True
except Exception as e:
    print("transformers/torch not available; using fallback token-frequency embeddings. Install 'transformers'+'torch' for better results.")
    USE_MODEL = False

# -------- Reference solutions (LIS example) --------
BRUTE_FORCE = """
def lis_recursive(nums):
    def helper(i, prev):
        if i == len(nums):
            return 0
        taken = 0
        if nums[i] > prev:
            taken = 1 + helper(i + 1, nums[i])
        not_taken = helper(i + 1, prev)
        return max(taken, not_taken)
    return helper(0, float('-inf'))
"""

TOP_DOWN_DP = """
def lis_top_down(nums):
    memo = {}
    def helper(i, prev):
        if i == len(nums):
            return 0
        if (i, prev) in memo:
            return memo[(i, prev)]
        taken = 0
        if nums[i] > prev:
            taken = 1 + helper(i + 1, nums[i])
        not_taken = helper(i + 1, prev)
        memo[(i, prev)] = max(taken, not_taken)
        return memo[(i, prev)]
    return helper(0, float('-inf'))
"""

BOTTOM_UP_DP = """
def lis_bottom_up(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
"""

BINARY_SEARCH_DP = """
from bisect import bisect_left
def lis_binary_search(nums):
    sub = []
    for x in nums:
        i = bisect_left(sub, x)
        if i == len(sub):
            sub.append(x)
        else:
            sub[i] = x
    return len(sub)
"""

REFERENCE_SOLUTIONS = {
    "Brute Force": [BRUTE_FORCE],
    "Top-Down DP": [TOP_DOWN_DP],
    "Bottom-Up DP": [BOTTOM_UP_DP],
    "Binary Search DP": [BINARY_SEARCH_DP],
}

# add near the top of your server file
import ast, re

def detect_recursion_ast(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fname = node.name
            # Search for a Call node inside function body that calls fname
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    # function call like fname(...)
                    if isinstance(n.func, ast.Name) and n.func.id == fname:
                        return True
                    # method-style call fname as attribute (rare but possible)
                    if isinstance(n.func, ast.Attribute) and n.func.attr == fname:
                        return True
    return False

def detect_recursion_regex(code: str) -> bool:
    # fallback for unparsable code: find a def name and same-name call
    m = re.search(r'def\s+([A-Za-z_]\w*)\s*\(', code)
    if not m:
        return False
    fname = m.group(1)
    # ignore the def line itself, search elsewhere for calls to fname(
    # ensure there is a call to fname after the def
    return bool(re.search(r'\b' + re.escape(fname) + r'\s*\(', code[m.end():]))

def detect_memoization(code: str) -> bool:
    # quick heuristic: look for 'memo', 'cache', 'lru_cache', or assignments like memo = {}
    if re.search(r'\bmemo\b|\bcache\b|\bvisited\b', code):
        return True
    if re.search(r'\bfrom\s+functools\s+import\s+lru_cache', code):
        return True
    if re.search(r'\b@lru_cache\b', code):
        return True
    # dict initialization heuristic
    if re.search(r'\b\w+\s*=\s*{}\b', code):
        return True
    return False

def detect_bottom_up_dp(code: str) -> bool:
    # look for dp-like patterns: dp = [1] * n, dp = [0]*n, "dp[" indexing or dp variable name
    if re.search(r'\bdp\s*=\s*\[.*\]\s*\*\s*\w+', code):
        return True
    if re.search(r'\bdp\s*\[', code):
        return True
    # nested for loops updating dp-like array
    if re.search(r'for\s+.*:\s*\n\s*for\s+.*:\s*\n\s*.*dp\[', code, re.M):
        return True
    return False

def detect_binary_search_pattern(code: str) -> bool:
    if 'bisect_left' in code or 'bisect' in code:
        return True
    # basic mid/low/high pattern
    if re.search(r'\blow\b.*\bmid\b.*\bhigh\b', code):
        return True
    return False

def structural_scores(code: str) -> dict:
    """Return structural heuristic scores in 0..1 for each approach."""
    # try AST for recursion first, fallback to regex
    is_rec = detect_recursion_ast(code) or detect_recursion_regex(code)
    has_memo = detect_memoization(code)
    has_bottom_dp = detect_bottom_up_dp(code)
    has_binsearch = detect_binary_search_pattern(code)

    scores = {
        "Brute Force": 0.0,
        "Top-Down DP": 0.0,
        "Bottom-Up DP": 0.0,
        "Binary Search DP": 0.0
    }
    # heuristics:
    if is_rec:
        # recursion strongly suggests brute force or top-down DP (depending on memo)
        scores["Brute Force"] = 0.9 if not has_memo else 0.2
        scores["Top-Down DP"] = 0.9 if has_memo else 0.1
    if has_memo:
        scores["Top-Down DP"] = max(scores["Top-Down DP"], 0.9)
    if has_bottom_dp:
        scores["Bottom-Up DP"] = 0.9
    if has_binsearch:
        scores["Binary Search DP"] = 0.95
    return scores


# -------- Embedding utilities --------
class SimpleTokenizerFallback:
    """Very small tokenizer and token-frequency 'embedding' fallback."""
    def __init__(self, refs: Dict[str, List[str]]):
        # Build vocabulary from references
        counter = Counter()
        for sols in refs.values():
            for s in sols:
                for tok in self._tokenize(s):
                    counter[tok] += 1
        # keep top N tokens to form vector
        top = [tok for tok, _ in counter.most_common(1024)]
        self.vocab = {tok: i for i, tok in enumerate(top)}
    
    def _tokenize(self, s: str):
        # naive tokenization: split on non-alnum
        import re
        toks = [t for t in re.split(r'[^0-9A-Za-z_]+', s) if t]
        return toks
    
    def embed(self, s: str):
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in self._tokenize(s):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

class UniXcoderEmbedder:
    """Wrapper around UniXcoder (transformers) to produce pooler_output embeddings."""
    def __init__(self, model_name="microsoft/unixcoder-base", device=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # move to CPU or GPU as available
        if device:
            self.model.to(device)
    
    def embed(self, code: str):
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # Move tensors to device if model is moved
        if hasattr(self.model, 'device'):
            device = next(self.model.parameters()).device
            inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.pooler_output.squeeze().cpu().numpy()
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

# Initialize chosen embedding backend
if USE_MODEL:
    try:
        EMBEDDER = UniXcoderEmbedder()
    except Exception as e:
        print("Failed to initialize UniXcoder; falling back to simple tokenizer:", e)
        USE_MODEL = False

if not USE_MODEL:
    EMBEDDER = SimpleTokenizerFallback(REFERENCE_SOLUTIONS)

# -------- Precompute reference prototypes --------
def compute_prototype_embeddings(refs: Dict[str, List[str]]):
    proto = {}
    for approach, sols in refs.items():
        vecs = []
        for s in sols:
            v = EMBEDDER.embed(s)
            vecs.append(v)
        # mean, then normalize
        mean = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(mean)
        if n > 0:
            mean = mean / n
        proto[approach] = mean
    return proto

PROTOTYPES = compute_prototype_embeddings(REFERENCE_SOLUTIONS)

# -------- Similarity function (cosine) --------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a and b assumed normalized; safe fallback
    if a is None or b is None:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def compute_similarities(emb: np.ndarray) -> Dict[str, float]:
    return {approach: cosine_sim(emb, ref) for approach, ref in PROTOTYPES.items()}

# -------- In-memory time-series store (per session) --------
MAX_HISTORY = 50  # how many recent snapshots to keep per session
SESSION_STORE: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
LOCK = threading.Lock()

# Observation model
class SnapshotPayload(BaseModel):
    session_id: str
    snapshot_id: str
    code: str
    timestamp: float = None  # unix seconds (optional)

# FastAPI app
app = FastAPI(title="Approach Detector Server", version="0.1")

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
def post_snapshot(payload: SnapshotPayload):
    ts = payload.timestamp or time.time()
    session = payload.session_id
    code = payload.code or ""
    if len(code.strip()) < 1:
        raise HTTPException(status_code=400, detail="code is empty")
    # embed (synchronously)
    emb = EMBEDDER.embed(code)
    sims = compute_similarities(emb)
    obs = {
        "snapshot_id": payload.snapshot_id,
        "timestamp": ts,
        "code_hash": hash(code),
        "similarities": sims,
        # optional diagnostics
        "len_code": len(code)
    }
    with LOCK:
        SESSION_STORE[session].append(obs)
    return {"ok": True, "session": session, "snapshot_id": payload.snapshot_id, "similarities": sims}

# Smoothing and detection helpers
def exponential_smoothing(observations: List[Dict[str, Any]], alpha=0.3) -> Dict[str, float]:
    if not observations:
        return {}
    # approaches set
    approaches = sorted(PROTOTYPES.keys())
    s = None
    for o in observations:
        arr = np.array([o["similarities"].get(a, 0.0) for a in approaches], dtype=np.float32)
        if s is None:
            s = arr
        else:
            s = alpha * arr + (1.0 - alpha) * s
    # normalize final vector to 0..1 (optional)
    # clip/scale to 0..1
    min_v = float(np.min(s))
    max_v = float(np.max(s))
    if max_v - min_v > 1e-6:
        s = (s - min_v) / (max_v - min_v)
    return dict(zip(approaches, s.tolist()))

def detect_label_for_session(session_id: str):
    with LOCK:
        obs_list = list(SESSION_STORE.get(session_id, []))
    if not obs_list:
        return {"error": "no snapshots"}, 404
    smoothed = exponential_smoothing(obs_list)
    if not smoothed:
        return {"error": "no data"}, 500
    # pick top approach and compute margin
    sorted_items = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    margin = top_score - second_score
    # confidence heuristics
    if top_score >= 0.75 and margin > 0.1:
        confidence = "high"
    elif top_score >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "approach": top,
        "score": float(top_score),
        "margin": float(margin),
        "confidence": confidence,
        "smoothed": smoothed
    }, 200

@app.get("/api/session/{session_id}/label")
def get_label(session_id: str):
    result, status = detect_label_for_session(session_id)
    if status != 200:
        raise HTTPException(status_code=status, detail=result.get("error"))
    return result

@app.get("/api/session/{session_id}/history")
def get_history(session_id: str, limit: int = 50):
    with LOCK:
        hist = list(SESSION_STORE.get(session_id, []))[-limit:]
    return {"session_id": session_id, "history": hist}

@app.get("/api/approaches")
def list_approaches():
    return {"approaches": list(PROTOTYPES.keys()), "use_model_backed_embedder": USE_MODEL}

# Simple health check
@app.get("/health")
def health():
    return {"ok": True, "use_model": USE_MODEL}
