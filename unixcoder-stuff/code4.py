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

# ------------------ Reference Solutions (Example) ------------------
REFERENCE_SOLUTIONS = {
    "LIS": {
        "Brute Force": ["...code1..."],
        "Top-Down DP": ["...code2..."],
        "Bottom-Up DP": ["...code3..."],
        "Binary Search DP": ["...code4..."]
    },
    "Knapsack": {
        "Recursive": ["..."],
        "Top-Down DP": ["..."],
        "Bottom-Up DP": ["..."]
    }
}

# Optional: structural heuristics per problem & approach
STRUCTURAL_RULES = {
    "LIS": {
        "Brute Force": ["recursion"],
        "Top-Down DP": ["recursion", "memoization"],
        "Bottom-Up DP": ["dp_array", "nested_loops"],
        "Binary Search DP": ["bisect"]
    },
    "Knapsack": {
        "Recursive": ["recursion"],
        "Top-Down DP": ["recursion","memoization"],
        "Bottom-Up DP": ["dp_array","nested_loops"]
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

# ------------------ Structural heuristics ------------------
def detect_recursion(code:str)->bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fname = node.name
                for n in ast.walk(node):
                    if isinstance(n, ast.Call):
                        if (hasattr(n.func,'id') and n.func.id==fname) or (hasattr(n.func,'attr') and n.func.attr==fname):
                            return True
        return False
    except:
        return False

def detect_memoization(code:str)->bool:
    return bool(re.search(r'\bmemo\b|\bcache\b|\blru_cache\b|\bvisited\b', code))

def detect_bottom_up_dp(code:str)->bool:
    return bool(re.search(r'\bdp\s*=\s*\[.*\]\s*\*\s*\w+|\bdp\[', code))

def detect_binary_search(code:str)->bool:
    return 'bisect_left' in code or 'bisect' in code

def structural_score(problem:str, approach:str, code:str)->float:
    heuristics = STRUCTURAL_RULES.get(problem, {}).get(approach, [])
    score = 0.0
    for h in heuristics:
        if h=='recursion' and detect_recursion(code): score += 0.9
        elif h=='memoization' and detect_memoization(code): score += 0.9
        elif h=='dp_array' and detect_bottom_up_dp(code): score += 0.9
        elif h=='bisect' and detect_binary_search(code): score += 0.9
    return min(score,1.0)

# ------------------ Similarity ------------------
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
            "structural_scores": {a:structural_score(prob,a,code) for a in PROTOTYPES[prob]},
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
