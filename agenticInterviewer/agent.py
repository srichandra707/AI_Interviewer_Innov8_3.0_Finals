from __future__ import annotations
import google.generativeai as genai
import os
import threading
import websockets
import json
import time
import random
import ast
import textwrap
import tempfile
import shutil
import os
from typing import Any, Dict, List, Tuple
import queue
import asyncio
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def gemini_chat_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    # ...existing code...
    # print(genai.list_models())
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    )
    return response.text.strip()

api_url = "http://127.0.0.1:5000"  # Replace with actual URL

transcription_queue = queue.Queue()

INTERVIEW_DURATION = 300

# Globals for hinting state with a lock
LAST_HINT_TIME = 0.0
HINT_LOCK = threading.Lock()
HINT_ATTEMPTS: Dict[str, int] = {}      # attempts per issue type
HINT_HISTORY: List[Dict[str, Any]] = []  # list of issued hints
SIM_HISTORY: List[Tuple[float, float, Dict[str, Any]]] = []  # (ts, score, details)

# Thresholds / config
IDLE_SECONDS_FOR_HINT = 30
SIM_IMPROVEMENT_THRESHOLD = 0.01
MIN_SECONDS_BETWEEN_HINTS = 10  # avoid spamming frontend too fast

SESSION_FILE = "session.txt"             # will contain concatenated iteration logs (user requested)
AUDIO_TRANSCRIPT_FILE = "audio_transcript.txt"  # file that stores all audio text collected during interview (change if needed)
FINAL_REPORT_FILE = "final_report.txt"   # LLM's textual report output
FINAL_REPORT_JSON = "final_report.json"  # structured JSON copy of the report (optional)

# ------------------ Utilities ------------------

def detect_top_issue(behav_details: Dict[str, Any]) -> str:
    if not behav_details:
        return "style"
    statuses = [c.get("status") for c in behav_details.get("cases", [])]
    if any(s == "syntax_error" for s in statuses):
        return "syntax"
    if any(s == "runtime_error" for s in statuses):
        return "runtime"
    if any(s == "timeout" for s in statuses):
        return "tle"
    if any(s == "wrong" for s in statuses):
        return "logic"
    return None

def connect_to_transcription_server():
    """
    Connects to a websocket transcription server and pushes messages to transcription_queue.
    Each message can be a string or a JSON string with timestamp/text.
    """
    print("Connecting to transcription server...")
    uri = "ws://localhost:8766"

    async def run():
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to transcription server at {uri}")
                async for message in websocket:
                    # message may be a plain string or JSON string like {"ts":..., "text":"..."}
                    try:
                        parsed = json.loads(message)
                        ts = parsed.get("ts", time.time())
                        txt = parsed.get("text", parsed.get("transcript", "") or "")
                        transcription_queue.put((ts, txt))
                        try:
                            with open(AUDIO_TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                                f.write(f"{ts}: {txt}\n")
                        except Exception as e:
                            print(f"[audio file append] failed: {e}")
                    except Exception:
                        # fallback to raw string with current timestamp
                        transcription_queue.put((time.time(), str(message)))
        except Exception as e:
            print(f"Error connecting to transcription server: {e}")

    # run websocket in blocking way within thread
    try:
        asyncio.run(run())
    except Exception as e:
        print(f"Transcription connection ended: {e}")

def fetch_snapshot():
    try:
        resp = requests.get(f"{api_url}/api/snapshots", timeout=5.0)
        resp.raise_for_status()
        snap=resp.json()  # Parse JSON response
        print(snap)
        # print(resp.text)
        return snap
    except Exception as e:
        print(f"[analysis] failed to fetch snapshot: {e}")
        return {}

def collect_recent_transcript(max_items=50):
    """
    Read up to max_items most recent transcript segments from transcription_queue.
    Each queue entry is expected to be (timestamp, text) or raw string.
    Returns combined text and last timestamp seen (or None).
    """
    parts = []
    last_ts = None
    temp_items = []
    # drain queue into temp list up to max_items
    while not transcription_queue.empty() and len(temp_items) < max_items:
        try:
            item = transcription_queue.get_nowait()
            temp_items.append(item)
        except queue.Empty:
            break

    # keep them (we consumed them) — if you want to preserve, you'd need to re-queue
    # Build parts from parsed items
    for it in temp_items:
        if isinstance(it, tuple) and len(it) == 2:
            ts, txt = it
            parts.append(txt)
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
        else:
            # raw string
            parts.append(str(it))
            last_ts = time.time()

    combined = " ".join(parts[-max_items:])
    return combined, last_ts

# ------------------ AST / Similarity ------------------

def normalize_and_rename_ast(src):
    tree = ast.parse(src)

    class Renamer(ast.NodeTransformer):
        def __init__(self):
            super().__init__()
            self.map = {}
            self.counter = 0

        def _new(self, orig: str) -> str:
            if orig in self.map:
                return self.map[orig]
            name = f"v{self.counter}"
            self.counter += 1
            self.map[orig] = name
            return name

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
                return ast.copy_location(ast.Name(id=self._new(node.id), ctx=node.ctx), node)
            return node

        def visit_arg(self, node):
            node.arg = self._new(node.arg)
            return node

        def visit_FunctionDef(self, node):
            node.name = self._new(node.name)
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            node.name = self._new(node.name)
            self.generic_visit(node)
            return node

    renamer = Renamer()
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)

def structural_similarity(src1, src2, precomputed_ast2=None):
    try:
        a = normalize_and_rename_ast(src1).split()
        b = precomputed_ast2.split() if precomputed_ast2 else normalize_and_rename_ast(src2).split()
    except Exception as e:
        # if AST fails, return 0 structural similarity
        return 0.0
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    uni = len(set_a | set_b)
    return inter / uni

def llm_semantic_similarity(code, optimal, question):
    prompt = f"""
You are an expert code reviewer. Rate How Similar the Candidate's code is to the optimal Solution for the following problem on a scale from 0 (completely different) to 1 (equivalent algorithm/logic)

Problem:
{question}

Optimal Solution:
{optimal}

Candidate Code (may not run):
{code}

Rules for Evaluation:
- Consider algorithmic structure, data structures used, and core approach. 
- Ignore variable names, whitespace, and comments. Return a single number between 0 and 1 (e.g., 0.0, 0.35, 0.9).
- return a one-line reason. AND NOTHING MORE
- Provide output as JSON: {{"score": number, "reason": "..."}}
"""
    response = gemini_chat_completion(prompt, temperature=0.0, max_tokens=200)
    try:
        parsed = json.loads(response)
        return float(parsed.get('score', 0.0)), parsed.get('reason', '')
    except Exception:
        import re
        m = re.search(r"([0-1](?:\.[0-9]+)?)", response)
        if m:
            return float(m.group(1)), response
        return 0.0, response

def combined_similarity(code, optimal, question, tests, alpha=0.2, optimal_ast=None):
    struct = 0.0
    try:
        struct = structural_similarity(code, optimal, precomputed_ast2=optimal_ast)
    except Exception:
        struct = 0.0

    semantic, semantic_reason = llm_semantic_similarity(code, optimal, question)
    w_struct = alpha
    w_sem = 1 - alpha
    combined = w_struct * struct + w_sem * semantic
    details = {
        "structural": struct,
        "semantic": semantic,
        "semantic_reason": semantic_reason,
        # provide placeholder for behavioural details (testcases) if available
        "behav_details": {
            "cases": tests if isinstance(tests, list) else []
        }
    }
    return combined, details

def recompute_similarity(snapshot: Dict[str, Any], optimal: str, tests: List[Dict[str, Any]], question: str):
    """
    Given a frontend snapshot (which should include candidate code), compute similarity and return (score, details)
    Expect snapshot to have e.g. snapshot.get('code') or snapshot.get('latest_code').
    """
    code = ""
    if isinstance(snapshot, dict):
        code = snapshot.get("code") or snapshot.get("latest_code") or snapshot.get("candidate_code") or ""
    else:
        code = str(snapshot)

    try:
        combined_score, details = combined_similarity(code, optimal, question, tests)
        return combined_score, details
    except Exception as e:
        print(f"[similarity] error computing similarity: {e}")
        return 0.0, {"structural": 0.0, "semantic": 0.0, "semantic_reason": str(e), "behav_details": {"cases": []}}

def communicate_hints_with_frontend(hints):
    payload = {"hints": hints}
    try:
        response = requests.post(f"{api_url}/api/hints", json=payload, timeout=5.0)
        if response.status_code == 200:
            print("Hints sent Successfully", hints)
            global LAST_HINT_TIME
            with HINT_LOCK:
                LAST_HINT_TIME = time.time()
        else:
            print(f"Failed to send hints. Status code: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending hints: {e}")

# Prompt template for hint
HINT_PROMPT_BASE = """
You are a senior technical interviewer and coach. Provide a concise hint (one or two sentences) at the requested level.
Levels:
- Nudge: short, conceptual idea (vague)
- Guide: more concrete, suggests functions/areas to inspect
- Direction: explicit targeted instruction to fix a single problem (not to give full solution)

Context (read carefully):
Problem: {title}
Optimal solution:
{optimal}
Testcases (representative):
{tests}

Candidate code (current snapshot):
{candidate}

Recent audio transcript (last 1 minute) from candidate:
{audio}

Now produce a hint at level {level} focused on the most pressing issue: {issue}.
Keep it brief and actionable.
Return only the hint text (no JSON wrapper).
"""

def build_hint(problem_title, optimal_code, tests, candidate_code, audio_last_min, issue, level):
    tests_preview = json.dumps(tests[:5], default=str)
    # failed_preview = json.dumps(failed_cases or [], default=str)
    prompt = HINT_PROMPT_BASE.format(
        title=problem_title,
        optimal=optimal_code,
        tests=tests_preview,
        candidate=candidate_code,
        audio=audio_last_min,
        level=level,
        issue=issue,
    )
    return gemini_chat_completion(prompt, temperature=0.2, max_tokens=200)

MIN_SECONDS_BETWEEN_HINTS = 10
SIM_IMPROVEMENT_THRESHOLD = 0.01

HINT_LOCK = threading.Lock()
SIM_HISTORY: List[Tuple[float, float, Dict[str, Any]]] = []
HINT_ATTEMPTS: Dict[str, int] = {}
HINT_HISTORY: List[Dict[str, Any]] = []
LAST_HINT_TIME: float = 0.0

def hint_analysis(
    recent_transcript: str,
    score: float,
    details: Dict[str, Any],
    current_time: float,
    snapshot: Dict[str, Any],
    optimal_code: str,
    testcases: List[Dict[str, Any]],
    question: str
) -> List[Dict[str, Any]]:
    """
    Decide whether to generate a hint. Returns a list of hint dict(s) or empty list.
    Logic:
    - If similarity improvement low OR no audio transcript -> issue hint.
    - Enforce minimum 10s between hints.
    - Track attempts per issue and escalate level: 0->Nudge,1->Guide,>=2->Direction
    """
    now = current_time

    # audio presence check (no timestamps available)
    audio_empty = (not recent_transcript) or (recent_transcript.strip() == "")

    # similarity improvement check
    last_improve = 0.0
    with HINT_LOCK:
        if len(SIM_HISTORY) >= 2:
            last = SIM_HISTORY[-1][1]
            prev = SIM_HISTORY[-2][1]
            last_improve = last - prev
    should_due_to_similarity = (last_improve <= SIM_IMPROVEMENT_THRESHOLD)

    # enforce global cooldown
    global LAST_HINT_TIME
    with HINT_LOCK:
        time_since_last_hint = time.time() - LAST_HINT_TIME if LAST_HINT_TIME else float("inf")
    if time_since_last_hint < MIN_SECONDS_BETWEEN_HINTS:
        return []

    # decide if we should issue a hint
    if not (audio_empty or should_due_to_similarity):
        return []

    # pick top issue
    behav = details.get("behav_details") if isinstance(details, dict) else {}
    top_issue = detect_top_issue(behav)

    # determine hint level
    with HINT_LOCK:
        attempts = HINT_ATTEMPTS.get(top_issue, 0)
    if attempts == 0:
        level = "Nudge"
    elif attempts == 1:
        level = "Guide"
    else:
        level = "Direction"

    # get candidate code
    candidate_code = snapshot.get("code") or snapshot.get("latest_code") or snapshot.get("candidate_code") or ""

    # call LLM to build hint
    try:
        hint_text = build_hint(
            question,
            optimal_code,
            testcases,
            candidate_code,
            recent_transcript or "",
            top_issue,
            level
        )
    except Exception as e:
        print(f"[hint] failed to produce hint: {e}")
        hint_text = "I couldn't generate a hint right now. Try making a small change to your approach."

    hint = {
        "time": now,
        "issue": top_issue,
        "level": level,
        "hint": hint_text,
        "score": score,
    }

    # update state
    with HINT_LOCK:
        HINT_HISTORY.append(hint)
        HINT_ATTEMPTS[top_issue] = HINT_ATTEMPTS.get(top_issue, 0) + 1
        LAST_HINT_TIME = time.time()

    return [hint]

# ------------------ Core loop / orchestration ------------------

def connect_to_frontend_server(question):
    print("Starting the core application logic...")
    optimal_code = "def maxSubArray(nums):\n    max_sum = nums[0]\n    cur_sum = nums[0]\n    for num in nums[1:]:\n        cur_sum = max(num, cur_sum + num)\n        max_sum = max(max_sum, cur_sum)\n    return max_sum"
    testcases = [
        {"input": "[-2,1,-3,4,-1,2,1,-5,4]", "output": 6},
        {"input": "[1]", "output": 1},
        {"input": "[5,4,-1,7,8]", "output": 23},
        {"input": "[-1,-2,-3]", "output": -1},
        {"input": "[2,-1,2,3,4,-5]", "output": 10}
    ]

    start_time = time.time()

    while time.time() - start_time < INTERVIEW_DURATION:
        try:
            snapshot = fetch_snapshot()  # expects dict with code + metadata
            score, details = recompute_similarity(snapshot, optimal_code, testcases, question)

            # Update SIM_HISTORY
            with HINT_LOCK:
                SIM_HISTORY.append((time.time(), score, details))
                # keep SIM_HISTORY modest size
                if len(SIM_HISTORY) > 200:
                    SIM_HISTORY.pop(0)

            recent_transcript, last_audio_ts = collect_recent_transcript(max_items=50)

            # For debugging / session log
            try:
                with open("session.txt", "a", encoding="utf-8") as f:  # append, not overwrite
                    f.write(f"--- ITERATION {len(SIM_HISTORY)} ---\n")
                    f.write(f"timestamp: {time.time()}\n")
                    f.write(f"score: {score}\n")
                    f.write(f"details: {json.dumps(details, default=str)}\n")
                    f.write("recent_transcript:\n")
                    f.write((recent_transcript or "") + "\n")
                    if isinstance(snapshot, dict):
                        f.write("snapshot_keys:\n")
                        f.write(json.dumps(list(snapshot.keys()), default=str) + "\n")
                        code = snapshot.get("code") or snapshot.get("latest_code") or snapshot.get("candidate_code") or ""
                        if code:
                            f.write("candidate_code_preview:\n")
                            f.write(code[:2000] + ("\n...TRUNCATED...\n" if len(code) > 2000 else "\n"))
                    else:
                        f.write(str(snapshot) + "\n")
                    f.write("\n")
            except Exception as e:
                print(f"[session append] failed: {e}")


            hints = hint_analysis(recent_transcript, score, details, time.time(), snapshot, optimal_code, testcases, question)
            if hints:
                communicate_hints_with_frontend(hints)

        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[main loop] error: {e}")

        # Poll interval
        time.sleep(5)

    end_time = time.time()
    # ensure we have the latest snapshot (if you have a real snapshot object, use that)
    final_snapshot = fetch_snapshot() or (frontend_url.get("snapshot") if isinstance(frontend_url, dict) else {})
    generate_report(start_time, end_time, SIM_HISTORY, HINT_HISTORY, question, optimal_code, final_snapshot)
    # End of interview handling (example)
    frontend_url = {"snapshot": "example_snapshot"}  # Replace with actual frontend URL object/dict
    snapshot = frontend_url.get('snapshot')

    # At end-of-interview in connect_to_frontend_server, after the loop:
    print(f"Final report written to {FINAL_REPORT_FILE} and {FINAL_REPORT_JSON}")

    print("Interview finished. Final snapshot:", snapshot)

def initialize_interview():
    question = """Find the maximum subarray sum in an integer array?"""
    payload = {"question": question}
    try:
        response = requests.post(f"{api_url}/api/question", json=payload, timeout=5.0)
        if response.status_code == 200:
            print("Question sent Successfully")
        else:
            print(f"Failed to send question. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending question: {e}")
    return question

def main():
    print("Connecting to transcription server and frontend server in threads...")

    question = initialize_interview()
    t1 = threading.Thread(target=connect_to_transcription_server, daemon=False)
    t2 = threading.Thread(target=connect_to_frontend_server, args=(question,), daemon=False)

    t1.start()
    t2.start()

    # Wait for interview duration then exit (or you can join threads)
    try:
        # main thread sleeps while worker threads run
        time.sleep(INTERVIEW_DURATION + 5)
        final_snapshot = {}  # or load last snapshot from queue/session
        # generate_report(time.time() - INTERVIEW_DURATION, time.time(), SIM_HISTORY, HINT_HISTORY, question, optimal_code, final_snapshot)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")


REPORT_PROMPT_TEMPLATE = """
You are a strict, forensic technical-interview summarizer. DO NOT ADD ANY FACTS THAT ARE NOT CONTAINED IN THE 'facts' JSON. Use ONLY the facts provided below to produce a report.

Output MUST be valid JSON (no extra text, not even the ```json top of the text) with this exact schema:
{{
  "dashboard_metrics": {{ ... }},                 # numeric metrics and short summary strings
  "human_summary": "...",                         # 3-6 short paragraphs describing the candidate's journey (facts-only)
  "session_playback": [                           # chronological list of notable events for playback (timestamps as unix seconds)
     {{ "ts": number, "type": "similarity"|"hint"|"snapshot", "summary": "..." }},
     ...
  ],
  "detailed_evaluation": {{                        # bullet-like structured evaluation
     "problem_understanding": "...",
     "algorithms_and_ds": "...",
     "debugging_and_iteration": "...",
     "communication": "...",
     "proactiveness": "..."
  }},
  "score_out_of_5": number,
  "hiring_decision": "hire"|"consider"|"reject",
  "decision_rationale": "..."                      # concise explanation referencing fields inside dashboard_metrics
}}

FACTS (use only these fields; do not hallucinate and do not invent extra timestamps or facts):
{facts_json}

INSTRUCTIONS:
- Use only the facts above. If something is unknown, say 'unknown' for that field.
- Keep the 'human_summary' factual, avoid adjectives that don't have evidence.
- Provide short, explicit references to evidence inside the 'decision_rationale' (e.g., 'final_similarity=0.73, hints_count=3').
- Make the output machine-parseable JSON exactly matching the schema.
- Temperature: 0.0. Be concise.
- GENERATE JSON ONLY AND NOTHING ELSE.
"""

def compute_metrics(sim_history: List[Tuple[float, float, Dict[str, Any]]], hint_history: List[Dict[str, Any]], start_time: float, end_time: float) -> Dict[str, Any]:
    """
    Returns deterministic numeric metrics derived from SIM_HISTORY and HINT_HISTORY.
    sim_history: list of (ts, score, details)
    hint_history: list of hints
    """
    metrics = {}
    if not sim_history:
        metrics.update({
            "final_similarity": 0.0,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "similarity_delta": 0.0,
            "measurements": 0
        })
    else:
        scores = [s for (_ts, s, _d) in sim_history]
        metrics["final_similarity"] = float(scores[-1])
        metrics["avg_similarity"] = float(sum(scores) / len(scores))
        metrics["min_similarity"] = float(min(scores))
        metrics["max_similarity"] = float(max(scores))
        metrics["similarity_delta"] = float(scores[-1] - scores[0]) if len(scores) >= 2 else 0.0
        metrics["measurements"] = len(scores)

    metrics["hints_count"] = len(hint_history or [])
    # hints per top_issue breakdown
    hints_by_issue = {}
    for h in (hint_history or []):
        issue = h.get("issue", "unknown")
        hints_by_issue[issue] = hints_by_issue.get(issue, 0) + 1
    metrics["hints_by_issue"] = hints_by_issue

    metrics["duration_seconds"] = float(max(0.0, end_time - start_time))
    # time to first hint:
    if hint_history:
        first_hint_ts = min(h.get("time", float("inf")) for h in hint_history)
        metrics["time_to_first_hint"] = float(first_hint_ts - start_time) if start_time and first_hint_ts != float("inf") else None
    else:
        metrics["time_to_first_hint"] = None

    # simple audio presence metric: fraction of iterations with non-empty transcript
    non_empty = 0
    for (_ts, _s, d) in sim_history:
        behav = d.get("behav_details") if isinstance(d, dict) else {}
        # can't rely on transcript here; we'll compute audio separately from audio file
        # keep placeholder
    return metrics

def compute_score_and_decision(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic mapping to a 0-5 score and hiring decision.
    - final_similarity heavily weighted
    - avg_similarity secondary
    - hint penalty: more hints reduce score
    Decision thresholds:
      score >= 4.0 -> 'hire'
      3.0 <= score < 4.0 -> 'consider'
      score < 3.0 -> 'reject'
    """
    final_sim = max(0.0, min(1.0, float(metrics.get("final_similarity", 0.0))))
    avg_sim = max(0.0, min(1.0, float(metrics.get("avg_similarity", final_sim))))
    hints = int(metrics.get("hints_count", 0))
    # penalty (0..1): each 2 hints -> 0.1 penalty, capped at 0.5
    hint_penalty = min(0.5, 0.05 * hints)
    # base combined raw 0..1
    raw = 0.6 * final_sim + 0.3 * avg_sim + 0.1 * (1.0 - hint_penalty)
    # Map to 0..5
    score_05 = round(max(0.0, min(5.0, raw * 5.0)), 2)

    if score_05 >= 4.0:
        decision = "hire"
    elif score_05 >= 3.0:
        decision = "consider"
    else:
        decision = "reject"

    rationale = {
        "weights": {"final_similarity": 0.6, "avg_similarity": 0.3, "hint_penalty_influence": 0.1},
        "final_sim": final_sim,
        "avg_sim": avg_sim,
        "hints": hints,
        "hint_penalty": hint_penalty,
        "raw_combined": raw
    }

    return {"score_out_of_5": score_05, "hiring_decision": decision, "rationale": rationale}

# --- Read audio transcript file safely ---
def load_audio_transcript(path: str = AUDIO_TRANSCRIPT_FILE) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
    
def generate_report(start_time: float, end_time: float, sim_history: List[Tuple[float, float, Dict[str, Any]]], hint_history: List[Dict[str, Any]], question: str, optimal_code: str, final_snapshot: Dict[str, Any]):
    # Build facts blob
    metrics = compute_metrics(sim_history, hint_history, start_time, end_time)
    decision = compute_score_and_decision(metrics)
    audio_text = load_audio_transcript()
    # minimal final candidate code snapshot:
    final_code = ""
    if isinstance(final_snapshot, dict):
        final_code = final_snapshot.get("code") or final_snapshot.get("latest_code") or final_snapshot.get("candidate_code") or ""
    facts = {
        "question": question,
        "optimal_code": optimal_code,
        "metrics": metrics,
        "decision_basis": decision,
        "sim_history": [
            {"ts": int(ts), "similarity": float(score), "details": details}
            for (ts, score, details) in sim_history
        ],
        "hint_history": hint_history,
        "audio_transcript": audio_text,
        "final_snapshot": {"keys": list(final_snapshot.keys()) if isinstance(final_snapshot, dict) else [], "code_preview": final_code[:500]}
    }

    prompt = REPORT_PROMPT_TEMPLATE.format(facts_json=json.dumps(facts, default=str))

    # call LLM deterministically
    try:
        report_text = gemini_chat_completion(prompt)
        # write the raw text and JSON
        with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report_text)
        # try to parse JSON and save machine-readable copy
        try:
            parsed = json.loads(report_text)
            with open(FINAL_REPORT_JSON, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, default=str)
        except Exception:
            # if parsing fails, still keep raw text
            print("[report] LLM output was not valid JSON; saved raw to final_report.txt")
    except Exception as e:
        print(f"[report] failed to generate report: {e}")

      
if __name__ == '__main__':
    main()
