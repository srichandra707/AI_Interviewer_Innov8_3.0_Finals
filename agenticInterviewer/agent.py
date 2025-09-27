"""
Agentic AI Interviewer - single-file prototype (updated)

Changes made per your request:
- Picks a question from an external `questions.json` with equal probability.
- ALWAYS performs both AST structural similarity AND an LLM semantic check (even when AST parses), then combines them.
- Replaced the unsafe sandbox implementation with a clear TODO placeholder: you must provide a secure sandbox `execute_in_sandbox` implementation before running untrusted code.
- Replaced the hardcoded LLM hint stub with real OpenAI API calls using GPT-4.1 family (recommended for coding tasks). See web citations in the chat for rationale.
- The LLM hint prompt includes: problem + optimal solution + tests (sent once when the problem is chosen), candidate's current code snapshot, the last 1 minute of voice transcript, and failing testcases when relevant.

Security: DO NOT run untrusted candidate code until you implement `execute_in_sandbox` using Docker/microVM/cgroups/seccomp as described in earlier messages.

Requirements:
- Python 3.10+
- websockets (pip install websockets)
- openai (pip install openai)

Environment:
- Set OPENAI_API_KEY in your environment before running.

Usage:
- Place your questions.json (the 10 Q-A-Testcases JSON) in the same directory.
- Run: python agentic_interviewer.py

"""

from __future__ import annotations
import asyncio
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
import openai

# --- Configuration: choose an LLM model suitable for coding tasks ---
# Based on recent benchmarks and model releases, GPT-4.1 (mini) provides
# strong coding and instruction-following performance with a reasonable cost/latency tradeoff.
# See OpenAI release notes and comparisons for rationale.
MODEL = "gpt-4.1-mini"  # change to "gpt-4.1" if you want the highest quality
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- Load questions --------------------
with open("questions.json", "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)

# Choose a question with equal probability
def pick_question() -> Dict[str, Any]:
    return random.choice(QUESTIONS)

# -------------------- AST NORMALIZATION & STRUCTURAL SIMILARITY --------------------

def normalize_and_rename_ast(src: str) -> str:
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


def structural_similarity(src1: str, src2: str= None, precomputed_ast2: str= None) -> float:
    try:
        a = normalize_and_rename_ast(src1).split()
        b = precomputed_ast2.split() if precomputed_ast2 else normalize_and_rename_ast(src2).split()

    except Exception as e:
        raise
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    uni = len(set_a | set_b)
    return inter / uni

# -------------------- SANDBOX: TODO (must be implemented securely) --------------------
async def execute_in_sandbox(code: str, func_name: str, args: tuple, timeout_s: float = 1.0) -> Tuple[str, Any]:
    """
    TODO: Replace this with your secure sandbox implementation.
    The function must run candidate code in a secure, isolated environment (Docker container / microVM)
    with network disabled, filesystem restrictions, CPU/memory limits, and timeouts.

    Return values (status, payload):
      - ("ok", result) when function executes and returns result
      - ("syntax_error", error_str) if code fails to parse
      - ("runtime_error", error_str) if runtime exception
      - ("timeout", "__TLE__") on timeout
      - ("other", message) for other problems
    """
    raise NotImplementedError("execute_in_sandbox must be implemented with a secure runner")

# -------------------- BEHAVIORAL SIMILARITY (uses sandbox) --------------------
async def behavioral_score(candidate_src: str, optimal_src: str, tests: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    details = {"total": len(tests), "candidate_passed": 0, "cases": []}
    try:
        tree = ast.parse(optimal_src)
        f = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        func_name = f.name if f else "_candidate_func"
    except Exception:
        func_name = "_candidate_func"

    for t in tests:
        args = t["input"]
        expected = t["output"]
        status_c, payload_c = await execute_in_sandbox(candidate_src, func_name, args, timeout_s=1.0)
        case = {"args": args, "expected": expected, "candidate_status": status_c, "candidate_payload": payload_c}
        if status_c == "ok" and payload_c == expected:
            details["candidate_passed"] += 1
            case["result"] = payload_c
            case["status"] = "pass"
        elif status_c == "ok":
            case["result"] = payload_c
            case["status"] = "wrong"
        else:
            case["status"] = status_c
        details["cases"].append(case)

    score = details["candidate_passed"] / max(1, details["total"])
    return score, details

# -------------------- LLM INTEGRATION (OpenAI GPT-4.1 family) --------------------
async def openai_chat_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    if openai.api_key is None:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Use the assistant's content
    return resp.choices[0].message.content.strip()

async def llm_semantic_similarity(candidate_code: str, optimal_code: str, question: Dict[str, Any]) -> float:
    """Ask LLM to rate semantic similarity 0..1. Returns float.
    We ask the LLM to ignore variable names/formatting and focus on algorithmic intent.
    """
    prompt = textwrap.dedent(f"""
    You are an expert code reviewer. Rate how similar the CANDIDATE's code is to the OPTIMAL solution
    for the following problem on a scale from 0 (completely different) to 1 (equivalent algorithm/logic).

    Problem:
    {question.get('title')}

    Optimal solution:
    {optimal_code}

    Candidate code (may not run):
    {candidate_code}

    Consider algorithmic structure, data structures used, and core approach. Ignore variable names,
    whitespace, and comments. Return a single number between 0 and 1 (e.g., 0.0, 0.35, 0.9).
    Also return a one-line reason.
    Provide output as JSON: {{"score": number, "reason": "..."}}
    """.format(optimal_code=optimal_code, candidate_code=candidate_code))

    out = await openai_chat_completion(prompt, temperature=0.0, max_tokens=200)
    try:
        parsed = json.loads(out)
        return float(parsed.get('score', 0.0)), parsed.get('reason', '')
    except Exception:
        # fallback: try to extract a number naively
        import re
        m = re.search(r"([0-1](?:\.[0-9]+)?)", out)
        if m:
            return float(m.group(1)), out
        return 0.0, out

# -------------------- LLM HINT PROMPT --------------------
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

If there are failing testcases, include them here:
{failed_cases}

Now produce a hint at level {level} focused on the most pressing issue: {issue}.
Keep it brief and actionable.
Return only the hint text (no JSON wrapper).
"""

async def llm_hint(problem: Dict[str, Any], candidate_code: str, audio_last_min: str, issue: str, level: str, failed_cases: List[Dict[str,Any]] = None) -> str:
    failed_cases = failed_cases or []
    tests_preview = ''.join([str(t) for t in problem.get('tests', [])[:5]])
    prompt = HINT_PROMPT_BASE.format(
        title=problem.get('title'),
        optimal=problem.get('optimal'),
        tests=tests_preview,
        candidate=candidate_code,
        audio=audio_last_min,
        failed_cases=''.join([str(fc) for fc in failed_cases]),
        level=level,
        issue=issue,
    )
    return await openai_chat_completion(prompt, temperature=0.2, max_tokens=200)

async def llm_verify_fix(issue: str, prev_hint: str, candidate_code: str, tests_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the LLM to verify whether a specific issue has been fixed in the candidate's latest code.
    
    Parameters:
    - issue: the issue identifier (e.g., "syntax", "logic")
    - prev_hint: the hint previously given to the candidate
    - candidate_code: the latest snapshot of candidate's code
    - tests_summary: dictionary of test cases and results (can include 'input' and 'expected')
    
    Returns:
    - JSON dict: {"fixed": True/False, "explanation": "..."}
    """
    # Prepare a concise test summary string for the LLM
    tests_str = "\n".join([f"Input: {t['input']}, Expected: {t['output']}" for t in tests_summary.get('cases', [])])

    prompt = textwrap.dedent(f"""
    You are a precise code reviewer. The candidate had an issue: {issue}.
    Previously suggested hint: {prev_hint}

    Candidate's new code:
    {candidate_code}

    Relevant test cases and expected results:
    {tests_str}

    Determine whether the specific issue has been fixed. Answer in strict JSON: {{"fixed": true/false, "explanation": "..."}}
    Focus only on the single issue.
    """)

    out = await openai_chat_completion(prompt, temperature=0.0, max_tokens=200)

    try:
        return json.loads(out)
    except Exception:
        # fallback if LLM output is not strict JSON
        return {"fixed": False, "explanation": out}

# -------------------- COMBINED SIMILARITY EVALUATOR (AST + LLM) --------------------
async def combined_similarity(candidate_src: str, optimal_src: str, tests: List[Dict[str, Any]], alpha: float = 0.2, beta: float = 0.5, optimal_ast: str = None) -> Tuple[float, Dict[str, Any]]:
    """Compute combined similarity metric:
       - struct_score: AST structural similarity (0..1) if AST parses, otherwise 0
       - semantic_score: LLM judgement (0..1) always computed
       - behav_score: fraction of tests passed (0..1)
       combined = alpha*struct + beta*semantic + (1-alpha-beta)*behav
    """
    struct = 0.0
    try:
        struct = structural_similarity(candidate_src, optimal_src, precomputed_ast2=optimal_ast)
    except Exception:
        struct = 0.0

    semantic, sem_reason = await llm_semantic_similarity(candidate_src, optimal_src, {})
    behav, behav_details = await behavioral_score(candidate_src, optimal_src, tests)

    # combine; ensure weights sum to 1
    w_struct = alpha
    w_sem = beta
    w_behav = 1 - (w_struct + w_sem)
    combined = w_struct * struct + w_sem * semantic + w_behav * behav
    details = {"struct": struct, "semantic": semantic, "semantic_reason": sem_reason, "behav": behav, "behav_details": behav_details}
    return combined, details

# -------------------- ISSUE DETECTION --------------------

def detect_top_issue(behav_details: Dict[str, Any]) -> str:
    statuses = [c.get("status") for c in behav_details.get("cases", [])]
    if any(s == "syntax_error" for s in statuses):
        return "syntax"
    if any(s == "runtime_error" for s in statuses):
        return "runtime"
    if any(s == "timeout" for s in statuses):
        return "tle"
    if any(s == "wrong" for s in statuses):
        return "logic"
    return "style"

# -------------------- HINT MANAGER --------------------

class HintManager:
    def __init__(self):
        self.attempts: Dict[str, int] = {}
        self.history: List[Dict[str, Any]] = []

    async def maybe_issue_hint(self, session: 'Session') -> Dict[str, Any]:
        idle = session.time_since_activity()
        # similarity delta check
        if len(session.similarity_history) >= 2:
            last_improve = session.similarity_history[-1][1] - session.similarity_history[-2][1]
        else:
            last_improve = 0.0

        if idle < 30 and last_improve > 0.01:
            return {"issued": False}

        combined, details = session.similarity_history[-1][1], session.similarity_history[-1][2]
        top_issue = detect_top_issue(details.get('behav_details', {}))
        attempts = self.attempts.get(top_issue, 0)
        if attempts>= 1:
            if attempts >= 3:
                priority = ["syntax", "runtime", "logic", "tle", "style"]
                idx = priority.index(top_issue) if top_issue in priority else -1
                top_issue = priority[(idx + 1) % len(priority)]
                attempts = self.attempts.get(top_issue, 0)
            else:
                prev_hint = next((h for h in reversed(self.history) if h['issue'] == top_issue), "")
                fix_status = await llm_verify_fix(top_issue, prev_hint, session.latest_code, details.get('behav_details', {}))
                if fix_status.get('fixed'):
                    priority = ["syntax", "runtime", "logic", "tle", "style"]
                    idx = priority.index(top_issue) if top_issue in priority else -1
                    top_issue = priority[(idx + 1) % len(priority)]
                    attempts = self.attempts.get(top_issue, 0)

        level = "Nudge" if attempts == 0 else ("Guide" if attempts == 1 else "Direction")

        # prepare audio last 1 minute
        audio_last_min = session.get_audio_last_minute()
        # failing cases
        failed = [c for c in details['behav_details']['cases'] if c['status'] not in ('pass',)] if 'behav_details' in details else []
        hint_text = await llm_hint(session.question, session.latest_code, audio_last_min, top_issue, level, failed)
        hint = {"time": time.time(), "issue": top_issue, "level": level, "hint": hint_text}
        self.history.append(hint)
        self.attempts[top_issue] = attempts + 1
        session.record_hint(hint)
        return {"issued": True, **hint}

# -------------------- SESSION CLASS --------------------

class Session:
    def __init__(self, session_id: str, question: Dict[str, Any]):
        self.session_id = session_id
        self.question = question
        self.optimal = question['optimal']
        self.tests = question['tests']
        self.latest_code = ""
        self.last_keystroke = time.time()
        self.last_audio = time.time()
        self.audio_buffer: List[Tuple[float,str]] = []  # (ts, text)
        self.similarity_history: List[Tuple[float, float, Dict[str, Any]]] = []
        self.events: List[Dict[str, Any]] = []
        self.hint_manager = HintManager()
        self.hint_history: List[Dict[str,Any]] = []
    
        # precompute normalized AST of optimal solution
        try:
            self.optimal_ast = normalize_and_rename_ast(self.optimal)
        except Exception:
            self.optimal_ast = ""

        # initial similarity
        loop = asyncio.get_event_loop()
        combined, details = loop.run_until_complete(combined_similarity("", self.optimal, self.tests, optimal_ast= self.optimal_ast))
        self.similarity_history.append((time.time(), combined, details))

    def latest_code_snippet(self) -> str:
        return self.latest_code[:20000]

    def record_event(self, evt: Dict[str, Any]):
        evt['ts'] = time.time()
        self.events.append(evt)
        if evt['type'] == 'keystroke':
            self.last_keystroke = time.time()
        if evt['type'] == 'audio':
            self.last_audio = time.time()
            self.audio_buffer.append((time.time(), evt.get('text','')))
        if evt['type'] == 'snapshot':
            self.latest_code = evt.get('code', self.latest_code)

    def get_audio_last_minute(self) -> str:
        cutoff = time.time() - 60.0
        recent = [txt for ts, txt in self.audio_buffer if ts >= cutoff]
        return "".join(recent[-20:])  # up to last 20 snippets

    def time_since_activity(self) -> float:
        return time.time() - max(self.last_keystroke, self.last_audio)

    async def recompute_similarity(self):
        combined, details = await combined_similarity(self.latest_code, self.optimal, self.tests)
        self.similarity_history.append((time.time(), combined, details))
        return combined, details

    def record_hint(self, hint: Dict[str, Any]):
        self.hint_history.append(hint)
        self.events.append({'type': 'hint', 'time': time.time(), 'hint': hint})

    def to_result(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'question_id': self.question.get('id'),
            'question_title': self.question.get('title'),
            'events': self.events,
            'similarity_history': self.similarity_history,
            'hints': self.hint_history,
        }

# -------------------- WEBSOCKET SERVER --------------------

SESSIONS: Dict[str, Session] = {}

async def handle_message(msg: Dict[str, Any], session: Session):
    typ = msg.get('type')
    if typ == 'keystroke':
        session.record_event({'type': 'keystroke', 'data': msg.get('data')})
    elif typ == 'audio':
        session.record_event({'type': 'audio', 'text': msg.get('text')})
    elif typ == 'snapshot':
        session.record_event({'type': 'snapshot', 'code': msg.get('code')})
        await session.recompute_similarity()
    else:
        session.record_event({'type': 'unknown', 'raw': msg})

async def session_monitor_loop(session: Session):
    try:
        await asyncio.sleep(5*60)  # initial delay
        while True:
            await session.recompute_similarity()
            hint_res = await session.hint_manager.maybe_issue_hint(session)
            if hint_res.get('issued'):
                print(f"Hint issued for session {session.session_id}: {hint_res['hint']}")
            await asyncio.sleep(50)
    except asyncio.CancelledError:
        return

async def websocket_handler(websocket, path):
    try:
        start_msg = await websocket.recv()
        parsed = json.loads(start_msg)
    except Exception as e:
        await websocket.send(json.dumps({'error': 'invalid start message'}))
        return

    if parsed.get('action') != 'start' or 'session_id' not in parsed:
        await websocket.send(json.dumps({'error': 'must send start with session_id'}))
        return

    sid = parsed['session_id']
    q = pick_question()
    session = Session(sid, q)
    SESSIONS[sid] = session

    # send chosen question and optimal once
    await websocket.send(json.dumps({'action': 'question', 'question': q['title'], 'signature': q.get('signature'), 'optimal': q.get('optimal'), 'tests': q.get('tests')}))

    monitor_task = asyncio.create_task(session_monitor_loop(session))

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except Exception:
                await websocket.send(json.dumps({'error': 'invalid json'}))
                continue
            await handle_message(msg, session)
            await websocket.send(json.dumps({'ack': True, 'ts': time.time()}))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        monitor_task.cancel()
        result = session.to_result()
        fname = f"session_{sid}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Session {sid} ended. Saved {fname}")

# -------------------- RUN SERVER --------------------

def main():
    host = 'localhost'
    port = 8765
    print(f"Starting WebSocket server on ws://{host}:{port}")
    start_server = websockets.serve(websocket_handler, host, port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    main()

