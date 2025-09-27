import time
import json
from typing import List, Dict, Any
import difflib
import ast
import subprocess
import tempfile
import os
import numpy as np
from openai import OpenAI
import random
from typing import Optional

###################################
# Memory Systems
###################################

    # PROBLEM: WHAT IF THE CANDIDATE DOES NOT FIX THE ISSUE EVEN AFTER 3 HINTS? - do we move onto the next issue?
    # PROBLEM 2: WHAT IF THE CANDIDATE FIXES THE ISSUE BUT INTRODUCES A NEW ISSUE?
    # PROBLEM 3: WHAT IF THE CANDIDATE DOES NOT SOLVE THE ISSUE BUT MAKES SIGNIFICANT PROGRESS TOWARDS IT? - HOW DO WE QUALITATIVELY TAKE NOTE OF THIS?
    # PROBLEM 4: WE HAVE A VOICE INPUT: WHAT IF THE CANDIDATE ASKS A QUESTION? HOW DO WE INCORPORATE THIS INTO THE WORKING MEMORY?
    # PROBLEM 5: WHAT IF THE CANDIDATE ASKS FOR A HINT? DO WE GIVE THEM A HINT IMMEDIATELY OR DO WE WAIT FOR THE IDLE TIMEOUT?
    # PROBLEM 6: VOICE INPUT: HOW DO WE QUALITATIVELY ASSESS THE CANDIDATE'S UNDERSTANDING BASED ON THEIR QUESTIONS/COMMENTS?

class WorkingMemory:
    """Short-term memory for LLM context (limited snapshots + summary)."""

    def __init__(self):
        self.snapshots: List[str] = []   # last 3 snapshots of unresolved issue
        self.progress_summary: str = ""  # evolving compressed summary
        self.current_issue: str = ""     # "syntax" | "correctness" | "efficiency"
        self.hint_level: str = "Nudge"   # "Nudge" -> "Help" -> "Obvious"

    def add_snapshot(self, code: str):
        self.snapshots.append(code)
        if len(self.snapshots) > 6: # changed to 6
            self.snapshots.pop(0)

    def escalate_hint(self):
        if self.hint_level == "Nudge":
            self.hint_level = "Help"
        elif self.hint_level == "Help":
            self.hint_level = "Obvious"

    def reset_hint_level(self):
        self.hint_level = "Nudge"
        self.snapshots = []

class LongTermMemory:
    """Long-term structured logs for final reporting and evaluation."""
    def __init__(self, candidate_id: str, problem: str):
        self.candidate_id = candidate_id
        self.problem = problem
        self.events: List[Dict[str, Any]] = []

    def log_event(self, action: str, details: Dict[str, Any]):
        event = {
            "time": time.time(),
            "action": action,
            "details": details
        }
        self.events.append(event)

    def export(self) -> str:
        return json.dumps({
            "candidate_id": self.candidate_id,
            "problem": self.problem,
            "events": self.events
        }, indent=2)

###################################
# Code Evaluation Engine
###################################

class CodeEvaluator: # DO WE USE LLM HERE?
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def run_code_on_input(self, code: str, input_str: str, timeout: int = 2) -> str:
        """Run candidate code with given input and return stdout (sandboxed)."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode())
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                input=input_str.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            )
            if result.returncode != 0:
                return "ERROR"
            return result.stdout.decode().strip()
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        finally:
            os.unlink(tmp_path)

    def ast_similarity(self, code: str, solution: str) -> float:
        """Compare code structure using AST dump + difflib."""
        try:
            cand_ast = ast.dump(ast.parse(code))
            sol_ast = ast.dump(ast.parse(solution))
            return difflib.SequenceMatcher(None, cand_ast, sol_ast).ratio()
        except Exception:
            return 0.0

    def test_case_accuracy(self, code: str, solution: str, test_cases: List[str]) -> float:
        """Run code on test cases, compare with reference outputs."""
        passed = 0
        for t in test_cases:
            cand_out = self.run_code_on_input(code, t)
            sol_out = self.run_code_on_input(solution, t)
            if cand_out == sol_out:
                passed += 1
        return passed / len(test_cases) if test_cases else 0.0

    def embedding_similarity(self, code: str, solution: str) -> float:
        """Compute cosine similarity using OpenAI embeddings."""
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[code, solution]
        )
        vec_code = np.array(response.data[0].embedding)
        vec_sol = np.array(response.data[1].embedding)

        # Cosine similarity
        sim = np.dot(vec_code, vec_sol) / (np.linalg.norm(vec_code) * np.linalg.norm(vec_sol))
        return float(sim)

    def compute_similarity(self, code: str, optimal_solutions: List[str], test_cases: List[str]) -> float:
        """Composite similarity score."""
        best_score = 0.0
        for sol in optimal_solutions:
            ast_score = self.ast_similarity(code, sol)
            test_score = self.test_case_accuracy(code, sol, test_cases)
            embed_score = self.embedding_similarity(code, sol)
            score = 0.4 * ast_score + 0.4 * test_score + 0.2 * embed_score
            best_score = max(best_score, score)
        return best_score

###################################
# LLM Orchestrator
###################################

class LLMHintGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client

    # PROBLEM: WHAT IF THE LLM WANTS TO FOCUS ON 1 ISSUE, BUT THE CANDIDATE'S VOICE/COMMENTS ARE FOCUSED ON SOME OTHER ISSUE?
        # In this case, we should prioritize what the candidate is focused on - need to send the LLM what exact problem to help fix, using Voice Transcription
        # But then, how do we ensuer that the Working Memory stores the previous unfinished sequence of the hints it was going to focus on, while also helping the cnadidate?
            # Simplest way: just reset the previous unfinished sequence, and use the one the candidate is focused on
        # Ideal improvement (not essential): If the candidate has localized the problem, giving a 'Nudge' hint won't really help- should directly jump to 'Help' or 'Obvious' stage as needed

    # PROBLEM 2: WHAT IF THE CANDIDATE THINKS THERE IS A PROBLEM WITH A PART OF THE CODE WHICH IS ACTUALLY CORRECT?
    
    def generate_hint(self, wm: WorkingMemory, code: str, eval_result: Dict[str, Any]) -> str:
        prompt = f"""
        You are an AI interviewer. 
        Candidate's current issue: {wm.current_issue or "unknown"} 
        Progress summary so far: {wm.progress_summary or "none"} 
        Hint level: {wm.hint_level}
        Candidate code snapshot:
        {code}

        Evaluation result:
        {json.dumps(eval_result,indent=2)}

        Rules:
        1. Give exactly ONE {wm.hint_level}-level hint.
        2. Focus only on the most important unresolved issue.
        3. Do not give full solutions, just a hint for the next step.
        """
        # Call your LLM API here
        return self.llm(prompt)

###################################
# Main Interviewer Loop
###################################

class AgenticInterviewer:
    # We should incorporate candidate's voice input/questions into working memory - where and how?
    # do we do it in update_progress_summary or update_current_issue functions?
    # How do we qualitatively assess their understanding based on their questions/comments?
    def __init__(self, candidate_id: str, problem: str, llm_client):
        self.wm = WorkingMemory()
        self.ltm = LongTermMemory(candidate_id, problem)
        self.evaluator = CodeEvaluator()
        self.hinter = LLMHintGenerator(llm_client)
        self.optimal_solutions: List[str] = [] 
        self.test_cases: List[str]= []         

    def evaluate_code(self, code: str) -> Dict[str, Any]:
        """Run code evaluation and return structured results."""
        similarity = self.evaluator.compute_similarity(code, self.optimal_solutions, self.test_cases)
        syntax_ok = True
        try:
            ast.parse(code)
        except SyntaxError:
            syntax_ok = False

        return {
            "similarity": similarity,
            "syntax_ok": syntax_ok,
            # Simplified correctness: pass rate against test cases
            "correctness": self.evaluator.test_case_accuracy(code, self.optimal_solutions[0], self.test_cases) 
                           if self.optimal_solutions else 0.0,
            # Efficiency is placeholder (could be profiled later) # WHAT IS THIS
            "efficiency_flag": "TODO"
        }

    def update_current_issue(self, eval_result: Dict[str, Any]):
        """Set current issue in working memory based on eval results."""
        if not eval_result["syntax_ok"]:
            self.wm.current_issue = "syntax"
        elif eval_result["correctness"] < 1.0:
            self.wm.current_issue = "correctness"
        elif eval_result["efficiency_flag"] != "OK":
            self.wm.current_issue = "efficiency"
        else:
            self.wm.current_issue = "none"

    def update_progress_summary(self, eval_result: Dict[str, Any]):
        """Update compressed progress summary for LLM context."""
        if self.wm.current_issue == "syntax":
            self.wm.progress_summary = "Currently resolving syntax issues."
        elif self.wm.current_issue == "correctness":
            self.wm.progress_summary = "Syntax resolved, working on correctness."
        elif self.wm.current_issue == "efficiency":
            self.wm.progress_summary = "Code works correctly, optimizing efficiency."
        else:
            self.wm.progress_summary = "All issues resolved."

    def trigger_hint(self, code: str, eval_result: Dict[str, Any]) -> str:
        """Called after idle period or failed progress."""
        hint = self.hinter.generate_hint(self.wm, code, eval_result)
        self.ltm.log_event("Hint", {"level": self.wm.hint_level, "text": hint})

        # Escalate for next time
        self.wm.escalate_hint()
        return hint

    def mark_issue_resolved(self): # , issue_type: str
        """When candidate fixes an issue, reset to Nudge."""
        self.ltm.log_event("IssueResolved", {"issue": self.wm.current_issue})
        self.wm.reset_hint_level()

    def finalize_report(self) -> str:
        """Export full interview log for evaluation/reporting."""
        return self.ltm.export()

###################################
# Simulated Candidate Input Source
###################################
def get_candidate_code_snapshot() -> Optional[str]:
    """
    Mock function for demo: in real system this would pull from
    live keystroke buffer or code editor snapshot.
    Returns None if no new code has been typed.
    """
    if random.random() < 0.7:  # 70% chance candidate typed something
        # Placeholder: return a piece of code string
        return "print('hello world')"
    return None


###################################
# Main Runner
###################################
def run_interview(candidate_id: str, problem: str, llm_client):
    interviewer = AgenticInterviewer(candidate_id, problem, llm_client)

    prompt = f"""
    You are an expert programmer. Provide the canonical solutions for this problem:
    {problem}
    Return only the Python code as a string.
    """
    optimal_solution = llm_client.chat(prompt)
    interviewer.optimal_solutions = [optimal_solution]

    interviewer.test_cases = ["", ""]  # use llm to generate test cases? or just except llm to know correct soln and interview without needing test cases?

    INTERVIEW_DURATION = 45 * 60  # 45 minutes
    HINT_IDLE_THRESHOLD = 30      # 30 seconds of no activity

    start_time = time.time()
    last_activity_time = start_time
    last_similarity = 0.0

    print(f"Interview started for candidate {candidate_id} on problem: {problem}")

    while time.time() - start_time < INTERVIEW_DURATION:
        code = get_candidate_code_snapshot() # every 5 secondS? some better way of doing this?
        if code:
            # Candidate typed new code
            last_activity_time = time.time()

            eval_result = interviewer.evaluate_code(code)
            interviewer.update_current_issue(eval_result)
            interviewer.update_progress_summary(eval_result)
            interviewer.ltm.log_event("CodeSubmission", eval_result)

            # PROBLEM: THIS DOES NOT ACTUALLY CHECK IF THAT PARTICULAR ISSUE IS RESOLVED
            # PROBLEM 2: THIS ONLY CHECKS IF SIMILARITY IMPROVED- IMPROVEMENT COULD BE TRIVIAL DUE TO RANDOM CODE ADDITIONS AND DOES NOT INDICATE SIGNIFICANCE
            # Detect progress
            if eval_result["similarity"] - last_similarity > 0.05:
                last_similarity = eval_result["similarity"]
                interviewer.mark_issue_resolved()  # issue_type=interviewer.wm.current_issue
        # Check idle timeout or no progress
        idle_time = time.time() - last_activity_time
        if idle_time > HINT_IDLE_THRESHOLD:
            hint = interviewer.trigger_hint(code or "", eval_result if code else {})
            print(f"[Hint Issued] {hint}")
            last_activity_time = time.time()  # reset after hint

        time.sleep(5)  # polling interval

    # End interview
    print("Interview finished. Generating report...")
    report = interviewer.finalize_report()
    print(report)
    return report


###################################
# Run Entry Point
###################################
if __name__ == "__main__":
    class DummyLLMClient:
        def chat(self, *args, **kwargs):
            return "This is a mock LLM response."

    llm_client = DummyLLMClient()
    run_interview("candidate_123", "Implement BFS traversal", llm_client)
