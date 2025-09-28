import os
import subprocess
import tempfile
import time
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Configuration Constants ---
DOCKER_IMAGE_MAP = {
    "python": "python-judge:latest",
    "cpp": "cpp-judge:latest",
}
DEFAULT_TIME_LIMIT = 5
DEFAULT_MEMORY_LIMIT = "256m"
DEFAULT_CPU_LIMIT = "0.5"

# --- Default Test Data ---
TEST_CASES = {
    "problem_1": [
        {"input": "5 3", "expected": "8"},
        {"input": "10 0", "expected": "10"}
    ]
}

# --- NEW: Global variables for agent communication ---
current_question = None
current_hints = None
code_snapshots = []
HINT_LOCK = threading.Lock()
LAST_HINT_TIME = 0

# ---------------- NEW: Agent Communication Endpoints ---------------- #

@app.route('/api/question', methods=['GET', 'POST'])
def handle_question():
    global current_question
    
    if request.method == 'POST':
        # Agent sends question to frontend via this endpoint
        data = request.json
        current_question = data.get('question')
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] ❓ Question received from agent: {current_question}")
        return jsonify({"status": "success"}), 200
    
    elif request.method == 'GET':
        # Frontend polls for new questions
        if current_question:
            question = current_question
            current_question = None  # Clear after sending
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] ❓ Question sent to frontend: {question}")
            return jsonify({"question": question}), 200
        return jsonify({"question": None}), 200

@app.route('/api/hints', methods=['GET', 'POST'])
def handle_hints():
    global current_hints, LAST_HINT_TIME
    
    if request.method == 'POST':
        # Agent sends hints to frontend via this endpoint
        data = request.json
        with HINT_LOCK:
            current_hints = data.get('hints')
            LAST_HINT_TIME = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] 💡 Hint received from agent: {current_hints}")
        return jsonify({"status": "success"}), 200
    
    elif request.method == 'GET':
        # Frontend polls for new hints
        if current_hints:
            hints = current_hints
            current_hints = None  # Clear after sending
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] 💡 Hint sent to frontend: {hints}")
            return jsonify({"hints": hints}), 200
        return jsonify({"hints": None}), 200

@app.route('/api/snapshots', methods=['GET', 'POST'])
def handle_snapshots():
    global code_snapshots
    
    if request.method == 'POST':
        # Frontend sends code snapshot every 10 seconds
        data = request.json
        snapshot = {
            'code': data.get('code'),
            'language': data.get('language'),
            'timestamp': data.get('timestamp', time.time())
        }
        code_snapshots.append(snapshot)
        
        # Keep only last 10 snapshots
        if len(code_snapshots) > 10:
            code_snapshots = code_snapshots[-10:]
        
        # Log snapshot receipt
        current_time = datetime.now().strftime("%H:%M:%S")
        code_length = len(snapshot['code']) if snapshot['code'] else 0
        print(f"[{current_time}] 📸 Snapshot received: {snapshot['language']}, {code_length} chars")
        return jsonify({"status": "success"}), 200
    
    elif request.method == 'GET':
        # Agent fetches latest snapshot
        if code_snapshots:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] 📸 Snapshot requested by agent")
            return jsonify(code_snapshots[-1]), 200
        return jsonify({}), 200

# --- NEW: Simple AI chat endpoint for frontend ---
@app.route('/api/ai', methods=['POST'])
def ai_chat():
    data = request.json
    message = data.get('message')
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] 🤖 AI chat received: {message}")
    
    # Simple response - you can enhance this
    return jsonify({"reply": f"AI received: {message}"}), 200

# ---------------- Utility: Run code inside Docker ---------------- #
def run_code_in_docker(code, language, input_data, expected_output):
    import tempfile, os, subprocess

    with tempfile.TemporaryDirectory() as temp_dir:
        code_filename = "solution." + ("py" if language == "python" else "cpp")
        code_path = os.path.join(temp_dir, code_filename)
        input_path = os.path.join(temp_dir, "input.txt")

        with open(code_path, 'w') as f:
            f.write(code)
        with open(input_path, 'w') as f:
            f.write(input_data)

        docker_image = DOCKER_IMAGE_MAP[language]
        script_path = "/usr/local/bin/execute_code.sh"
        container_command = f"/bin/bash {script_path} {language} {code_filename} {DEFAULT_TIME_LIMIT}"

        docker_run_command = [
            "docker", "run",
            "--rm",
            "--network", "none",
            "--cap-drop", "ALL",
            "-u", "sandboxuser",
            "-m", DEFAULT_MEMORY_LIMIT,
            f"--cpus={DEFAULT_CPU_LIMIT}",
            "-v", f"{temp_dir}:/home/sandboxuser/code",
            "-v", f"{os.path.abspath('execute_code.sh')}:/usr/local/bin/execute_code.sh:ro",
            docker_image,
            "/bin/bash", "-c", container_command
        ]

        try:
            result = subprocess.run(
                docker_run_command,
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TIME_LIMIT + 5
            )
            user_output = result.stdout.strip()

            if "STATUS:TIME_LIMIT_EXCEEDED" in user_output:
                status = "Time Limit Exceeded (TLE)"
                test_passed = False
            elif "STATUS:RUNTIME_ERROR" in user_output:
                status = "Runtime Error (RTE)"
                test_passed = False
            elif result.returncode == 110:
                status = "Compilation Error (CE)"
                test_passed = False
            else:
                test_passed = (user_output == expected_output)
                status = "Accepted" if test_passed else "Wrong Answer"

            return {
                "status": status,
                "test_passed": test_passed,
                "output": user_output,
                "expected": expected_output
            }

        except subprocess.TimeoutExpired:
            return {"status": "System Error: Host timeout", "test_passed": False, "output": "", "expected": expected_output}
        except subprocess.CalledProcessError as e:
            return {"status": f"System Error: {e.stderr.strip()}", "test_passed": False, "output": e.stdout.strip(), "expected": expected_output}

# ---------------- Endpoint: Run single or multiple test cases ---------------- #
@app.route('/api/run_code', methods=['POST'])
def run_code():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language')
    testcases = data.get('testcases')  # Optional list of {"input_data":..., "expected_output":...}
    print(code)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] 🏃 Code execution requested: {language}")

    if not code or language not in DOCKER_IMAGE_MAP:
        return jsonify({"status": "error", "message": "Invalid code or language."}), 400

    results = []

    # If no testcases provided, use default first test case
    if not testcases:
        test_data = TEST_CASES.get("problem_1", [])
        if not test_data:
            return jsonify({"status": "error", "message": "No test cases found."}), 500
        input_data = test_data[0]['input']
        expected_output = test_data[0]['expected']
        results.append(run_code_in_docker(code, language, input_data, expected_output))
    else:
        for idx, tc in enumerate(testcases, start=1):
            print(tc)
            input_data = tc.get("input_data", "")
            expected_output = tc.get("expected_output", "")
            res = run_code_in_docker(code, language, input_data, expected_output)
            res["testcase_number"] = idx
            res["input_data"] = input_data
            print("result issss", res)
            results.append(res)
    
    print(f"[{current_time}] 🏃 Code execution completed: {len(results)} test cases")
    return jsonify({"status": "success", "results": results})

# --- NEW: Health check endpoint ---
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "Backend server is running!",
        "timestamp": time.time(),
        "endpoints": [
            "POST /api/question - Agent sends questions",
            "GET /api/question - Frontend polls for questions", 
            "POST /api/hints - Agent sends hints",
            "GET /api/hints - Frontend polls for hints",
            "POST /api/snapshots - Frontend sends code snapshots",
            "GET /api/snapshots - Agent fetches code snapshots",
            "POST /api/run_code - Frontend runs code",
            "POST /api/ai - Frontend AI chat"
        ]
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Enhanced Flask Backend...")
    print("📍 Server will run on http://127.0.0.1:5000")
    print("📊 Endpoints added:")
    print("  • POST/GET /api/question - Question communication")
    print("  • POST/GET /api/hints - Hint communication") 
    print("  • POST/GET /api/snapshots - Code snapshot communication")
    print("  • POST /api/ai - AI chat")
    print("  • POST /api/run_code - Code execution (existing)")
    print("  • GET /api/health - Health check")
    print("📸 Will log all communication activities")
    print("=" * 60)
    
    os.chmod('execute_code.sh', 0o755)
    app.run(debug=True, host='127.0.0.1', port=5000)