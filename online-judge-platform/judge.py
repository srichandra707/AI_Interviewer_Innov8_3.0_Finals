import os 
import subprocess
import tempfile
import uuid
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Configuration Constants (Modify as needed) ---
DOCKER_IMAGE_MAP = {
    "python": "python-judge:latest",
    "cpp": "cpp-judge:latest",
}
DEFAULT_TIME_LIMIT = 5 # seconds
DEFAULT_MEMORY_LIMIT = "256m" # 256 megabytes
DEFAULT_CPU_LIMIT = "0.5" # 50% of one core

# --- Default Test Data ---
TEST_CASES = {
    "problem_1": [
        {"input": "5 3", "expected": "8"},
        {"input": "10 0", "expected": "10"}
    ]
}

@app.route('/api/run_code', methods=['POST'])
def run_code():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language')
    input_data = data.get('input_data')           # Optional custom input
    expected_output = data.get('expected_output') # Optional custom expected output
    
    if not code or language not in DOCKER_IMAGE_MAP:
        return jsonify({"status": "error", "message": "Invalid code or language."}), 400

    # Use default first test case if custom input/output is not provided
    if input_data is None or expected_output is None:
        test_data = TEST_CASES.get("problem_1", [])
        if not test_data:
            return jsonify({"status": "error", "message": "No test cases found."}), 500
        if input_data is None:
            input_data = test_data[0]['input']
        if expected_output is None:
            expected_output = test_data[0]['expected']
    
    # Use a temporary directory for secure file handling
    with tempfile.TemporaryDirectory() as temp_dir:
        code_filename = "solution." + ("py" if language == "python" else "cpp")
        code_path = os.path.join(temp_dir, code_filename)
        input_path = os.path.join(temp_dir, "input.txt")

        # Write the user's code and input to temp files
        with open(code_path, 'w') as f:
            f.write(code)
        with open(input_path, 'w') as f:
            f.write(input_data)

        # Build the Docker command
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

            # Determine status
            if "STATUS:TIME_LIMIT_EXCEEDED" in user_output:
                status = "Time Limit Exceeded (TLE)"
                test_passed = False
            elif "STATUS:RUNTIME_ERROR" in user_output:
                status = "Runtime Error (RTE)"
                test_passed = False
            elif result.returncode == 110: # Compilation error
                status = "Compilation Error (CE)"
                test_passed = False
            else:
                test_passed = (user_output == expected_output)
                status = "Accepted" if test_passed else "Wrong Answer"

            return jsonify({
                "status": status,
                "test_passed": test_passed,
                "output": user_output,
                "expected": expected_output
            })

        except subprocess.CalledProcessError as e:
            return jsonify({
                "status": "System Error",
                "message": f"Execution failed: {e.stderr.strip()}",
                "output": e.stdout.strip()
            }), 500
        except subprocess.TimeoutExpired:
            return jsonify({
                "status": "System Error",
                "message": "Host system timed out waiting for the container."
            }), 500


if __name__ == '__main__':
    os.chmod('execute_code.sh', 0o755)
    app.run(debug=True, port=5000)
