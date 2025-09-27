import os
import subprocess
import tempfile
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
            input_data = tc.get("input_data", "")
            expected_output = tc.get("expected_output", "")
            res = run_code_in_docker(code, language, input_data, expected_output)
            res["testcase_number"] = idx
            res["input_data"] = input_data
            results.append(res)
    print(results)
    return jsonify({"status": "success", "results": results})

if __name__ == '__main__':
    os.chmod('execute_code.sh', 0o755)
    app.run(debug=True, port=5000)
