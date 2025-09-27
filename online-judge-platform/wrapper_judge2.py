import subprocess
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask_cors import CORS
app = Flask(__name__)
CORS(app)


def run_code_in_judge_parallel(code, language, test_cases, docker_image_map=None, time_limit=5, max_workers=4):
    """
    Run code in Docker-based sandbox with multiple test cases in parallel.
    
    Args:
        code (str): Source code to execute.
        language (str): 'python' or 'cpp'.
        test_cases (list of dict): Each dict has 'input_data' and 'expected_output'.
        docker_image_map (dict, optional): Maps language to Docker image name.
        time_limit (int, optional): Max seconds per test case.
        max_workers (int, optional): Max parallel threads.
    
    Returns:
        list of dicts: Each dict contains testcase_number, input, output, expected, status, test_passed.
    """
    if docker_image_map is None:
        docker_image_map = {
            "python": "python-judge:latest",
            "cpp": "cpp-judge:latest"
        }

    def run_single_test(idx, test):
        input_data = test.get("input_data", "")
        expected_output = test.get("expected_output", "")

        with tempfile.TemporaryDirectory() as temp_dir:
            code_filename = "solution." + ("py" if language == "python" else "cpp")
            code_path = os.path.join(temp_dir, code_filename)
            input_path = os.path.join(temp_dir, "input.txt")

            # Write code and input
            with open(code_path, 'w') as f:
                f.write(code)
            with open(input_path, 'w') as f:
                f.write(input_data)

            # Docker command
            script_path = "/usr/local/bin/execute_code.sh"
            container_command = f"/bin/bash {script_path} {language} {code_filename} {time_limit}"

            docker_run_command = [
                "docker", "run",
                "--rm",
                "--network", "none",
                "--cap-drop", "ALL",
                "-u", "sandboxuser",
                "-m", "256m",
                f"--cpus=0.5",
                "-v", f"{temp_dir}:/home/sandboxuser/code",
                "-v", f"{os.path.abspath('execute_code.sh')}:/usr/local/bin/execute_code.sh:ro",
                docker_image_map[language],
                "/bin/bash", "-c", container_command
            ]

            try:
                result = subprocess.run(
                    docker_run_command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=time_limit + 5
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
                    "testcase_number": idx,
                    "input": input_data,
                    "output": user_output,
                    "expected": expected_output,
                    "status": status,
                    "test_passed": test_passed
                }

            except subprocess.TimeoutExpired:
                return {
                    "testcase_number": idx,
                    "input": input_data,
                    "output": "",
                    "expected": expected_output,
                    "status": "Host Timeout",
                    "test_passed": False
                }

    # Run tests in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(run_single_test, idx+1, test): idx for idx, test in enumerate(test_cases)}
        for future in as_completed(future_to_idx):
            results.append(future.result())

    # Sort results by testcase_number to maintain order
    results.sort(key=lambda x: x["testcase_number"])
    return results
