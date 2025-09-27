import subprocess
import json

def run_code_in_judge(code, language, input_data=None, expected_output=None):
    """
    Wrapper to send code to judge API and return the response.

    Args:
        code (str): Code to execute
        language (str): 'python' or 'cpp'
        input_data (str, optional): Custom stdin input. Defaults to None.
        expected_output (str, optional): Expected output to compare. Defaults to None.

    Returns:
        dict: JSON response from the API
    """
    url = "http://127.0.0.1:5000/api/run_code"

    payload = {
        "language": language,
        "code": code
    }

    # Include custom input/output if provided
    if input_data is not None:
        payload["input_data"] = input_data
    if expected_output is not None:
        payload["expected_output"] = expected_output

    # Convert payload to JSON string
    payload_str = json.dumps(payload)

    # Use curl to send POST request
    cmd = [
        "curl",
        "-s",  # silent
        "-X", "POST",
        url,
        "-H", "Content-Type: application/json",
        "-d", payload_str
    ]

    # Run curl command
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        response = {"status": "error", "message": "Failed to parse response", "raw": result.stdout}

    return response

resp = run_code_in_judge(
    code="""
import sys
for line in sys.stdin:
    nums = list(map(int, line.split()))
    print(sum(nums))
""",
    language="python",
    input_data="1 2\n3 4\n5 6",
    expected_output="3\n7\n11"
)
print(resp)

resp = run_code_in_judge(
    code="""
#include <iostream>
using namespace std;
int main() {
    int a, b;
    while(cin >> a >> b) {
        cout << a * b << endl;
    }
    return 0;
}
""",
    language="cpp",
    input_data="2 3\n4 5\n6 7",
    expected_output="6\n20\n42"
)
print(resp)