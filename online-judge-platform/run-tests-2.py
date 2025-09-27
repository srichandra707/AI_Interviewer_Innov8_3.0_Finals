from wrapper_judge2 import run_code_in_judge_parallel

python_code_correct = """
import sys
try:
    a, b = map(int, sys.stdin.read().split())
    print(a + b)
except ValueError:
    pass
"""

python_code_runtime_error = """
x = 1 / 0
"""

python_code_infinite_loop = """
while True:
    pass
"""

python_code_syntax_error = """
for i in range(5)
    print(i)
"""

python_tests = [
    {"input_data": "2 3", "expected_output": "5"},          # Correct answer
    {"input_data": "2 2", "expected_output": "5"},          # Wrong answer
    {"input_data": "", "expected_output": ""},              # Runtime error (division by zero in code)
    {"input_data": "0 0", "expected_output": ""},           # Time limit exceeded (infinite loop)
    {"input_data": "", "expected_output": ""}               # Compilation/syntax error
]

# Mapping each test to a code variant
python_code_mapping = [
    python_code_correct,
    python_code_correct,
    python_code_runtime_error,
    python_code_infinite_loop,
    python_code_syntax_error
]

cpp_code_correct = """
#include <iostream>
using namespace std;
int main() {
    int a, b;
    if (!(cin >> a >> b)) return 0;
    cout << a + b << endl;
    return 0;
}
"""

cpp_code_runtime_error = """
#include <iostream>
using namespace std;
int main() {
    int x = 0;
    cout << 10 / x << endl; // division by zero
    return 0;
}
"""

cpp_code_infinite_loop = """
#include <iostream>
using namespace std;
int main() {
    while(true) {}
    return 0;
}
"""

cpp_code_compile_error = """
#include <iostream>
using namespace std;
int main() {
    int x
    cout << x << endl;
    return 0;
}
"""

cpp_tests = [
    {"input_data": "2 3", "expected_output": "5"},         # Correct answer
    {"input_data": "2 2", "expected_output": "5"},         # Wrong answer
    {"input_data": "", "expected_output": ""},             # Runtime error (div by zero)
    {"input_data": "0 0", "expected_output": ""},          # Time limit exceeded
    {"input_data": "", "expected_output": ""}              # Compilation error
]

cpp_code_mapping = [
    cpp_code_correct,
    cpp_code_correct,
    cpp_code_runtime_error,
    cpp_code_infinite_loop,
    cpp_code_compile_error
]

# Example for Python
results_py = []
for code, test in zip(python_code_mapping, python_tests):
    result = run_code_in_judge_parallel(
        code=code,
        language="python",
        test_cases=[test]
    )
    results_py.extend(result)

print("=== Python Test Results ===")
for r in results_py:
    print(f"Test Case {r['testcase_number']}: {r['status']}, Output: '{r['output']}', Expected: '{r['expected']}'")

# Example for C++
results_cpp = []
for code, test in zip(cpp_code_mapping, cpp_tests):
    result = run_code_in_judge_parallel(
        code=code,
        language="cpp",
        test_cases=[test]
    )
    results_cpp.extend(result)

print("=== C++ Test Results ===")
for r in results_cpp:
    print(f"Test Case {r['testcase_number']}: {r['status']}, Output: '{r['output']}', Expected: '{r['expected']}'")
