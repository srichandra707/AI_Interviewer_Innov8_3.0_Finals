#!/bin/bash
# test_cpp.sh - Runs multiple C++ test cases against your judge API

URL="http://127.0.0.1:5000/api/run_code"

echo "=== C++ Test Cases ==="

# 1. Correct Answer
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "cpp",
    "code": "#include <iostream>\nusing namespace std;\nint main() { int a,b; cin >> a >> b; cout << a+b << endl; return 0; }"
}' -w "\n[CORRECT ANSWER]\n"

# 2. Wrong Answer
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "cpp",
    "code": "#include <iostream>\nusing namespace std;\nint main() { int a,b; cin >> a >> b; cout << a-b << endl; return 0; }"
}' -w "\n[WRONG ANSWER]\n"

# 3. Runtime Error (division by zero)
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "cpp",
    "code": "#include <iostream>\nusing namespace std;\nint main() { int a,b; cin >> a >> b; cout << a/0 << endl; return 0; }"
}' -w "\n[RUNTIME ERROR]\n"

# 4. Time Limit Exceeded (infinite loop)
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "cpp",
    "code": "#include <iostream>\nusing namespace std;\nint main() { while(true){} return 0; }"
}' -w "\n[TIME LIMIT EXCEEDED]\n"

# 5. Compilation Error (invalid C++)
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "cpp",
    "code": "#include <iostream>\nint main() { syntax error }"
}' -w "\n[COMPILATION ERROR]\n"
