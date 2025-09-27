#!/bin/bash
# test_python.sh - Runs multiple Python test cases against your judge API

URL="http://127.0.0.1:5000/api/run_code"

echo "=== Python Test Cases ==="

# 1. Correct Answer
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "python",
    "code": "a, b = map(int, input().split())\nprint(a + b)"
}' -w "\n[CORRECT ANSWER]\n"

# 2. Wrong Answer
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "python",
    "code": "a, b = map(int, input().split())\nprint(a - b)"
}' -w "\n[WRONG ANSWER]\n"

# 3. Runtime Error (division by zero)
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "python",
    "code": "a, b = map(int, input().split())\nprint(a / 0)"
}' -w "\n[RUNTIME ERROR]\n"

# 4. Time Limit Exceeded (infinite loop)
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "python",
    "code": "while True: pass"
}' -w "\n[TIME LIMIT EXCEEDED]\n"

# 5. Compilation/Syntax Error
curl -s -X POST "$URL" -H "Content-Type: application/json" -d '{
    "language": "python",
    "code": "print(Hello)"
}' -w "\n[COMPILATION ERROR]\n"
