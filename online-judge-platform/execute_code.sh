#!/bin/bash
# execute_code.sh - Orchestrates the sandboxed execution

# Arguments:
# $1: Language (python or cpp)
# $2: Code file name (e.g., solution.py)
# $3: Time limit (e.g., 5)

LANGUAGE=$1
CODE_FILE=$2
TIME_LIMIT=$3s

# Exit code 124 is the standard signal from the 'timeout' command indicating TLE.
TLE_EXIT_CODE=124
COMPILATION_ERROR_CODE=110

if [ "$LANGUAGE" == "cpp" ]; then
    # --- C++ Compilation Step ---
    g++ "$CODE_FILE" -o /tmp/solution_exe 2> /tmp/compile_error.txt
    if [ $? -ne 0 ]; then
        # Return a custom error code for compilation failure
        cat /tmp/compile_error.txt
        exit $COMPILATION_ERROR_CODE
    fi
    # --- C++ Execution Step ---
    # Use 'timeout' to enforce the Time Limit
    /usr/bin/timeout "$TIME_LIMIT" /tmp/solution_exe < input.txt
    EXECUTION_EXIT_CODE=$?

elif [ "$LANGUAGE" == "python" ]; then
    ERROR_LOG="/tmp/runtime_error.log"

    /usr/bin/timeout "$TIME_LIMIT" python3 "$CODE_FILE" < input.txt 2> "$ERROR_LOG"
    EXECUTION_EXIT_CODE=$?

    if [ "$EXECUTION_EXIT_CODE" -eq $TLE_EXIT_CODE ]; then
        echo "STATUS:TIME_LIMIT_EXCEEDED"
        exit 0
    fi

    # If non-timeout error, print stderr
    if [ -s "$ERROR_LOG" ]; then
        cat "$ERROR_LOG"
    fi

else
    echo "Unsupported language: $LANGUAGE"
    exit 1
fi

# Check the exit code of the execution
if [ "$EXECUTION_EXIT_CODE" -eq "$TLE_EXIT_CODE" ]; then
    # Return a special TLE message (handled by the backend)
    echo "STATUS:TIME_LIMIT_EXCEEDED"
    exit 0 # Exit with 0 so Docker does not report an error, 
           # but the output indicates TLE
elif [ "$EXECUTION_EXIT_CODE" -ne 0 ]; then
    # Return standard output/errors for Runtime Errors
    echo "STATUS:RUNTIME_ERROR"
    exit 0
else
    # Successful execution, output is already in stdout
    exit 0
fi