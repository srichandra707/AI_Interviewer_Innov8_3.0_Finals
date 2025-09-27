# test_dsa_cases.py
from wrapper_judge import run_code_in_judge

# --- Python DSA Test Cases ---
python_tests = [
    {
        "description": "Correct Answer",
        "code": "a, b = map(int, input().split()); print(a+b)",
        "input_data": "5 3",
        "expected_output": "8"
    },
    {
        "description": "Wrong Answer",
        "code": "a, b = map(int, input().split()); print(a-b)",
        "input_data": "5 3",
        "expected_output": "8"
    },
    {
        "description": "Runtime Error",
        "code": "print(1/0)",
        "input_data": "",
        "expected_output": ""
    },
    {
        "description": "Time Limit Exceeded",
        "code": "while True: pass",
        "input_data": "",
        "expected_output": ""
    },
    {
        "description": "Syntax Error",
        "code": "print(\"Hello\"",
        "input_data": "",
        "expected_output": ""
    }
]

# --- C++ DSA Test Cases ---
cpp_tests = [
    {
        "description": "Correct Answer",
        "code": "#include <iostream>\nusing namespace std;\nint main(){int a,b; cin>>a>>b; cout<<a+b<<endl; return 0;}",
        "input_data": "5 3",
        "expected_output": "8"
    },
    {
        "description": "Wrong Answer",
        "code": "#include <iostream>\nusing namespace std;\nint main(){int a,b; cin>>a>>b; cout<<a-b<<endl; return 0;}",
        "input_data": "5 3",
        "expected_output": "8"
    },
    {
        "description": "Runtime Error",
        "code": "#include <iostream>\nusing namespace std;\nint main(){ int x=0; cout<<10/x<<endl; return 0; }",
        "input_data": "",
        "expected_output": ""
    },
    {
        "description": "Time Limit Exceeded",
        "code": "#include <iostream>\nusing namespace std;\nint main(){ while(true){} return 0; }",
        "input_data": "",
        "expected_output": ""
    },
    {
        "description": "Compilation Error",
        "code": "#include <iostream>\nusing namespace std;\nint main( { cout<<1; return 0; }",
        "input_data": "",
        "expected_output": ""
    }
]

python_multiline_tests = [
    {
        "description": "Correct Answer - Matrix Sum",
        "code": """
rows, cols = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]
for row in matrix:
    print(sum(row))
""",
        "input_data": "2 3\n1 2 3\n4 5 6",
        "expected_output": "6\n15"
    },
    {
        "description": "Correct Answer - BFS traversal of graph",
        "code": """
from collections import deque
n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
visited = [False]*n
queue = deque([0])
visited[0]=True
while queue:
    node = queue.popleft()
    print(node)
    for nei in graph[node]:
        if not visited[nei]:
            visited[nei]=True
            queue.append(nei)
""",
        "input_data": "5 4\n0 1\n0 2\n1 3\n3 4",
        "expected_output": "0\n1\n2\n3\n4"
    },
    {
        "description": "Wrong Answer - Matrix Sum +1",
        "code": """
rows, cols = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(rows)]
for row in matrix:
    print(sum(row)+1)
""",
        "input_data": "2 3\n1 2 3\n4 5 6",
        "expected_output": "6\n15"
    },
    {
        "description": "Runtime Error - BFS empty graph",
        "code": """
from collections import deque
n, m = map(int, input().split())
graph = [[] for _ in range(n)]
queue = deque([0])
visited = [False]*n
while queue:
    node = queue.popleft()
    print(node)
    for nei in graph[node+1]:  # This will cause IndexError
        queue.append(nei)
""",
        "input_data": "2 0",
        "expected_output": ""
    },
    {
        "description": "Time Limit Exceeded - Infinite loop",
        "code": "while True: pass",
        "input_data": "",
        "expected_output": ""
    }
]

cpp_multiline_tests = [
    {
        "description": "Correct Answer - Matrix Sum",
        "code": """
#include <iostream>
#include <vector>
using namespace std;
int main() {
    int rows, cols; cin >> rows >> cols;
    vector<vector<int>> mat(rows, vector<int>(cols));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            cin >> mat[i][j];
    for(int i=0;i<rows;i++){
        int s=0;
        for(int j=0;j<cols;j++) s+=mat[i][j];
        cout << s << endl;
    }
    return 0;
}
""",
        "input_data": "2 3\n1 2 3\n4 5 6",
        "expected_output": "6\n15"
    },
    {
        "description": "Correct Answer - BFS Traversal",
        "code": """
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m; cin >> n >> m;
    vector<vector<int>> g(n);
    for(int i=0;i<m;i++){
        int u,v; cin>>u>>v;
        g[u].push_back(v);
    }
    vector<bool> vis(n,false);
    queue<int> q;
    q.push(0); vis[0]=true;
    while(!q.empty()){
        int node = q.front(); q.pop();
        cout<<node<<endl;
        for(int nei: g[node]){
            if(!vis[nei]){
                vis[nei]=true;
                q.push(nei);
            }
        }
    }
    return 0;
}
""",
        "input_data": "5 4\n0 1\n0 2\n1 3\n3 4",
        "expected_output": "0\n1\n2\n3\n4"
    },
    {
        "description": "Wrong Answer - Matrix Sum +1",
        "code": """
#include <iostream>
#include <vector>
using namespace std;
int main() {
    int rows, cols; cin >> rows >> cols;
    vector<vector<int>> mat(rows, vector<int>(cols));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            cin >> mat[i][j];
    for(int i=0;i<rows;i++){
        int s=0;
        for(int j=0;j<cols;j++) s+=mat[i][j];
        cout << s+1 << endl;
    }
    return 0;
}
""",
        "input_data": "2 3\n1 2 3\n4 5 6",
        "expected_output": "6\n15"
    },
    {
        "description": "Runtime Error - Out of bounds",
        "code": "#include <iostream>\n\nint main() {\n    // Attempt to access memory at address 0 (NULL pointer)\n    int *ptr = nullptr;\n    *ptr = 10; // Segmentation Fault here\n    return 0;\n}",
        "input_data": "2 0",
        "expected_output": ""
    },
    {
        "description": "Time Limit Exceeded - Infinite loop",
        "code": "#include <iostream>\nint main(){while(true){} return 0;}",
        "input_data": "",
        "expected_output": ""
    }
]


# --- Run Tests ---
def run_tests(tests, language):
    print(f"\n=== {language.upper()} DSA Test Cases ===\n")
    for test in tests:
        print(f"--- {test['description']} ---")
        result = run_code_in_judge(
            code=test["code"],
            language=language,
            input_data=test.get("input_data", ""),
            expected_output=test.get("expected_output", "")
        )
        print(result, "\n")

def run_multiline_dsa_tests(tests, language):
    print(f"\n=== {language.upper()} Multi-line DSA Tests ===\n")
    for test in tests:
        print(f"--- {test['description']} ---")
        result = run_code_in_judge(
            code=test["code"],
            language=language,
            input_data=test.get("input_data", ""),
            expected_output=test.get("expected_output", "")
        )
        print(result, "\n")


if __name__ == "__main__":
    #run_tests(python_tests, "python")
    #run_tests(cpp_tests, "cpp")
    run_multiline_dsa_tests(python_multiline_tests, "python")
    run_multiline_dsa_tests(cpp_multiline_tests, "cpp")
