import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !key) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  process.exit(1);
}

const supabase = createClient(url, key);

// 문제 제목 → 테스트케이스 매핑
const testCasesByTitle = {
  "Two Sum": {
    function_name: "two_sum",
    call_type: "function",
    test_cases: [
      { input: "[2, 7, 11, 15], 9", expected: "True" },
      { input: "[3, 2, 4], 6", expected: "True" },
      { input: "[1, 2, 3], 7", expected: "False" },
      { input: "[1, 1], 2", expected: "True" },
      { input: "[], 5", expected: "False" },
    ],
  },
  "Longest Consecutive Sequence": {
    function_name: "longestConsecutive",
    call_type: "function",
    test_cases: [
      { input: "[100, 4, 200, 1, 3, 2]", expected: "4" },
      { input: "[0, 3, 7, 2, 5, 8, 4, 6, 0, 1]", expected: "9" },
      { input: "[]", expected: "0" },
      { input: "[1]", expected: "1" },
    ],
  },
  "Valid Parentheses": {
    function_name: "isValid",
    call_type: "function",
    test_cases: [
      { input: '"()"', expected: "True" },
      { input: '"()[]{}"', expected: "True" },
      { input: '"(]"', expected: "False" },
      { input: '"([)]"', expected: "False" },
      { input: '"{[]}"', expected: "True" },
      { input: '""', expected: "True" },
    ],
  },
  "Daily Temperatures": {
    function_name: "dailyTemperatures",
    call_type: "function",
    test_cases: [
      { input: "[73, 74, 75, 71, 69, 72, 76, 73]", expected: "[1, 1, 4, 2, 1, 1, 0, 0]" },
      { input: "[30, 40, 50, 60]", expected: "[1, 1, 1, 0]" },
      { input: "[30, 60, 90]", expected: "[1, 1, 0]" },
    ],
  },
  "Fibonacci (DP)": {
    function_name: "fibo",
    call_type: "function",
    test_cases: [
      { input: "1", expected: "1" },
      { input: "2", expected: "1" },
      { input: "5", expected: "5" },
      { input: "10", expected: "55" },
      { input: "20", expected: "6765" },
    ],
  },
  "Climbing Stairs (DP)": {
    function_name: "cs",
    call_type: "function",
    test_cases: [
      { input: "1", expected: "1" },
      { input: "2", expected: "2" },
      { input: "3", expected: "3" },
      { input: "5", expected: "8" },
      { input: "10", expected: "89" },
    ],
  },
  "Unique Paths": {
    function_name: "uniquePaths",
    call_type: "function",
    test_cases: [
      { input: "3, 7", expected: "28" },
      { input: "3, 2", expected: "3" },
      { input: "1, 1", expected: "1" },
      { input: "7, 3", expected: "28" },
    ],
  },
  "Permutations (순열)": {
    function_name: "permute",
    call_type: "function",
    test_cases: [
      { input: "[1, 2, 3]", expected: "[[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]" },
      { input: "[1]", expected: "[[1]]" },
      { input: "[0, 1]", expected: "[[0, 1], [1, 0]]" },
    ],
  },
  "Number of Islands": {
    function_name: "numIslands",
    call_type: "function",
    test_cases: [
      {
        input: '[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]',
        expected: "1",
      },
      {
        input: '[["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]',
        expected: "3",
      },
    ],
  },
  "Dijkstra": {
    function_name: "dijkstra",
    call_type: "script",
    test_cases: [
      {
        input: `import heapq
import inspect as __insp
graph = {
    'A': [(1, 'B'), (4, 'C')],
    'B': [(2, 'C'), (5, 'D')],
    'C': [(1, 'D')],
    'D': []
}
__sig = __insp.signature(dijkstra)
__n = len(__sig.parameters)
if __n >= 4:
    __res = dijkstra(graph, 'A', 'D', 4)
elif __n == 3:
    __res = dijkstra(graph, 'A', 'D')
elif __n == 2:
    __res = dijkstra(graph, 'A')
else:
    __res = dijkstra(graph)
if isinstance(__res, dict):
    __res = __res.get('D', __res)
if isinstance(__res, tuple):
    __res = __res[0] if isinstance(__res[0], (int, float)) else __res
print(__res)`,
        expected: "4",
      },
      {
        input: `import heapq
import inspect as __insp
graph2 = {
    'A': [(2, 'B'), (6, 'C')],
    'B': [(3, 'C'), (1, 'D')],
    'C': [(1, 'D')],
    'D': []
}
__sig = __insp.signature(dijkstra)
__n = len(__sig.parameters)
if __n >= 4:
    __res = dijkstra(graph2, 'A', 'D', 4)
elif __n == 3:
    __res = dijkstra(graph2, 'A', 'D')
elif __n == 2:
    __res = dijkstra(graph2, 'A')
else:
    __res = dijkstra(graph2)
if isinstance(__res, dict):
    __res = __res.get('D', __res)
if isinstance(__res, tuple):
    __res = __res[0] if isinstance(__res[0], (int, float)) else __res
print(__res)`,
        expected: "3",
      },
    ],
  },
  "Combinations (조합)": {
    function_name: "solution",
    call_type: "function",
    test_cases: [
      { input: "[1, 2, 3, 4], 2", expected: "[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]" },
      { input: "[1, 2, 3], 1", expected: "[[1], [2], [3]]" },
    ],
  },
  "Lowest Common Ancestor": {
    function_name: "LCA",
    call_type: "script",
    test_cases: [
      {
        input: `class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
root.left.left = TreeNode(6)
root.left.right = TreeNode(2)
result = LCA(root, root.left, root.right)
print(result.val)`,
        expected: "3",
      },
    ],
  },
  "Browser History": {
    function_name: "BrowserHistory",
    call_type: "script",
    test_cases: [
      {
        input: `bh = BrowserHistory("leetcode.com")
bh.visit("google.com")
bh.visit("facebook.com")
bh.visit("youtube.com")
print(bh.back(1))
print(bh.back(1))
print(bh.forward(1))
bh.visit("linkedin.com")
print(bh.forward(2))
print(bh.back(2))
print(bh.back(7))`,
        expected: "facebook.com\ngoogle.com\nfacebook.com\nlinkedin.com\ngoogle.com\nleetcode.com",
      },
    ],
  },
  "Keys and Rooms": {
    function_name: "canVisitAllRooms",
    call_type: "function",
    test_cases: [
      { input: "[[1], [2], [3], []]", expected: "True" },
      { input: "[[1, 3], [3, 0, 1], [2], [0]]", expected: "False" },
    ],
  },
  "Network Delay Time": {
    function_name: "networkDelayTime",
    call_type: "script",
    test_cases: [
      {
        input: `import heapq
from collections import defaultdict
print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]], 4, 2))`,
        expected: "2",
      },
      {
        input: `import heapq
from collections import defaultdict
print(networkDelayTime([[1,2,1]], 2, 2))`,
        expected: "-1",
      },
    ],
  },
};

async function seedTestCases() {
  const { data: problems, error } = await supabase
    .from("problems")
    .select("id, title");

  if (error) {
    console.error("Error fetching problems:", error);
    return;
  }

  let updated = 0;
  for (const problem of problems) {
    const testCases = testCasesByTitle[problem.title];
    if (testCases) {
      const { error: updateError } = await supabase
        .from("problems")
        .update({ test_cases: testCases })
        .eq("id", problem.id);

      if (updateError) {
        console.error(`Error updating ${problem.title}:`, updateError);
      } else {
        console.log(`✅ ${problem.title}`);
        updated++;
      }
    } else {
      console.log(`⏭️  ${problem.title} (테스트케이스 없음)`);
    }
  }

  console.log(`\n${updated}/${problems.length} 문제에 테스트케이스 추가 완료`);
}

seedTestCases();
