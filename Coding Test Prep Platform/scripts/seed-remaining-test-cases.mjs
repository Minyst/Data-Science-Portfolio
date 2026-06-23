import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !key) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  process.exit(1);
}

const supabase = createClient(url, key);

// 트리 노드 헬퍼 코드 (테스트에서 공통 사용)
const TREE_HELPER = `class TreeNode:
    def __init__(self, value):
        self.value = value
        self.data = value
        self.left = None
        self.right = None
`;

// 나머지 10문제 테스트케이스
const testCasesByTitle = {
  "LinkedList 구현": {
    function_name: "LinkedList",
    call_type: "script",
    test_cases: [
      {
        input: `ll = LinkedList(1)
ll.append(2)
ll.append(3)
ll.print_all()`,
        expected: "1\n2\n3",
      },
      {
        input: `ll = LinkedList(10)
ll.append(20)
ll.append(30)
ll.add_node(1, 15)
ll.print_all()`,
        expected: "10\n15\n20\n30",
      },
      {
        input: `ll = LinkedList(1)
ll.append(2)
ll.append(3)
ll.delete_node(1)
ll.print_all()`,
        expected: "1\n3",
      },
    ],
  },

  "Maximum Depth of Binary Tree (BFS)": {
    function_name: "maxDepth",
    call_type: "script",
    test_cases: [
      {
        input: `${TREE_HELPER}from collections import deque
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)
print(maxDepth(root))`,
        expected: "3",
      },
      {
        input: `${TREE_HELPER}from collections import deque
root = TreeNode(1)
root.right = TreeNode(2)
print(maxDepth(root))`,
        expected: "2",
      },
      {
        input: `${TREE_HELPER}from collections import deque
print(maxDepth(None))`,
        expected: "0",
      },
    ],
  },

  "Maximum Depth of Binary Tree (Recursive)": {
    function_name: "maxDepth",
    call_type: "script",
    test_cases: [
      {
        input: `${TREE_HELPER}root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)
print(maxDepth(root))`,
        expected: "3",
      },
      {
        input: `${TREE_HELPER}root = TreeNode(1)
root.right = TreeNode(2)
print(maxDepth(root))`,
        expected: "2",
      },
      {
        input: `${TREE_HELPER}print(maxDepth(None))`,
        expected: "0",
      },
    ],
  },

  "BFS (Tree)": {
    function_name: "bfs",
    call_type: "script",
    test_cases: [
      {
        input: `${TREE_HELPER}from collections import deque
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
print(bfs(root))`,
        expected: "[1, 2, 3, 4, 5]",
      },
      {
        input: `${TREE_HELPER}from collections import deque
root = TreeNode(10)
root.left = TreeNode(20)
root.right = TreeNode(30)
print(bfs(root))`,
        expected: "[10, 20, 30]",
      },
      {
        input: `${TREE_HELPER}from collections import deque
print(bfs(None))`,
        expected: "0",
      },
    ],
  },

  "BFS (Graph)": {
    function_name: "bfs",
    call_type: "script",
    test_cases: [
      {
        input: `from collections import deque
graph = {
    1: [2, 3],
    2: [4, 5],
    3: [],
    4: [],
    5: []
}
print(bfs(graph, 1))`,
        expected: "[1, 2, 3, 4, 5]",
      },
      {
        input: `from collections import deque
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}
print(bfs(graph, 'A'))`,
        expected: "['A', 'B', 'C', 'D']",
      },
    ],
  },

  "DFS (Preorder/Inorder/Postorder)": {
    function_name: "preorder",
    call_type: "script",
    test_cases: [
      {
        input: `${TREE_HELPER}root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
print("preorder:")
preorder(root)`,
        expected: "preorder:\n1\n2\n4\n5\n3",
      },
      {
        input: `${TREE_HELPER}root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
print("inorder:")
inorder(root)`,
        expected: "inorder:\n4\n2\n5\n1\n3",
      },
      {
        input: `${TREE_HELPER}root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
print("postorder:")
postorder(root)`,
        expected: "postorder:\n4\n5\n2\n3\n1",
      },
    ],
  },

  "DFS (Graph)": {
    function_name: "dfs",
    call_type: "script",
    test_cases: [
      {
        input: `graph = {
    1: [2, 3],
    2: [4, 5],
    3: [6],
    4: [],
    5: [],
    6: []
}
visited = []
dfs(1)
print(visited)`,
        expected: "[1, 2, 4, 5, 3, 6]",
      },
      {
        input: `graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': [],
    'D': []
}
visited = []
dfs('A')
print(visited)`,
        expected: "['A', 'B', 'D', 'C']",
      },
    ],
  },

  "Min Cost Climbing Stairs": {
    function_name: "dp",
    call_type: "script",
    test_cases: [
      {
        input: `cost = [10, 15, 20, 17, 1]
print(dp(len(cost)))`,
        expected: "28",
      },
      {
        input: `cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
print(dp(len(cost)))`,
        expected: "6",
      },
      {
        input: `cost = [10, 15, 20]
print(dp(len(cost)))`,
        expected: "15",
      },
    ],
  },

  "Max Heap": {
    function_name: "heapq",
    call_type: "script",
    test_cases: [
      {
        input: `import heapq
max_heap = [5, 3, 9, 4, 1, 2, 6]
max_heap = [(-1 * i, i) for i in max_heap]
heapq.heapify(max_heap)
weight, value = heapq.heappop(max_heap)
print(value)`,
        expected: "9",
      },
      {
        input: `import heapq
max_heap = [10, 20, 30, 5, 15]
max_heap = [(-1 * i, i) for i in max_heap]
heapq.heapify(max_heap)
weight, value = heapq.heappop(max_heap)
print(value)
weight2, value2 = heapq.heappop(max_heap)
print(value2)`,
        expected: "30\n20",
      },
    ],
  },

  "Subsets (부분집합)": {
    function_name: "solution",
    call_type: "function",
    test_cases: [
      {
        input: "[1, 2, 3]",
        expected: "[[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]",
      },
      {
        input: "[0]",
        expected: "[[], [0]]",
      },
    ],
  },
};

async function seed() {
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
        console.error(`❌ ${problem.title}: ${updateError.message}`);
      } else {
        console.log(`✅ ${problem.title}`);
        updated++;
      }
    }
  }

  console.log(`\n${updated} 문제에 테스트케이스 추가 완료`);
}

seed();
