import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !key) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables.");
  console.error("Usage: SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... node scripts/seed-from-notion.mjs");
  process.exit(1);
}

const supabase = createClient(url, key);

// Parsed from Notion "Coding Test" page
const problems = [
  {
    title: "Two Sum",
    category: "Hash",
    difficulty: "Easy",
    description: "주어진 배열에서 두 수의 합이 target이 되는지 확인",
    code: `def two_sum(nums, target):
    memo = {}
    for i, num in enumerate(nums):
        needed = target - num
        if needed in memo:
            return True
        memo[num] = i
    return False`,
  },
  {
    title: "Browser History",
    category: "LinkedList",
    difficulty: "Medium",
    description: "이중 연결 리스트로 브라우저 히스토리 구현",
    code: `class ListNode(object):
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev
class BrowserHistory(object):
    def __init__(self, homepage):
        self.head = self.current = ListNode(val=homepage)
    def visit(self, url):
        self.current.next = ListNode(val=url, prev=self.current)
        self.current = self.current.next
        return None
    def back(self, steps):
        while steps > 0 and self.current.prev != None:
            steps -= 1
            self.current = self.current.prev
        return self.current.val
    def forward(self, steps):
        while steps > 0 and self.current.next != None:
            steps -= 1
            self.current = self.current.next
        return self.current.val`,
  },
  {
    title: "LinkedList 구현",
    category: "LinkedList",
    difficulty: "Easy",
    description: "단일 연결 리스트 정의 (append, print, get, add, delete)",
    code: `class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self, value):
        self.head = Node(value)

    def append(self, value):
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = Node(value)

    def print_all(self):
        cur = self.head
        while cur is not None:
            print(cur.data)
            cur = cur.next

    def get_node(self, index):
        cur = self.head
        cur_index = 0
        while cur_index != index:
            cur = cur.next
            cur_index += 1
        return cur

    def add_node(self, index, value):
        new_node = Node(value)
        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return
        prev_node = self.get_node(index - 1)
        next_node = prev_node.next
        prev_node.next = new_node
        new_node.next = next_node

    def delete_node(self, index):
        if index == 0:
            self.head = self.head.next
            return
        prev_node = self.get_node(index - 1)
        index_node = self.get_node(index)
        prev_node.next = index_node.next`,
  },
  {
    title: "Valid Parentheses",
    category: "Stack",
    difficulty: "Easy",
    description: "괄호 유효성 검사",
    code: `def isValid(s):
    stack = []
    for p in s:
        if p == "(":
            stack.append(")")
        elif p == "{":
            stack.append("}")
        elif p == "[":
            stack.append("]")
        elif not stack or stack.pop() != p:
            return False
    return not stack`,
  },
  {
    title: "Daily Temperatures",
    category: "Stack",
    difficulty: "Medium",
    description: "각 날짜에 대해 더 따뜻한 날까지 며칠을 기다려야 하는지 계산",
    code: `def dailyTemperatures(temperatures):
    ans = [0] * len(temperatures)
    stack = []
    for cur_day, cur_temp in enumerate(temperatures):
        while stack and stack[-1][1] < cur_temp:
            prev_day, _ = stack.pop()
            ans[prev_day] = cur_day - prev_day
        stack.append((cur_day, cur_temp))
    return ans`,
  },
  {
    title: "Longest Consecutive Sequence",
    category: "Hash",
    difficulty: "Medium",
    description: "정렬 없이 가장 긴 연속 수열의 길이를 찾기",
    code: `def longestConsecutive(nums):
    longest = 0
    num_dict = {}
    for num in nums:
        num_dict[num] = True
    for num in num_dict:
        if num - 1 not in num_dict:
            cnt = 1
            target = num + 1
            while target in num_dict:
                target += 1
                cnt += 1
            longest = max(longest, cnt)
    return longest`,
  },
  {
    title: "Maximum Depth of Binary Tree (BFS)",
    category: "Tree",
    difficulty: "Easy",
    description: "이진 트리의 최대 깊이 구하기 - BFS 풀이",
    code: `from collections import deque

def maxDepth(root):
    max_depth = 0
    if root is None:
        return max_depth
    q = deque()
    q.append((root, 1))
    while q:
        cur_node, cur_depth = q.popleft()
        max_depth = max(max_depth, cur_depth)
        if cur_node.left:
            q.append((cur_node.left, cur_depth + 1))
        if cur_node.right:
            q.append((cur_node.right, cur_depth + 1))
    return max_depth`,
  },
  {
    title: "Maximum Depth of Binary Tree (Recursive)",
    category: "Tree",
    difficulty: "Easy",
    description: "이진 트리의 최대 깊이 구하기 - 재귀 풀이",
    code: `def maxDepth(root):
    max_depth = 0
    if root is None:
        return max_depth
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    max_depth = max(left_depth, right_depth) + 1
    return max_depth`,
  },
  {
    title: "BFS (Tree)",
    category: "BFS",
    difficulty: "Medium",
    description: "이진 트리 BFS (레벨 순회)",
    code: `from collections import deque

def bfs(root):
    visited = []
    if root is None:
        return 0
    q = deque()
    q.append(root)
    while q:
        cur_node = q.popleft()
        visited.append(cur_node.value)
        if cur_node.left:
            q.append(cur_node.left)
        if cur_node.right:
            q.append(cur_node.right)
    return visited`,
  },
  {
    title: "BFS (Graph)",
    category: "BFS",
    difficulty: "Medium",
    description: "그래프 BFS 탐색",
    code: `from collections import deque

def bfs(graph, start_v):
    visited = [start_v]
    q = deque(start_v)
    while q:
        cur_v = q.popleft()
        for v in graph[cur_v]:
            if v not in visited:
                visited.append(v)
                q.append(v)
    return visited`,
  },
  {
    title: "DFS (Preorder/Inorder/Postorder)",
    category: "DFS",
    difficulty: "Medium",
    description: "트리 DFS 전위/중위/후위 순회",
    code: `def preorder(cur_node):
    if cur_node is None:
        return
    print(cur_node.value)
    preorder(cur_node.left)
    preorder(cur_node.right)

def inorder(cur_node):
    if cur_node is None:
        return
    inorder(cur_node.left)
    print(cur_node.value)
    inorder(cur_node.right)

def postorder(cur_node):
    if cur_node is None:
        return
    postorder(cur_node.left)
    postorder(cur_node.right)
    print(cur_node.value)`,
  },
  {
    title: "DFS (Graph)",
    category: "DFS",
    difficulty: "Medium",
    description: "그래프 DFS 탐색 (재귀)",
    code: `graph = { ... }
visited = []

def dfs(cur_v):
    visited.append(cur_v)
    for v in graph[cur_v]:
        if v not in visited:
            dfs(v)`,
  },
  {
    title: "Number of Islands",
    category: "BFS",
    difficulty: "Medium",
    description: "2D 그리드에서 섬의 개수 세기",
    code: `from collections import deque

def numIslands(grid):
    number_of_islands = 0
    m = len(grid)
    n = len(grid[0])
    visited = [[False]*n for _ in range(m)]
    def bfs(x, y):
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        visited[x][y] = True
        queue = deque()
        queue.append((x, y))
        while queue:
            cur_x, cur_y = queue.popleft()
            for i in range(4):
                next_x = cur_x + dx[i]
                next_y = cur_y + dy[i]
                if next_x >= 0 and next_x < m and next_y >= 0 and next_y < n:
                    if grid[next_x][next_y] == "1" and not visited[next_x][next_y]:
                        visited[next_x][next_y] = True
                        queue.append((next_x, next_y))
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1" and not visited[i][j]:
                bfs(i, j)
                number_of_islands += 1
    return number_of_islands`,
  },
  {
    title: "Lowest Common Ancestor",
    category: "Tree",
    difficulty: "Medium",
    description: "이진 트리의 최소 공통 조상 찾기",
    code: `def LCA(root, p, q):
    if root == None:
        return None
    left = LCA(root.left, p, q)
    right = LCA(root.right, p, q)
    if root == p or root == q:
        return root
    elif left and right:
        return root
    return left or right`,
  },
  {
    title: "Keys and Rooms",
    category: "BFS",
    difficulty: "Medium",
    description: "모든 방을 방문할 수 있는지 확인 (BFS)",
    code: `from collections import deque

def canVisitAllRooms(rooms):
    visited = [False] * len(rooms)
    def bfs(v):
        queue = deque()
        queue.append(v)
        visited[v] = True
        while queue:
            cur_v = queue.popleft()
            for next_v in rooms[cur_v]:
                if visited[next_v] == False:
                    queue.append(next_v)
                    visited[next_v] = True
    bfs(0)
    return all(visited)`,
  },
  {
    title: "Fibonacci (DP)",
    category: "DP",
    difficulty: "Easy",
    description: "피보나치 수열 (Bottom-up DP)",
    code: `memo = {1: 1, 2: 1}
def fibo(n):
    if n == 1 or n == 2:
        return 1
    for i in range(3, n+1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]`,
  },
  {
    title: "Climbing Stairs (DP)",
    category: "DP",
    difficulty: "Easy",
    description: "계단 오르기 (1칸 또는 2칸)",
    code: `memo = {1: 1, 2: 2}
def cs(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    for i in range(3, n+1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]`,
  },
  {
    title: "Min Cost Climbing Stairs",
    category: "DP",
    difficulty: "Easy",
    description: "최소 비용으로 계단 오르기",
    code: `cost = [10, 15, 20, 17, 1]
def dp(n):
    memo = [-1]*n
    memo[0] = 0
    memo[1] = 0
    for i in range(2, n+1):
        memo[i] = min(memo[i-1] + cost[i-1], memo[i-2] + cost[i-2])
    return memo[n]`,
  },
  {
    title: "Unique Paths",
    category: "DP",
    difficulty: "Medium",
    description: "m x n 그리드의 고유 경로 수",
    code: `def uniquePaths(m, n):
    memo = [[-1] * n for _ in range(m)]
    for r in range(m):
        memo[r][0] = 1
    for c in range(n):
        memo[0][c] = 1
    for r in range(1, m):
        for c in range(1, n):
            memo[r][c] = memo[r-1][c] + memo[r][c-1]
    return memo[m-1][n-1]`,
  },
  {
    title: "Max Heap",
    category: "Heap",
    difficulty: "Easy",
    description: "Python heapq를 이용한 최대 힙",
    code: `import heapq

max_heap = [5, 3, 9, 4, 1, 2, 6]
max_heap = [(-1 * i, i) for i in max_heap]
heapq.heapify(max_heap)
weight, value = heapq.heappop(max_heap)`,
  },
  {
    title: "Dijkstra",
    category: "Dijkstra",
    difficulty: "Medium",
    description: "다익스트라 최단 경로 알고리즘",
    code: `def dijkstra(graph, start, final, n):
    costs = {}
    pq = []
    heapq.heappush(pq, (0, start))
    while pq:
        cur_cost, cur_v = heapq.heappop(pq)
        if cur_v not in costs:
            costs[cur_v] = cur_cost
            for cost, next_v in graph[cur_v]:
                next_cost = cur_cost + cost
                heapq.heappush(pq, (next_cost, next_v))
    return costs[final]`,
  },
  {
    title: "Network Delay Time",
    category: "Dijkstra",
    difficulty: "Medium",
    description: "네트워크 지연 시간 (다익스트라 응용)",
    code: `def networkDelayTime(times, n, k):
    graph = defaultdict(list)
    for time in times:
        graph[time[0]].append((time[2], time[1]))
    costs = {}
    pq = []
    heapq.heappush(pq, (0, k))
    while pq:
        cur_cost, cur_node = heapq.heappop(pq)
        if cur_node not in costs:
            costs[cur_node] = cur_cost
            for cost, next_node in graph[cur_node]:
                next_cost = cur_cost + cost
                heapq.heappush(pq, (next_cost, next_node))
    for node in range(1, n+1):
        if node not in costs:
            return -1
    return max(costs.values())`,
  },
  {
    title: "Permutations (순열)",
    category: "Backtracking",
    difficulty: "Medium",
    description: "주어진 배열의 모든 순열 반환",
    code: `def permute(nums):
    def backtrack(curr):
        if len(curr) == len(nums):
            ans.append(curr[:])
            return
        for num in nums:
            if num not in curr:
                curr.append(num)
                backtrack(curr)
                curr.pop()
    ans = []
    backtrack([])
    return ans`,
  },
  {
    title: "Combinations (조합)",
    category: "Backtracking",
    difficulty: "Medium",
    description: "n개 중 k개 선택하는 모든 조합",
    code: `def solution(nums, k):
    result = []
    def backtracking(start, curr):
        if len(curr) == k:
            result.append(curr[:])
            return
        for i in range(start, len(nums)):
            curr.append(nums[i])
            backtracking(i+1, curr)
            curr.pop()
    backtracking(start = 0, curr = [])
    return result`,
  },
  {
    title: "Subsets (부분집합)",
    category: "Backtracking",
    difficulty: "Medium",
    description: "주어진 배열의 모든 부분집합 반환",
    code: `def solution(nums):
    result = []
    def backtracking(start, curr):
        if len(curr) == k:
            result.append(curr[:])
            return
        for i in range(start, len(nums)):
            curr.append(nums[i])
            backtracking(i+1, curr)
            curr.pop()
    for k in range(len(nums)+1):
        backtracking(start = 0, curr = [])
    return result`,
  },
];

async function seed() {
  console.log(`Seeding ${problems.length} problems...`);
  await supabase.from("problems").delete().neq("id", "00000000-0000-0000-0000-000000000000");
  const { data, error } = await supabase.from("problems").insert(problems).select();
  if (error) {
    console.error("Seed error:", error);
  } else {
    console.log(`Successfully seeded ${data.length} problems!`);
    const cats = {};
    for (const p of data) {
      cats[p.category] = (cats[p.category] || 0) + 1;
    }
    console.table(cats);
  }
}

seed();
