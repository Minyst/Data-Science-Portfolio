export interface TestCase {
  input: string;
  expected: string;
}

export interface TestCaseConfig {
  function_name: string;
  call_type: "function" | "class" | "script";
  test_cases: TestCase[];
}

export interface TestResult {
  input: string;
  expected: string;
  actual: string;
  passed: boolean;
  error?: string;
}

export interface Problem {
  id: string;
  title: string;
  category: string;
  difficulty: string;
  code: string;
  description: string | null;
  notes: string | null;
  test_cases: TestCaseConfig | null;
  images: string[] | null;
  notion_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface Submission {
  id: string;
  problem_id: string;
  user_code: string;
  similarity_score: number;
  is_correct: boolean;
  passed_count: number;
  total_count: number;
  test_results: TestResult[] | null;
  created_at: string;
}

export const CATEGORIES = [
  "Hash",
  "LinkedList",
  "Stack",
  "Queue",
  "BFS",
  "DFS",
  "DP",
  "Heap",
  "Dijkstra",
  "Backtracking",
  "Tree",
  "Graph",
] as const;

export const DIFFICULTIES = ["Easy", "Medium", "Hard"] as const;
