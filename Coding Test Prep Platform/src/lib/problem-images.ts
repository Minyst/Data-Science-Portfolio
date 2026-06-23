// 문제 제목 → 정적 이미지 경로 매핑
const PROBLEM_IMAGES: Record<string, string> = {
  "Two Sum": "/images/problems/two-sum.png",
  "Design Browser History": "/images/problems/design-browser-history.png",
  "Valid Parentheses": "/images/problems/valid-parentheses.png",
  "Daily Temperatures": "/images/problems/daily-temperatures.png",
  "Longest Consecutive Sequence": "/images/problems/longest-consecutive-sequence.png",
  "Maximum Depth of Binary Tree (BFS)": "/images/problems/maximum-depth-of-binary-tree.png",
  "Maximum Depth of Binary Tree (Recursive)": "/images/problems/maximum-depth-of-binary-tree.png",
  "Number of Islands": "/images/problems/number-of-islands.png",
  "Lowest Common Ancestor": "/images/problems/lowest-common-ancestor.png",
  "Shortest Path in Binary Matrix": "/images/problems/shortest-path-in-binary-matrix.png",
  "Keys and Rooms": "/images/problems/keys-and-rooms.png",
  "Climbing Stairs (DP)": "/images/problems/climbing-stairs.png",
  "Min Cost Climbing Stairs": "/images/problems/min-cost-climbing-stairs.png",
  "Unique Paths": "/images/problems/unique-paths.png",
  "Network Delay Time": "/images/problems/network-delay-time.png",
};

export function getProblemImage(title: string): string | null {
  return PROBLEM_IMAGES[title] || null;
}
