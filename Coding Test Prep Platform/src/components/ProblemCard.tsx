import Link from "next/link";
import type { Problem } from "@/lib/types";

interface ProblemCardProps {
  problem: Problem;
  lastScore?: number | null;
  isSolved?: boolean;
}

const difficultyColors: Record<string, string> = {
  Easy: "bg-green-500/15 text-green-400",
  Medium: "bg-yellow-500/15 text-yellow-400",
  Hard: "bg-red-500/15 text-red-400",
};

const categoryColors: Record<string, string> = {
  Hash: "bg-purple-500/15 text-purple-400",
  LinkedList: "bg-blue-500/15 text-blue-400",
  Stack: "bg-cyan-500/15 text-cyan-400",
  Queue: "bg-teal-500/15 text-teal-400",
  BFS: "bg-orange-500/15 text-orange-400",
  DFS: "bg-amber-500/15 text-amber-400",
  DP: "bg-pink-500/15 text-pink-400",
  Heap: "bg-indigo-500/15 text-indigo-400",
  Dijkstra: "bg-rose-500/15 text-rose-400",
  Backtracking: "bg-emerald-500/15 text-emerald-400",
  Tree: "bg-lime-500/15 text-lime-400",
  Graph: "bg-sky-500/15 text-sky-400",
};

export default function ProblemCard({ problem, lastScore, isSolved = false }: ProblemCardProps) {
  return (
    <Link
      href={`/problems/${problem.id}`}
      className="group block rounded-xl border border-gray-800 bg-black p-5 transition-all hover:border-gray-600"
    >
      <div className="flex items-start justify-between">
        <h3 className="text-lg font-semibold text-white group-hover:text-blue-400 transition-colors">
          {problem.title}
        </h3>
        {lastScore !== null && lastScore !== undefined ? (
          <span
            className={`text-sm font-bold ${
              isSolved ? "text-green-400" : lastScore >= 50 ? "text-yellow-400" : "text-red-400"
            }`}
          >
            {isSolved ? "✅ 통과" : `${lastScore}%`}
          </span>
        ) : null}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <span className={`rounded-full px-3 py-1 text-xs font-medium ${categoryColors[problem.category] || "bg-gray-700 text-gray-300"}`}>
          {problem.category}
        </span>
        <span className={`rounded-full px-3 py-1 text-xs font-medium ${difficultyColors[problem.difficulty] || "bg-gray-700 text-gray-300"}`}>
          {problem.difficulty}
        </span>
      </div>

      {problem.description && (
        <p className="mt-3 text-sm text-gray-400 line-clamp-2">{problem.description}</p>
      )}

      <div className="mt-4 flex items-center text-sm text-gray-500">
        <span>풀어보기 →</span>
      </div>
    </Link>
  );
}
