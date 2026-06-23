"use client";

import type { Problem } from "@/lib/types";

interface CategoryStatsProps {
  problems: Problem[];
  scores: Record<string, number>;
  solved: Record<string, boolean>;
}

export default function CategoryStats({ problems, scores, solved }: CategoryStatsProps) {
  const stats = new Map<string, { total: number; solved: number }>();

  for (const p of problems) {
    if (!stats.has(p.category)) {
      stats.set(p.category, { total: 0, solved: 0 });
    }
    const s = stats.get(p.category)!;
    s.total++;
    if (solved[p.id]) s.solved++;
  }

  const entries = Array.from(stats.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  if (entries.length === 0) return null;

  const totalProblems = problems.length;
  const totalSolved = Object.keys(solved).length;
  const totalAttempted = Object.keys(scores).length;

  return (
    <div className="rounded-xl bg-white p-5 space-y-4">
      <div className="flex items-center gap-4">
        <span className="text-sm font-semibold text-black">총 {totalProblems}문제</span>
        <span className="text-sm font-semibold text-green-600">{totalSolved}문제 통과</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {entries.map(([category, { total, solved: solvedCount }]) => {
          const pct = total > 0 ? Math.round((solvedCount / total) * 100) : 0;
          return (
            <div key={category} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="text-black font-medium">{category}</span>
                <span className="text-black font-medium">{solvedCount}/{total}</span>
              </div>
              <div className="h-1.5 rounded-full bg-gray-200 overflow-hidden">
                <div
                  className="h-full rounded-full bg-green-500 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
