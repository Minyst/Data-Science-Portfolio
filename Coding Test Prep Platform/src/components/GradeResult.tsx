"use client";

interface GradeResultProps {
  grade: "perfect" | "partial" | "wrong";
  message: string;
  emoji: string;
  passedCount: number;
  totalCount: number;
}

export default function GradeResult({
  grade,
  message,
  emoji,
  passedCount,
  totalCount,
}: GradeResultProps) {
  const bgColors = {
    perfect: "bg-green-500/10 border-green-500/30",
    partial: "bg-yellow-500/10 border-yellow-500/30",
    wrong: "bg-red-500/10 border-red-500/30",
  };

  const pct = totalCount > 0 ? Math.round((passedCount / totalCount) * 100) : 0;
  const strokeColor =
    grade === "perfect" ? "#22c55e" : grade === "partial" ? "#eab308" : "#ef4444";

  return (
    <div className={`rounded-xl border p-5 ${bgColors[grade]}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-3xl font-bold">
            <span className="mr-2 text-lg">{emoji}</span>
            {passedCount}/{totalCount}
            <span className="ml-3 text-base font-medium text-gray-400">{pct}%</span>
          </p>
          <p className="mt-1 text-sm text-gray-300">{message}</p>
        </div>
        <div className="relative h-16 w-16 shrink-0">
          <svg className="h-16 w-16 -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50" cy="50" r="40"
              fill="none" stroke="currentColor"
              strokeWidth="8" className="text-gray-700"
            />
            <circle
              cx="50" cy="50" r="40"
              fill="none" strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={`${pct * 2.51} 251`}
              style={{ stroke: strokeColor }}
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
