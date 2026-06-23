"use client";

interface CodeDiffProps {
  userCode: string;
  answerCode: string;
}

interface DiffLine {
  type: "same" | "added" | "removed";
  content: string;
  lineNum?: number;
}

// LCS 기반 라인 diff 알고리즘
function computeDiff(userLines: string[], answerLines: string[]): DiffLine[] {
  const m = userLines.length;
  const n = answerLines.length;

  // LCS 테이블 생성
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (userLines[i - 1].trim() === answerLines[j - 1].trim()) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  // Backtrack으로 diff 생성
  const result: DiffLine[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && userLines[i - 1].trim() === answerLines[j - 1].trim()) {
      result.push({ type: "same", content: answerLines[j - 1], lineNum: j });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      result.push({ type: "added", content: answerLines[j - 1], lineNum: j });
      j--;
    } else {
      result.push({ type: "removed", content: userLines[i - 1] });
      i--;
    }
  }

  return result.reverse();
}

export default function CodeDiff({ userCode, answerCode }: CodeDiffProps) {
  const userLines = userCode.split("\n");
  const answerLines = answerCode.split("\n");
  const diffLines = computeDiff(userLines, answerLines);

  return (
    <div className="rounded-lg overflow-hidden border border-gray-700">
      {/* Legend */}
      <div className="flex gap-4 px-4 py-2 bg-gray-800 border-b border-gray-700 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-red-500/30 border border-red-500/50" />
          내 코드에만 있는 부분
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm bg-green-500/30 border border-green-500/50" />
          정답 코드에만 있는 부분
        </span>
      </div>
      {/* Diff Lines */}
      <div className="overflow-x-auto bg-[#1e1e1e] p-0">
        <pre className="text-sm font-mono leading-6">
          {diffLines.map((line, i) => {
            let bgClass = "";
            let textClass = "text-gray-300";
            let prefix = "  ";

            if (line.type === "removed") {
              bgClass = "bg-red-500/15";
              textClass = "text-red-300";
              prefix = "- ";
            } else if (line.type === "added") {
              bgClass = "bg-green-500/15";
              textClass = "text-green-300";
              prefix = "+ ";
            }

            return (
              <div key={i} className={`px-4 ${bgClass}`}>
                <span className="text-gray-600 select-none w-8 inline-block text-right mr-3">
                  {line.lineNum || ""}
                </span>
                <span className={`${textClass} select-none opacity-60 mr-1`}>{prefix}</span>
                <span className={textClass}>{line.content}</span>
              </div>
            );
          })}
        </pre>
      </div>
    </div>
  );
}
