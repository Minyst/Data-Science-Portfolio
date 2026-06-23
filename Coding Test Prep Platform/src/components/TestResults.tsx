"use client";

import type { TestResult } from "@/lib/types";

interface TestResultsProps {
  results: TestResult[];
  passedCount: number;
  totalCount: number;
}

export default function TestResults({
  results,
  passedCount,
  totalCount,
}: TestResultsProps) {
  return (
    <div className="space-y-3">
      {/* 헤더 */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold text-gray-300">테스트 결과</span>
        <span
          className={`rounded-full px-3 py-1 text-xs font-semibold ${
            passedCount === totalCount
              ? "bg-green-500/15 text-green-400"
              : passedCount > 0
              ? "bg-yellow-500/15 text-yellow-400"
              : "bg-red-500/15 text-red-400"
          }`}
        >
          {passedCount}/{totalCount} 통과
        </span>
      </div>

      {/* 테스트 카드 그리드 — 균등 너비, 테트리스 쌓기 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {results.map((result, i) => (
          <div
            key={i}
            className={`rounded-lg border p-4 ${
              result.passed
                ? "border-green-500/30 bg-green-500/5"
                : "border-red-500/30 bg-red-500/5"
            }`}
          >
            {/* 테스트 번호 */}
            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-700/50">
              <span className="text-base">{result.passed ? "✅" : "❌"}</span>
              <span className="text-sm font-semibold text-gray-200">
                테스트 {i + 1}
              </span>
            </div>

            {/* 입력 / 기대값 / 실제값 — 균등 행 */}
            <div className="space-y-2 text-xs font-mono">
              <div className="grid grid-cols-[56px_1fr] gap-2">
                <span className="text-gray-500 font-sans font-medium">입력</span>
                <span className="text-gray-300 break-all bg-black/20 rounded px-2 py-1 overflow-x-auto">
                  {formatInput(result.input)}
                </span>
              </div>
              <div className="grid grid-cols-[56px_1fr] gap-2">
                <span className="text-gray-500 font-sans font-medium">기대값</span>
                <span className="text-green-400 break-all bg-black/20 rounded px-2 py-1">
                  {result.expected}
                </span>
              </div>
              {!result.passed && (
                <div className="grid grid-cols-[56px_1fr] gap-2">
                  <span className="text-gray-500 font-sans font-medium">실제값</span>
                  <span className="text-red-400 break-all bg-black/20 rounded px-2 py-1">
                    {result.actual}
                  </span>
                </div>
              )}
              {result.error && (
                <div className="mt-1 rounded bg-red-500/10 p-2 text-xs text-red-400 break-all">
                  {result.error}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** 긴 스크립트 입력은 축약 표시 */
function formatInput(input: string): string {
  // 멀티라인 스크립트인 경우 첫 줄만 + ... 표시
  const lines = input.split("\n").filter(Boolean);
  if (lines.length <= 2) return input;
  return `${lines[0]}  ... (${lines.length}줄)`;
}
