"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import type { Problem, TestResult } from "@/lib/types";
import type { GradeResult as GradeResultType } from "@/lib/grading";
import CodeEditor from "@/components/CodeEditor";
import GradeResult from "@/components/GradeResult";
import TestResults from "@/components/TestResults";
import ProblemEditForm from "@/components/ProblemEditForm";
import CodeDiff from "@/components/CodeDiff";
import CodeBlock from "@/components/CodeBlock";
import { getProblemImage } from "@/lib/problem-images";

export default function ProblemDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [problem, setProblem] = useState<Problem | null>(null);
  const [userCode, setUserCode] = useState("");
  const [result, setResult] = useState<(GradeResultType & { testResults: TestResult[] }) | null>(null);
  const [grading, setGrading] = useState(false);
  const [running, setRunning] = useState(false);
  const [runOutput, setRunOutput] = useState<{ stdout: string; stderr: string; error: string | null; expected: string } | null>(null);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [pyodideLoading, setPyodideLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    title: "", category: "", difficulty: "", code: "", description: "", notes: "",
  });

  useEffect(() => {
    const supabase = createClient();
    supabase
      .from("problems")
      .select("*")
      .eq("id", params.id)
      .single()
      .then(({ data }) => {
        setProblem(data);
        if (data) {
          setEditForm({
            title: data.title,
            category: data.category,
            difficulty: data.difficulty,
            code: data.code,
            description: data.description || "",
            notes: data.notes || "",
          });
        }
        setLoading(false);
      });
  }, [params.id]);

  const loadPyodide = useCallback(async () => {
    if (pyodideReady || pyodideLoading) return;
    setPyodideLoading(true);
    try {
      const { getPyodide } = await import("@/lib/pyodide-runner");
      await getPyodide();
      setPyodideReady(true);
    } catch {
      console.error("Pyodide 로드 실패");
    } finally {
      setPyodideLoading(false);
    }
  }, [pyodideReady, pyodideLoading]);

  useEffect(() => {
    if (problem) loadPyodide();
  }, [problem, loadPyodide]);

  const handleRun = async () => {
    if (!problem || !userCode.trim() || !problem.test_cases) return;
    setRunning(true);
    setRunOutput(null);
    try {
      if (!pyodideReady) await loadPyodide();
      const { dryRunTestCase } = await import("@/lib/test-runner");
      const output = await dryRunTestCase(userCode, problem.test_cases, 0);
      setRunOutput(output);
    } catch {
      setRunOutput({ stdout: "", stderr: "", error: "실행 중 오류가 발생했습니다.", expected: "" });
    } finally {
      setRunning(false);
    }
  };

  const handleGrade = async () => {
    if (!problem || !userCode.trim()) return;
    if (!problem.test_cases) {
      alert("이 문제에는 아직 테스트케이스가 없습니다.");
      return;
    }

    setGrading(true);
    setResult(null);

    try {
      if (!pyodideReady) await loadPyodide();
      const { runTestCases } = await import("@/lib/test-runner");
      const { gradeByTestResults } = await import("@/lib/grading");

      const runResult = await runTestCases(userCode, problem.test_cases);
      const gradeResult = gradeByTestResults(runResult.testResults);

      setResult({ ...gradeResult, testResults: runResult.testResults });

      await fetch("/api/grade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          problemId: problem.id,
          userCode,
          testResults: runResult.testResults,
          passedCount: runResult.passedCount,
          totalCount: runResult.totalCount,
          isCorrect: runResult.passed,
        }),
      });
    } catch {
      alert("채점 중 오류가 발생했습니다.");
    } finally {
      setGrading(false);
    }
  };

  const handleDelete = async () => {
    if (!problem || !confirm("정말 삭제하시겠습니까?")) return;
    const supabase = createClient();
    await supabase.from("problems").delete().eq("id", problem.id);
    router.push("/");
  };

  const handleUpdate = async () => {
    if (!problem) return;
    const supabase = createClient();
    await supabase.from("problems").update({
      ...editForm,
      notes: editForm.notes || null,
      description: editForm.description || null,
    }).eq("id", problem.id);
    setProblem({ ...problem, ...editForm });
    setIsEditing(false);
  };

  const handleRetry = () => {
    setResult(null);
  };

  if (loading) return <div className="py-20 text-center text-gray-500">로딩 중...</div>;
  if (!problem) return <div className="py-20 text-center text-gray-500">문제를 찾을 수 없습니다</div>;

  if (isEditing) {
    return (
      <ProblemEditForm
        title="문제 수정"
        form={editForm}
        onChange={setEditForm}
        onSave={handleUpdate}
        onCancel={() => setIsEditing(false)}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold">{problem.title}</h1>
          <div className="mt-3 flex gap-2">
            <span className="rounded-full bg-blue-500/15 px-3 py-1 text-sm text-blue-400">
              {problem.category}
            </span>
            <span className="rounded-full bg-yellow-500/15 px-3 py-1 text-sm text-yellow-400">
              {problem.difficulty}
            </span>
            {!problem.test_cases && (
              <span className="rounded-full bg-gray-500/15 px-3 py-1 text-sm text-gray-400">
                테스트케이스 없음
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setIsEditing(true)}
            className="rounded-lg bg-blue-200 border border-blue-300 px-4 py-2 text-sm text-blue-900 hover:bg-blue-300 hover:border-blue-400 transition-colors font-medium"
          >
            수정
          </button>
          <button
            onClick={handleDelete}
            className="rounded-lg bg-red-200 border border-red-300 px-4 py-2 text-sm text-red-900 hover:bg-red-300 hover:border-red-400 transition-colors font-medium"
          >
            삭제
          </button>
        </div>
      </div>

      {/* Problem Image */}
      {getProblemImage(problem.title) && (
        <div>
          <img
            src={getProblemImage(problem.title)!}
            alt={`${problem.title} 문제`}
            className="rounded-lg max-w-full border border-gray-700"
          />
        </div>
      )}

      {/* ── 채점 전: 에디터 + 채점 버튼 ── */}
      {!result ? (
        <>
          <div>
            <h2 className="mb-3 text-lg font-semibold">풀이 작성</h2>
            <CodeEditor
              value={userCode}
              onChange={setUserCode}
              placeholder="여기에 코드를 작성하고 채점하기를 누르세요..."
            />
          </div>
          <div className="flex gap-3">
            <button
              onClick={handleRun}
              disabled={running || grading || !userCode.trim() || !problem.test_cases}
              className="flex-1 rounded-xl bg-gray-800 border border-gray-600 py-4 text-lg font-bold text-white transition-all hover:border-gray-400 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {running ? "실행 중..." : "코드 실행"}
            </button>
            <button
              onClick={handleGrade}
              disabled={grading || running || !userCode.trim() || !problem.test_cases}
              className="flex-1 rounded-xl bg-black border border-gray-700 py-4 text-lg font-bold text-white transition-all hover:border-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {grading ? "채점 중..." : "채점하기"}
            </button>
          </div>

          {/* 코드 실행 결과 */}
          {runOutput && (
            <div className="rounded-xl border border-gray-700 bg-gray-900 p-4 space-y-3">
              <h3 className="text-sm font-semibold text-gray-300">실행 결과</h3>
              {runOutput.error ? (
                <pre className="rounded-lg bg-red-500/10 p-3 text-sm text-red-400 whitespace-pre-wrap break-all">
                  {runOutput.error}
                </pre>
              ) : (
                <div className="space-y-2">
                  <div>
                    <span className="text-xs text-gray-500">내 출력</span>
                    <pre className="mt-1 rounded-lg bg-black p-3 text-sm text-gray-200 whitespace-pre-wrap break-all">
                      {runOutput.stdout || "(출력 없음)"}
                    </pre>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">기대 출력</span>
                    <pre className="mt-1 rounded-lg bg-black p-3 text-sm text-green-400 whitespace-pre-wrap break-all">
                      {runOutput.expected || "(없음)"}
                    </pre>
                  </div>
                  {runOutput.stdout.trim() === runOutput.expected.trim() ? (
                    <p className="text-sm text-green-400 font-semibold">✅ 출력이 일치합니다</p>
                  ) : (
                    <p className="text-sm text-red-400 font-semibold">❌ 출력이 다릅니다</p>
                  )}
                </div>
              )}
              {runOutput.stderr && (
                <div>
                  <span className="text-xs text-gray-500">stderr</span>
                  <pre className="mt-1 rounded-lg bg-yellow-500/10 p-3 text-sm text-yellow-400 whitespace-pre-wrap break-all">
                    {runOutput.stderr}
                  </pre>
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        /* ── 채점 후: 블록 쌓기 레이아웃 ── */
        <div className="space-y-4">

          {/* ▌1층: 채점 결과 스코어 카드 (풀 와이드) */}
          <GradeResult
            grade={result.grade}
            message={result.message}
            emoji={result.emoji}
            passedCount={result.passedCount}
            totalCount={result.totalCount}
          />

          {/* ▌2층: 내 코드 | 정답 코드/코드 비교 (좌우 균등 분할) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* 왼쪽: 내 코드 */}
            <div className="flex flex-col">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-300">내 코드</span>
                <button
                  onClick={handleRetry}
                  className="rounded-md bg-gray-700/80 border border-gray-600 px-3 py-1 text-xs text-gray-300 hover:bg-gray-600 transition-colors"
                >
                  다시 풀기
                </button>
              </div>
              <div className="flex-1 min-h-0">
                <CodeEditor
                  value={userCode}
                  onChange={setUserCode}
                  readOnly
                  height="360px"
                />
              </div>
            </div>

            {/* 오른쪽: 정답이면 정답 코드 / 틀리면 코드 비교 */}
            <div className="flex flex-col">
              <div className="mb-2">
                <span className="text-sm font-semibold text-gray-300">
                  {result.grade === "perfect" ? "정답 코드" : "코드 비교 (내 코드 vs 정답)"}
                </span>
              </div>
              <div className="flex-1 min-h-0">
                {result.grade === "perfect" ? (
                  <CodeEditor
                    value={problem.code}
                    onChange={() => {}}
                    readOnly
                    height="360px"
                  />
                ) : (
                  <div className="h-[360px] overflow-auto">
                    <CodeDiff userCode={userCode} answerCode={problem.code} />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ▌3층: 테스트 결과 상세 (풀 와이드, 가로 균등 그리드) */}
          <TestResults
            results={result.testResults}
            passedCount={result.passedCount}
            totalCount={result.totalCount}
          />

          {/* ▌4층: 다시 풀기 버튼 */}
          <button
            onClick={handleRetry}
            className="w-full rounded-xl bg-black border border-gray-700 py-4 text-lg font-bold text-white transition-all hover:border-gray-500"
          >
            다시 풀기
          </button>
        </div>
      )}
    </div>
  );
}
