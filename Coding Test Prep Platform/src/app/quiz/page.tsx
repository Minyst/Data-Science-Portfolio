"use client";

import { useEffect, useState, useCallback } from "react";
import { createClient } from "@/lib/supabase/client";
import type { Problem, TestResult } from "@/lib/types";
import type { GradeResult as GradeResultType } from "@/lib/grading";
import CodeEditor from "@/components/CodeEditor";
import GradeResult from "@/components/GradeResult";
import TestResults from "@/components/TestResults";
import CodeDiff from "@/components/CodeDiff";
import { getProblemImage } from "@/lib/problem-images";
import {
  getTodayKey,
  getDailyResult,
  saveDailyResult,
  getMonthResults,
  getWrongAnswers,
  removeWrongAnswer,
  type QuizAnswer,
  type DailyQuizResult,
  type WrongAnswer,
} from "@/lib/quiz-storage";

type Tab = "quiz" | "calendar" | "wrong";

export default function QuizPage() {
  const [tab, setTab] = useState<Tab>("quiz");
  const [quizProblems, setQuizProblems] = useState<Problem[]>([]);
  const [loading, setLoading] = useState(true);
  const [todayDone, setTodayDone] = useState(false);
  const [todayResult, setTodayResult] = useState<DailyQuizResult | null>(null);

  // Quiz state
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userCodes, setUserCodes] = useState<string[]>(["", ""]);
  const [gradeResults, setGradeResults] = useState<((GradeResultType & { testResults: TestResult[] }) | null)[]>([null, null]);
  const [grading, setGrading] = useState(false);
  const [quizFinished, setQuizFinished] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);

  // Calendar state
  const [calYear, setCalYear] = useState(new Date().getFullYear());
  const [calMonth, setCalMonth] = useState(new Date().getMonth() + 1);
  const [monthResults, setMonthResults] = useState<Record<string, DailyQuizResult>>({});

  // Wrong answers
  const [wrongAnswers, setWrongAnswers] = useState<Record<string, WrongAnswer>>({});
  const [expandedWrong, setExpandedWrong] = useState<string | null>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase
      .from("problems")
      .select("*")
      .not("test_cases", "is", null)
      .then(({ data }) => {
        const all = data || [];

        const today = getTodayKey();
        const existing = getDailyResult(today);
        if (existing) {
          setTodayDone(true);
          setTodayResult(existing);
        } else if (all.length >= 2) {
          const shuffled = [...all].sort(() => Math.random() - 0.5);
          setQuizProblems(shuffled.slice(0, 2));
        } else {
          setQuizProblems(all.slice(0, 2));
        }

        setLoading(false);
      });
  }, []);

  useEffect(() => {
    setMonthResults(getMonthResults(calYear, calMonth));
  }, [calYear, calMonth]);

  useEffect(() => {
    setWrongAnswers(getWrongAnswers());
  }, [tab]);

  const loadPyodide = useCallback(async () => {
    if (pyodideReady) return;
    const { getPyodide } = await import("@/lib/pyodide-runner");
    await getPyodide();
    setPyodideReady(true);
  }, [pyodideReady]);

  useEffect(() => {
    if (quizProblems.length > 0) loadPyodide();
  }, [quizProblems, loadPyodide]);

  const handleGradeOne = async (index: number) => {
    const problem = quizProblems[index];
    const code = userCodes[index];
    if (!problem || !code.trim() || !problem.test_cases) return;

    setGrading(true);
    try {
      if (!pyodideReady) await loadPyodide();
      const { runTestCases } = await import("@/lib/test-runner");
      const { gradeByTestResults } = await import("@/lib/grading");

      const runResult = await runTestCases(code, problem.test_cases);
      const gradeResult = gradeByTestResults(runResult.testResults);

      const newResults = [...gradeResults];
      newResults[index] = { ...gradeResult, testResults: runResult.testResults };
      setGradeResults(newResults);
    } catch {
      // 에러 시에도 결과 표시
      const newResults = [...gradeResults];
      newResults[index] = {
        grade: "wrong",
        message: "채점 중 오류가 발생했습니다.",
        emoji: "💪",
        passedCount: 0,
        totalCount: 1,
        isCorrect: false,
        testResults: [],
      };
      setGradeResults(newResults);
    } finally {
      setGrading(false);
    }
  };

  const handleRetryOne = (index: number) => {
    const newResults = [...gradeResults];
    newResults[index] = null;
    setGradeResults(newResults);
  };

  const finishQuiz = () => {
    const today = getTodayKey();
    const answers: QuizAnswer[] = quizProblems.map((p, i) => ({
      problemId: p.id,
      problemTitle: p.title,
      category: p.category,
      correct: gradeResults[i]?.grade === "perfect",
      userCode: userCodes[i],
      answerCode: p.code,
    }));
    const result: DailyQuizResult = {
      date: today,
      answers,
      completedAt: new Date().toISOString(),
    };
    saveDailyResult(result);
    setTodayDone(true);
    setTodayResult(result);
    setQuizFinished(true);
  };

  // Calendar helpers
  const daysInMonth = new Date(calYear, calMonth, 0).getDate();
  const firstDayOfWeek = new Date(calYear, calMonth - 1, 1).getDay();
  const calDays = Array.from({ length: daysInMonth }, (_, i) => i + 1);
  const padDays = Array.from({ length: firstDayOfWeek }, () => 0);

  const prevMonth = () => {
    if (calMonth === 1) { setCalYear(calYear - 1); setCalMonth(12); }
    else setCalMonth(calMonth - 1);
  };
  const nextMonth = () => {
    if (calMonth === 12) { setCalYear(calYear + 1); setCalMonth(1); }
    else setCalMonth(calMonth + 1);
  };

  const currentProblem = quizProblems[currentIndex];
  const currentResult = gradeResults[currentIndex];
  const gradedCount = gradeResults.filter((r) => r !== null).length;
  const allGraded = gradedCount === quizProblems.length && quizProblems.length > 0;

  if (loading) return <div className="py-20 text-center text-gray-500">로딩 중...</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Daily Quiz</h1>

      {/* Tabs */}
      <div className="flex gap-2">
        {(["quiz", "calendar", "wrong"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${
              tab === t
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {t === "quiz" ? "오늘의 퀴즈" : t === "calendar" ? "달력" : "오답노트"}
          </button>
        ))}
      </div>

      {/* ════════════════ Quiz Tab ════════════════ */}
      {tab === "quiz" && (
        <div className="space-y-6">
          {todayDone && todayResult ? (
            /* ── 오늘 퀴즈 완료 상태 ── */
            <div className="space-y-4">
              <div className="rounded-xl border border-gray-700 bg-gray-900 p-6 text-center">
                <p className="text-xl font-bold">오늘의 퀴즈 완료!</p>
                <p className="mt-2 text-gray-400">
                  {todayResult.answers.filter((a) => a.correct).length} / {todayResult.answers.length} 정답
                </p>
              </div>
              {todayResult.answers.map((ans, i) => (
                <div
                  key={ans.problemId}
                  className={`rounded-xl border p-4 ${
                    ans.correct ? "border-green-700 bg-green-900/20" : "border-red-700 bg-red-900/20"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">
                      {i + 1}. {ans.problemTitle}
                    </span>
                    <span className={`text-sm font-bold ${ans.correct ? "text-green-400" : "text-red-400"}`}>
                      {ans.correct ? "정답" : "오답"}
                    </span>
                  </div>
                  <span className="mt-1 inline-block rounded-full bg-gray-700/50 px-2 py-0.5 text-xs text-gray-400">
                    {ans.category}
                  </span>
                </div>
              ))}
            </div>
          ) : quizProblems.length === 0 ? (
            <div className="rounded-xl border border-gray-700 bg-gray-900 p-6 text-center">
              <p className="text-gray-400">테스트케이스가 있는 문제가 부족합니다.</p>
            </div>
          ) : (
            <>
              {/* ── Progress indicator ── */}
              <div className="flex items-center gap-3">
                {quizProblems.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => setCurrentIndex(i)}
                    className={`flex h-10 w-10 items-center justify-center rounded-full text-sm font-bold transition-colors ${
                      gradeResults[i]?.grade === "perfect"
                        ? "bg-green-600 text-white"
                        : gradeResults[i] !== null
                        ? "bg-red-600 text-white"
                        : currentIndex === i
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 text-gray-300"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
                <span className="ml-auto text-sm text-gray-400">
                  {gradedCount} / {quizProblems.length} 채점됨
                </span>
              </div>

              {/* ── 현재 문제 (기존 문제 페이지와 동일한 레이아웃) ── */}
              {currentProblem && (
                <div className="space-y-6">
                  {/* Header */}
                  <div>
                    <h2 className="text-3xl font-bold">{currentProblem.title}</h2>
                    <div className="mt-3 flex gap-2">
                      <span className="rounded-full bg-blue-500/15 px-3 py-1 text-sm text-blue-400">
                        {currentProblem.category}
                      </span>
                      <span className="rounded-full bg-yellow-500/15 px-3 py-1 text-sm text-yellow-400">
                        {currentProblem.difficulty}
                      </span>
                    </div>
                    {currentProblem.description && (
                      <p className="mt-3 text-sm text-gray-400">{currentProblem.description}</p>
                    )}
                  </div>

                  {/* Problem Image */}
                  {getProblemImage(currentProblem.title) && (
                    <div>
                      <img
                        src={getProblemImage(currentProblem.title)!}
                        alt={`${currentProblem.title} 문제`}
                        className="rounded-lg max-w-full border border-gray-700"
                      />
                    </div>
                  )}

                  {/* ── 채점 전: 에디터 + 채점 버튼 ── */}
                  {!currentResult ? (
                    <>
                      <div>
                        <h3 className="mb-3 text-lg font-semibold">풀이 작성</h3>
                        <CodeEditor
                          value={userCodes[currentIndex]}
                          onChange={(v) => {
                            const next = [...userCodes];
                            next[currentIndex] = v;
                            setUserCodes(next);
                          }}
                          placeholder="여기에 코드를 작성하고 채점하기를 누르세요..."
                        />
                      </div>
                      <button
                        onClick={() => handleGradeOne(currentIndex)}
                        disabled={grading || !userCodes[currentIndex].trim()}
                        className="w-full rounded-xl bg-black border border-gray-700 py-4 text-lg font-bold text-white transition-all hover:border-gray-500 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {grading ? "채점 중..." : "채점하기"}
                      </button>
                    </>
                  ) : (
                    /* ── 채점 후: 기존 문제 페이지와 동일한 블록 쌓기 레이아웃 ── */
                    <div className="space-y-4">

                      {/* 1층: 채점 결과 스코어 카드 */}
                      <GradeResult
                        grade={currentResult.grade}
                        message={currentResult.message}
                        emoji={currentResult.emoji}
                        passedCount={currentResult.passedCount}
                        totalCount={currentResult.totalCount}
                      />

                      {/* 2층: 내 코드 | 정답 코드 (좌우 균등 분할) */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {/* 왼쪽: 내 코드 */}
                        <div className="flex flex-col">
                          <div className="mb-2 flex items-center justify-between">
                            <span className="text-sm font-semibold text-gray-300">내 코드</span>
                            <button
                              onClick={() => handleRetryOne(currentIndex)}
                              className="rounded-md bg-gray-700/80 border border-gray-600 px-3 py-1 text-xs text-gray-300 hover:bg-gray-600 transition-colors"
                            >
                              다시 풀기
                            </button>
                          </div>
                          <div className="flex-1 min-h-0">
                            <CodeEditor
                              value={userCodes[currentIndex]}
                              onChange={() => {}}
                              readOnly
                              height="360px"
                            />
                          </div>
                        </div>

                        {/* 오른쪽: 정답이면 정답 코드 / 틀리면 코드 비교 */}
                        <div className="flex flex-col">
                          <div className="mb-2">
                            <span className="text-sm font-semibold text-gray-300">
                              {currentResult.grade === "perfect" ? "정답 코드" : "코드 비교 (내 코드 vs 정답)"}
                            </span>
                          </div>
                          <div className="flex-1 min-h-0">
                            {currentResult.grade === "perfect" ? (
                              <CodeEditor
                                value={currentProblem.code}
                                onChange={() => {}}
                                readOnly
                                height="360px"
                              />
                            ) : (
                              <div className="h-[360px] overflow-auto">
                                <CodeDiff userCode={userCodes[currentIndex]} answerCode={currentProblem.code} />
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* 3층: 테스트 결과 상세 */}
                      <TestResults
                        results={currentResult.testResults}
                        passedCount={currentResult.passedCount}
                        totalCount={currentResult.totalCount}
                      />

                      {/* 4층: 다시 풀기 */}
                      <button
                        onClick={() => handleRetryOne(currentIndex)}
                        className="w-full rounded-xl bg-black border border-gray-700 py-4 text-lg font-bold text-white transition-all hover:border-gray-500"
                      >
                        다시 풀기
                      </button>
                    </div>
                  )}

                  {/* ── Navigation ── */}
                  <div className="flex gap-3">
                    {currentIndex > 0 && (
                      <button
                        onClick={() => setCurrentIndex(currentIndex - 1)}
                        className="flex-1 rounded-lg bg-gray-800 py-2 text-sm font-semibold text-gray-300 hover:bg-gray-700 transition-colors"
                      >
                        ← 이전 문제
                      </button>
                    )}
                    {currentIndex < quizProblems.length - 1 && (
                      <button
                        onClick={() => setCurrentIndex(currentIndex + 1)}
                        className="flex-1 rounded-lg bg-gray-800 py-2 text-sm font-semibold text-gray-300 hover:bg-gray-700 transition-colors"
                      >
                        다음 문제 →
                      </button>
                    )}
                  </div>

                  {/* Finish button */}
                  {allGraded && !quizFinished && (
                    <button
                      onClick={finishQuiz}
                      className="w-full rounded-xl bg-blue-600 py-4 text-lg font-bold text-white hover:bg-blue-500 transition-colors"
                    >
                      퀴즈 완료
                    </button>
                  )}
                  {gradedCount > 0 && !allGraded && (
                    <button
                      onClick={finishQuiz}
                      className="w-full rounded-xl bg-gray-700 py-3 text-sm font-semibold text-gray-300 hover:bg-gray-600 transition-colors"
                    >
                      나머지 문제 건너뛰고 완료
                    </button>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ════════════════ Calendar Tab ════════════════ */}
      {tab === "calendar" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <button onClick={prevMonth} className="rounded-lg bg-gray-800 px-3 py-1 text-sm text-gray-300 hover:bg-gray-700">
              ←
            </button>
            <span className="text-lg font-bold">{calYear}년 {calMonth}월</span>
            <button onClick={nextMonth} className="rounded-lg bg-gray-800 px-3 py-1 text-sm text-gray-300 hover:bg-gray-700">
              →
            </button>
          </div>

          <div className="grid grid-cols-7 gap-1 text-center text-sm">
            {["일", "월", "화", "수", "목", "금", "토"].map((d) => (
              <div key={d} className="py-2 text-gray-500 font-medium">{d}</div>
            ))}
            {padDays.map((_, i) => (
              <div key={`pad-${i}`} />
            ))}
            {calDays.map((day) => {
              const dateStr = `${calYear}-${String(calMonth).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
              const result = monthResults[dateStr];
              const isToday = dateStr === getTodayKey();
              const correctCount = result?.answers.filter((a) => a.correct).length ?? 0;
              const totalCount = result?.answers.length ?? 0;

              return (
                <div
                  key={day}
                  className={`relative rounded-lg py-2 text-sm ${
                    isToday ? "ring-2 ring-blue-500" : ""
                  } ${result ? "bg-gray-800" : ""}`}
                >
                  <span className={isToday ? "font-bold text-blue-400" : "text-gray-300"}>
                    {day}
                  </span>
                  {result && (
                    <div className="mt-0.5 text-xs">
                      <span className={correctCount === totalCount ? "text-green-400" : correctCount > 0 ? "text-yellow-400" : "text-red-400"}>
                        {correctCount}/{totalCount}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="rounded-xl border border-gray-700 bg-gray-900 p-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">이번 달 기록</h3>
            {Object.keys(monthResults).length === 0 ? (
              <p className="text-sm text-gray-500">이번 달 퀴즈 기록이 없습니다.</p>
            ) : (
              <div className="space-y-2">
                {Object.entries(monthResults)
                  .sort(([a], [b]) => b.localeCompare(a))
                  .map(([date, r]) => (
                    <div key={date} className="flex items-center justify-between rounded-lg bg-gray-800 px-3 py-2 text-sm">
                      <span className="text-gray-300">{date}</span>
                      <div className="flex gap-2">
                        {r.answers.map((a, i) => (
                          <span
                            key={i}
                            className={`rounded px-2 py-0.5 text-xs ${
                              a.correct ? "bg-green-900/50 text-green-400" : "bg-red-900/50 text-red-400"
                            }`}
                          >
                            {a.problemTitle.length > 15 ? a.problemTitle.slice(0, 15) + "..." : a.problemTitle}
                            {a.correct ? " ✓" : " ✗"}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ════════════════ Wrong Answers Tab ════════════════ */}
      {tab === "wrong" && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold">오답노트</h2>
          {Object.keys(wrongAnswers).length === 0 ? (
            <div className="rounded-xl border border-gray-700 bg-gray-900 p-6 text-center">
              <p className="text-gray-400">틀린 문제가 없습니다. 대단해요!</p>
            </div>
          ) : (
            Object.values(wrongAnswers)
              .sort((a, b) => b.lastDate.localeCompare(a.lastDate))
              .map((w) => (
                <div key={w.problemId} className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden">
                  <button
                    onClick={() => setExpandedWrong(expandedWrong === w.problemId ? null : w.problemId)}
                    className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-800 transition-colors"
                  >
                    <div>
                      <span className="font-semibold text-white">{w.problemTitle}</span>
                      <div className="mt-1 flex gap-2">
                        <span className="rounded-full bg-blue-500/15 px-2 py-0.5 text-xs text-blue-400">{w.category}</span>
                        <span className="text-xs text-gray-500">마지막 오답: {w.lastDate}</span>
                      </div>
                    </div>
                    <span className="text-gray-500">{expandedWrong === w.problemId ? "▲" : "▼"}</span>
                  </button>
                  {expandedWrong === w.problemId && (
                    <div className="border-t border-gray-700 p-4 space-y-3">
                      <div>
                        <span className="text-sm font-semibold text-red-400">내가 작성한 코드</span>
                        <pre className="mt-1 rounded-lg bg-black p-3 text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                          {w.userCode || "(코드 없음)"}
                        </pre>
                      </div>
                      <div>
                        <span className="text-sm font-semibold text-green-400">정답 코드</span>
                        <pre className="mt-1 rounded-lg bg-black p-3 text-sm text-gray-300 overflow-x-auto whitespace-pre-wrap">
                          {w.answerCode}
                        </pre>
                      </div>
                      <button
                        onClick={() => {
                          removeWrongAnswer(w.problemId);
                          setWrongAnswers(getWrongAnswers());
                        }}
                        className="rounded-lg bg-gray-700 px-3 py-1 text-xs text-gray-300 hover:bg-gray-600 transition-colors"
                      >
                        오답노트에서 제거
                      </button>
                    </div>
                  )}
                </div>
              ))
          )}
        </div>
      )}
    </div>
  );
}
