import type { TestResult } from "./types";

export interface GradeResult {
  grade: "perfect" | "partial" | "wrong";
  message: string;
  emoji: string;
  passedCount: number;
  totalCount: number;
  isCorrect: boolean;
}

/** 테스트 결과 기반 채점 */
export function gradeByTestResults(testResults: TestResult[]): GradeResult {
  const passedCount = testResults.filter((r) => r.passed).length;
  const totalCount = testResults.length;
  const isCorrect = passedCount === totalCount && totalCount > 0;

  if (isCorrect) {
    return {
      grade: "perfect",
      message: `정답입니다! 모든 테스트를 통과했습니다.`,
      emoji: "🎉",
      passedCount,
      totalCount,
      isCorrect: true,
    };
  }

  if (passedCount > 0) {
    return {
      grade: "partial",
      message: `${totalCount}개 중 ${passedCount}개 테스트를 통과했습니다.`,
      emoji: "🤔",
      passedCount,
      totalCount,
      isCorrect: false,
    };
  }

  return {
    grade: "wrong",
    message: "테스트를 통과하지 못했습니다. 다시 시도해보세요!",
    emoji: "💪",
    passedCount: 0,
    totalCount,
    isCorrect: false,
  };
}
