export interface QuizAnswer {
  problemId: string;
  problemTitle: string;
  category: string;
  correct: boolean;
  userCode: string;
  answerCode: string;
}

export interface DailyQuizResult {
  date: string; // YYYY-MM-DD
  answers: QuizAnswer[];
  completedAt: string;
}

export interface WrongAnswer {
  problemId: string;
  problemTitle: string;
  category: string;
  userCode: string;
  answerCode: string;
  lastDate: string; // 마지막으로 틀린 날짜
}

const DAILY_KEY_PREFIX = "quiz-daily-";
const WRONG_KEY = "quiz-wrong-answers";

export function getTodayKey(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function getDailyResult(date: string): DailyQuizResult | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(DAILY_KEY_PREFIX + date);
  return raw ? JSON.parse(raw) : null;
}

export function saveDailyResult(result: DailyQuizResult) {
  localStorage.setItem(DAILY_KEY_PREFIX + result.date, JSON.stringify(result));
  // 틀린 문제를 오답노트에 저장
  const wrongs = getWrongAnswers();
  for (const ans of result.answers) {
    if (!ans.correct) {
      wrongs[ans.problemId] = {
        problemId: ans.problemId,
        problemTitle: ans.problemTitle,
        category: ans.category,
        userCode: ans.userCode,
        answerCode: ans.answerCode,
        lastDate: result.date,
      };
    }
  }
  localStorage.setItem(WRONG_KEY, JSON.stringify(wrongs));
}

export function getWrongAnswers(): Record<string, WrongAnswer> {
  if (typeof window === "undefined") return {};
  const raw = localStorage.getItem(WRONG_KEY);
  return raw ? JSON.parse(raw) : {};
}

export function removeWrongAnswer(problemId: string) {
  const wrongs = getWrongAnswers();
  delete wrongs[problemId];
  localStorage.setItem(WRONG_KEY, JSON.stringify(wrongs));
}

/** 특정 월의 퀴즈 결과 목록 가져오기 */
export function getMonthResults(year: number, month: number): Record<string, DailyQuizResult> {
  if (typeof window === "undefined") return {};
  const results: Record<string, DailyQuizResult> = {};
  const prefix = DAILY_KEY_PREFIX;
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(prefix)) {
      const date = key.slice(prefix.length);
      const [y, m] = date.split("-").map(Number);
      if (y === year && m === month) {
        const raw = localStorage.getItem(key);
        if (raw) results[date] = JSON.parse(raw);
      }
    }
  }
  return results;
}
