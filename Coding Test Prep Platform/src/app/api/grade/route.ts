import { NextRequest, NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";

export async function POST(request: NextRequest) {
  const { problemId, userCode, testResults, passedCount, totalCount, isCorrect } =
    await request.json();

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    { cookies: { getAll: () => [], setAll: () => {} } }
  );

  await supabase.from("submissions").insert({
    problem_id: problemId,
    user_code: userCode,
    similarity_score: totalCount > 0 ? Math.round((passedCount / totalCount) * 100) : 0,
    is_correct: isCorrect,
    passed_count: passedCount,
    total_count: totalCount,
    test_results: testResults,
  });

  return NextResponse.json({ success: true });
}
