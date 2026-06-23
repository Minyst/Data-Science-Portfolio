import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const supabase = createClient(url, key);

async function fix() {
  const { data: problems, error } = await supabase
    .from("problems")
    .select("id, title, code, test_cases");

  if (error) {
    console.error("Error:", error);
    return;
  }

  let fixed = 0;

  for (const p of problems) {
    const updates = {};

    // ──────────────────────────────────────
    // FIX 1: BFS (Graph) — deque(start_v) → deque([start_v])
    // ──────────────────────────────────────
    if (p.title === "BFS (Graph)") {
      if (p.code.includes("deque(start_v)")) {
        updates.code = p.code.replace("deque(start_v)", "deque([start_v])");
        console.log(`🔧 [${p.title}] 정답 코드: deque(start_v) → deque([start_v])`);
      }
    }

    // ──────────────────────────────────────
    // FIX 2: Min Cost Climbing Stairs — memo 크기 버그 + 테스트 기대값
    // ──────────────────────────────────────
    if (p.title === "Min Cost Climbing Stairs") {
      // 정답 코드 수정: memo = [-1]*n → memo = [-1]*(n+1)
      if (p.code.includes("memo = [-1]*n")) {
        updates.code = p.code.replace("memo = [-1]*n", "memo = [-1]*(n+1)");
        console.log(`🔧 [${p.title}] 정답 코드: memo = [-1]*n → memo = [-1]*(n+1)`);
      }

      // 테스트 기대값 수정: cost=[10,15,20,17,1] → 정답은 31 (28은 오류)
      if (p.test_cases && p.test_cases.test_cases) {
        const tc = p.test_cases.test_cases;
        let testFixed = false;
        for (const t of tc) {
          if (t.input.includes("[10, 15, 20, 17, 1]") && t.expected === "28") {
            t.expected = "31";
            testFixed = true;
            console.log(`🔧 [${p.title}] 테스트 기대값: "28" → "31"`);
          }
        }
        if (testFixed) {
          updates.test_cases = p.test_cases;
        }
      }
    }

    // DB 업데이트
    if (Object.keys(updates).length > 0) {
      const { error: updateError } = await supabase
        .from("problems")
        .update(updates)
        .eq("id", p.id);

      if (updateError) {
        console.error(`❌ ${p.title}: ${updateError.message}`);
      } else {
        console.log(`✅ ${p.title} 수정 완료`);
        fixed++;
      }
    }
  }

  console.log(`\n${"=".repeat(50)}`);
  console.log(`총 ${fixed}개 문제 수정 완료`);
}

fix();
