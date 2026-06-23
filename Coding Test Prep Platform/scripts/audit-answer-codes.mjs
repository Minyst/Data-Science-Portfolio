import { createClient } from "@supabase/supabase-js";

const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const supabase = createClient(url, key);

async function audit() {
  const { data: problems, error } = await supabase
    .from("problems")
    .select("id, title, code, test_cases, category")
    .order("category");

  if (error) {
    console.error("Error:", error);
    return;
  }

  console.log(`\n총 ${problems.length}개 문제 점검\n`);
  console.log("=".repeat(80));

  for (const p of problems) {
    console.log(`\n📌 [${p.category}] ${p.title}`);
    console.log("-".repeat(60));
    console.log(p.code);
    console.log("-".repeat(60));
    if (p.test_cases) {
      console.log(`call_type: ${p.test_cases.call_type}`);
      console.log(`function_name: ${p.test_cases.function_name}`);
      console.log(`test_cases: ${p.test_cases.test_cases.length}개`);
    } else {
      console.log("⚠️ 테스트케이스 없음");
    }
    console.log("=".repeat(80));
  }
}

audit();
