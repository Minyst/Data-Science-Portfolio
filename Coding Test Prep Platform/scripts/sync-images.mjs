/**
 * Notion 이미지 → Supabase Storage 동기화 스크립트
 * 환경변수: NOTION_API_KEY, NOTION_PAGE_ID, NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY
 */

const NOTION_API_KEY = process.env.NOTION_API_KEY;
const PAGE_ID = process.env.NOTION_PAGE_ID;
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!NOTION_API_KEY || !PAGE_ID || !SUPABASE_URL || !SUPABASE_KEY) {
  console.error("필수 환경변수가 없습니다. .env.local에 다음을 설정하세요:");
  console.error("  NOTION_API_KEY, NOTION_PAGE_ID, NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY");
  process.exit(1);
}

/** Notion 페이지의 모든 블록 가져오기 */
async function fetchAllBlocks() {
  const blocks = [];
  let cursor;
  do {
    const url = `https://api.notion.com/v1/blocks/${PAGE_ID}/children?page_size=100${cursor ? "&start_cursor=" + cursor : ""}`;
    const res = await fetch(url, {
      headers: { Authorization: `Bearer ${NOTION_API_KEY}`, "Notion-Version": "2022-06-28" },
    });
    const data = await res.json();
    blocks.push(...data.results);
    cursor = data.has_more ? data.next_cursor : null;
  } while (cursor);
  return blocks;
}

/** Supabase에서 모든 문제 가져오기 */
async function fetchProblems() {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/problems?select=id,title,code,images&order=created_at`, {
    headers: { apikey: SUPABASE_KEY, Authorization: `Bearer ${SUPABASE_KEY}` },
  });
  return res.json();
}

/** 코드 정규화: 주석 제거 + 공백 통일 */
function normalizeCode(code) {
  return code
    .split("\n")
    .filter((line) => !line.trim().startsWith("#")) // 주석 제거
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

/** 함수/클래스명 추출 */
function extractNames(code) {
  const names = [];
  const funcMatch = code.matchAll(/def\s+(\w+)/g);
  for (const m of funcMatch) names.push(m[1]);
  const classMatch = code.matchAll(/class\s+(\w+)/g);
  for (const m of classMatch) names.push(m[1]);
  return names;
}

/** 두 코드가 같은 문제인지 비교 */
function codesMatch(notionCode, supabaseCode) {
  const normNotion = normalizeCode(notionCode);
  const normSupabase = normalizeCode(supabaseCode);

  // 1. 정규화 후 앞부분 비교
  if (normNotion.substring(0, 80) === normSupabase.substring(0, 80)) return true;

  // 2. 함수/클래스명 비교 + 코드 일부 포함 여부
  const notionNames = extractNames(notionCode);
  const supabaseNames = extractNames(supabaseCode);
  if (notionNames.length > 0 && supabaseNames.length > 0) {
    const commonNames = notionNames.filter((n) => supabaseNames.includes(n));
    if (commonNames.length > 0) {
      // 주요 로직 키워드도 비교
      const notionBody = normNotion.substring(0, 200);
      const supabaseBody = normSupabase.substring(0, 200);
      if (notionBody.includes(supabaseBody.substring(0, 60)) || supabaseBody.includes(notionBody.substring(0, 60))) {
        return true;
      }
    }
  }

  // 3. 긴 부분 문자열 포함 여부
  const chunk = normSupabase.substring(0, 120);
  if (chunk.length > 40 && normNotion.includes(chunk)) return true;

  return false;
}

/** 이미지를 Supabase Storage에 업로드 */
async function uploadImage(imageUrl, filename) {
  const imgRes = await fetch(imageUrl);
  if (!imgRes.ok) {
    console.warn(`  ⚠️ 이미지 다운로드 실패: ${imgRes.status}`);
    return null;
  }
  const buffer = await imgRes.arrayBuffer();
  const uint8 = new Uint8Array(buffer);
  const path = `problem-images/${filename}`;

  const uploadRes = await fetch(`${SUPABASE_URL}/storage/v1/object/problems/${path}`, {
    method: "PUT",
    headers: {
      Authorization: `Bearer ${SUPABASE_KEY}`,
      "Content-Type": "image/png",
      "x-upsert": "true",
    },
    body: uint8,
  });

  if (!uploadRes.ok) {
    // 이미 존재하면 기존 URL 반환
    console.log(`  ℹ️ 이미 Storage에 존재, DB만 업데이트`);
  }

  return `${SUPABASE_URL}/storage/v1/object/public/problems/${path}`;
}

/** Supabase 문제 업데이트 */
async function updateProblemImages(problemId, images) {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/problems?id=eq.${problemId}`, {
    method: "PATCH",
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify({ images }),
  });
  return res.ok;
}

async function main() {
  console.log("🔄 Notion 이미지 → Supabase 동기화 시작...\n");

  const [blocks, problems] = await Promise.all([fetchAllBlocks(), fetchProblems()]);
  console.log(`📄 Notion 블록: ${blocks.length}개`);
  console.log(`📦 Supabase 문제: ${problems.length}개\n`);

  // Notion 블록에서 [이미지, 코드] 쌍 추출
  // 패턴: image → heading_3("💻 최종 코드") → code
  const pairs = [];
  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i].type === "image") {
      for (let j = i + 1; j < Math.min(i + 4, blocks.length); j++) {
        if (blocks[j].type === "code") {
          const imgUrl = blocks[i].image.type === "file" ? blocks[i].image.file.url : blocks[i].image.external?.url;
          const codeText = blocks[j].code.rich_text.map((t) => t.plain_text).join("");
          if (imgUrl && codeText) {
            pairs.push({ imageUrl: imgUrl, code: codeText });
          }
          break;
        }
      }
    }
  }

  console.log(`🔗 이미지-코드 쌍: ${pairs.length}개\n`);

  let matched = 0, uploaded = 0, failed = 0;
  const matchedIds = new Set();

  for (const pair of pairs) {
    const problem = problems.find((p) => !matchedIds.has(p.id) && codesMatch(pair.code, p.code));

    if (!problem) {
      console.log(`⏭️  매칭 실패 (code: ${pair.code.substring(0, 50).replace(/\n/g, " ")}...)`);
      failed++;
      continue;
    }

    matchedIds.add(problem.id);
    matched++;
    console.log(`✅ 매칭: ${problem.title}`);

    const safeName = problem.title.replace(/[^a-zA-Z0-9가-힣_-]/g, "_");
    const filename = `${safeName}_0.png`;
    const publicUrl = await uploadImage(pair.imageUrl, filename);

    if (publicUrl) {
      const updated = await updateProblemImages(problem.id, [publicUrl]);
      if (updated) {
        console.log(`  📤 업로드 완료: ${filename}`);
        uploaded++;
      } else {
        console.log(`  ❌ DB 업데이트 실패`);
      }
    }
  }

  // 매칭 안 된 문제 리포트
  const unmatched = problems.filter((p) => !matchedIds.has(p.id) && !p.images);
  if (unmatched.length > 0) {
    console.log(`\n⚠️ Notion에 이미지 없는 문제 (${unmatched.length}개):`);
    unmatched.forEach((p) => console.log(`  - ${p.title}`));
  }

  console.log(`\n📊 결과: 매칭 ${matched}, 업로드 ${uploaded}, 실패 ${failed}`);
}

main().catch(console.error);
