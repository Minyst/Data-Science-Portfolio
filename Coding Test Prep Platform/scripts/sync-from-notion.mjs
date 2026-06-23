import { Client } from "@notionhq/client";
import { createClient } from "@supabase/supabase-js";

// Environment variables
const NOTION_API_KEY = process.env.NOTION_API_KEY;
const NOTION_DATABASE_ID = process.env.NOTION_DATABASE_ID;
const SUPABASE_URL = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!NOTION_API_KEY || !NOTION_DATABASE_ID) {
  console.error("Missing NOTION_API_KEY or NOTION_DATABASE_ID");
  console.error("Usage: NOTION_API_KEY=... NOTION_DATABASE_ID=... SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... node scripts/sync-from-notion.mjs");
  process.exit(1);
}

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY");
  process.exit(1);
}

const notion = new Client({ auth: NOTION_API_KEY });
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

/** Notion 데이터베이스의 모든 페이지 가져오기 */
async function fetchNotionPages() {
  const pages = [];
  let cursor = undefined;

  do {
    const response = await notion.databases.query({
      database_id: NOTION_DATABASE_ID,
      start_cursor: cursor,
    });
    pages.push(...response.results);
    cursor = response.has_more ? response.next_cursor : undefined;
  } while (cursor);

  return pages;
}

/** Notion 페이지의 블록(content) 가져오기 */
async function fetchPageBlocks(pageId) {
  const blocks = [];
  let cursor = undefined;

  do {
    const response = await notion.blocks.children.list({
      block_id: pageId,
      start_cursor: cursor,
    });
    blocks.push(...response.results);
    cursor = response.has_more ? response.next_cursor : undefined;
  } while (cursor);

  return blocks;
}

/** Rich text 배열에서 plain text 추출 */
function richTextToPlain(richTextArray) {
  if (!richTextArray) return "";
  return richTextArray.map((t) => t.plain_text).join("");
}

/** Notion 속성에서 값 추출 (다양한 타입 지원) */
function getPropertyValue(page, propertyName) {
  const prop = page.properties[propertyName];
  if (!prop) return null;

  switch (prop.type) {
    case "title":
      return richTextToPlain(prop.title);
    case "rich_text":
      return richTextToPlain(prop.rich_text);
    case "select":
      return prop.select?.name || null;
    case "multi_select":
      return prop.multi_select?.map((s) => s.name) || [];
    case "number":
      return prop.number;
    case "checkbox":
      return prop.checkbox;
    case "url":
      return prop.url;
    default:
      return null;
  }
}

/** 블록에서 코드, 설명, 이미지 추출 */
function extractContentFromBlocks(blocks) {
  let description = "";
  let code = "";
  let testCasesJson = null;
  const images = [];

  for (const block of blocks) {
    switch (block.type) {
      case "paragraph":
        description += richTextToPlain(block.paragraph.rich_text) + "\n";
        break;
      case "heading_1":
      case "heading_2":
      case "heading_3":
        description += richTextToPlain(block[block.type].rich_text) + "\n";
        break;
      case "bulleted_list_item":
        description += "• " + richTextToPlain(block.bulleted_list_item.rich_text) + "\n";
        break;
      case "numbered_list_item":
        description += "- " + richTextToPlain(block.numbered_list_item.rich_text) + "\n";
        break;
      case "code": {
        const codeText = richTextToPlain(block.code.rich_text);
        const caption = richTextToPlain(block.code.caption).toLowerCase();

        if (caption.includes("test") || caption.includes("테스트")) {
          // 테스트케이스 JSON
          try {
            testCasesJson = JSON.parse(codeText);
          } catch {
            console.warn("  ⚠️ 테스트케이스 JSON 파싱 실패");
          }
        } else if (!code) {
          // 첫 번째 코드 블록 = 정답 코드
          code = codeText;
        }
        break;
      }
      case "image": {
        const imageUrl =
          block.image.type === "file"
            ? block.image.file.url
            : block.image.external?.url;
        if (imageUrl) images.push(imageUrl);
        break;
      }
    }
  }

  return {
    description: description.trim(),
    code: code.trim(),
    test_cases: testCasesJson,
    images,
  };
}

/** 이미지를 Supabase Storage에 업로드 */
async function uploadImageToSupabase(imageUrl, problemTitle, index) {
  try {
    const response = await fetch(imageUrl);
    if (!response.ok) return null;

    const buffer = await response.arrayBuffer();
    const ext = imageUrl.includes(".png") ? "png" : "jpg";
    const filename = `${problemTitle.replace(/[^a-zA-Z0-9가-힣]/g, "_")}_${index}.${ext}`;
    const path = `problem-images/${filename}`;

    const { error } = await supabase.storage
      .from("problems")
      .upload(path, buffer, {
        contentType: `image/${ext}`,
        upsert: true,
      });

    if (error) {
      console.warn(`  ⚠️ 이미지 업로드 실패: ${error.message}`);
      return null;
    }

    const { data: urlData } = supabase.storage
      .from("problems")
      .getPublicUrl(path);

    return urlData.publicUrl;
  } catch (err) {
    console.warn(`  ⚠️ 이미지 다운로드 실패: ${err.message}`);
    return null;
  }
}

/** 메인 동기화 함수 */
async function sync() {
  console.log("🔄 Notion → Supabase 동기화 시작...\n");

  const pages = await fetchNotionPages();
  console.log(`📄 ${pages.length}개 페이지 발견\n`);

  let created = 0;
  let updated = 0;
  let failed = 0;

  for (const page of pages) {
    // Notion 속성에서 기본 정보 추출
    // 속성명은 Notion DB 구조에 따라 조정 필요
    const title =
      getPropertyValue(page, "Title") ||
      getPropertyValue(page, "Name") ||
      getPropertyValue(page, "이름") ||
      getPropertyValue(page, "제목") ||
      "";

    if (!title) {
      console.log(`⏭️  제목 없는 페이지 건너뜀 (${page.id})`);
      continue;
    }

    const category =
      getPropertyValue(page, "Category") ||
      getPropertyValue(page, "카테고리") ||
      "Uncategorized";

    const difficulty =
      getPropertyValue(page, "Difficulty") ||
      getPropertyValue(page, "난이도") ||
      "Medium";

    // 페이지 블록에서 상세 내용 추출
    const blocks = await fetchPageBlocks(page.id);
    const content = extractContentFromBlocks(blocks);

    if (!content.code) {
      console.log(`⏭️  ${title} - 코드 블록 없음, 건너뜀`);
      continue;
    }

    // 이미지 업로드 (Notion URL → Supabase Storage)
    const permanentImages = [];
    for (let i = 0; i < content.images.length; i++) {
      const uploaded = await uploadImageToSupabase(content.images[i], title, i);
      if (uploaded) permanentImages.push(uploaded);
    }

    // Supabase upsert (notion_id 기반)
    const problemData = {
      notion_id: page.id,
      title,
      category,
      difficulty,
      code: content.code,
      description: content.description || null,
      test_cases: content.test_cases || null,
      images: permanentImages.length > 0 ? permanentImages : null,
    };

    // notion_id로 기존 문제 확인
    const { data: existing } = await supabase
      .from("problems")
      .select("id")
      .eq("notion_id", page.id)
      .single();

    if (existing) {
      const { error } = await supabase
        .from("problems")
        .update(problemData)
        .eq("notion_id", page.id);

      if (error) {
        console.error(`❌ ${title}: ${error.message}`);
        failed++;
      } else {
        console.log(`🔄 ${title} (업데이트)`);
        updated++;
      }
    } else {
      const { error } = await supabase.from("problems").insert(problemData);

      if (error) {
        console.error(`❌ ${title}: ${error.message}`);
        failed++;
      } else {
        console.log(`✅ ${title} (새로 추가)`);
        created++;
      }
    }
  }

  console.log(`\n📊 동기화 완료: 추가 ${created}, 업데이트 ${updated}, 실패 ${failed}`);
}

sync().catch(console.error);
