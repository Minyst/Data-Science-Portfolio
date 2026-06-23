# Design-Implementation Gap Analysis Report

> **Summary**: Gap analysis of 5-phase refactoring plan for coding-test-web (Pyodide grading, Monaco editor, Notion sync)
>
> **Author**: bkit-gap-detector
> **Created**: 2026-03-15
> **Last Modified**: 2026-03-15
> **Status**: Draft

---

## Analysis Overview
- **Analysis Target**: AlgoViz coding-test-web major refactoring (5 phases)
- **Design Document**: `C:\Users\USER\.claude\plans\compiled-orbiting-harp.md`
- **Implementation Path**: `C:\Users\USER\Desktop\vibecoding\coding-test-web\src\`
- **Analysis Date**: 2026-03-15

---

## Overall Scores

| Category | Score | Status |
|----------|:-----:|:------:|
| Phase 1: Schema & Types | 100% | OK |
| Phase 2: Monaco Editor | 100% | OK |
| Phase 3: Pyodide + Output Grading | 95% | Warning |
| Phase 4: Notion Integration | 100% | OK |
| Phase 5: Solved Count Display | 70% | Problem |
| Leftover Cleanup | 60% | Problem |
| **Overall** | **87%** | Warning |

---

## Phase-by-Phase Analysis

### Phase 1: Schema & Types (100%)

| Requirement | Status | Notes |
|-------------|:------:|-------|
| `supabase/migrations/002_add_test_cases_and_images.sql` exists | OK | All 6 ALTER TABLE statements present with `IF NOT EXISTS` |
| `TestCase` interface in `types.ts` | OK | `{ input: string; expected: string }` |
| `TestCaseConfig` interface | OK | Includes `call_type` field (extends plan with `"function" \| "class" \| "script"`) |
| `TestResult` interface | OK | Has `error?` optional field (good addition) |
| `Problem` extended with `test_cases`, `images`, `notion_id` | OK | Matches plan |
| `Submission` extended with `passed_count`, `total_count`, `test_results` | OK | Matches plan |

### Phase 2: Monaco Editor (100%)

| Requirement | Status | Notes |
|-------------|:------:|-------|
| Uses `@monaco-editor/react` | OK | `package.json` has `"@monaco-editor/react": "^4.7.0"` |
| Dynamic import with SSR disabled | OK | `dynamic(() => import("@monaco-editor/react"), { ssr: false })` |
| Python language | OK | `language="python"` |
| vs-dark theme | OK | `theme="vs-dark"` |
| Minimap off | OK | `minimap: { enabled: false }` |
| tabSize 4 | OK | `tabSize: 4` |
| fontSize 14 | OK | `fontSize: 14` |
| Mac-style dot header | OK | Red/yellow/green dots preserved |
| Props interface maintained | OK | `value`, `onChange`, `placeholder`, `readOnly` + `height` |
| `handleKeyDown` removed | OK | No keyboard handler |

### Phase 3: Pyodide + Output-based Judging (95%)

| Requirement | Status | Notes |
|-------------|:------:|-------|
| `pyodide-runner.ts` exists | OK | Singleton pattern, CDN load, stdout capture, timeout |
| `test-runner.ts` exists | OK | function/class/script types, output comparison |
| `grading.ts` completely rewritten | OK | Only `gradeByTestResults`, no similarity/diff code |
| `GradeResult.tsx` shows test results | OK | Pass/fail per test, no diff view |
| `TestResults.tsx` new component | OK | Per-test display with input/expected/actual |
| `grade/route.ts` saves results only | OK | Receives `testResults`, `passedCount`, `totalCount` |
| Problem page has Pyodide loading indicator | OK | "Python loading..." message |
| Problem page has "Run" button | OK | `handleRun` with `executeCode` |
| Problem page has "Grade" button with test cases | OK | `handleGrade` with `runTestCases` |
| Output panel | OK | `runOutput` displayed in code block |
| `diff` package removed from package.json | OK | Not present in dependencies |
| `seed-test-cases.mjs` exists | OK | 15 problems with test case data |

**Minor Issue**: `grade/route.ts` line 17 still writes `similarity_score` field (computed from passedCount/totalCount ratio). This is backward-compatible but represents legacy naming.

### Phase 4: Notion Integration (100%)

| Requirement | Status | Notes |
|-------------|:------:|-------|
| `scripts/sync-from-notion.mjs` exists | OK | Full Notion API + Supabase upsert logic |
| `@notionhq/client` in package.json | OK | `"@notionhq/client": "^5.13.0"` |
| `.github/workflows/sync-notion.yml` exists | OK | Cron every 6 hours + manual trigger |
| Image download and Supabase Storage upload | OK | `uploadImageToSupabase` function |
| `notion_id`-based upsert | OK | Check existing, then update or insert |
| Environment variables documented | OK | `NOTION_API_KEY`, `NOTION_DATABASE_ID`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` |
| `next.config.ts` has Supabase Storage domain | OK | `gjpfmduphizgujaawmwc.supabase.co` configured |
| Problem page renders images | OK | `problem.images` array rendered with `<img>` tags |

### Phase 5: Main Page Solved Count (70%)

| Requirement | Status | Notes |
|-------------|:------:|-------|
| `CategoryStats.tsx` shows "N solved" in green | OK | `"{totalSolved} solved"` in green-600 |
| White card background + black text | OK | `bg-white`, `text-black` |
| Solved criteria changed to `is_correct` | MISSING | Still uses `scores[p.id] >= 95` (similarity-based) |
| `page.tsx` uses `is_correct` field | MISSING | Still queries `similarity_score`, uses `>= 95` threshold |

---

## Differences Found

### Missing Features (Design O, Implementation X)

| Item | Design Location | Description |
|------|-----------------|-------------|
| `is_correct`-based solved count | Plan Phase 5.1-5.2 | `page.tsx` and `CategoryStats.tsx` still use `similarity_score >= 95` instead of `is_correct === true` |

### Added Features (Design X, Implementation O)

| Item | Implementation Location | Description |
|------|------------------------|-------------|
| `call_type` field in TestCaseConfig | `src/lib/types.ts:8` | Plan only mentions `function_name` and `test_cases`, implementation adds `call_type: "function" \| "class" \| "script"` discriminator (good addition) |
| `error` field in TestResult | `src/lib/types.ts:17` | Optional error message per test (good addition) |
| `height` prop in CodeEditor | `src/components/CodeEditor.tsx:20` | Configurable editor height (good addition) |
| `normalizeOutput` in test-runner | `src/lib/test-runner.ts:69` | Quote and whitespace normalization for comparison (good addition) |
| `executeCode` function | `src/lib/test-runner.ts:77` | Simple code execution without grading (used by "Run" button) |

### Changed/Inconsistent Features (Design != Implementation)

| Item | Design | Implementation | Impact |
|------|--------|----------------|--------|
| Solved count logic | `is_correct === true` | `similarity_score >= 95` | **High** - old grading logic leaks through |
| `similarity_score` in Submission type | Should be removed or deprecated | Still present in `types.ts:39` | Medium - legacy field persists |
| `similarity_score` in grade API | Not mentioned in plan | Computed as `passedCount/totalCount * 100` in `route.ts:17` | Medium - backward compat but confusing |
| `scores` variable in page.tsx | Should query `is_correct` | Queries `similarity_score` | **High** - main page shows wrong solved count |

---

## Leftover Cleanup Issues

| File | Line | Issue |
|------|------|-------|
| `src/app/page.tsx` | 30-31 | Queries `similarity_score` from submissions instead of `is_correct` |
| `src/app/page.tsx` | 51 | Uses `>= 95` threshold for solved count |
| `src/components/CategoryStats.tsx` | 19 | Uses `scores[p.id] >= 95` instead of `is_correct` |
| `src/lib/types.ts` | 39 | `similarity_score: number` still in Submission interface |
| `src/app/api/grade/route.ts` | 17 | Still writes `similarity_score` field |
| `src/components/ProblemCard.tsx` | 43 | Uses `lastScore >= 95` for color coding |

---

## Recommended Actions

### Immediate Actions (Match Rate Impact)

1. **Rewrite `page.tsx` data fetching** - Query `is_correct` from submissions instead of `similarity_score`. Change solved count to `submissions.filter(s => s.is_correct).length`
2. **Update `CategoryStats.tsx`** - Accept submission correctness data (boolean map) instead of score map. Use `is_correct` for solved count
3. **Update `ProblemCard.tsx`** - Use `is_correct` boolean for status color instead of score threshold

### Documentation/Cleanup Actions

4. Consider removing `similarity_score` from `Submission` interface or marking it as `@deprecated`
5. The `similarity_score` write in `grade/route.ts` can be kept for backward compatibility but should be documented as legacy

### Estimated Match Rate After Fixes

Fixing items 1-3 would bring the match rate from **87% to ~97%**.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-15 | Initial gap analysis | bkit-gap-detector |
