# Coding Test Prep Platform

비전공자도 자료구조·알고리즘을 쉽게 이해하고 코딩테스트를 준비할 수 있도록 돕는 올인원 학습 웹앱입니다.
브라우저에서 Python 코드를 직접 작성·실행하고, 자동 채점 결과를 즉시 확인할 수 있습니다.

## Features
- **문제 풀이 (Practice)**: 난이도별 코딩테스트 문제, Monaco 기반 코드 에디터
- **브라우저 내 실행/채점**: Pyodide로 클라이언트에서 Python 실행, 테스트 케이스 기반 자동 채점
- **퀴즈 (Quiz)**: 개념 점검용 퀴즈 모드
- **문제 관리**: 문제 등록/편집, 이미지 첨부, 카테고리 필터 및 통계
- **콘텐츠 동기화**: Notion → Supabase 시드/싱크 스크립트로 문제 데이터 관리

## Tech Stack
- **Framework**: Next.js 15 (App Router) + TypeScript
- **Styling**: Tailwind CSS
- **Code Editor / Execution**: Monaco Editor + Pyodide (브라우저 내 Python 실행)
- **Backend**: Supabase (PostgreSQL, Auth, Storage)
- **Content Pipeline**: Notion API → Supabase 동기화 스크립트
- **Deployment**: Vercel

## Project Structure
```
src/
  app/            # 페이지 & API 라우트 (problems, quiz, grade)
  components/     # UI 컴포넌트 (CodeEditor, GradeResult, TestResults 등)
  lib/            # grading, pyodide-runner, test-runner, supabase 클라이언트
scripts/          # Notion→Supabase 시드/싱크 스크립트
supabase/         # config 및 마이그레이션 SQL
```

## Getting Started
```bash
npm install
cp .env.example .env.local   # 환경 변수 값 채우기
npm run dev
# http://localhost:3000
```

## Environment Variables
`.env.example`를 참고하여 `.env.local`을 구성합니다. 실제 키 값은 **커밋하지 않습니다**.
- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY` (시드 스크립트 전용)
- `NOTION_API_KEY`, `NOTION_DATABASE_ID` (콘텐츠 동기화용)
