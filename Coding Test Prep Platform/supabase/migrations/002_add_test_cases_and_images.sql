-- Add test cases, images, and notion sync support
ALTER TABLE problems ADD COLUMN IF NOT EXISTS test_cases JSONB;
ALTER TABLE problems ADD COLUMN IF NOT EXISTS images TEXT[];
ALTER TABLE problems ADD COLUMN IF NOT EXISTS notion_id TEXT UNIQUE;

-- Add test result tracking to submissions
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS passed_count INTEGER DEFAULT 0;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS total_count INTEGER DEFAULT 0;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS test_results JSONB;
