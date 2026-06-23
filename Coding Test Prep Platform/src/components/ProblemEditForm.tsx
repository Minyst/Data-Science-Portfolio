"use client";

import { CATEGORIES, DIFFICULTIES } from "@/lib/types";
import CodeEditor from "@/components/CodeEditor";

interface EditFormData {
  title: string;
  category: string;
  difficulty: string;
  code: string;
  description: string;
  notes: string;
}

interface ProblemEditFormProps {
  form: EditFormData;
  onChange: (form: EditFormData) => void;
  onSave: () => void;
  onCancel: () => void;
  title: string;
  saveLabel?: string;
  saving?: boolean;
}

export default function ProblemEditForm({
  form,
  onChange,
  onSave,
  onCancel,
  title,
  saveLabel = "저장",
  saving = false,
}: ProblemEditFormProps) {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{title}</h1>
      <div className="space-y-4">
        <input
          value={form.title}
          onChange={(e) => onChange({ ...form, title: e.target.value })}
          className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-blue-500"
          placeholder="제목"
        />
        <div className="grid grid-cols-2 gap-4">
          <select
            value={form.category}
            onChange={(e) => onChange({ ...form, category: e.target.value })}
            className="rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-blue-500"
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select
            value={form.difficulty}
            onChange={(e) => onChange({ ...form, difficulty: e.target.value })}
            className="rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-blue-500"
          >
            {DIFFICULTIES.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
        <textarea
          value={form.description}
          onChange={(e) => onChange({ ...form, description: e.target.value })}
          className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-blue-500 min-h-[80px]"
          placeholder="문제 설명 (선택)"
        />
        <textarea
          value={form.notes}
          onChange={(e) => onChange({ ...form, notes: e.target.value })}
          className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white outline-none focus:border-blue-500 min-h-[60px]"
          placeholder="메모 (선택) - 풀이 힌트, 시간복잡도 등"
        />
        <div>
          <label className="mb-2 block text-sm text-gray-400">정답 코드</label>
          <CodeEditor value={form.code} onChange={(v) => onChange({ ...form, code: v })} />
        </div>
        <div className="flex gap-3">
          <button
            onClick={onSave}
            disabled={saving || !form.title.trim() || !form.code.trim()}
            className="rounded-lg bg-blue-600 px-6 py-3 font-medium hover:bg-blue-500 transition-colors disabled:opacity-50"
          >
            {saving ? "저장 중..." : saveLabel}
          </button>
          <button
            onClick={onCancel}
            className="rounded-lg border border-gray-700 px-6 py-3 text-gray-300 hover:bg-gray-800 transition-colors"
          >
            취소
          </button>
        </div>
      </div>
    </div>
  );
}
