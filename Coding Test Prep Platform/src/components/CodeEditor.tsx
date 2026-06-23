"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <div className="flex h-[300px] items-center justify-center bg-gray-900 text-gray-500 text-sm">
      에디터 로딩 중...
    </div>
  ),
});

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  readOnly?: boolean;
  height?: string;
}

export default function CodeEditor({
  value,
  onChange,
  placeholder = "여기에 코드를 작성하세요...",
  readOnly = false,
  height = "300px",
}: CodeEditorProps) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div
      className={`relative rounded-lg border ${
        isFocused ? "border-blue-500 ring-2 ring-blue-500/20" : "border-gray-700"
      } bg-gray-900 transition-all overflow-hidden`}
    >
      <div className="flex items-center gap-2 border-b border-gray-700 px-4 py-2">
        <div className="flex gap-1.5">
          <div className="h-3 w-3 rounded-full bg-red-500" />
          <div className="h-3 w-3 rounded-full bg-yellow-500" />
          <div className="h-3 w-3 rounded-full bg-green-500" />
        </div>
        <span className="text-xs text-gray-400">Python</span>
      </div>
      <MonacoEditor
        height={height}
        language="python"
        theme="vs-dark"
        value={value}
        onChange={(v) => onChange(v ?? "")}
        onMount={(editor) => {
          editor.onDidFocusEditorText(() => setIsFocused(true));
          editor.onDidBlurEditorText(() => setIsFocused(false));
          if (!value && placeholder) {
            editor.updateOptions({
              unicodeHighlight: { ambiguousCharacters: false },
            });
          }
        }}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          tabSize: 4,
          insertSpaces: true,
          wordWrap: "on",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          readOnly,
          lineNumbers: "on",
          renderLineHighlight: "line",
          scrollbar: {
            verticalScrollbarSize: 8,
            horizontalScrollbarSize: 8,
          },
          padding: { top: 12, bottom: 12 },
        }}
      />
    </div>
  );
}
