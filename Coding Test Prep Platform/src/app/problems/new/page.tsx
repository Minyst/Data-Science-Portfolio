"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import ProblemEditForm from "@/components/ProblemEditForm";

export default function NewProblemPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    title: "",
    category: "Hash",
    difficulty: "Medium",
    code: "",
    description: "",
    notes: "",
  });
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!form.title.trim() || !form.code.trim()) return;
    setSaving(true);
    const supabase = createClient();
    const { error } = await supabase.from("problems").insert({
      title: form.title,
      category: form.category,
      difficulty: form.difficulty,
      code: form.code,
      description: form.description || null,
      notes: form.notes || null,
    });
    if (!error) router.push("/");
    setSaving(false);
  };

  return (
    <ProblemEditForm
      title="새 문제 추가"
      form={form}
      onChange={setForm}
      onSave={handleSave}
      onCancel={() => router.push("/")}
      saveLabel="문제 추가"
      saving={saving}
    />
  );
}
