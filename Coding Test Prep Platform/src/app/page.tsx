"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import type { Problem } from "@/lib/types";
import ProblemCard from "@/components/ProblemCard";
import CategoryFilter from "@/components/CategoryFilter";
import CategoryStats from "@/components/CategoryStats";

export default function HomePage() {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [scores, setScores] = useState<Record<string, number>>({});
  const [solved, setSolved] = useState<Record<string, boolean>>({});
  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const supabase = createClient();

    async function fetchData() {
      let query = supabase.from("problems").select("*").order("category").order("title");
      if (category) query = query.eq("category", category);
      if (search) query = query.ilike("title", `%${search}%`);

      const { data: problemsData } = await query;
      setProblems(problemsData || []);

      const { data: subsData } = await supabase
        .from("submissions")
        .select("problem_id, similarity_score, is_correct")
        .order("created_at", { ascending: false });

      if (subsData) {
        const scoreMap: Record<string, number> = {};
        const solvedMap: Record<string, boolean> = {};
        for (const sub of subsData) {
          if (!scoreMap[sub.problem_id]) {
            scoreMap[sub.problem_id] = sub.similarity_score;
          }
          if (sub.is_correct) {
            solvedMap[sub.problem_id] = true;
          }
        }
        setScores(scoreMap);
        setSolved(solvedMap);
      }

      setLoading(false);
    }

    fetchData();
  }, [category, search]);

  const stats = {
    total: problems.length,
    solved: Object.keys(solved).length,
    attempted: Object.keys(scores).length,
  };

  return (
    <div className="space-y-8">
      {/* Search */}
      <input
        type="text"
        placeholder="문제 검색..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full rounded-lg border border-gray-800 bg-black px-4 py-3 text-white placeholder-gray-500 outline-none focus:border-gray-600 transition-colors"
      />

      {/* Category Stats */}
      {!loading && <CategoryStats problems={problems} scores={scores} solved={solved} />}

      {/* Category Filter */}
      <CategoryFilter selected={category} onChange={setCategory} />

      {/* Problem List */}
      {loading ? (
        <div className="py-20 text-center text-gray-500">로딩 중...</div>
      ) : problems.length === 0 ? (
        <div className="py-20 text-center text-gray-500">
          <p className="text-lg">문제가 없습니다</p>
          <p className="mt-2 text-sm">+ 문제 추가 버튼으로 첫 번째 문제를 등록해보세요!</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {problems.map((problem) => (
            <ProblemCard
              key={problem.id}
              problem={problem}
              lastScore={scores[problem.id]}
              isSolved={!!solved[problem.id]}
            />
          ))}
        </div>
      )}
    </div>
  );
}
