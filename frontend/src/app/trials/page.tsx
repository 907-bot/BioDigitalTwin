"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

export default function TrialsPage() {
  const [q, setQ] = useState("metformin type 2 diabetes");
  const [by, setBy] = useState<"condition" | "drug">("condition");
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r: any = await api.trialsSearch(q, by, 15);
      setResults(r);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const trials = results?.trials || [];
  const phaseLabel = (ph: string[] | string) => {
    if (!ph) return "?";
    if (Array.isArray(ph)) return ph.length ? ph.join("/") : "N/A";
    return ph;
  };
  const phaseChip = (ph: string[] | string) => {
    const lbl = phaseLabel(ph);
    if (!lbl || lbl === "N/A") return "chip-info";
    if (lbl.includes("4") || lbl.includes("2/3") || lbl.includes("3")) return "chip-ok";
    if (lbl.includes("2") || lbl.includes("1")) return "chip-info";
    return "chip-info";
  };
  const statusChip = (s: string) => {
    if (!s) return "chip-info";
    if (s === "RECRUITING") return "chip-ok";
    if (s === "COMPLETED") return "chip-info";
    if (s === "ACTIVE_NOT_RECRUITING") return "chip-info";
    if (s === "TERMINATED" || s === "WITHDRAWN" || s === "SUSPENDED") return "chip-warn";
    return "chip-info";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Clinical trials</h1>
          <p className="text-sm text-muted mt-1">
            Live ClinicalTrials.gov v2 API. 24-hour disk cache for stability.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
      </div>

      <Card>
        <div className="flex gap-3 items-end flex-wrap">
          <label className="text-xs text-muted flex-1 min-w-[20rem]">
            Search query
            <input value={q} onChange={e => setQ(e.target.value)}
                   onKeyDown={e => e.key === "Enter" && run()}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-full" />
          </label>
          <label className="text-xs text-muted">
            By
            <select value={by} onChange={e => setBy(e.target.value as any)}
                    className="block mt-1 px-3 py-1.5 bg-bg border border-border
                               rounded-md text-text w-full">
              <option value="condition">condition</option>
              <option value="drug">drug / intervention</option>
            </select>
          </label>
          <button onClick={run} disabled={loading}
                  className="px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                             font-medium hover:bg-teal/80 disabled:opacity-50">
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </Card>

      {err && <ErrorBox err={err} />}

      {results && (
        <Card>
          <div className="text-sm text-muted mb-3">
            <span className="text-text font-mono">{results.n_results}</span> results ·{" "}
            query: <span className="text-text">"{results.query}"</span> by {results.by}
          </div>
          {trials.length === 0 && <div className="text-xs text-muted">No trials found.</div>}
          <div className="space-y-2">
            {trials.map((t: any) => (
              <div key={t.nct_id} className="p-3 rounded border border-border bg-bg">
                <div className="flex items-center gap-2 flex-wrap">
                  <a href={`https://clinicaltrials.gov/study/${t.nct_id}`}
                     target="_blank" rel="noreferrer"
                     className="text-teal font-mono text-sm hover:underline">
                    {t.nct_id}
                  </a>
                  <span className={"chip " + phaseChip(t.phase)}>
                    Phase {phaseLabel(t.phase)}
                  </span>
                  <span className={"chip " + statusChip(t.overall_status)}>
                    {t.overall_status}
                  </span>
                  <span className="text-xs text-muted ml-auto">
                    {t.enrollment ? `n=${t.enrollment}` : ""}
                    {t.start_date ? ` · start ${t.start_date}` : ""}
                  </span>
                </div>
                <div className="text-text text-sm mt-1 font-medium">
                  {t.brief_title || t.title}
                </div>
                {t.official_title && t.official_title !== t.brief_title && (
                  <div className="text-xs text-muted mt-0.5 italic">{t.official_title}</div>
                )}
                <div className="text-xs text-muted mt-1">
                  {t.conditions?.slice(0, 3).join(" · ")}
                </div>
                {t.study_type && (
                  <div className="text-[10px] text-muted mt-1">
                    type: {t.study_type}
                    {t.completion_date ? ` · complete ${t.completion_date}` : ""}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
