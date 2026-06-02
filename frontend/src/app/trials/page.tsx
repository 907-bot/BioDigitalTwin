"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

export default function TrialsPage() {
  const [q, setQ] = useState("metformin type 2 diabetes");
  const [by, setBy] = useState<"condition" | "drug">("condition");
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r = await api.trialsSearch(q, by, 15);
      setResults(r);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const phaseColor = (ph: string) => {
    if (!ph) return "chip-info";
    if (ph.includes("4")) return "chip-ok";
    if (ph.includes("3")) return "chip-info";
    if (ph.includes("2")) return "chip-info";
    return "chip-warn";
  };

  const statusColor = (s: string) => {
    if (s === "RECRUITING") return "chip-ok";
    if (s === "COMPLETED") return "chip-info";
    if (s === "TERMINATED" || s === "WITHDRAWN") return "chip-warn";
    return "chip-info";
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Clinical trials</h1>
        <p className="text-sm text-muted mt-1">
          Live ClinicalTrials.gov v2 API. 24-hour disk cache.
        </p>
      </div>

      <Card>
        <div className="flex gap-3 items-end flex-wrap">
          <label className="text-xs text-muted flex-1 min-w-[20rem]">
            Search query
            <input value={q} onChange={e => setQ(e.target.value)}
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
            <span className="text-text font-mono">{results.trials.length}</span> trials ·{" "}
            source: {results.source}{results.cached && " (cached)"}
          </div>
          <div className="space-y-2">
            {results.trials.map((t: any) => (
              <div key={t.nct_id} className="p-3 rounded border border-border bg-bg">
                <div className="flex items-center gap-2 flex-wrap">
                  <a href={`https://clinicaltrials.gov/study/${t.nct_id}`}
                     target="_blank" rel="noreferrer"
                     className="text-teal font-mono text-sm hover:underline">
                    {t.nct_id}
                  </a>
                  <span className={"chip " + phaseColor(t.phase)}>
                    Phase {t.phase || "?"}
                  </span>
                  <span className={"chip " + statusColor(t.status)}>{t.status}</span>
                  <span className="text-xs text-muted ml-auto">
                    {t.enrollment ? `n=${t.enrollment}` : ""}
                    {t.start_date ? ` · start ${t.start_date}` : ""}
                  </span>
                </div>
                <div className="text-text text-sm mt-1 font-medium">{t.title}</div>
                <div className="text-xs text-muted mt-1">
                  {t.conditions?.slice(0, 3).join(" · ")}
                </div>
                {t.sponsor && (
                  <div className="text-[10px] text-muted mt-1">sponsor: {t.sponsor}</div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
