"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";
import { Narrative, NarrativeList } from "@/components/Narrative";

const SEV_BG: Record<string, string> = {
  contraindicated: "bg-rose/20 border-rose/40",
  major: "bg-amber/20 border-amber/40",
  moderate: "bg-yellow/10 border-yellow/30",
  minor: "bg-bg border-border",
  none: "bg-bg border-border",
};
const SEV_TEXT: Record<string, string> = {
  contraindicated: "text-rose",
  major: "text-amber",
  moderate: "text-yellow",
  minor: "text-muted",
  none: "text-muted",
};

export default function PolypharmacyPage() {
  const [drugs, setDrugs] = useState(
    "warfarin, ciprofloxacin, aspirin, ibuprofen, clarithromycin, simvastatin, atorvastatin"
  );
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const list = drugs.split(",").map(s => s.trim()).filter(Boolean);
      const r = await api.ddiCheck(list);
      setResult(r);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const matrix: string[][] = [];
  if (result) {
    for (let i = 0; i < result.drugs.length; i++) {
      const row: string[] = [];
      for (let j = 0; j < result.drugs.length; j++) {
        if (i === j) { row.push("self"); continue; }
        const m = result.interactions.find(
          (x: any) => (x.drug_a === result.drugs[i] && x.drug_b === result.drugs[j])
                    || (x.drug_a === result.drugs[j] && x.drug_b === result.drugs[i])
        );
        row.push(m ? m.severity : "none");
      }
      matrix.push(row);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Polypharmacy checker</h1>
        <p className="text-sm text-muted mt-1">
          Drug-drug interactions from a curated 60+ pair table plus
          transitive inference from the CYP/transporter graph.
        </p>
      </div>

      <Card>
        <div className="flex gap-3 items-end flex-wrap">
          <label className="text-xs text-muted flex-1 min-w-[20rem]">
            Drug list (comma-separated, up to 30)
            <textarea value={drugs} onChange={e => setDrugs(e.target.value)}
                      rows={2}
                      className="block mt-1 px-3 py-1.5 bg-bg border border-border
                                 rounded-md text-text w-full font-mono text-xs" />
          </label>
          <button onClick={run} disabled={loading}
                  className="px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                             font-medium hover:bg-teal/80 disabled:opacity-50">
            {loading ? "Checking..." : "Check interactions"}
          </button>
        </div>
      </Card>

      {err && <ErrorBox err={err} />}

      {result?.narrative && (
        <Narrative data={result.narrative} title="Polypharmacy summary" />
      )}

      {result && (
        <>
          <Card>
            <div className="flex items-center gap-6 flex-wrap text-sm">
              <div>
                <span className="text-muted text-xs">Drugs</span>{" "}
                <span className="text-text font-mono">{result.n_drugs}</span>
              </div>
              <div>
                <span className="text-muted text-xs">Interactions</span>{" "}
                <span className="text-text font-mono">{result.n_interactions}</span>
                <span className="text-xs text-muted ml-1">
                  ({result.n_direct} direct, {result.n_inferred} inferred)
                </span>
              </div>
              <div>
                <span className="text-muted text-xs">Overall severity</span>{" "}
                <span className={SEV_TEXT[result.overall_severity] || "text-muted"}>
                  {result.overall_severity}
                </span>
              </div>
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Interaction matrix</h2>
            <div className="overflow-x-auto">
              <table className="text-xs">
                <thead>
                  <tr>
                    <th className="p-1"></th>
                    {result.drugs.map((d: string) => (
                      <th key={d} className="p-1 text-muted font-mono">{d}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="p-1 text-muted font-mono text-right pr-2">
                        {result.drugs[i]}
                      </td>
                      {row.map((cell, j) => (
                        <td key={j} className="p-0.5">
                          <div className={
                            "w-7 h-7 rounded flex items-center justify-center text-[10px] border " +
                            (cell === "self" ? "bg-bg2 border-border text-muted" : SEV_BG[cell] || SEV_BG.none)
                          }>
                            {cell === "self" ? "·" : cell[0].toUpperCase()}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex gap-3 text-xs text-muted mt-3 flex-wrap">
              <span><span className="inline-block w-3 h-3 rounded bg-rose/30 border border-rose/40 align-middle"></span> Contraindicated</span>
              <span><span className="inline-block w-3 h-3 rounded bg-amber/30 border border-amber/40 align-middle"></span> Major</span>
              <span><span className="inline-block w-3 h-3 rounded bg-yellow/10 border border-yellow/30 align-middle"></span> Moderate</span>
              <span><span className="inline-block w-3 h-3 rounded bg-bg2 border border-border align-middle"></span> None / Self</span>
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Interactions detail</h2>
            <div className="space-y-3">
              {result.interactions.map((i: any, idx: number) => (
                <div key={idx} className={"rounded border " + (SEV_BG[i.severity] || SEV_BG.none)}>
                  <div className="p-3">
                    <div className="flex items-center gap-2 text-sm">
                      <span className="font-mono">{i.drug_a}</span>
                      <span className="text-muted">+</span>
                      <span className="font-mono">{i.drug_b}</span>
                      <span className={SEV_TEXT[i.severity] + " ml-auto text-xs"}>
                        {i.severity.toUpperCase()}
                      </span>
                      {i.inferred && (
                        <span className="text-[10px] text-muted bg-bg2 px-1.5 py-0.5 rounded">
                          inferred
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-muted mt-1">
                      <span className="text-text">{i.mechanism}</span> — {i.clinical_effect}
                    </div>
                    <div className="text-[10px] text-muted mt-1">source: {i.source}</div>
                  </div>
                  {i.narrative && (
                    <div className="px-3 pb-3">
                      <Narrative data={i.narrative} title="" collapsible={false} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
