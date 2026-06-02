"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, Stat, ErrorBox } from "@/components/Panels";

const phases = [
  { n: 1, t: "Synthetic Patient Generator", d: "Cohort of 500 patients with correlated biomarkers" },
  { n: 2, t: "Graph-Based Digital Twin",   d: "GCN/GAT encoder, similar patients, cluster summary" },
  { n: 3, t: "Disease Dynamics",            d: "ODE + LIF simulator, counterfactual, bifurcation" },
  { n: 4, t: "Causal AI",                   d: "SCM, ATE, CATE, 3-step patient counterfactual" },
  { n: 5, t: "LLM Agent",                   d: "Ollama llama3.1, 9 tools, conversation memory" },
  { n: 6, t: "Dashboard",                   d: "This UI — Next.js + Tailwind + Recharts" },
];

export default function Home() {
  const [health, setHealth] = useState<any>(null);
  const [err, setErr] = useState<any>(null);
  useEffect(() => {
    api.health().then(setHealth).catch(setErr);
  }, []);
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-text">Bio-Digital Twin</h1>
        <p className="text-sm text-muted mt-1">
          End-to-end platform from synthetic cohort to LLM-augmented counterfactuals.
        </p>
      </div>

      {err && <ErrorBox err={err} />}
      {health && (
        <Card>
          <div className="flex items-center gap-4">
            <span className="chip chip-info">{health.status}</span>
            <span className="text-sm text-muted">{health.phase}</span>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-3 gap-4">
        {phases.map((p) => (
          <Link key={p.n} href={
            p.n === 6 ? "/" : p.n === 5 ? "/chat" :
            p.n === 4 ? "/causal" : p.n === 3 ? "/simulate" :
            p.n === 2 ? "/cohort" : "/cohort"
          }>
            <Card>
              <div className="text-xs text-teal font-medium mb-1">Phase {p.n}</div>
              <div className="text-text font-medium">{p.t}</div>
              <div className="text-xs text-muted mt-1">{p.d}</div>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
