"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

const phases = [
  { n: 1,  t: "Synthetic Patient Generator", d: "Cohort of 500 patients with correlated biomarkers" },
  { n: 2,  t: "Graph-Based Digital Twin",    d: "GCN/GAT encoder, similar patients, cluster summary" },
  { n: 3,  t: "Disease Dynamics",            d: "ODE + LIF simulator, counterfactual, bifurcation" },
  { n: 4,  t: "Causal AI",                   d: "SCM, ATE, CATE, 3-step patient counterfactual" },
  { n: 5,  t: "LLM Agent",                   d: "Ollama llama3.1, 13+ tools, conversation memory" },
  { n: 6,  t: "Dashboard",                   d: "Next.js 14 + Tailwind + Recharts" },
  { n: 8,  t: "Pharmacogenomics",            d: "CYP panel, drug-gene rules, PGx-aware counterfactuals" },
  { n: 9,  t: "Drug-Drug Interactions",      d: "60+ curated pairs + transitive CYP graph" },
  { n: 10, t: "PK / PD",                     d: "2-compartment + sigmoid Emax, 31 drugs, MC sim" },
  { n: 11, t: "Uncertainty Quantification",  d: "Bootstrap CIs on patient counterfactuals" },
  { n: 12, t: "Clinical Trials",             d: "Live ClinicalTrials.gov v2 with 24h cache" },
  { n: 13, t: "Regulatory",                  d: "FDA + OpenFDA FAERS + RxNorm + warnings" },
  { n: 14, t: "Wet-Lab",                     d: "PAINS/Brenk/SAS/IC50/tox on candidate SMILES" },
  { n: 15, t: "Disease Registry",            d: "Postgres-backed CRUD for the disease catalog" },
];

const linkFor = (n: number) => {
  switch (n) {
    case 1:  return "/cohort";
    case 2:  return "/cohort";
    case 3:  return "/simulate";
    case 4:  return "/causal";
    case 5:  return "/chat";
    case 6:  return "/";
    case 8:  return "/pharmacogenomics";
    case 9:  return "/polypharmacy";
    case 10: return "/pkpd";
    case 11: return "/uncertainty";
    case 12: return "/trials";
    case 13: return "/regulatory";
    case 14: return "/wetlab";
    case 15: return "/registry";
    default: return "/";
  }
};

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
          End-to-end drug discovery platform — synthetic cohort, GNN, dynamics,
          causality, LLM agent, plus 8 advancement modules for clinical realism.
        </p>
      </div>

      {err && <ErrorBox err={err} />}
      {health && (
        <Card>
          <div className="flex items-center gap-4 flex-wrap">
            <span className="chip chip-info">{health.status}</span>
            <span className="text-sm text-muted">{health.phase}</span>
            <span className="text-xs text-muted ml-auto">
              Open the <span className="text-teal">≡</span> menu for all 15 modules
            </span>
          </div>
        </Card>
      )}

      <div>
        <div className="text-xs uppercase tracking-wider text-muted font-semibold mb-2">
          Core platform — Phases 1-6
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {phases.filter(p => p.n <= 6).map((p) => (
            <Link key={p.n} href={linkFor(p.n)}>
              <Card>
                <div className="text-xs text-teal font-medium mb-1">Phase {p.n}</div>
                <div className="text-text font-medium text-sm">{p.t}</div>
                <div className="text-xs text-muted mt-1">{p.d}</div>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      <div>
        <div className="text-xs uppercase tracking-wider text-muted font-semibold mb-2 flex items-center gap-2">
          <span>Advancements — Phases 8-15</span>
          <span className="px-1.5 py-0.5 rounded bg-teal/10 text-teal normal-case tracking-normal font-medium">
            new
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {phases.filter(p => p.n >= 8).map((p) => (
            <Link key={p.n} href={linkFor(p.n)}>
              <Card>
                <div className="text-xs text-teal font-medium mb-1">Phase {p.n}</div>
                <div className="text-text font-medium text-sm">{p.t}</div>
                <div className="text-xs text-muted mt-1">{p.d}</div>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
