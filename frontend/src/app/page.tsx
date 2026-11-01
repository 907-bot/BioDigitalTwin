"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";

/* ─── Phase definitions ──────────────────────────────────── */
const CORE_PHASES = [
  {
    n: 1, t: "Synthetic Patients",   href: "/cohort",
    blurb: "500-patient cohort with correlated biomarkers and pathophysiology",
    icon: "⬡", color: "#00d4a0",
  },
  {
    n: 2, t: "Graph Digital Twin",   href: "/cohort",
    blurb: "GCN/GAT encoder, patient similarity graph, cluster summaries",
    icon: "◎", color: "#00bcd4",
  },
  {
    n: 3, t: "Disease Dynamics",     href: "/simulate",
    blurb: "ODE + LIF simulator, counterfactual trajectories, bifurcation",
    icon: "⚡", color: "#a78bfa",
  },
  {
    n: 4, t: "Causal AI",            href: "/causal",
    blurb: "Structural causal models, ATE/CATE estimation, do-calculus",
    icon: "◈", color: "#60a5fa",
  },
  {
    n: 5, t: "LLM Agent",            href: "/chat",
    blurb: "Ollama llama3.1 with 13 tools and persistent session memory",
    icon: "◑", color: "#ffa94d",
  },
  {
    n: 6, t: "Dashboard",            href: "/",
    blurb: "Real-time monitoring hub across all 16 phases",
    icon: "▦", color: "#4ade80",
  },
];

const ADV_PHASES = [
  { n: 8,  t: "Pharmacogenomics",  href: "/pharmacogenomics", blurb: "CYP panel, drug-gene rules, PGx counterfactuals",    icon: "⬟", color: "#00d4a0" },
  { n: 9,  t: "Drug-Drug Interact",href: "/polypharmacy",     blurb: "60+ curated DDI pairs + transitive CYP graph",        icon: "⬡", color: "#ff6b5b" },
  { n: 10, t: "PK / PD",           href: "/pkpd",             blurb: "2-compartment + sigmoid Emax, 31 drugs, MC sim",       icon: "⌑", color: "#a78bfa" },
  { n: 11, t: "Uncertainty QA",    href: "/uncertainty",      blurb: "Bootstrap CIs on patient counterfactuals",            icon: "≋", color: "#60a5fa" },
  { n: 12, t: "Clinical Trials",   href: "/trials",           blurb: "Live ClinicalTrials.gov v2 with 24h cache",           icon: "⊕", color: "#ffa94d" },
  { n: 13, t: "Regulatory",        href: "/regulatory",       blurb: "FDA + OpenFDA FAERS + RxNorm + safety warnings",     icon: "⊛", color: "#4ade80" },
  { n: 14, t: "Wet-Lab",           href: "/wetlab",           blurb: "PAINS / Brenk / SAS / IC50 / toxicity on SMILES",    icon: "⎔", color: "#00bcd4" },
  { n: 15, t: "Disease Registry",  href: "/registry",         blurb: "Postgres-backed disease catalog with CRUD API",       icon: "▤", color: "#a78bfa" },
  { n: 16, t: "Explainable AI",    href: "/explain",          blurb: "SHAP-lite + reasoning chains across all phases",      icon: "◬", color: "#60a5fa" },
  { n: 18, t: "Real-World Twin",   href: "/samsung-twin",     blurb: "Samsung Health CSVs → feature engineering → 4 ML models → HITL predictions", icon: "⌚", color: "#00bcd4" },
];

/* ─── System metrics ─────────────────────────────────────── */
const SYSTEM_STATS = [
  { label: "Active Phases",    value: "16",    unit: "",     accent: "#00d4a0" },
  { label: "API Endpoints",    value: "60+",   unit: "",     accent: "#a78bfa" },
  { label: "State Dimensions", value: "30",    unit: "dim",  accent: "#60a5fa" },
  { label: "Observation Space",value: "16",    unit: "dim",  accent: "#ffa94d" },
];

/* ─── Recent activity ────────────────────────────────────── */
const ACTIVITY = [
  { label: "Core engine validated",    time: "Engine test",    status: "pass" },
  { label: "E2E patient workflow",     time: "API workflow",   status: "pass" },
  { label: "Biomarker serialization",  time: "Bug fix",        status: "fixed" },
  { label: "Observation dim updated",  time: "Phase 3 → 16D",  status: "updated" },
];

/* ─── Component ──────────────────────────────────────────── */
export default function Home() {
  const [health, setHealth] = useState<any>(null);
  const [err, setErr] = useState<any>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    api.health()
      .then((h) => { setHealth(h); setLoaded(true); })
      .catch((e) => { setErr(e); setLoaded(true); });
  }, []);

  return (
    <div className="space-y-12 animate-fade-in">

      {/* ── Hero Section ───────────────────────────────── */}
      <section className="relative overflow-hidden rounded-2xl">
        {/* Background layers */}
        <div className="absolute inset-0 bg-grid opacity-30" style={{ backgroundSize: "28px 28px" }} />
        <div className="absolute inset-0" style={{
          background: "radial-gradient(ellipse 70% 60% at 30% 0%, rgba(0,212,160,0.12) 0%, transparent 60%), radial-gradient(ellipse 50% 50% at 80% 50%, rgba(167,139,250,0.08) 0%, transparent 60%)"
        }} />
        <div className="absolute inset-0 rounded-2xl border border-[#1e2d42]" />

        <div className="relative px-8 py-12 md:py-16 md:px-12">
          <div className="flex flex-col md:flex-row md:items-center gap-8">
            <div className="flex-1 space-y-5">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#00d4a0]/25 text-[11px] font-medium"
                   style={{ background: "rgba(0,212,160,0.08)", color: "#00d4a0" }}>
                <div className="pulse-dot" style={{ "--teal": "#00d4a0" } as React.CSSProperties} />
                Drug Discovery Platform · 16 Phases Active
              </div>

              {/* Heading */}
              <h1 className="text-3xl md:text-4xl font-bold leading-tight tracking-tight">
                <span className="text-[#e2eaf5]">Bio-Digital</span>
                {" "}
                <span style={{
                  background: "linear-gradient(135deg, #00d4a0 0%, #00bcd4 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}>
                  Twin
                </span>
                {" "}
                <span className="text-[#e2eaf5]">Platform</span>
              </h1>

              {/* Description */}
              <p className="text-[#7a93b0] text-sm md:text-base leading-relaxed max-w-xl">
                End-to-end drug discovery with synthetic patient cohorts, GNN-based digital twins,
                causal AI, pharmacogenomics, PK/PD simulation, and explainable AI — all in one platform.
              </p>

              {/* CTAs */}
              <div className="flex flex-wrap gap-3 pt-2">
                <Link href="/cohort" id="cta-cohort" className="btn-primary text-sm">
                  Generate Cohort →
                </Link>
                <Link href="/causal" id="cta-causal" className="btn text-sm">
                  Explore Causal AI
                </Link>
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
                   className="btn text-sm">
                  API Docs ↗
                </a>
              </div>
            </div>

            {/* System Status Panel */}
            <div className="flex-shrink-0 w-full md:w-72">
              <div className="rounded-xl border border-[#1e2d42] p-5 space-y-4"
                   style={{ background: "rgba(13,20,34,0.8)" }}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#7a93b0]">System Status</span>
                  {loaded && !err ? (
                    <span className="chip chip-online text-[10px]">● Healthy</span>
                  ) : loaded ? (
                    <span className="chip chip-high text-[10px]">● Offline</span>
                  ) : (
                    <span className="chip chip-info text-[10px]">Connecting…</span>
                  )}
                </div>

                {health && (
                  <div className="text-[11px] text-[#4d6580] font-mono bg-[#0d1422] rounded-lg px-3 py-2 truncate">
                    {health.phase || "phases: 1-16"}
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3">
                  {SYSTEM_STATS.map((s) => (
                    <div key={s.label} className="rounded-lg p-3 border border-[#1e2d42]"
                         style={{ background: "rgba(21,32,53,0.5)" }}>
                      <div className="text-xl font-bold font-mono" style={{ color: s.accent }}>
                        {s.value}
                        {s.unit && <span className="text-xs text-[#4d6580] ml-1">{s.unit}</span>}
                      </div>
                      <div className="text-[10px] text-[#4d6580] mt-0.5 leading-tight">{s.label}</div>
                    </div>
                  ))}
                </div>

                {/* Recent activity */}
                <div className="space-y-2 pt-1">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-[#4d6580]">Recent Updates</div>
                  {ACTIVITY.map((a, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{
                        background: a.status === "pass" ? "#00d4a0" :
                                    a.status === "fixed" ? "#a78bfa" : "#60a5fa"
                      }} />
                      <span className="text-[#7a93b0] flex-1 truncate">{a.label}</span>
                      <span className="text-[#4d6580] text-[10px]">{a.time}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Core Platform ─────────────────────────────── */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#e2eaf5]">Core Platform</h2>
            <p className="text-xs text-[#4d6580] mt-0.5">Phases 1–6 · Foundation pipeline</p>
          </div>
          <span className="chip chip-online text-[10px]">6 modules</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {CORE_PHASES.map((p) => (
            <PhaseCard key={p.n} phase={p} isAdv={false} />
          ))}
        </div>
      </section>

      {/* ── Advanced Modules ───────────────────────────── */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#e2eaf5]">Clinical Modules</h2>
            <p className="text-xs text-[#4d6580] mt-0.5">Phases 8–16 · Advanced drug discovery</p>
          </div>
          <span className="chip chip-info text-[10px]">9 modules</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3">
          {ADV_PHASES.map((p) => (
            <PhaseCard key={p.n} phase={p} isAdv={true} />
          ))}
        </div>
      </section>

      {/* ── Workflow Strip ─────────────────────────────── */}
      <section className="rounded-2xl border border-[#1e2d42] overflow-hidden">
        <div className="px-6 py-4 border-b border-[#1e2d42]" style={{ background: "rgba(13,20,34,0.6)" }}>
          <h2 className="text-sm font-semibold text-[#e2eaf5]">Typical Workflow</h2>
          <p className="text-xs text-[#4d6580] mt-0.5">End-to-end drug discovery pipeline</p>
        </div>
        <div className="p-6">
          <div className="flex flex-wrap gap-2 items-center">
            {[
              { label: "Generate Cohort",    href: "/cohort",          n: 1 },
              { label: "Explore Similarity", href: "/cohort",          n: 2 },
              { label: "Simulate Dynamics",  href: "/simulate",        n: 3 },
              { label: "Causal Analysis",    href: "/causal",          n: 4 },
              { label: "LLM Consultation",   href: "/chat",            n: 5 },
              { label: "PK/PD Modelling",    href: "/pkpd",            n: 10 },
              { label: "UQ Assessment",      href: "/uncertainty",     n: 11 },
              { label: "PGx Check",          href: "/pharmacogenomics",n: 8  },
              { label: "DDI Screen",         href: "/polypharmacy",    n: 9  },
              { label: "Trial Search",       href: "/trials",          n: 12 },
              { label: "Regulatory Review",  href: "/regulatory",      n: 13 },
              { label: "Wet-Lab Triage",     href: "/wetlab",          n: 14 },
              { label: "XAI Explanation",    href: "/explain",         n: 16 },
            ].map((step, i, arr) => (
              <div key={step.n} className="flex items-center gap-2">
                <Link
                  href={step.href}
                  id={`workflow-step-${step.n}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#1e2d42] hover:border-[#00d4a0]/30 hover:bg-[#152035] transition-all text-xs text-[#7a93b0] hover:text-[#e2eaf5]"
                  style={{ background: "rgba(15,25,41,0.8)" }}
                >
                  <span className="font-mono text-[#4d6580] text-[10px]">{step.n}</span>
                  {step.label}
                </Link>
                {i < arr.length - 1 && (
                  <span className="text-[#253548] text-sm">→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

    </div>
  );
}

/* ─── Phase Card Component ───────────────────────────────── */
function PhaseCard({ phase, isAdv }: {
  phase: { n: number; t: string; href: string; blurb: string; icon: string; color: string };
  isAdv: boolean;
}) {
  return (
    <Link
      href={phase.href}
      id={`phase-card-${phase.n}`}
      className="module-card group"
      style={isAdv ? {} : {}}
    >
      {/* Decorative glow corner */}
      <div
        className="absolute top-0 right-0 w-24 h-24 rounded-bl-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{ background: `radial-gradient(circle at top right, ${phase.color}12 0%, transparent 70%)` }}
      />

      <div className="relative flex items-start gap-3">
        {/* Phase icon badge */}
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center text-sm flex-shrink-0 transition-all"
          style={{
            background: `${phase.color}12`,
            border: `1px solid ${phase.color}25`,
            color: phase.color,
          }}
        >
          {phase.icon}
        </div>

        <div className="flex-1 min-w-0">
          {/* Phase number */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-mono font-semibold" style={{ color: phase.color }}>
              Phase {phase.n}
            </span>
          </div>
          {/* Title */}
          <div className="text-[13px] font-semibold text-[#e2eaf5] leading-tight mb-1 group-hover:text-white transition-colors">
            {phase.t}
          </div>
          {/* Description */}
          <div className="text-[11px] text-[#4d6580] leading-relaxed">
            {phase.blurb}
          </div>
        </div>

        {/* Arrow */}
        <svg
          className="w-3.5 h-3.5 text-[#253548] group-hover:text-[#7a93b0] transition-all group-hover:translate-x-0.5 mt-1 flex-shrink-0"
          viewBox="0 0 14 14" fill="none"
        >
          <path d="M2 7h10M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
    </Link>
  );
}
