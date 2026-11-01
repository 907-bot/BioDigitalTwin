"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export type Module = {
  n: number;
  title: string;
  blurb: string;
  href: string;
  group: "core" | "advancements";
  icon: string;
};

export const MODULES: Module[] = [
  // ---- Core (Phases 1-6) ----
  { n: 1,  title: "Synthetic Patients",  blurb: "500-patient cohort generator",  href: "/cohort",          group: "core",         icon: "⬡" },
  { n: 2,  title: "Graph Digital Twin",  blurb: "GCN/GAT encoder & similarity",  href: "/cohort",          group: "core",         icon: "◎" },
  { n: 3,  title: "Disease Dynamics",    blurb: "ODE + LIF neural simulator",    href: "/simulate",        group: "core",         icon: "⚡" },
  { n: 4,  title: "Causal AI",           blurb: "SCM · ATE · CATE analysis",     href: "/causal",          group: "core",         icon: "◈" },
  { n: 5,  title: "LLM Agent",           blurb: "Ollama llama3.1 + 13 tools",    href: "/chat",            group: "core",         icon: "◑" },
  { n: 6,  title: "Dashboard",           blurb: "Live monitoring hub",           href: "/",                group: "core",         icon: "▦" },
  // ---- Advancements (Phases 8-16) ----
  { n: 8,  title: "Pharmacogenomics",    blurb: "CYP panel + drug-gene rules",   href: "/pharmacogenomics", group: "advancements", icon: "⬟" },
  { n: 9,  title: "Drug-Drug Interact.", blurb: "60+ DDI pairs + CYP graph",     href: "/polypharmacy",     group: "advancements", icon: "⬡" },
  { n: 10, title: "PK / PD",             blurb: "2-cpt model · sigmoid Emax",    href: "/pkpd",             group: "advancements", icon: "⌑" },
  { n: 11, title: "Uncertainty QA",      blurb: "Bootstrap CIs · UQ pipeline",   href: "/uncertainty",      group: "advancements", icon: "≋" },
  { n: 12, title: "Clinical Trials",     blurb: "ClinicalTrials.gov v2 live",    href: "/trials",           group: "advancements", icon: "⊕" },
  { n: 13, title: "Regulatory",          blurb: "FDA · FAERS · RxNorm checks",   href: "/regulatory",       group: "advancements", icon: "⊛" },
  { n: 14, title: "Wet-Lab",             blurb: "PAINS · SAS · IC50 · tox",      href: "/wetlab",           group: "advancements", icon: "⎔" },
  { n: 15, title: "Disease Registry",    blurb: "Postgres-backed CRUD catalog",  href: "/registry",         group: "advancements", icon: "▤" },
  { n: 16, title: "Explainable AI",      blurb: "SHAP-lite + reasoning chains",  href: "/explain",          group: "advancements", icon: "◬" },
  { n: 17, title: "Personal Twin Studio",blurb: "Real metrics · Workouts · OTC · Cell", href: "/personal-twin", group: "advancements", icon: "🧬" },
  { n: 18, title: "Real-World Twin",   blurb: "Samsung Health CSVs · ML Models · HITL", href: "/samsung-twin", group: "advancements", icon: "⌚" },
];

export function Sidebar({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const path = usePathname();
  useEffect(() => { onClose(); }, [path]); // close on navigation

  const core = MODULES.filter((m) => m.group === "core");
  const adv  = MODULES.filter((m) => m.group === "advancements");

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-40 transition-all duration-300 ${
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        }`}
        style={{ background: "rgba(7, 12, 20, 0.7)", backdropFilter: "blur(4px)" }}
      />

      {/* Drawer */}
      <aside
        className={`fixed left-0 top-0 bottom-0 w-80 z-50
                    transform transition-transform duration-300 ease-out
                    ${open ? "translate-x-0" : "-translate-x-full"}`}
        style={{ background: "#0d1422", borderRight: "1px solid #1e2d42" }}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[#1e2d42]">
          <div className="relative w-9 h-9 flex-shrink-0">
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-[#00d4a0] to-[#00bcd4]" />
            <div className="absolute inset-0 rounded-xl flex items-center justify-center text-[11px] font-bold text-[#070c14]">
              BDT
            </div>
          </div>
          <div className="flex-1">
            <div className="text-sm font-semibold text-[#e2eaf5]">Bio-Digital Twin</div>
            <div className="text-[11px] text-[#4d6580] font-mono">v1.0 · 16 phases</div>
          </div>
          <button
            onClick={onClose}
            id="sidebar-close"
            className="w-8 h-8 rounded-lg flex items-center justify-center text-[#4d6580] hover:text-[#e2eaf5] hover:bg-[#152035] transition-all"
            aria-label="Close sidebar"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        {/* Nav */}
        <nav className="overflow-y-auto h-[calc(100vh-65px)] py-4 flex flex-col gap-0.5 px-3">
          {/* Core Section */}
          <SectionHeader label="Core Platform" subtitle="Phases 1–6" />
          <div className="flex flex-col gap-0.5 mb-3">
            {core.map((m) => (
              <SidebarItem key={m.n} m={m} active={path === m.href} isAdv={false} />
            ))}
          </div>

          {/* Advancements */}
          <SectionHeader label="Clinical Modules" subtitle="Phases 8–16" badge="16 phases" />
          <div className="flex flex-col gap-0.5 mb-3">
            {adv.map((m) => (
              <SidebarItem key={m.n} m={m} active={path === m.href} isAdv={true} />
            ))}
          </div>

          {/* Footer */}
          <div className="mt-auto mx-1 px-3 py-4 rounded-xl border border-[#1e2d42] text-[11px] text-[#4d6580]">
            <div className="flex items-center gap-2 mb-2">
              <div className="pulse-dot" style={{ "--teal": "#00d4a0" } as React.CSSProperties} />
              <span className="text-[#00d4a0] font-medium">All systems operational</span>
            </div>
            <div>FastAPI · Next.js · Ollama</div>
            <div>DoWhy · RDKit · SciPy</div>
          </div>
        </nav>
      </aside>
    </>
  );
}

function SectionHeader({ label, subtitle, badge }: { label: string; subtitle: string; badge?: string }) {
  return (
    <div className="px-2 pt-1 pb-2 flex items-center justify-between">
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[#7a93b0]">{label}</div>
        <div className="text-[10px] text-[#4d6580] font-mono">{subtitle}</div>
      </div>
      {badge && (
        <span className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-[#00d4a0]/10 text-[#00d4a0] border border-[#00d4a0]/20 uppercase tracking-wider">
          {badge}
        </span>
      )}
    </div>
  );
}

function SidebarItem({ m, active, isAdv }: { m: Module; active: boolean; isAdv: boolean }) {
  const accentColor = isAdv ? "#a78bfa" : "#00d4a0";
  return (
    <Link
      href={m.href}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150 group
        ${active
          ? `text-[${accentColor}]`
          : "text-[#7a93b0] hover:text-[#e2eaf5] hover:bg-[#152035]"
        }`}
      style={active ? {
        background: `${accentColor}12`,
        border: `1px solid ${accentColor}25`,
        color: accentColor,
      } : {}}
    >
      {/* Phase number */}
      <div
        className="w-7 h-7 rounded-lg flex items-center justify-center text-[10px] font-bold font-mono flex-shrink-0 transition-all"
        style={active ? {
          background: `${accentColor}20`,
          color: accentColor,
          border: `1px solid ${accentColor}30`,
        } : {
          background: "#152035",
          color: "#4d6580",
          border: "1px solid #1e2d42",
        }}
      >
        {m.n}
      </div>
      {/* Text */}
      <div className="flex-1 min-w-0">
        <div className="text-[13px] font-medium leading-tight truncate">{m.title}</div>
        <div className="text-[10px] text-[#4d6580] leading-tight mt-0.5 truncate">{m.blurb}</div>
      </div>
      {/* Active indicator */}
      {active && (
        <div
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ background: accentColor }}
        />
      )}
    </Link>
  );
}
