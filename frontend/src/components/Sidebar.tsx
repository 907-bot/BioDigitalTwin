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
};

export const MODULES: Module[] = [
  // ---- Core (Phases 1-5) ----
  { n: 1, title: "Synthetic Patients",   blurb: "Cohort generation",      href: "/cohort",          group: "core" },
  { n: 2, title: "Graph Digital Twin",    blurb: "GNN + similarity",       href: "/cohort",          group: "core" },
  { n: 3, title: "Disease Dynamics",     blurb: "ODE + LIF simulator",    href: "/simulate",        group: "core" },
  { n: 4, title: "Causal AI",            blurb: "SCM + ATE/CATE",         href: "/causal",          group: "core" },
  { n: 5, title: "LLM Agent",            blurb: "Ollama llama3.1",        href: "/chat",            group: "core" },
  { n: 6, title: "Dashboard",            blurb: "Next.js + Tailwind",     href: "/",                group: "core" },
  // ---- Advancements (Phases 8-15) ----
  { n: 8,  title: "Pharmacogenomics",     blurb: "CYP panel + drug-gene",  href: "/pharmacogenomics", group: "advancements" },
  { n: 9,  title: "Drug-Drug Interact.",  blurb: "Polypharmacy checker",   href: "/polypharmacy",     group: "advancements" },
  { n: 10, title: "PK / PD",              blurb: "2-cpt + sigmoid Emax",   href: "/pkpd",             group: "advancements" },
  { n: 11, title: "Uncertainty",          blurb: "Bootstrap CIs",          href: "/uncertainty",      group: "advancements" },
  { n: 12, title: "Clinical Trials",      blurb: "ClinicalTrials.gov",     href: "/trials",           group: "advancements" },
  { n: 13, title: "Regulatory",           blurb: "FDA + FAERS + RxNorm",   href: "/regulatory",       group: "advancements" },
  { n: 14, title: "Wet-Lab",              blurb: "PAINS/SAS/IC50",         href: "/wetlab",           group: "advancements" },
  { n: 15, title: "Disease Registry",     blurb: "Postgres-backed CRUD",   href: "/registry",         group: "advancements" },
];

export function Sidebar({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const path = usePathname();
  useEffect(() => { onClose(); }, [path]); // close on nav

  const core = MODULES.filter((m) => m.group === "core");
  const adv = MODULES.filter((m) => m.group === "advancements");

  return (
    <>
      {/* backdrop */}
      <div
        onClick={onClose}
        className={`fixed inset-0 bg-black/40 z-40 transition-opacity ${
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        }`}
      />
      <aside
        className={`fixed left-0 top-0 bottom-0 w-72 bg-panel border-r border-border
                    z-50 transform transition-transform duration-200
                    ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="px-5 py-4 border-b border-border flex items-center gap-2">
          <span className="text-teal text-lg">●</span>
          <div>
            <div className="text-text font-semibold text-sm">Bio-Digital Twin</div>
            <div className="text-xs text-muted">v0.8 · 15 phases</div>
          </div>
          <button
            onClick={onClose}
            className="ml-auto text-muted hover:text-text text-xl"
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>
        <nav className="overflow-y-auto h-[calc(100vh-57px)] py-3">
          <SectionHeader label="Core platform" />
          {core.map((m) => (
            <SidebarItem key={m.n} m={m} active={path === m.href} />
          ))}
          <SectionHeader label="Advancements" badge="new" />
          {adv.map((m) => (
            <SidebarItem key={m.n} m={m} active={path === m.href} />
          ))}
          <div className="px-5 pt-4 pb-6 text-xs text-muted">
            Built with FastAPI · Next.js · Ollama · DoWhy · RDKit · CT.gov
          </div>
        </nav>
      </aside>
    </>
  );
}

function SectionHeader({ label, badge }: { label: string; badge?: string }) {
  return (
    <div className="px-5 pt-3 pb-1 text-[10px] uppercase tracking-wider
                    text-muted font-semibold flex items-center gap-2">
      <span>{label}</span>
      {badge && (
        <span className="px-1.5 py-0.5 rounded bg-teal/10 text-teal normal-case
                         tracking-normal font-medium">
          {badge}
        </span>
      )}
    </div>
  );
}

function SidebarItem({ m, active }: { m: Module; active: boolean }) {
  return (
    <Link
      href={m.href}
      className={`flex items-start gap-3 px-5 py-2 text-sm transition-colors
                  ${active
                    ? "bg-panel2 text-text"
                    : "text-muted hover:text-text hover:bg-panel2/50"}`}
    >
      <span className={`mt-0.5 text-xs font-mono w-5 text-right
                        ${active ? "text-teal" : "text-muted"}`}>
        {m.n}
      </span>
      <span className="flex-1">
        <div className="text-text leading-tight">{m.title}</div>
        <div className="text-[11px] text-muted leading-tight">{m.blurb}</div>
      </span>
    </Link>
  );
}
