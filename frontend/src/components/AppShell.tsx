"use client";
import Link from "next/link";
import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Sidebar open={open} onClose={() => setOpen(false)} />
      <header className="sticky top-0 z-30 bg-bg/80 backdrop-blur border-b border-border">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-4">
          <button
            onClick={() => setOpen(true)}
            className="text-muted hover:text-text text-lg"
            aria-label="Open menu"
          >
            ≡
          </button>
          <Link href="/" className="font-semibold text-text text-sm">
            <span className="text-teal">●</span> Bio-Digital Twin
            <span className="ml-2 text-xs text-muted font-normal">v0.9</span>
          </Link>
          <nav className="hidden md:flex gap-1 text-sm ml-4 flex-wrap">
            <Link href="/cohort"        className="px-3 py-1.5 rounded-md text-muted hover:text-text hover:bg-panel2">Cohort</Link>
            <Link href="/simulate"      className="px-3 py-1.5 rounded-md text-muted hover:text-text hover:bg-panel2">Simulate</Link>
            <Link href="/causal"        className="px-3 py-1.5 rounded-md text-muted hover:text-text hover:bg-panel2">Causal</Link>
            <Link href="/chat"          className="px-3 py-1.5 rounded-md text-muted hover:text-text hover:bg-panel2">Chat</Link>
            <span className="text-muted text-xs self-center mx-1">·</span>
            <Link href="/pharmacogenomics" className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">PGx</Link>
            <Link href="/polypharmacy"  className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">DDI</Link>
            <Link href="/pkpd"          className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">PK/PD</Link>
            <Link href="/trials"        className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">Trials</Link>
            <Link href="/regulatory"    className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">Regulatory</Link>
            <Link href="/explain"       className="px-3 py-1.5 rounded-md text-teal/80 hover:text-teal hover:bg-teal/10 text-xs">XAI</Link>
          </nav>
          <div className="ml-auto text-xs text-muted hidden md:block">
            API: <code className="text-text">/api</code>{" "}
            <a href="http://localhost:8000/docs" target="_blank"
               className="text-teal hover:underline">docs ↗</a>
          </div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-6">{children}</main>
      <footer className="max-w-6xl mx-auto px-6 py-6 text-xs text-muted">
        Bio-Digital Twin v0.9 · 16 phases · FastAPI + Next.js + Ollama + DoWhy + RDKit + XAI
      </footer>
    </>
  );
}
