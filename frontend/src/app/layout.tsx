import "./globals.css";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Bio-Digital Twin",
  description: "Phases 1-6 — synthetic patients, GNN, dynamics, causality, agent",
};

const nav = [
  { href: "/",                label: "Home" },
  { href: "/cohort",          label: "Cohort" },
  { href: "/simulate",        label: "Simulate" },
  { href: "/counterfactual",  label: "Counterfactual" },
  { href: "/causal",          label: "Causal" },
  { href: "/chat",            label: "Chat" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="sticky top-0 z-50 bg-bg/80 backdrop-blur border-b border-border">
          <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-6">
            <Link href="/" className="font-semibold text-text text-sm">
              <span className="text-teal">●</span> Bio-Digital Twin
              <span className="ml-2 text-xs text-muted font-normal">v0.6</span>
            </Link>
            <nav className="flex gap-1 text-sm">
              {nav.map((n) => (
                <Link key={n.href} href={n.href}
                      className="px-3 py-1.5 rounded-md text-muted
                                 hover:text-text hover:bg-panel2 transition-colors">
                  {n.label}
                </Link>
              ))}
            </nav>
            <div className="ml-auto text-xs text-muted">
              API: <code className="text-text">/api</code>{" "}
              <a href="http://localhost:8000/docs" target="_blank"
                 className="text-teal hover:underline">docs ↗</a>
            </div>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-6 py-6">{children}</main>
        <footer className="max-w-6xl mx-auto px-6 py-6 text-xs text-muted">
          Built with FastAPI · Next.js · Ollama · PyTorch Geometric
        </footer>
      </body>
    </html>
  );
}
