"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";

const NAV_LINKS = [
  { href: "/cohort",    label: "Cohort",    icon: "⬡" },
  { href: "/simulate",  label: "Simulate",  icon: "⚡" },
  { href: "/causal",    label: "Causal AI", icon: "◎" },
  { href: "/chat",      label: "LLM Agent", icon: "◈" },
];

const ADV_LINKS = [
  { href: "/pharmacogenomics", label: "PGx"        },
  { href: "/polypharmacy",     label: "DDI"        },
  { href: "/pkpd",             label: "PK/PD"      },
  { href: "/trials",           label: "Trials"     },
  { href: "/regulatory",       label: "Regulatory" },
  { href: "/explain",          label: "XAI"        },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const pathname = usePathname();

  return (
    <>
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-[#1e2d42]"
        style={{ background: "rgba(7, 12, 20, 0.85)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)" }}>
        <div className="max-w-7xl mx-auto px-5 h-14 flex items-center gap-4">
          {/* Hamburger */}
          <button
            id="sidebar-toggle"
            onClick={() => setSidebarOpen(true)}
            className="flex flex-col gap-[5px] p-2 rounded-lg hover:bg-[#152035] transition-colors group"
            aria-label="Open menu"
          >
            <span className="w-[18px] h-[1.5px] bg-[#7a93b0] group-hover:bg-[#00d4a0] transition-colors" />
            <span className="w-[14px] h-[1.5px] bg-[#7a93b0] group-hover:bg-[#00d4a0] transition-colors" />
            <span className="w-[18px] h-[1.5px] bg-[#7a93b0] group-hover:bg-[#00d4a0] transition-colors" />
          </button>

          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="relative w-7 h-7">
              <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-[#00d4a0] to-[#00bcd4] opacity-80 group-hover:opacity-100 transition-opacity" />
              <div className="absolute inset-0 rounded-lg flex items-center justify-center text-[10px] font-bold text-[#070c14]">
                BDT
              </div>
            </div>
            <div className="flex items-baseline gap-1.5">
              <span className="font-semibold text-sm text-[#e2eaf5]">Bio-Digital Twin</span>
              <span className="text-[10px] text-[#4d6580] font-mono">v1.0</span>
            </div>
          </Link>

          {/* Core Nav */}
          <nav className="hidden md:flex items-center gap-0.5 ml-3">
            {NAV_LINKS.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150
                    ${active
                      ? "bg-[#00d4a0]/10 text-[#00d4a0] border border-[#00d4a0]/20"
                      : "text-[#7a93b0] hover:text-[#e2eaf5] hover:bg-[#152035]"
                    }`}
                >
                  <span className="text-[10px]">{link.icon}</span>
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Divider */}
          <div className="hidden md:block w-px h-4 bg-[#1e2d42] ml-1" />

          {/* Advanced Modules */}
          <nav className="hidden lg:flex items-center gap-0.5">
            {ADV_LINKS.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all duration-150
                    ${active
                      ? "bg-[#a78bfa]/10 text-[#a78bfa]"
                      : "text-[#4d6580] hover:text-[#7a93b0] hover:bg-[#152035]"
                    }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* Right side */}
          <div className="ml-auto flex items-center gap-3">
            <div className="hidden md:flex items-center gap-1.5 text-[11px]">
              <div className="pulse-dot" style={{ "--teal": "#00d4a0" } as React.CSSProperties} />
              <span className="text-[#00d4a0] font-medium">API</span>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#4d6580] hover:text-[#7a93b0] transition-colors"
              >
                docs ↗
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-5 py-8 animate-fade-in">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1e2d42] mt-16">
        <div className="max-w-7xl mx-auto px-5 py-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-gradient-to-br from-[#00d4a0] to-[#00bcd4] opacity-70 flex items-center justify-center text-[8px] font-bold text-[#070c14]">
              B
            </div>
            <span className="text-[11px] text-[#4d6580]">Bio-Digital Twin v1.0 · 16 Phases</span>
          </div>
          <div className="text-[11px] text-[#4d6580] font-mono">
            FastAPI · Next.js · Ollama · DoWhy · RDKit · SciPy
          </div>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="chip chip-online">System Online</span>
          </div>
        </div>
      </footer>
    </>
  );
}
