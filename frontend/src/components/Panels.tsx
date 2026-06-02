"use client";
import clsx from "clsx";

export function Card({ title, children, action }: {
  title?: string; children: React.ReactNode; action?: React.ReactNode;
}) {
  return (
    <div className="card">
      {(title || action) && (
        <div className="flex items-center justify-between mb-3">
          {title && <h2 className="text-sm font-medium text-text">{title}</h2>}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function RiskChip({ label }: { label?: string }) {
  const k = (label || "low").toLowerCase();
  return (
    <span className={clsx("chip", {
      "chip-low": k === "low",
      "chip-moderate": k === "moderate",
      "chip-high": k === "high",
      "chip-critical": k === "critical" || k === "decompensated",
      "chip-info": k === "preclinical" || k === "healthy" || k === "clinical",
    })}>{label}</span>
  );
}

export function Stat({ label, value, sub }: {
  label: string; value: React.ReactNode; sub?: string;
}) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="text-2xl font-semibold text-text mt-1">{value}</div>
      {sub && <div className="text-xs text-muted mt-0.5">{sub}</div>}
    </div>
  );
}

export function ErrorBox({ err }: { err: unknown }) {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    <div className="card border-red/30 bg-red/5 text-sm text-red p-3">
      {msg}
    </div>
  );
}

export function Live({ ts, label = "Live" }: { ts: number; label?: string }) {
  const ago = (() => {
    const s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s/60)}m ago`;
    return `${Math.floor(s/3600)}h ago`;
  })();
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald opacity-75"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald"></span>
      </span>
      <span className="text-emerald font-medium">{label}</span>
      <span className="text-muted">{ago}</span>
    </div>
  );
}
