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
