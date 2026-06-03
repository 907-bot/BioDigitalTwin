"use client";
import { useState } from "react";
import clsx from "clsx";

export type NarrativePayload = {
  headline?: string;
  lay?: string;
  scientist?: string;
  risk_level?: "low" | "moderate" | "high" | "critical" | string;
};

const RISK_COLORS: Record<string, string> = {
  low: "bg-emerald/15 text-emerald border-emerald/30",
  moderate: "bg-amber/15 text-amber border-amber/30",
  high: "bg-orange/15 text-orange border-orange/30",
  critical: "bg-red/15 text-red border-red/30",
};

const RISK_LABELS: Record<string, string> = {
  low: "Low risk",
  moderate: "Moderate risk",
  high: "High risk",
  critical: "Critical",
};

export function Narrative({
  data,
  title = "Summary",
  defaultMode = "lay",
  collapsible = true,
}: {
  data?: NarrativePayload | null;
  title?: string;
  defaultMode?: "lay" | "scientist";
  collapsible?: boolean;
}) {
  const [mode, setMode] = useState<"lay" | "scientist">(defaultMode);
  const [open, setOpen] = useState(true);

  if (!data) return null;

  const risk = (data.risk_level || "low").toLowerCase();
  const riskColor = RISK_COLORS[risk] || RISK_COLORS.low;
  const riskLabel = RISK_LABELS[risk] || risk;

  const headline = data.headline || (mode === "scientist" ? data.scientist : data.lay) || "";
  const body = mode === "scientist" ? data.scientist : data.lay;

  return (
    <div className={clsx("rounded-lg border bg-card/60 p-4", riskColor, "border-l-4")}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[10px] uppercase tracking-wider opacity-70 font-medium">
              {title}
            </span>
            <span className={clsx("text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded border", riskColor)}>
              {riskLabel}
            </span>
          </div>
          <p className="text-sm font-medium text-text leading-snug">
            {headline}
          </p>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <div className="inline-flex rounded-md border border-border bg-bg/40 p-0.5 text-[10px] font-medium">
            <button
              onClick={() => setMode("lay")}
              className={clsx("px-2 py-1 rounded transition-colors", {
                "bg-text/10 text-text": mode === "lay",
                "text-muted hover:text-text": mode !== "lay",
              })}
            >
              Plain English
            </button>
            <button
              onClick={() => setMode("scientist")}
              className={clsx("px-2 py-1 rounded transition-colors", {
                "bg-text/10 text-text": mode === "scientist",
                "text-muted hover:text-text": mode !== "scientist",
              })}
            >
              For scientists
            </button>
          </div>
          {collapsible && (
            <button
              onClick={() => setOpen(!open)}
              className="text-muted hover:text-text px-2 py-1 text-xs"
              aria-label="Toggle details"
            >
              {open ? "−" : "+"}
            </button>
          )}
        </div>
      </div>
      {open && body && body !== headline && (
        <div className="mt-3 pt-3 border-t border-current/10">
          <p className="text-xs text-muted leading-relaxed whitespace-pre-wrap">
            {body}
          </p>
        </div>
      )}
    </div>
  );
}

export function NarrativeList({
  items,
  title = "Items",
  emptyText = "No items",
}: {
  items: { narrative?: NarrativePayload; label?: string }[] | undefined;
  title?: string;
  emptyText?: string;
}) {
  if (!items || items.length === 0) {
    return (
      <div className="text-xs text-muted italic">{emptyText}</div>
    );
  }
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex gap-3 items-start">
          {item.label && (
            <div className="text-xs font-mono text-muted shrink-0 pt-2 min-w-[80px]">
              {item.label}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <Narrative data={item.narrative} title="" collapsible={false} />
          </div>
        </div>
      ))}
    </div>
  );
}
