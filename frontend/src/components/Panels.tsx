"use client";

/* ─── Card ──────────────────────────────────────────────── */
export function Card({
  title,
  subtitle,
  children,
  action,
  accent,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  accent?: string;
  className?: string;
}) {
  return (
    <div
      className={`card ${className}`}
      style={accent ? { borderColor: `${accent}30`, background: `linear-gradient(135deg, ${accent}05 0%, rgba(15,25,41,0.95) 60%)` } : {}}
    >
      {(title || action) && (
        <div className="flex items-start justify-between mb-4 gap-3">
          <div>
            {title && (
              <h2 className="text-sm font-semibold text-[#e2eaf5] leading-tight">{title}</h2>
            )}
            {subtitle && (
              <p className="text-[11px] text-[#4d6580] mt-0.5">{subtitle}</p>
            )}
          </div>
          {action && <div className="flex-shrink-0">{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

/* ─── Risk Chip ─────────────────────────────────────────── */
export function RiskChip({ label }: { label?: string }) {
  const k = (label || "low").toLowerCase();
  const cls =
    k === "low"                                  ? "chip-low"      :
    k === "moderate"                             ? "chip-moderate" :
    k === "high"                                 ? "chip-high"     :
    k === "critical" || k === "decompensated"    ? "chip-critical" :
    k === "healthy"                              ? "chip-healthy"  :
    k === "preclinical" || k === "clinical"      ? "chip-info"     :
                                                   "chip-info";
  return <span className={`chip ${cls}`}>{label}</span>;
}

/* ─── Stat ───────────────────────────────────────────────── */
export function Stat({
  label,
  value,
  sub,
  hint,
  accent,
  unit,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  hint?: string;
  accent?: string;
  unit?: string;
}) {
  const subText = sub || hint;
  return (
    <div className="space-y-1">
      <div className="label">{label}</div>
      <div
        className="text-2xl font-bold tracking-tight leading-none"
        style={{ color: accent || "#e2eaf5" }}
      >
        {value}
        {unit && <span className="text-sm font-normal text-[#4d6580] ml-1">{unit}</span>}
      </div>
      {subText && <div className="text-[11px] text-[#4d6580] leading-tight">{subText}</div>}
    </div>
  );
}

/* ─── Error Box ─────────────────────────────────────────── */
export function ErrorBox({ err }: { err: unknown }) {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    <div className="rounded-xl border border-[#f87171]/20 bg-[#f87171]/5 px-4 py-3 text-[13px] text-[#f87171] flex items-start gap-2">
      <span className="mt-0.5 flex-shrink-0 text-[#f87171]/70">⚠</span>
      <span>{msg}</span>
    </div>
  );
}

/* ─── Live Indicator ────────────────────────────────────── */
export function Live({ ts, label = "Live" }: { ts: number; label?: string }) {
  const ago = (() => {
    const s = Math.max(0, Math.floor((Date.now() - ts) / 1000));
    if (s < 60)   return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  })();

  return (
    <div className="flex items-center gap-2 text-[11px]">
      <div className="pulse-dot" style={{ "--teal": "#00d4a0" } as React.CSSProperties} />
      <span className="text-[#00d4a0] font-medium">{label}</span>
      <span className="text-[#4d6580]">{ago}</span>
    </div>
  );
}

/* ─── Spinner ────────────────────────────────────────────── */
export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      className="animate-spin text-[#00d4a0]"
    >
      <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeOpacity="0.2" strokeWidth="2" />
      <path d="M10 2a8 8 0 0 1 8 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/* ─── Empty State ────────────────────────────────────────── */
export function Empty({ message, icon = "○" }: { message: string; icon?: string }) {
  return (
    <div className="py-12 text-center space-y-2">
      <div className="text-4xl text-[#1e2d42]">{icon}</div>
      <div className="text-sm text-[#4d6580]">{message}</div>
    </div>
  );
}

/* ─── Section Header ─────────────────────────────────────── */
export function SectionTitle({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 mb-5">
      <div>
        <h1 className="text-xl font-bold text-[#e2eaf5]">{title}</h1>
        {subtitle && <p className="text-sm text-[#4d6580] mt-0.5">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

/* ─── Key-Value Row ─────────────────────────────────────── */
export function KVRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1e2d42] last:border-0">
      <span className="text-xs text-[#7a93b0]">{label}</span>
      <span className="text-xs text-[#e2eaf5] font-mono">{value}</span>
    </div>
  );
}

/* ─── Tag ────────────────────────────────────────────────── */
export function Tag({ label, color = "#7a93b0" }: { label: string; color?: string }) {
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium border"
      style={{
        color,
        borderColor: `${color}30`,
        background: `${color}10`,
      }}
    >
      {label}
    </span>
  );
}
