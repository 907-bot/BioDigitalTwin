"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api, CausalGraph } from "@/lib/api";
import { Card, ErrorBox, Stat } from "@/components/Panels";
import { Narrative } from "@/components/Narrative";

const Causal3D = dynamic(
  () => import("@/components/Causal3D").then((m) => m.Causal3D),
  { ssr: false, loading: () => <div className="text-muted text-sm p-8">Loading 3D view…</div> }
);

const KIND_COLOR: Record<string, string> = {
  biomarker:   "#1D9E75",
  organ:       "#7F77DD",
  disease:     "#D85A30",
  demographic: "#BA7517",
};

export default function CausalPage() {
  const [g, setG] = useState<CausalGraph | null>(null);
  const [err, setErr] = useState<any>(null);
  const [view, setView] = useState<"3d" | "2d">("3d");

  useEffect(() => {
    api.causalGraph().then(setG).catch(setErr);
  }, []);

  if (err) return <ErrorBox err={err} />;
  if (!g) return <div className="text-muted">Loading…</div>;

  // 2D radial layout
  const W = 760, H = 520, cx = W / 2, cy = H / 2;
  const byKind: Record<string, any[]> = {};
  g.nodes.forEach(n => {
    (byKind[n.kind] = byKind[n.kind] || []).push(n);
  });
  const kinds = Object.keys(byKind);
  const positioned: Record<string, { x: number; y: number; n: any }> = {};
  kinds.forEach((k, ki) => {
    const arr = byKind[k];
    const r = 90 + ki * 90;
    arr.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / arr.length + ki * 0.4;
      positioned[n.id] = { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), n };
    });
  });

  const unmapped = g.nodes.filter(n => !positioned[n.id]).length;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold text-text">Causal graph</h1>
          <p className="text-sm text-muted mt-1">
            {g.n_nodes} nodes · {g.n_edges} directed edges from the biological
            ontology + clinical driver edges (age/BMI → biomarkers).
            <br />
            <span className="text-xs">
              3D view places graph nodes at the anatomical positions of
              their human-body counterparts.
            </span>
          </p>
        </div>
        <div className="inline-flex rounded-md border border-border bg-bg/40 p-0.5 text-xs font-medium">
          <button
            onClick={() => setView("3d")}
            className={"px-3 py-1.5 rounded transition-colors " +
              (view === "3d" ? "bg-text/10 text-text" : "text-muted hover:text-text")}
          >
            3D (anatomy)
          </button>
          <button
            onClick={() => setView("2d")}
            className={"px-3 py-1.5 rounded transition-colors " +
              (view === "2d" ? "bg-text/10 text-text" : "text-muted hover:text-text")}
          >
            2D (topology)
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {kinds.map(k =>
          <Card key={k}><Stat label={k} value={byKind[k].length} /></Card>)}
      </div>

      {view === "3d" ? (
        <Card title="3D causal graph on anatomy">
          <Causal3D graph={g} height={560} />
          {unmapped > 0 && (
            <div className="text-[10px] text-muted mt-2">
              ⚠ {unmapped} node(s) not yet mapped to anatomy (shown unmapped in the 3D view).
            </div>
          )}
        </Card>
      ) : (
        <Card title="DAG layout (2D)">
          <div className="flex gap-6">
            <svg viewBox={`0 0 ${W} ${H}`} className="bg-bg rounded-md flex-1" style={{ maxWidth: W }}>
              {g.edges.map((e, i) => {
                const s = positioned[e.src], d = positioned[e.dst];
                if (!s || !d) return null;
                const w = Math.max(0.5, Math.min(2, e.weight));
                return (
                  <g key={i}>
                    <defs>
                      <marker id={`a-${i}`} viewBox="0 0 10 10" refX="9" refY="5"
                              markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />
                      </marker>
                    </defs>
                    <line x1={s.x} y1={s.y} x2={d.x} y2={d.y}
                          stroke="#475569" strokeWidth={w}
                          markerEnd={`url(#a-${i})`} />
                  </g>
                );
              })}
              {Object.values(positioned).map((p: any) => (
                <g key={p.n.id} transform={`translate(${p.x}, ${p.y})`}>
                  <circle r="20" fill={KIND_COLOR[p.n.kind] || "#1f2937"}
                          stroke="#0b0f17" strokeWidth="2" />
                  <text y="32" textAnchor="middle" fontSize="10" fill="#e5e7eb">
                    {p.n.id}
                  </text>
                </g>
              ))}
            </svg>
            <div className="text-xs text-muted space-y-2 w-48">
              <div className="label">Legend</div>
              {Object.entries(KIND_COLOR).map(([k, c]) => (
                <div key={k} className="flex items-center gap-2">
                  <span className="inline-block w-3 h-3 rounded-full" style={{ background: c }} />
                  <span>{k}</span>
                </div>
              ))}
              <div className="pt-3 label">Edge types</div>
              <ul className="space-y-1">
                {Array.from(new Set(g.edges.map(e => e.rel))).map(rel => (
                  <li key={rel}>· {rel}</li>
                ))}
              </ul>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
