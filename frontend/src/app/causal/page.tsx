"use client";
import { useEffect, useState } from "react";
import { api, CausalGraph } from "@/lib/api";
import { Card, ErrorBox, Stat } from "@/components/Panels";

const KIND_COLOR: Record<string, string> = {
  biomarker:   "#1D9E75",
  organ:       "#7F77DD",
  disease:     "#D85A30",
  demographic: "#BA7517",
};

export default function CausalPage() {
  const [g, setG] = useState<CausalGraph | null>(null);
  const [err, setErr] = useState<any>(null);

  useEffect(() => {
    api.causalGraph().then(setG).catch(setErr);
  }, []);

  if (err) return <ErrorBox err={err} />;
  if (!g) return <div className="text-muted">Loading…</div>;

  // Simple force-free radial layout: arrange by topo order in concentric rings
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

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-text">Causal graph</h1>
      <p className="text-sm text-muted">
        {g.n_nodes} nodes · {g.n_edges} directed edges derived from the biological ontology
        + clinical driver edges (age/BMI → biomarkers).
      </p>

      <div className="grid grid-cols-4 gap-4">
        {kinds.map(k =>
          <Card key={k}><Stat label={k} value={byKind[k].length} /></Card>)}
      </div>

      <Card title="DAG layout">
        <div className="flex gap-6">
          <svg viewBox={`0 0 ${W} ${H}`} className="bg-bg rounded-md flex-1" style={{ maxWidth: W }}>
            {/* edges */}
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
            {/* nodes */}
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
    </div>
  );
}
