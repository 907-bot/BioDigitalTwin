"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api } from "@/lib/api";
import { Card, Stat, RiskChip, ErrorBox } from "@/components/Panels";

const RISK_COLORS: Record<string, string> = {
  low: "#1D9E75", moderate: "#BA7517", high: "#D85A30", critical: "#E24B4A",
  preclinical: "#7F77DD", healthy: "#1D9E75", clinical: "#D85A30", decompensated: "#E24B4A",
};

export default function CohortPage() {
  const [stats, setStats] = useState<any>(null);
  const [clusters, setClusters] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string>("");
  const [err, setErr] = useState<any>(null);

  async function refresh() {
    try {
      const [s, c] = await Promise.all([api.graphStats(), api.clusterSummary(4)]);
      setStats(s); setClusters(c);
    } catch (e) { setErr(e); }
  }

  useEffect(() => { refresh(); }, []);

  async function pipeline() {
    setBusy(true); setMsg("generating patients…"); setErr(null);
    try {
      await api.generate(500); setMsg("building graph…");
      await api.buildGraph(true); setMsg("training GNN…");
      await api.trainGnn(50); setMsg("done");
      await refresh();
    } catch (e) { setErr(e); }
    finally { setBusy(false); }
  }

  const distData = stats && clusters ? [
    ...(clusters.clusters || []).map((c: any) => ({
      name: `Cluster ${c.cluster_id}`,
      size: c.patient_count, age: c.avg_age, bmi: c.avg_bmi, glu: c.avg_glucose,
    })),
  ] : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-text">Cohort overview</h1>
        <button onClick={pipeline} disabled={busy} className="btn-primary">
          {busy ? "running…" : "Run full pipeline"}
        </button>
      </div>
      {msg && <div className="text-xs text-muted">{msg}</div>}
      {err && <ErrorBox err={err} />}

      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <Card><Stat label="Patients" value={stats.node_count} /></Card>
          <Card><Stat label="Edges" value={stats.edge_count} /></Card>
          <Card><Stat label="Avg degree" value={stats.avg_degree} /></Card>
          <Card><Stat label="Density" value={stats.density} /></Card>
        </div>
      )}

      {clusters && (
        <>
          <Card title="Cluster distribution (KMeans on GNN embeddings)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={distData}>
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
                <Bar dataKey="size" fill="#1D9E75" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Clusters">
            <div className="grid grid-cols-2 gap-3">
              {clusters.clusters?.map((c: any) => (
                <div key={c.cluster_id} className="bg-panel2 rounded-md p-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Cluster {c.cluster_id}</span>
                    <span className="text-xs text-muted">{c.patient_count} patients</span>
                  </div>
                  <div className="text-xs text-muted mt-2 grid grid-cols-2 gap-1">
                    <div>avg age: <span className="text-text">{c.avg_age}</span></div>
                    <div>avg BMI: <span className="text-text">{c.avg_bmi}</span></div>
                    <div>avg glu: <span className="text-text">{c.avg_glucose}</span></div>
                    <div>avg HR: <span className="text-text">{c.avg_hr}</span></div>
                    <div>avg SpO2: <span className="text-text">{c.avg_spo2}</span></div>
                    <div>% female: <span className="text-text">{c.pct_female}</span></div>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {c.sample_ids?.slice(0, 4).map((id: string) => (
                      <Link key={id} href={`/cohort?patient=${id}`}
                            className="text-xs text-teal hover:underline">{id}</Link>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      {!stats && !err && <div className="text-muted text-sm">No graph yet — click "Run full pipeline".</div>}
    </div>
  );
}
