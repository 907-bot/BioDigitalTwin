"use client";
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid, ReferenceArea } from "recharts";
import { api, Simulation } from "@/lib/api";
import { Card, Stat, RiskChip, ErrorBox } from "@/components/Panels";

const BIOMARKER_COLORS: Record<string, string> = {
  hr: "#7F77DD", hrv: "#1D9E75", spo2: "#0EA5E9", glucose: "#D85A30",
  systolic_bp: "#E24B4A", diastolic_bp: "#BA7517", bmi: "#A855F7",
};

export default function SimulatePage() {
  const [diseases, setDiseases] = useState<any>(null);
  const [interventions, setInterventions] = useState<any>(null);
  const [disease, setDisease] = useState("t2d");
  const [intervention, setIntervention] = useState<string>("");
  const [horizon, setHorizon] = useState(365);
  const [pid, setPid] = useState("P000001");
  const [usePatient, setUsePatient] = useState(true);
  const [initial, setInitial] = useState({
    hr: 72, hrv: 45, spo2: 97, glucose: 110, systolic_bp: 130, diastolic_bp: 82, bmi: 32.0,
  });
  const [sim, setSim] = useState<Simulation | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<any>(null);

  useEffect(() => {
    api.diseases().then(d => setDiseases(d.diseases));
    api.interventions().then(i => setInterventions(i.interventions));
  }, []);

  async function run() {
    setBusy(true); setErr(null); setSim(null);
    try {
      let result: Simulation;
      if (usePatient) {
        result = await api.simulatePatient(pid, disease, horizon, intervention || undefined);
      } else {
        result = await api.simulate({
          initial_state: initial, disease, horizon_days: horizon,
          dt_hours: 6, sample_every_hours: 24, rng_seed: 0,
          intervention_name: intervention || undefined,
        });
      }
      setSim(result);
    } catch (e) { setErr(e); }
    finally { setBusy(false); }
  }

  // Merge all biomarkers into a single series keyed by day
  const chartData = sim ? (() => {
    if (!sim.biomarkers[0]) return [];
    const days = sim.biomarkers[0].trajectory.map(p => p.day);
    return days.map((d, i) => {
      const row: any = { day: d };
      sim.biomarkers.forEach(b => {
        row[b.name] = b.trajectory[i]?.value;
      });
      return row;
    });
  })() : [];

  const biomarkerKeys = sim?.biomarkers.map(b => b.name) || [];

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-text">Disease simulation</h1>
      {err && <ErrorBox err={err} />}

      <Card>
        <div className="grid grid-cols-5 gap-3 items-end">
          <div>
            <div className="label">Disease</div>
            <select className="select w-full mt-1" value={disease}
                    onChange={e => setDisease(e.target.value)}>
              {diseases?.map((d: any) =>
                <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          <div>
            <div className="label">Intervention</div>
            <select className="select w-full mt-1" value={intervention}
                    onChange={e => setIntervention(e.target.value)}>
              <option value="">(none)</option>
              {interventions?.map((i: any) =>
                <option key={i.name} value={i.name}>{i.name}</option>)}
            </select>
          </div>
          <div>
            <div className="label">Horizon (days)</div>
            <input className="input w-full mt-1" type="number"
                   min={7} max={1825} value={horizon}
                   onChange={e => setHorizon(+e.target.value)} />
          </div>
          {usePatient ? (
            <div>
              <div className="label">Patient</div>
              <input className="input w-full mt-1" value={pid}
                     onChange={e => setPid(e.target.value)} />
            </div>
          ) : (
            <div className="col-span-2 grid grid-cols-4 gap-2">
              {Object.entries(initial).map(([k, v]) => (
                <div key={k}>
                  <div className="label">{k}</div>
                  <input className="input w-full mt-1" type="number"
                         value={v} onChange={e => setInitial({ ...initial, [k]: +e.target.value })} />
                </div>
              ))}
            </div>
          )}
          <div className="flex flex-col gap-2">
            <label className="text-xs text-muted flex items-center gap-1">
              <input type="checkbox" checked={usePatient}
                     onChange={e => setUsePatient(e.target.checked)} />
              Use patient
            </label>
            <button onClick={run} disabled={busy} className="btn-primary">
              {busy ? "running…" : "Simulate"}
            </button>
          </div>
        </div>
      </Card>

      {sim && (
        <>
          <div className="grid grid-cols-4 gap-4">
            <Card><Stat label="Initial risk" value={sim.initial_risk.toFixed(3)} /></Card>
            <Card><Stat label="Final risk"   value={sim.final_risk.toFixed(3)} /></Card>
            <Card><Stat label="Final state"  value={<RiskChip label={sim.disease_state} />} /></Card>
            <Card><Stat label="Spike rate"   value={`${sim.spike_view.spike_rate_hz} Hz`}
                          sub={`LIF on ${sim.spike_view.dominant_biomarker}`} /></Card>
          </div>

          <Card title="Risk evolution">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={sim.risk_evolution}>
                <CartesianGrid stroke="#1f2937" />
                <XAxis dataKey="day" stroke="#94a3b8" fontSize={11}
                       label={{ value: "day", position: "insideBottom", offset: -2, fill: "#94a3b8", fontSize: 11 }} />
                <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
                <ReferenceArea y1={0}    y2={0.25} fill="#1D9E75" fillOpacity={0.06} />
                <ReferenceArea y1={0.25} y2={0.55} fill="#BA7517" fillOpacity={0.06} />
                <ReferenceArea y1={0.55} y2={0.80} fill="#D85A30" fillOpacity={0.06} />
                <ReferenceArea y1={0.80} y2={1.01} fill="#E24B4A" fillOpacity={0.06} />
                <Line type="monotone" dataKey="risk" stroke="#7F77DD" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Biomarker trajectories">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#1f2937" />
                <XAxis dataKey="day" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
                <Legend />
                {biomarkerKeys.map(k =>
                  <Line key={k} type="monotone" dataKey={k}
                        stroke={BIOMARKER_COLORS[k] || "#7F77DD"} dot={false} strokeWidth={1.5} />)}
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}
    </div>
  );
}
