"use client";
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api, Counterfactual } from "@/lib/api";
import { Card, Stat, RiskChip, ErrorBox } from "@/components/Panels";

export default function CounterfactualPage() {
  const [diseases, setDiseases] = useState<any>(null);
  const [interventions, setInterventions] = useState<any>(null);
  const [disease, setDisease] = useState("t2d");
  const [horizon, setHorizon] = useState(365);
  const [intervention, setIntervention] = useState("metformin");
  const [initial, setInitial] = useState({
    hr: 72, hrv: 45, spo2: 97, glucose: 110, systolic_bp: 130, diastolic_bp: 82, bmi: 32.0,
  });
  const [cf, setCf] = useState<Counterfactual | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<any>(null);

  useEffect(() => {
    api.diseases().then(d => setDiseases(d.diseases));
    api.interventions().then(i => setInterventions(i.interventions));
  }, []);

  async function run() {
    setBusy(true); setErr(null); setCf(null);
    try {
      const res = await api.counterfactual({
        initial_state: initial, disease, horizon_days: horizon,
        intervention_name: intervention || undefined,
      });
      setCf(res);
    } catch (e) { setErr(e); }
    finally { setBusy(false); }
  }

  const biomarkerData = cf ? Object.keys(cf.control.final_state).map(k => ({
    name: k,
    control: +cf.control.final_state[k].toFixed(2),
    treated: +cf.treated.final_state[k].toFixed(2),
  })) : [];

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-text">Counterfactual explorer</h1>
      <p className="text-sm text-muted">
        Compare a patient&apos;s projected biomarkers under control vs a named intervention.
      </p>
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
                   value={horizon} onChange={e => setHorizon(+e.target.value)} />
          </div>
          <div className="col-span-2 grid grid-cols-4 gap-2">
            {Object.entries(initial).map(([k, v]) => (
              <div key={k}>
                <div className="label">{k}</div>
                <input className="input w-full mt-1" type="number"
                       value={v} onChange={e => setInitial({ ...initial, [k]: +e.target.value })} />
              </div>
            ))}
          </div>
          <button onClick={run} disabled={busy} className="btn-primary">
            {busy ? "running…" : "Run counterfactual"}
          </button>
        </div>
      </Card>

      {cf && (
        <>
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <div className="label">Absolute risk reduction</div>
              <div className="text-3xl font-semibold text-teal mt-2">
                {cf.counterfactual_effect.absolute_risk_reduction.toFixed(3)}
              </div>
              <div className="text-xs text-muted mt-1">control − treated</div>
            </Card>
            <Card>
              <div className="label">Relative risk reduction</div>
              <div className="text-3xl font-semibold text-teal mt-2">
                {(cf.counterfactual_effect.relative_risk_reduction * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-muted mt-1">vs control baseline</div>
            </Card>
            <Card>
              <div className="label">State change</div>
              <div className="text-lg mt-2 flex items-center gap-2">
                <RiskChip label={cf.counterfactual_effect.from_state} />
                <span>→</span>
                <RiskChip label={cf.counterfactual_effect.to_state} />
              </div>
              <div className="text-xs text-muted mt-2">
                {cf.counterfactual_effect.state_changed ? "✓ changed" : "× no change"}
              </div>
            </Card>
          </div>

          <Card title="Final biomarker state — control vs treated">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={biomarkerData}>
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
                <Bar dataKey="control" fill="#D85A30" />
                <Bar dataKey="treated" fill="#1D9E75" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}
    </div>
  );
}
