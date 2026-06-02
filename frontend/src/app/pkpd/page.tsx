"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";
import { Narrative } from "@/components/Narrative";

export default function PKPDPage() {
  const [drug, setDrug] = useState("warfarin");
  const [doseMg, setDoseMg] = useState(5);
  const [doseInterval, setDoseInterval] = useState(24);
  const [nDoses, setNDoses] = useState(7);
  const [weight, setWeight] = useState(70);
  const [creat, setCreat] = useState(1.0);
  const [age, setAge] = useState(60);
  const [sex, setSex] = useState<"male" | "female">("male");
  const [biomarker, setBiomarker] = useState("glucose");
  const [baseline, setBaseline] = useState(120);
  const [pk, setPK] = useState<any>(null);
  const [pd, setPD] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const patient = { weight_kg: weight, serum_creatinine_mg_dl: creat, age, sex };
      const pkR: any = await api.pkpdSimulate({
        drug, dose_mg: doseMg, n_doses: nDoses, dose_interval_h: doseInterval,
        patient,
      });
      setPK(pkR);
      const peak = pkR.pk_metrics?.cmax ?? pkR.pk_metrics?.cmax_ss ?? 1.0;
      const pdR: any = await api.pkpdSim({
        drug, target_biomarker: biomarker, baseline_value: baseline,
        concentration_mg_L: peak, dose_mg: doseMg, patient,
      });
      setPD(pdR);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  // SVG line chart for concentration curve
  function PKChart({ data, cmax, cmin }: { data: any[]; cmax: number; cmin: number }) {
    if (!data || data.length === 0) return null;
    const W = 720, H = 200, P = 30;
    const xs = data.map(p => p.t_h);
    const ys = data.map(p => p.c_mg_per_L);
    const xMax = Math.max(...xs);
    const yMax = Math.max(cmax * 1.15, ...ys);
    const x = (v: number) => P + (v / xMax) * (W - 2*P);
    const y = (v: number) => H - P - (v / yMax) * (H - 2*P);
    const path = data.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.t_h).toFixed(1)} ${y(p.c_mg_per_L).toFixed(1)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-48">
        <line x1={P} y1={H-P} x2={W-P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        <line x1={P} y1={P} x2={P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        <line x1={P} y1={y(cmax)} x2={W-P} y2={y(cmax)} stroke="#fbbf24" strokeOpacity="0.4" strokeDasharray="3 3"/>
        <line x1={P} y1={y(cmin)} x2={W-P} y2={y(cmin)} stroke="#94a3b8" strokeOpacity="0.4" strokeDasharray="3 3"/>
        <text x={W-P} y={y(cmax)-4} fill="#fbbf24" fontSize="10" textAnchor="end">Cmax {cmax.toFixed(3)}</text>
        <text x={W-P} y={y(cmin)-4} fill="#94a3b8" fontSize="10" textAnchor="end">Cmin {cmin.toFixed(3)}</text>
        <path d={path} fill="none" stroke="#2dd4bf" strokeWidth="1.5"/>
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <text key={f} x={P + f*(W-2*P)} y={H-P+15} fill="currentColor" fillOpacity="0.5" fontSize="10" textAnchor="middle">
            {(f*xMax).toFixed(0)}h
          </text>
        ))}
      </svg>
    );
  }

  function PDChart({ data, baseline }: { data: any[]; baseline: number }) {
    if (!data || data.length === 0) return null;
    const W = 720, H = 200, P = 30;
    const xs = data.map(p => p.t_h);
    const ys = data.map(p => p.effect);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys, baseline);
    const yMax = Math.max(...ys, baseline);
    const yRange = yMax - yMin || 1;
    const x = (v: number) => P + (v / xMax) * (W - 2*P);
    const y = (v: number) => H - P - ((v - yMin) / yRange) * (H - 2*P);
    const path = data.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.t_h).toFixed(1)} ${y(p.effect).toFixed(1)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-48">
        <line x1={P} y1={H-P} x2={W-P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        <line x1={P} y1={P} x2={P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        <line x1={P} y1={y(baseline)} x2={W-P} y2={y(baseline)} stroke="#94a3b8" strokeOpacity="0.5" strokeDasharray="3 3"/>
        <text x={W-P} y={y(baseline)-4} fill="#94a3b8" fontSize="10" textAnchor="end">baseline {baseline}</text>
        <path d={path} fill="none" stroke="#a78bfa" strokeWidth="1.5"/>
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <text key={f} x={P + f*(W-2*P)} y={H-P+15} fill="currentColor" fillOpacity="0.5" fontSize="10" textAnchor="middle">
            {(f*xMax).toFixed(0)}h
          </text>
        ))}
      </svg>
    );
  }

  const m = pk?.pk_metrics;
  const validation = pk?.validation_checks || [];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">PK / PD simulator</h1>
          <p className="text-sm text-muted mt-1">
            Industry-grade 2-compartment PK with allometric scaling and
            Cockcroft-Gault renal adjustment, plus sigmoid-Emax PD.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
      </div>

      <Card>
        <h2 className="text-sm font-semibold text-text mb-3">Inputs</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <Field label="Drug" v={drug} onChange={setDrug} />
          <Field label="Dose (mg)" type="number" v={doseMg} onChange={n => setDoseMg(+n)} />
          <Field label="Interval (h)" type="number" v={doseInterval} onChange={n => setDoseInterval(+n)} />
          <Field label="# doses" type="number" v={nDoses} onChange={n => setNDoses(+n)} />
          <Field label="Weight (kg)" type="number" v={weight} onChange={n => setWeight(+n)} />
          <Field label="Serum creatinine (mg/dL)" type="number" v={creat} step="0.1" onChange={n => setCreat(+n)} />
          <Field label="Age" type="number" v={age} onChange={n => setAge(+n)} />
          <label className="text-muted">Sex
            <select value={sex} onChange={e => setSex(e.target.value as any)}
                    className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full">
              <option value="male">male</option>
              <option value="female">female</option>
            </select>
          </label>
          <Field label="Biomarker (PD)" v={biomarker} onChange={setBiomarker} />
          <Field label="Baseline value" type="number" v={baseline} onChange={n => setBaseline(+n)} />
        </div>
        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Simulating..." : "Run PK + PD"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {pk?.narrative && (
        <Narrative data={pk.narrative} title="PK simulation" />
      )}
      {pd?.narrative && (
        <Narrative data={pd.narrative} title="PD response" />
      )}

      {m && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">
            PK summary — {pk.drug} {pk.regimen.dose_mg}mg q{pk.regimen.interval_h}h × {pk.regimen.n_doses}
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Cmax (last dose)" v={`${m.cmax_ss.toFixed(3)} mg/L`} />
            <Stat k="Tmax" v={`${m.tmax.toFixed(1)} h`} />
            <Stat k="Half-life" v={`${m.half_life_h.toFixed(1)} h`} />
            <Stat k="AUC 0-∞" v={`${m.auc_0_inf.toFixed(2)} mg·h/L`} />
            <Stat k="Cmin ss" v={`${m.cmin_ss.toFixed(3)} mg/L`} />
            <Stat k="Cmax ss" v={`${m.cmax_ss.toFixed(3)} mg/L`} />
            <Stat k="CL/F" v={`${m.clearance_L_per_h.toFixed(3)} L/h`} />
            <Stat k="Vss/F" v={`${m.vd_ss_L.toFixed(2)} L`} />
            <Stat k="Accumulation" v={`${m.accumulation_ratio.toFixed(2)}×`} />
            <Stat k="Time to SS" v={`${m.time_to_steady_state_h.toFixed(0)} h`} />
          </div>
        </Card>
      )}

      {pk?.concentration_curve && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">Concentration-time profile</h2>
          <PKChart data={pk.concentration_curve} cmax={m?.cmax_ss ?? 1} cmin={m?.cmin_ss ?? 0} />
          <div className="text-[10px] text-muted mt-1">
            {pk.concentration_curve.length} timesteps · ODE solved by scipy LSODA
          </div>
        </Card>
      )}

      {validation.length > 0 && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-2">Validation checks</h2>
          <div className="space-y-1">
            {validation.map((c: any, i: number) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={
                  c.status === "pass" ? "text-emerald" :
                  c.status === "info" ? "text-muted" : "text-rose"
                }>
                  {c.status === "pass" ? "✓" : c.status === "info" ? "i" : "✗"}
                </span>
                <span className="text-text">{c.description}</span>
                <span className="text-muted font-mono ml-auto">
                  {typeof c.value === "number" ? c.value.toFixed(3) : c.value}
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {pd && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">
            PD response — {pd.drug} → {pd.target_biomarker}
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Baseline" v={pd.min_effect.toFixed(1)} />
            <Stat k="At Tmax" v={pd.effect_at_tmax.toFixed(1)} />
            <Stat k="Max effect" v={pd.max_effect.toFixed(1)} />
            <Stat k="Model" v={pd.pd_model} />
            <Stat k="Δ from baseline" v={`${(pd.min_effect - pd.max_effect).toFixed(1)} ${pd.pd_unit}`} />
            <Stat k="% reduction" v={`${(((pd.min_effect - pd.max_effect) / pd.min_effect) * 100).toFixed(1)}%`} />
          </div>
          {pd.effect_curve && (
            <div className="mt-4">
              <PDChart data={pd.effect_curve} baseline={baseline} />
              <div className="text-[10px] text-muted mt-1">
                {pd.effect_curve.length} timesteps
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function Field({ label, v, onChange, type = "text", step }: {
  label: string; v: any; onChange: (s: string) => void; type?: string; step?: string;
}) {
  return (
    <label className="text-muted">
      {label}
      <input type={type} value={v} step={step} onChange={e => onChange(e.target.value)}
             className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
    </label>
  );
}
function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="bg-bg rounded p-2 border border-border">
      <div className="text-[10px] text-muted uppercase">{k}</div>
      <div className="text-text text-sm font-mono mt-0.5">{v}</div>
    </div>
  );
}
