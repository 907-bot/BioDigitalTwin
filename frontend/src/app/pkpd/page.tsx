"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

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

  async function run() {
    setLoading(true); setErr(null);
    try {
      const patient = { weight_kg: weight, serum_creatinine_mg_dl: creat, age, sex };
      const pkR = await api.pkpdSimulate({
        drug, dose_mg: doseMg, n_doses: nDoses, dose_interval_h: doseInterval,
        patient,
      });
      setPK(pkR);
      const peak = pkR.summary.cmax_mg_L || 1.0;
      const pdR = await api.pkpdSim({
        drug, target_biomarker: biomarker, baseline_value: baseline,
        concentration_mg_L: peak, patient,
      });
      setPD(pdR);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">PK / PD simulator</h1>
        <p className="text-sm text-muted mt-1">
          Industry-grade 2-compartment PK with allometric scaling and
          Cockcroft-Gault renal adjustment, plus sigmoid-Emax PD.
        </p>
      </div>

      <Card>
        <h2 className="text-sm font-semibold text-text mb-3">Inputs</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <label className="text-muted">Drug
            <input value={drug} onChange={e => setDrug(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Dose (mg)
            <input type="number" value={doseMg} onChange={e => setDoseMg(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Interval (h)
            <input type="number" value={doseInterval} onChange={e => setDoseInterval(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted"># doses
            <input type="number" value={nDoses} onChange={e => setNDoses(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Weight (kg)
            <input type="number" value={weight} onChange={e => setWeight(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Serum creatinine (mg/dL)
            <input type="number" step="0.1" value={creat} onChange={e => setCreat(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Age
            <input type="number" value={age} onChange={e => setAge(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Sex
            <select value={sex} onChange={e => setSex(e.target.value as any)}
                    className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full">
              <option value="male">male</option>
              <option value="female">female</option>
            </select>
          </label>
          <label className="text-muted">Biomarker (PD)
            <input value={biomarker} onChange={e => setBiomarker(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Baseline value
            <input type="number" value={baseline} onChange={e => setBaseline(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
        </div>
        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Simulating..." : "Run PK + PD"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {pk && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">PK summary — {pk.drug}</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Cmax" v={`${pk.summary.cmax_mg_L.toFixed(3)} mg/L`} />
            <Stat k="Tmax" v={`${pk.summary.tmax_h.toFixed(2)} h`} />
            <Stat k="Half-life" v={`${pk.summary.half_life_h.toFixed(2)} h`} />
            <Stat k="AUC last dose" v={`${pk.summary.auc_last_dose.toFixed(2)} mg·h/L`} />
            <Stat k="CL/F" v={`${pk.summary.apparent_clearance_L_h.toFixed(2)} L/h`} />
            <Stat k="Vss/F" v={`${pk.summary.volume_distribution_L.toFixed(1)} L`} />
            <Stat k="Steady-state" v={pk.summary.steady_state_reached ? "Yes" : "No"} />
            <Stat k="Accumulation" v={`${pk.summary.accumulation_ratio.toFixed(2)}×`} />
          </div>
          <div className="mt-3 text-xs text-muted">
            CL adjusted for renal function via Cockcroft-Gault (CrCl ≈ {pk.adjustments.crcl_ml_min.toFixed(0)} mL/min) ·
            Vd scaled allometrically from reference 70kg (factor {pk.adjustments.allometric_factor.toFixed(2)}) ·
            PK model: {pk.model} · source: {pk.source}
          </div>
        </Card>
      )}

      {pd && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">PD response — {pd.drug} → {pd.target_biomarker}</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Baseline" v={`${pd.effect.baseline.toFixed(1)}`} />
            <Stat k="At Cmax" v={`${pd.effect.at_peak_concentration.toFixed(1)}`} />
            <Stat k="Change" v={`${pd.effect.delta.toFixed(1)} (${(pd.effect.percent_change*100).toFixed(1)}%)`} />
            <Stat k="Model" v={pd.model} />
          </div>
          <div className="text-xs text-muted mt-2">
            Emax: {pd.params.emax.toFixed(2)} · EC50: {pd.params.ec50_mg_L.toFixed(3)} mg/L · Hill: {pd.params.hill_coefficient}
          </div>
        </Card>
      )}
    </div>
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
