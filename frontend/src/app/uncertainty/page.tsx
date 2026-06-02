"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

export default function UncertaintyPage() {
  const [pid, setPid] = useState("P000001");
  const [treatment, setTreatment] = useState("metformin");
  const [biomarker, setBiomarker] = useState("glucose");
  const [value, setValue] = useState(500);
  const [outcome, setOutcome] = useState("glucose");
  const [n, setN] = useState(50);
  const [cf, setCf] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r = await api.uqCounterfactual({
        patient_id: pid,
        treatment, biomarker, value,
        n_bootstrap: n,
      });
      setCf(r);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Uncertainty quantification</h1>
        <p className="text-sm text-muted mt-1">
          Bootstrap confidence intervals on individual patient counterfactuals
          — re-fit the SCM on N resamples to get 5/50/95 percentiles
          and a confidence label (high/medium/low) + direction stability.
        </p>
      </div>

      <Card>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs">
          <label className="text-muted">Patient
            <input value={pid} onChange={e => setPid(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full font-mono" />
          </label>
          <label className="text-muted">Treatment
            <input value={treatment} onChange={e => setTreatment(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Biomarker
            <input value={biomarker} onChange={e => setBiomarker(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Value
            <input type="number" value={value} onChange={e => setValue(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">Outcome
            <input value={outcome} onChange={e => setOutcome(e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
          <label className="text-muted">N bootstrap
            <input type="number" value={n} onChange={e => setN(+e.target.value)}
                   className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full" />
          </label>
        </div>
        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Bootstrapping..." : "Compute CI"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {cf && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">Result</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Point estimate" v={cf.effect.toFixed(3)} />
            <Stat k="Median (50%)" v={cf.ci.median.toFixed(3)} />
            <Stat k="CI 5-95" v={`[${cf.ci.p5.toFixed(2)}, ${cf.ci.p95.toFixed(2)}]`} />
            <Stat k="CI 25-75" v={`[${cf.ci.p25.toFixed(2)}, ${cf.ci.p75.toFixed(2)}]`} />
            <Stat k="Mean" v={cf.ci.mean.toFixed(3)} />
            <Stat k="Std" v={cf.ci.std.toFixed(3)} />
            <Stat k="Direction stability" v={`${(cf.direction_stability * 100).toFixed(0)}%`} />
            <Stat k="Confidence" v={
              <span className={
                "chip " +
                (cf.confidence === "high" ? "chip-ok" :
                 cf.confidence === "medium" ? "chip-info" : "chip-warn")
              }>{cf.confidence}</span>
            } />
          </div>
          <div className="text-xs text-muted mt-3">
            N bootstrap = {cf.n_bootstrap} · effect on <span className="text-text">{cf.outcome}</span>
            {" "}when forcing <span className="text-text">{cf.treatment}</span> on {cf.biomarker} to <span className="text-text">{cf.value}</span>
          </div>
        </Card>
      )}
    </div>
  );
}

function Stat({ k, v }: { k: string; v: any }) {
  return (
    <div className="bg-bg rounded p-2 border border-border">
      <div className="text-[10px] text-muted uppercase">{k}</div>
      <div className="text-text text-sm font-mono mt-0.5">{v}</div>
    </div>
  );
}
