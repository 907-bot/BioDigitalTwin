"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

export default function UncertaintyPage() {
  const [pid, setPid] = useState("P000001");
  const [treatment, setTreatment] = useState("metformin");
  const [biomarker, setBiomarker] = useState("glucose");
  const [value, setValue] = useState(500);
  const [outcome, setOutcome] = useState("hba1c");
  const [n, setN] = useState(50);
  const [cf, setCf] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r: any = await api.uqCounterfactual({
        patient_id: pid, treatment, biomarker, value, outcome, n_bootstrap: n,
      });
      setCf(r);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  function CIBar({ lo, mid, hi, mn }: { lo: number; mid: number; hi: number; mn: number }) {
    const lo1 = Math.min(lo, mid, hi, mn);
    const hi1 = Math.max(lo, mid, hi, mn);
    const range = hi1 - lo1 || 1;
    const pct = (v: number) => ((v - lo1) / range) * 100;
    return (
      <div className="relative h-6 bg-bg2 rounded my-2">
        <div className="absolute h-full bg-teal/30 rounded"
             style={{ left: `${pct(lo)}%`, width: `${pct(hi) - pct(lo)}%` }} />
        <div className="absolute h-6 w-1 bg-teal"
             style={{ left: `${pct(mid)}%` }} />
        <div className="absolute h-6 w-1 bg-amber"
             style={{ left: `${pct(mn)}%` }} />
        <div className="absolute -bottom-5 text-[10px] text-muted" style={{ left: `${pct(lo)}%` }}>{lo.toFixed(2)}</div>
        <div className="absolute -bottom-5 text-[10px] text-muted" style={{ left: `${pct(hi)}%`, transform: "translateX(-100%)" }}>{hi.toFixed(2)}</div>
      </div>
    );
  }

  const e = cf?.effect;
  const conf = cf?.confidence_label;
  const dir = cf?.direction_stability;
  const wRel = cf?.ci_width_relative;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Uncertainty quantification</h1>
          <p className="text-sm text-muted mt-1">
            Bootstrap confidence intervals on individual patient counterfactuals.
            Re-fits the SCM on N resamples and reports the effect distribution
            with a confidence label and direction-stability score.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
      </div>

      <Card>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs">
          <Field label="Patient" v={pid} onChange={setPid} mono />
          <Field label="Treatment" v={treatment} onChange={setTreatment} />
          <Field label="Biomarker" v={biomarker} onChange={setBiomarker} />
          <Field label="Value" v={value} type="number" onChange={n => setValue(+n)} />
          <Field label="Outcome" v={outcome} onChange={setOutcome} />
          <Field label="N bootstrap" v={n} type="number" onChange={n => setN(+n)} />
        </div>
        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Bootstrapping..." : "Compute CI"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {e && (
        <>
          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">
              Effect: {cf.treatment} → {cf.outcome}
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm mb-2">
              <Stat k="Mean effect" v={e.mean.toFixed(3)} />
              <Stat k="Std" v={e.std.toFixed(3)} />
              <Stat k="Median (50%)" v={e.ci_50.toFixed(3)} />
              <Stat k="CI [lo, hi]" v={`[${e.ci_lo.toFixed(2)}, ${e.ci_hi.toFixed(2)}]`} />
            </div>
            <div className="text-[10px] text-muted mb-1">
              CI level: {e.ci_level} · teal = bootstrap range · amber = mean
            </div>
            <CIBar lo={e.ci_lo} mid={e.ci_50} hi={e.ci_hi} mn={e.mean} />
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Confidence assessment</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3 rounded border border-border bg-bg">
                <div className="text-[10px] text-muted uppercase mb-1">Confidence label</div>
                <span className={
                  "chip " +
                  (conf === "high" ? "chip-ok" : conf === "medium" ? "chip-info" : "chip-warn")
                }>{conf}</span>
              </div>
              <div className="p-3 rounded border border-border bg-bg">
                <div className="text-[10px] text-muted uppercase mb-1">Direction stability</div>
                <div className="text-2xl font-semibold text-text font-mono">
                  {((dir ?? 0) * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-muted mt-1">
                  fraction of bootstrap samples with same effect sign
                </div>
              </div>
              <div className="p-3 rounded border border-border bg-bg">
                <div className="text-[10px] text-muted uppercase mb-1">CI width (relative)</div>
                <div className="text-2xl font-semibold text-text font-mono">
                  {((wRel ?? 0) * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-muted mt-1">
                  {(hi1 => hi1 < 0.5 ? "narrow — reliable" : hi1 < 1.0 ? "moderate" : "wide — uncertain")(wRel ?? 0)}
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Methodology</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <Stat k="Method" v={cf.ci_method} />
              <Stat k="N requested" v={cf.n_bootstrap} />
              <Stat k="N attempted" v={cf.n_bootstrap_attempted} />
              <Stat k="Treatment value" v={`${cf.treatment_value} (forced)`} />
            </div>
            <div className="text-xs text-muted mt-3">
              The model resamples the cohort {cf.n_bootstrap}×, re-fits the SCM for each
              resample, and computes the effect of forcing <code className="text-text">{cf.treatment}</code> on
              {" "}<code className="text-text">{cf.biomarker}</code> to <code className="text-text">{cf.value}</code>{" "}
              for patient <code className="text-text">{cf.patient_id}</code>.
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Field({ label, v, onChange, type = "text", mono }: {
  label: string; v: any; onChange: (s: string) => void; type?: string; mono?: boolean;
}) {
  return (
    <label className="text-muted">
      {label}
      <input type={type} value={v} onChange={e => onChange(e.target.value)}
             className={"block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full "
                        + (mono ? "font-mono" : "")} />
    </label>
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
