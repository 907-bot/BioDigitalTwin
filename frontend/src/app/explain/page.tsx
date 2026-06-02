"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

type Mode = "counterfactual" | "ddi" | "pk" | "pgx" | "patient";

export default function XAIPage() {
  const [mode, setMode] = useState<Mode>("counterfactual");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  // counterfactual
  const [pid, setPid] = useState("P000001");
  const [treatment, setTreatment] = useState("metformin");
  const [biomarker, setBiomarker] = useState("glucose");
  const [value, setValue] = useState(500);
  const [outcome, setOutcome] = useState("hba1c");
  const [nBoot, setNBoot] = useState(20);
  // ddi
  const [drugA, setDrugA] = useState("warfarin");
  const [drugB, setDrugB] = useState("ciprofloxacin");
  // pk
  const [pkDrug, setPkDrug] = useState("metformin");
  const [pkDose, setPkDose] = useState(1000);
  const [pkWeight, setPkWeight] = useState(85);
  const [pkAge, setPkAge] = useState(72);
  const [pkCreat, setPkCreat] = useState(1.4);
  // pgx
  const [pgxDrug, setPgxDrug] = useState("codeine");

  async function run() {
    setLoading(true); setErr(null);
    try {
      let r: any;
      if (mode === "counterfactual") {
        r = await api.xaiCounterfactual({ patient_id: pid, treatment, biomarker,
                                          value, outcome, n_bootstrap: nBoot });
      } else if (mode === "ddi") {
        r = await api.xaiDDI(drugA, drugB);
      } else if (mode === "pk") {
        r = await api.xaiPK({
          drug: pkDrug, dose_mg: pkDose,
          patient: { weight_kg: pkWeight, age: pkAge, sex: "male",
                     serum_creatinine_mg_dl: pkCreat },
        });
      } else if (mode === "pgx") {
        r = await api.xaiPGx(pid, pgxDrug);
      } else {
        r = await api.xaiPatient({ patient_id: pid,
                                    drugs: [drugA, drugB, pgxDrug, treatment],
                                    treatment, outcome });
      }
      setResult(r);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const conf = result?.confidence;
  const chain = result?.reasoning_chain;
  const features = result?.feature_attribution || [];

  const confChip = (lbl: string) => {
    if (lbl === "high") return "chip-ok";
    if (lbl === "medium") return "chip-info";
    return "chip-warn";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Explainable AI</h1>
          <p className="text-sm text-muted mt-1">
            <span className="text-teal">Phase 16</span> — composes all
            upstream modules into structured reasoning chains. Every
            prediction gets a feature-attribution breakdown, a confidence
            label, and a question → evidence → conclusion chain.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
      </div>

      <Card>
        <div className="flex gap-1 flex-wrap mb-3">
          {(["counterfactual", "ddi", "pk", "pgx", "patient"] as Mode[]).map(m => (
            <button key={m} onClick={() => { setMode(m); setResult(null); }}
                    className={
                      "text-xs px-3 py-1.5 rounded font-medium " +
                      (mode === m
                        ? "bg-teal text-bg"
                        : "bg-bg2 text-muted hover:text-text")
                    }>
              {m === "counterfactual" ? "Counterfactual" :
               m === "ddi" ? "Drug interaction" :
               m === "pk" ? "PK/PD prediction" :
               m === "pgx" ? "PGx warning" : "Patient composite"}
            </button>
          ))}
        </div>

        {mode === "counterfactual" && (
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs">
            <Field label="Patient" v={pid} onChange={setPid} mono />
            <Field label="Treatment" v={treatment} onChange={setTreatment} />
            <Field label="Biomarker" v={biomarker} onChange={setBiomarker} />
            <Field label="Value" v={value} type="number" onChange={n => setValue(+n)} />
            <Field label="Outcome" v={outcome} onChange={setOutcome} />
            <Field label="N bootstrap" v={nBoot} type="number" onChange={n => setNBoot(+n)} />
          </div>
        )}
        {mode === "ddi" && (
          <div className="grid grid-cols-2 gap-3 text-xs">
            <Field label="Drug A" v={drugA} onChange={setDrugA} />
            <Field label="Drug B" v={drugB} onChange={setDrugB} />
          </div>
        )}
        {mode === "pk" && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            <Field label="Drug" v={pkDrug} onChange={setPkDrug} />
            <Field label="Dose (mg)" v={pkDose} type="number" onChange={n => setPkDose(+n)} />
            <Field label="Weight (kg)" v={pkWeight} type="number" onChange={n => setPkWeight(+n)} />
            <Field label="Age" v={pkAge} type="number" onChange={n => setPkAge(+n)} />
            <Field label="Creatinine" v={pkCreat} type="number" step="0.1" onChange={n => setPkCreat(+n)} />
          </div>
        )}
        {mode === "pgx" && (
          <div className="grid grid-cols-2 gap-3 text-xs">
            <Field label="Patient" v={pid} onChange={setPid} mono />
            <Field label="Drug" v={pgxDrug} onChange={setPgxDrug} />
          </div>
        )}
        {mode === "patient" && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <Field label="Patient" v={pid} onChange={setPid} mono />
            <Field label="Drug A" v={drugA} onChange={setDrugA} />
            <Field label="Drug B" v={drugB} onChange={setDrugB} />
            <Field label="Treatment" v={treatment} onChange={setTreatment} />
          </div>
        )}

        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Explaining..." : "Explain"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {result && (
        <>
          {conf && (
            <Card>
              <div className="flex items-center gap-4 flex-wrap">
                <div>
                  <div className="text-[10px] text-muted uppercase">Confidence</div>
                  <span className={"chip " + confChip(conf.label)}>{conf.label}</span>
                  <div className="text-[10px] text-muted mt-1">
                    score {(conf.score * 100).toFixed(0)}%
                  </div>
                </div>
                {conf.reasons && (
                  <div className="flex-1 min-w-[20rem]">
                    <div className="text-[10px] text-muted uppercase mb-1">Why this confidence?</div>
                    <ul className="text-xs text-text space-y-0.5">
                      {conf.reasons.map((r: string, i: number) => (
                        <li key={i}>· {r}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Card>
          )}

          {features.length > 0 && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Feature attribution</h2>
              <p className="text-xs text-muted mb-3">
                SHAP-lite: leave-one-out marginal contribution of each feature to the prediction.
                Bar length = |contribution|; teal = positive (drives effect up), rose = negative.
              </p>
              <div className="space-y-2">
                {features.map((f: any, i: number) => {
                  if (f.feature === "__total") {
                    return (
                      <div key={i} className="flex items-center gap-2 text-xs border-t border-border pt-2">
                        <span className="text-muted font-mono">__total (sum)</span>
                        <span className="ml-auto font-mono text-text">{f.contribution.toFixed(3)}</span>
                      </div>
                    );
                  }
                  const c = f.contribution;
                  const pct = Math.min(100, Math.abs(c) / 8);  // normalize
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-text font-mono w-32 truncate" title={f.feature}>
                        {f.feature}
                      </span>
                      <div className="flex-1 bg-bg2 rounded h-3 overflow-hidden relative">
                        <div className={"h-3 " + (c >= 0 ? "bg-teal" : "bg-rose")}
                             style={{ width: `${pct}%`, marginLeft: c >= 0 ? "0" : "auto" }} />
                      </div>
                      <span className={"font-mono w-20 text-right " + (c >= 0 ? "text-teal" : "text-rose")}>
                        {c >= 0 ? "+" : ""}{c.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {chain && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Reasoning chain</h2>
              <div className="space-y-3">
                <div>
                  <div className="text-[10px] text-muted uppercase">Question</div>
                  <div className="text-text text-sm mt-1">{chain.question}</div>
                </div>
                <div>
                  <div className="text-[10px] text-muted uppercase">Evidence ({chain.n_evidence})</div>
                  <div className="mt-2 space-y-1">
                    {chain.evidence.map((e: any, i: number) => (
                      <div key={i} className="flex items-start gap-2 text-xs bg-bg rounded p-2 border border-border">
                        <span className="text-teal font-mono w-6 text-right">{i+1}.</span>
                        <div className="flex-1">
                          <div className="text-text">{e.fact}</div>
                          <div className="text-[10px] text-muted mt-0.5">
                            source: {e.source} · weight: {typeof e.weight === "number" ? e.weight.toFixed(2) : e.weight}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="border-t border-border pt-3">
                  <div className="text-[10px] text-muted uppercase">Conclusion</div>
                  <div className="text-text text-sm mt-1 p-3 rounded bg-teal/5 border border-teal/20">
                    {chain.conclusion}
                  </div>
                </div>
                {chain.alternative_hypotheses?.length > 0 && (
                  <div>
                    <div className="text-[10px] text-muted uppercase">Alternative hypotheses</div>
                    <ul className="text-xs text-muted mt-1 space-y-0.5">
                      {chain.alternative_hypotheses.map((h: string, i: number) => (
                        <li key={i}>· {h}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </Card>
          )}

          {result.direct_interaction && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">Direct interaction</h2>
              <div className="text-sm">
                <div>
                  <span className="text-muted">Severity:</span>{" "}
                  <span className="text-text">{result.direct_interaction.severity}</span>
                </div>
                <div className="mt-1">
                  <span className="text-muted">Mechanism:</span>{" "}
                  <span className="text-text">{result.direct_interaction.mechanism}</span>
                </div>
                <div className="mt-1">
                  <span className="text-muted">Clinical effect:</span>{" "}
                  <span className="text-text">{result.direct_interaction.clinical_effect}</span>
                </div>
                {result.transitive_path && (
                  <div className="mt-2 p-2 rounded bg-bg2 text-xs text-muted">
                    Inferred path: {result.transitive_path.join(" → ")}
                  </div>
                )}
              </div>
            </Card>
          )}

          {result.layers && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Layers</h2>
              <div className="space-y-2">
                {result.layers.map((l: any, i: number) => (
                  <div key={i} className="p-3 rounded border border-border bg-bg">
                    <div className="text-xs text-muted uppercase">{l.layer}</div>
                    <div className="text-sm text-text mt-1">{l.summary}</div>
                    {l.detail && (
                      <details className="mt-2">
                        <summary className="text-[10px] text-muted cursor-pointer">
                          show details
                        </summary>
                        <pre className="text-[10px] text-muted mt-1 whitespace-pre-wrap">
                          {JSON.stringify(l.detail, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          <Card>
            <div className="text-[10px] text-muted flex items-center gap-2 flex-wrap">
              <span>method: {result.method}</span>
              <span>·</span>
              <span>latency: {result.latency_ms} ms</span>
              {result.point_estimate !== undefined && (
                <>
                  <span>·</span>
                  <span>point estimate: {result.point_estimate.toFixed(3)}</span>
                </>
              )}
              {result.cmax_approx !== undefined && (
                <>
                  <span>·</span>
                  <span>Cmax: {result.cmax_approx.toFixed(3)} mg/L</span>
                </>
              )}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Field({ label, v, onChange, type = "text", step, mono }: {
  label: string; v: any; onChange: (s: string) => void; type?: string; step?: string; mono?: boolean;
}) {
  return (
    <label className="text-muted">
      {label}
      <input type={type} value={v} step={step} onChange={e => onChange(e.target.value)}
             className={"block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full "
                        + (mono ? "font-mono" : "")} />
    </label>
  );
}
