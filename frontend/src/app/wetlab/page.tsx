"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

const PRESETS = [
  { name: "Aspirin",      smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Atorvastatin", smiles: "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccc(F)cc3)n(CC(O)CC(O)CC(=O)O)c1c4ccccc4" },
  { name: "Metformin",    smiles: "CN(C)C(=N)NC(=N)N" },
  { name: "Imatinib",     smiles: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1ncc(-c2cccnc2)c(Cl)n1" },
  { name: "Ibuprofen",    smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
];

export default function WetlabPage() {
  const [smiles, setSmiles] = useState(PRESETS[0].smiles);
  const [name, setName] = useState(PRESETS[0].name);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);

  function pick(p: typeof PRESETS[number]) {
    setName(p.name);
    setSmiles(p.smiles);
  }

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r = await api.wetlabValidate(smiles);
      setResult(r);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const scoreColor = (s: number) => {
    if (s >= 80) return "text-emerald";
    if (s >= 60) return "text-amber";
    return "text-rose";
  };
  const verdictColor = (v: string) => {
    if (v?.includes("ready")) return "chip-ok";
    if (v?.includes("caution")) return "chip-warn";
    return "chip-warn";
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Wet-lab validation</h1>
        <p className="text-sm text-muted mt-1">
          RDKit-powered molecule triage: Lipinski Ro5, Veber, PAINS, Brenk,
          synthetic accessibility, predicted IC50, target inference, and
          rule-based hepatotoxicity / cardiotoxicity flags.
        </p>
      </div>

      <Card>
        <div className="flex gap-1 flex-wrap mb-3">
          {PRESETS.map(p => (
            <button key={p.name} onClick={() => pick(p)}
                    className="text-xs px-2 py-1 rounded bg-bg2 text-muted
                               hover:text-text hover:bg-bg2/70">
              {p.name}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="text-xs text-muted md:col-span-1">
            Name
            <input value={name} onChange={e => setName(e.target.value)}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-full" />
          </label>
          <label className="text-xs text-muted md:col-span-2">
            SMILES
            <input value={smiles} onChange={e => setSmiles(e.target.value)}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-full font-mono text-xs" />
          </label>
        </div>
        <button onClick={run} disabled={loading}
                className="mt-4 px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                           font-medium hover:bg-teal/80 disabled:opacity-50">
          {loading ? "Validating..." : "Validate"}
        </button>
      </Card>

      {err && <ErrorBox err={err} />}

      {result && (
        <>
          <Card>
            <div className="flex items-center gap-4 flex-wrap">
              <div>
                <div className="text-[10px] text-muted uppercase">Score</div>
                <div className={"text-3xl font-bold font-mono " + scoreColor(result.score)}>
                  {result.score}
                </div>
                <div className="text-[10px] text-muted">/100</div>
              </div>
              <div className="flex-1">
                <div className="text-[10px] text-muted uppercase mb-1">Verdict</div>
                <span className={"chip " + verdictColor(result.verdict)}>
                  {result.verdict.replace(/_/g, " ")}
                </span>
                <div className="text-xs text-muted mt-2">{result.recommendation}</div>
              </div>
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Physico-chemical</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <Stat k="MW" v={result.properties.molecular_weight.toFixed(1) + " Da"} />
              <Stat k="LogP" v={result.properties.logp.toFixed(2)} />
              <Stat k="HBD" v={result.properties.h_bond_donors} />
              <Stat k="HBA" v={result.properties.h_bond_acceptors} />
              <Stat k="TPSA" v={result.properties.tpsa.toFixed(1) + " Å²"} />
              <Stat k="Rot. bonds" v={result.properties.rotatable_bonds} />
              <Stat k="Rings" v={result.properties.ring_count} />
              <Stat k="Aromatic rings" v={result.properties.aromatic_rings} />
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-text mb-3">Rule checks</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
              <Rule k="Lipinski Ro5" pass={result.rules.lipinski.pass} detail={result.rules.lipinski.detail} />
              <Rule k="Veber"        pass={result.rules.veber.pass}   detail={result.rules.veber.detail} />
              <Rule k="PAINS"        pass={!result.rules.pains.has_pains} detail={result.rules.pains.has_pains ? `${result.rules.pains.matched_filters.length} filter(s)` : "clean"} />
              <Rule k="Brenk"        pass={!result.rules.brenk.has_brenk} detail={result.rules.brenk.has_brenk ? `${result.rules.brenk.matched_alerts.length} alert(s)` : "clean"} />
              <Rule k="Egan"         pass={result.rules.egan.pass}    detail={`TPSA≤131.6 & LogP 5.88`} />
              <Rule k="Synthetic access." v={`SAS = ${result.rules.sas.score.toFixed(2)}`} />
            </div>
          </Card>

          {result.target_inference && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">Predicted target</h2>
              <div className="text-sm">
                <div>
                  <span className="text-muted">Top hit:</span>{" "}
                  <span className="text-text">{result.target_inference.top_target.name}</span>{" "}
                  <span className="text-muted text-xs">
                    (Tanimoto {result.target_inference.top_target.similarity.toFixed(3)})
                  </span>
                </div>
                <div className="text-xs text-muted mt-1">
                  Class: <span className="text-text">{result.target_inference.top_target.target_class}</span> ·
                  Method: {result.target_inference.method}
                </div>
              </div>
            </Card>
          )}

          {result.activity_prediction && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">Predicted potency</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                <Stat k="Predicted IC50" v={`${result.activity_prediction.predicted_ic50_nM.toFixed(0)} nM`} />
                <Stat k="Hill coefficient" v={result.activity_prediction.hill_coefficient.toFixed(2)} />
                <Stat k="Activity class" v={result.activity_prediction.activity_class} />
                <Stat k="Method" v={result.activity_prediction.method} />
              </div>
            </Card>
          )}

          {result.toxicity_flags?.length > 0 && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">Toxicity flags</h2>
              <div className="space-y-1">
                {result.toxicity_flags.map((f: string, i: number) => (
                  <div key={i} className="text-sm text-rose">⚠ {f}</div>
                ))}
              </div>
            </Card>
          )}
        </>
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
function Rule({ k, pass, detail }: { k: string; pass?: boolean; detail?: string; v?: string }) {
  return (
    <div className="bg-bg rounded p-2 border border-border">
      <div className="flex items-center gap-2 text-xs">
        <span className={
          pass === undefined ? "text-muted" :
          pass ? "text-emerald" : "text-rose"
        }>
          {pass === undefined ? "·" : pass ? "✓" : "✗"}
        </span>
        <span className="text-text">{k}</span>
      </div>
      {detail && <div className="text-[10px] text-muted mt-1">{detail}</div>}
    </div>
  );
}
