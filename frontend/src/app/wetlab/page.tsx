"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

const PRESETS = [
  { name: "Aspirin",      smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Atorvastatin", smiles: "CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccc(F)cc3)n(CC(O)CC(O)CC(=O)O)c1c4ccccc4" },
  { name: "Metformin",    smiles: "CN(C)C(=N)NC(=N)N" },
  { name: "Imatinib",     smiles: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1ncc(-c2cccnc2)c(Cl)n1" },
  { name: "Ibuprofen",    smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
  { name: "Diazepam",     smiles: "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21" },
];

export default function WetlabPage() {
  const [smiles, setSmiles] = useState(PRESETS[0].smiles);
  const [name, setName] = useState(PRESETS[0].name);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  function pick(p: typeof PRESETS[number]) {
    setName(p.name); setSmiles(p.smiles);
  }

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r = await api.wetlabValidate(smiles);
      setResult(r);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  function DRChart({ curve, ic50 }: { curve: any[]; ic50: number }) {
    if (!curve || curve.length === 0) return null;
    const W = 540, H = 240, P = 40;
    const xs = curve.map(p => p.concentration_M);
    const ys = curve.map(p => p.response_pct);
    const xMax = Math.max(...xs);
    const yMax = 100;
    const xLog = (v: number) => {
      const lmin = -9, lmax = Math.log10(xMax);
      const lv = Math.log10(Math.max(v, 1e-12));
      return P + ((lv - lmin) / (lmax - lmin)) * (W - 2*P);
    };
    const y = (v: number) => H - P - (v / yMax) * (H - 2*P);
    const path = curve.map((p, i) => `${i === 0 ? "M" : "L"} ${xLog(p.concentration_M).toFixed(1)} ${y(p.response_pct).toFixed(1)}`).join(" ");
    // IC50 line
    const ic50X = xLog(ic50 * 1e-9);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-56">
        <line x1={P} y1={H-P} x2={W-P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        <line x1={P} y1={P} x2={P} y2={H-P} stroke="currentColor" strokeOpacity="0.2"/>
        {/* 50% response line */}
        <line x1={P} y1={y(50)} x2={W-P} y2={y(50)} stroke="#94a3b8" strokeOpacity="0.4" strokeDasharray="3 3"/>
        <text x={W-P-2} y={y(50)-4} fill="#94a3b8" fontSize="9" textAnchor="end">50% response</text>
        {ic50X >= P && ic50X <= W-P && (
          <>
            <line x1={ic50X} y1={P} x2={ic50X} y2={H-P} stroke="#f472b6" strokeOpacity="0.6" strokeDasharray="3 3"/>
            <text x={ic50X+2} y={P+10} fill="#f472b6" fontSize="9">IC50 {(ic50/1e3).toFixed(1)} µM</text>
          </>
        )}
        <path d={path} fill="none" stroke="#a78bfa" strokeWidth="1.8"/>
        {curve.map((p, i) => (
          <circle key={i} cx={xLog(p.concentration_M)} cy={y(p.response_pct)} r="2.5" fill="#a78bfa"/>
        ))}
        {/* x-axis labels: log concentrations */}
        {[-9,-8,-7,-6,-5,-4].map(lv => {
          const px = xLog(Math.pow(10, lv));
          if (px < P || px > W-P) return null;
          return (
            <text key={lv} x={px} y={H-P+14} fill="currentColor" fillOpacity="0.5" fontSize="9" textAnchor="middle">
              10<sup>{lv}</sup>
            </text>
          );
        })}
        <text x={W/2} y={H-4} fill="currentColor" fillOpacity="0.4" fontSize="10" textAnchor="middle">[Concentration] M</text>
        <text x={10} y={H/2} fill="currentColor" fillOpacity="0.4" fontSize="10" textAnchor="middle" transform={`rotate(-90 10 ${H/2})`}>Response %</text>
      </svg>
    );
  }

  const scoreColor = (s: number) =>
    s >= 80 ? "text-emerald" : s >= 60 ? "text-amber" : "text-rose";
  const verdictChip = (v: string) => {
    if (!v) return "chip-info";
    if (v.includes("ready")) return "chip-ok";
    if (v.includes("caution") || v.includes("moderate")) return "chip-warn";
    return "chip-warn";
  };

  const p = result?.properties;
  const dl = result?.drug_likeness;
  const f = result?.filters;
  const dr = result?.dose_response;
  const tgts = result?.probable_targets || [];
  const tox = result?.toxicity_alerts || [];
  const v = result?.verdict;
  const score = result?.overall_score;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Wet-lab validation</h1>
          <p className="text-sm text-muted mt-1">
            RDKit-powered molecule triage: Lipinski Ro5, Veber, PAINS, Brenk,
            synthetic accessibility, predicted IC50, target inference, and
            toxicity flags.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
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
          <label className="text-xs text-muted">
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
            <div className="flex items-center gap-6 flex-wrap">
              <div>
                <div className="text-[10px] text-muted uppercase">Score</div>
                <div className={"text-4xl font-bold font-mono " + scoreColor(score ?? 0)}>
                  {score}
                </div>
                <div className="text-[10px] text-muted">/100</div>
              </div>
              <div className="flex-1">
                <div className="text-[10px] text-muted uppercase mb-1">Verdict</div>
                <span className={"chip " + verdictChip(v)}>
                  {v?.replace(/_/g, " ")}
                </span>
                <div className="text-xs text-muted mt-2">
                  RDKit {result.rdkit_version} · {p?.mw?.toFixed(1)} Da · {p?.logp?.toFixed(2)} LogP
                </div>
              </div>
            </div>
          </Card>

          {p && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Physico-chemical</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                <Stat k="MW" v={`${p.mw.toFixed(1)} Da`} />
                <Stat k="LogP" v={p.logp.toFixed(2)} />
                <Stat k="HBD" v={p.hbd} />
                <Stat k="HBA" v={p.hba} />
                <Stat k="TPSA" v={`${p.tpsa.toFixed(1)} Å²`} />
                <Stat k="Rot. bonds" v={p.rotatable_bonds} />
                <Stat k="Rings" v={p.rings} />
                <Stat k="Aromatic rings" v={p.aromatic_rings} />
              </div>
            </Card>
          )}

          {dl && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Drug-likeness</h2>
              <div className="grid grid-cols-2 md:grid-cols-2 gap-3 text-sm">
                <Rule k="Lipinski Ro5"
                      pass={dl.n_lipinski_violations === 0}
                      detail={dl.n_lipinski_violations === 0
                              ? "No violations"
                              : `${dl.n_lipinski_violations} violation(s): ${dl.lipinski_violations.join(", ")}`} />
                <Rule k="Veber"
                      pass={dl.n_veber_violations === 0}
                      detail={dl.n_veber_violations === 0
                              ? "Good oral bioavailability"
                              : `${dl.n_veber_violations} violation(s)`} />
              </div>
            </Card>
          )}

          {f && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Filters</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <Rule k="PAINS"
                      pass={f.pains_clean}
                      detail={f.pains_clean ? "No pan-assay interference" : `${f.pains_matches.length} match(es)`} />
                <Rule k="Brenk"
                      pass={f.brenk_clean}
                      detail={f.brenk_clean ? "Clean" : `Alerts: ${f.brenk_matches.join(", ")}`} />
                <Stat k="SAS" v={f.sas.toFixed(2)} />
              </div>
            </Card>
          )}

          {dr && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Dose-response prediction</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm mb-3">
                <Stat k="Predicted IC50" v={`${dr.ic50_nM >= 1000 ? (dr.ic50_nM/1000).toFixed(1)+' µM' : dr.ic50_nM.toFixed(0)+' nM'}`} />
                <Stat k="Hill coefficient" v={dr.hill_coefficient.toFixed(2)} />
                <Stat k="Estimated target" v={dr.estimated_target} />
                <Stat k="Curve points" v={dr.curve.length} />
              </div>
              <DRChart curve={dr.curve} ic50={dr.ic50_nM} />
            </Card>
          )}

          {tgts.length > 0 && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Probable targets (Tanimoto similarity)</h2>
              <div className="space-y-2">
                {tgts.map((t: any, i: number) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="text-text w-32 font-mono">{t.target}</span>
                    <div className="flex-1 bg-bg2 rounded h-2 overflow-hidden">
                      <div className="bg-teal h-2"
                           style={{ width: `${Math.min(100, t.tanimoto_similarity * 200)}%` }} />
                    </div>
                    <span className="text-muted text-xs w-16 text-right">
                      sim {t.tanimoto_similarity.toFixed(3)}
                    </span>
                    <span className="text-muted text-[10px] w-32 text-right">
                      ref: {t.reference_drug}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {tox.length > 0 ? (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">Toxicity alerts</h2>
              <div className="space-y-1">
                {tox.map((a: string, i: number) => (
                  <div key={i} className="text-sm text-rose">⚠ {a}</div>
                ))}
              </div>
            </Card>
          ) : (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-1">Toxicity</h2>
              <div className="text-sm text-emerald">✓ No rule-based toxicity alerts</div>
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
function Rule({ k, pass, detail }: { k: string; pass?: boolean; detail?: string }) {
  return (
    <div className="bg-bg rounded p-2 border border-border">
      <div className="flex items-center gap-2 text-xs">
        <span className={pass === undefined ? "text-muted" : pass ? "text-emerald" : "text-rose"}>
          {pass === undefined ? "·" : pass ? "✓" : "✗"}
        </span>
        <span className="text-text">{k}</span>
      </div>
      {detail && <div className="text-[10px] text-muted mt-1">{detail}</div>}
    </div>
  );
}
