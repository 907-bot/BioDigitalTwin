"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";
import { Narrative, NarrativeList } from "@/components/Narrative";

export default function PharmacogenomicsPage() {
  const [pid, setPid] = useState("P000001");
  const [drugs, setDrugs] = useState("warfarin, codeine, clopidogrel, sertraline");
  const [profile, setProfile] = useState<any>(null);
  const [check, setCheck] = useState<any>(null);
  const [err, setErr] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const [prof, chk] = await Promise.all([
        api.pgxProfile(pid),
        api.pgxCheck(pid, drugs.split(",").map(s => s.trim()).filter(Boolean)),
      ]);
      setProfile(prof);
      setCheck(chk);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Pharmacogenomics</h1>
        <p className="text-sm text-muted mt-1">
          8-gene CYP / Phase-II panel. Per-patient metabolizer status (PM/IM/EM/UM)
          modulates the drug effect in counterfactuals.
        </p>
      </div>
      <Card>
        <div className="flex gap-3 items-end flex-wrap">
          <label className="text-xs text-muted">
            Patient
            <input value={pid} onChange={e => setPid(e.target.value)}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-32 font-mono" />
          </label>
          <label className="text-xs text-muted flex-1 min-w-[20rem]">
            Drug list (comma-separated)
            <input value={drugs} onChange={e => setDrugs(e.target.value)}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-full" />
          </label>
          <button onClick={run} disabled={loading}
                  className="px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                             font-medium hover:bg-teal/80 disabled:opacity-50">
            {loading ? "Running..." : "Run PGx check"}
          </button>
        </div>
      </Card>

      {err && <ErrorBox err={err} />}

      {profile?.narrative && (
        <Narrative data={profile.narrative} title="PGx profile" />
      )}

      {check?.narrative && (
        <Narrative data={check.narrative} title="Drug-gene check" />
      )}

      {profile && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">
            PGx profile — {profile.patient_id}
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {profile.genes.map((g: any) => (
              <div key={g.gene} className="bg-bg rounded p-2 border border-border">
                <div className="text-xs text-muted font-mono">{g.gene}</div>
                <div className="flex items-center justify-between mt-1">
                  <span className={
                    "chip " +
                    (g.status === "PM" ? "chip-warn" :
                     g.status === "UM" ? "chip-info" : "chip-ok")
                  }>
                    {g.status}
                  </span>
                  <span className="text-xs text-text">a={g.activity.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {check && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">
            Drug-gene warnings — {check.patient_id}
            {check.n_warnings > 0 && (
              <span className="ml-2 text-xs text-muted">
                {check.n_warnings} warning{check.n_warnings !== 1 ? "s" : ""}
                {" · "}
                highest: <span className={
                  check.highest_severity === "critical" ? "text-rose" :
                  check.highest_severity === "major" ? "text-amber" :
                  "text-muted"
                }>{check.highest_severity}</span>
              </span>
            )}
          </h2>
          {check.warnings.length === 0 ? (
            <div className="text-sm text-muted">No pharmacogenomic warnings for this regimen.</div>
          ) : (
            <>
              <table className="w-full text-sm mb-4">
                <thead className="text-xs text-muted">
                  <tr>
                    <th className="text-left py-1">Drug</th>
                    <th className="text-left py-1">Gene</th>
                    <th className="text-left py-1">Status</th>
                    <th className="text-left py-1">Severity</th>
                    <th className="text-left py-1">Impact</th>
                    <th className="text-left py-1">Clinical note</th>
                  </tr>
                </thead>
                <tbody>
                  {check.warnings.map((w: any, i: number) => (
                    <tr key={i} className="border-t border-border">
                      <td className="py-2 font-mono">{w.drug}</td>
                      <td className="py-2 font-mono">{w.gene}</td>
                      <td className="py-2">{w.patient_status}</td>
                      <td className="py-2">
                        <span className={
                          "chip " +
                          (w.severity === "critical" ? "chip-warn" :
                           w.severity === "major" ? "chip-warn" : "chip-info")
                        }>{w.severity}</span>
                      </td>
                      <td className="py-2 font-mono">{w.impact_factor.toFixed(2)}×</td>
                      <td className="py-2 text-muted text-xs">{w.clinical_note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="mt-4 space-y-2">
                <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-2">
                  What this means
                </h3>
                <NarrativeList
                  items={check.warnings.map((w: any) => ({
                    label: `${w.drug} · ${w.gene}`,
                    narrative: w.narrative,
                  }))}
                />
              </div>
            </>
          )}
        </Card>
      )}
    </div>
  );
}
