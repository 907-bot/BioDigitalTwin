"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

export default function RegulatoryPage() {
  const [drug, setDrug] = useState("warfarin");
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r = await api.regProfile(drug);
      setProfile(r);
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Regulatory profile</h1>
        <p className="text-sm text-muted mt-1">
          FDA black-box warnings, contraindications, AEs, FAERS top reactions
          (live OpenFDA), approval history (curated Orange Book),
          and RxNorm normalization.
        </p>
      </div>

      <Card>
        <div className="flex gap-3 items-end flex-wrap">
          <label className="text-xs text-muted flex-1 min-w-[20rem]">
            Drug name
            <input value={drug} onChange={e => setDrug(e.target.value)}
                   onKeyDown={e => e.key === "Enter" && run()}
                   className="block mt-1 px-3 py-1.5 bg-bg border border-border
                              rounded-md text-text w-full" />
          </label>
          <button onClick={run} disabled={loading}
                  className="px-4 py-1.5 bg-teal text-bg rounded-md text-sm
                             font-medium hover:bg-teal/80 disabled:opacity-50">
            {loading ? "Loading..." : "Get profile"}
          </button>
        </div>
      </Card>

      {err && <ErrorBox err={err} />}

      {profile && (
        <>
          {profile.rxnorm && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">RxNorm</h2>
              <div className="text-sm">
                <span className="text-muted">Input:</span>{" "}
                <span className="text-text">{profile.rxnorm.input}</span>{" "}
                <span className="text-muted">→</span>{" "}
                <span className="text-text">{profile.rxnorm.name || "no match"}</span>
                {profile.rxnorm.rxcui && (
                  <span className="text-muted text-xs ml-2">rxcui={profile.rxnorm.rxcui}</span>
                )}
              </div>
            </Card>
          )}

          {profile.warnings && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Curated warnings</h2>
              {profile.warnings.black_box ? (
                <div className="p-3 rounded border border-rose/40 bg-rose/10 text-sm mb-3">
                  <div className="text-rose font-semibold text-xs uppercase mb-1">
                    ⚠ Black-box warning
                  </div>
                  <div className="text-text">{profile.warnings.black_box}</div>
                </div>
              ) : (
                <div className="text-xs text-muted mb-3">No black-box warning.</div>
              )}
              {profile.warnings.contraindications?.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs text-muted font-semibold mb-1">Contraindications</div>
                  <ul className="text-sm space-y-1">
                    {profile.warnings.contraindications.map((c: string, i: number) => (
                      <li key={i} className="text-text">· {c}</li>
                    ))}
                  </ul>
                </div>
              )}
              {profile.warnings.common_adverse_events?.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs text-muted font-semibold mb-1">Common AEs</div>
                  <div className="flex gap-1 flex-wrap">
                    {profile.warnings.common_adverse_events.map((a: string, i: number) => (
                      <span key={i} className="chip chip-info">{a}</span>
                    ))}
                  </div>
                </div>
              )}
              {profile.warnings.pregnancy_category && (
                <div className="text-xs text-muted">
                  Pregnancy category: <span className="text-text">{profile.warnings.pregnancy_category}</span>
                </div>
              )}
            </Card>
          )}

          {profile.faers && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">FAERS (live OpenFDA)</h2>
              {profile.faers.error ? (
                <div className="text-xs text-amber">
                  OpenFDA error: {profile.faers.error}
                </div>
              ) : (
                <>
                  <div className="text-xs text-muted mb-2">
                    {profile.faers.total_reports.toLocaleString()} total reports
                    {profile.faers.cached && " (cached)"}
                  </div>
                  <div>
                    <div className="text-xs text-muted font-semibold mb-1">Top reactions</div>
                    <div className="space-y-1">
                      {profile.faers.top_reactions.slice(0, 8).map((r: any, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-xs">
                          <span className="text-text w-48 truncate">{r.reaction}</span>
                          <div className="flex-1 bg-bg2 rounded h-2 overflow-hidden">
                            <div className="bg-rose h-2"
                                 style={{ width: `${(r.count / profile.faers.top_reactions[0].count) * 100}%` }} />
                          </div>
                          <span className="text-muted font-mono w-16 text-right">{r.count.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </Card>
          )}

          {profile.approval && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">FDA approval</h2>
              {profile.approval.found ? (
                <div className="text-sm space-y-1">
                  <div>
                    <span className="text-muted">Approved:</span>{" "}
                    <span className="text-text">{profile.approval.approval_date}</span>
                  </div>
                  <div>
                    <span className="text-muted">Form:</span>{" "}
                    <span className="text-text">{profile.approval.dosage_form}</span>
                  </div>
                  <div>
                    <span className="text-muted">Applicant:</span>{" "}
                    <span className="text-text">{profile.approval.applicant}</span>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-muted">No approval record in our curated snapshot.</div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
