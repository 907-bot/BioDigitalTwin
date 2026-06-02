"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

export default function RegulatoryPage() {
  const [drug, setDrug] = useState("warfarin");
  const [profile, setProfile] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<any>(null);
  const [ts, setTs] = useState<number>(0);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const r: any = await api.regProfile(drug);
      setProfile(r);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }

  const ob = profile?.orange_book;
  const safety = profile?.safety;
  const faers = profile?.faers;
  const topReactions = faers?.top_reactions || [];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Regulatory profile</h1>
          <p className="text-sm text-muted mt-1">
            FDA orange book (curated), safety warnings, FAERS top reactions
            (live OpenFDA), and pregnancy/typical-dose information.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
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
          {ob && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-2">FDA Orange Book</h2>
              {ob.approved && ob.entries?.length > 0 ? (
                <div className="space-y-2">
                  {ob.entries.map((e: any, i: number) => (
                    <div key={i} className="p-3 rounded border border-border bg-bg text-sm">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-text font-medium">{e.trade_name}</span>
                        <span className="text-muted text-xs">({e.ingredient})</span>
                        <span className="chip chip-ok ml-auto">approved {e.approval_date}</span>
                      </div>
                      <div className="text-xs text-muted mt-1">applicant: {e.applicant}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted">No approval record in our curated snapshot.</div>
              )}
            </Card>
          )}

          {safety && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">Safety</h2>
              {safety.black_box_warnings?.length > 0 && (
                <div className="p-3 rounded border border-rose/40 bg-rose/10 mb-3">
                  <div className="text-rose font-semibold text-xs uppercase mb-1">
                    ⚠ Black-box warning
                  </div>
                  {safety.black_box_warnings.map((w: string, i: number) => (
                    <div key={i} className="text-text text-sm">{w}</div>
                  ))}
                </div>
              )}
              {safety.contraindications?.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs text-muted font-semibold mb-1">Contraindications</div>
                  <ul className="text-sm space-y-1">
                    {safety.contraindications.map((c: string, i: number) => (
                      <li key={i} className="text-text">· {c}</li>
                    ))}
                  </ul>
                </div>
              )}
              {safety.common_adverse_events?.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs text-muted font-semibold mb-1">Common adverse events</div>
                  <div className="space-y-1">
                    {safety.common_adverse_events.map((a: any, i: number) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="text-text">{a.ae}</span>
                        <span className="text-muted ml-auto font-mono">{a.frequency}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-2 text-xs mt-3">
                {safety.pregnancy_category && (
                  <div className="p-2 rounded bg-bg border border-border">
                    <div className="text-[10px] text-muted uppercase">Pregnancy</div>
                    <div className="text-text text-sm font-mono mt-0.5">{safety.pregnancy_category}</div>
                  </div>
                )}
                {safety.typical_dose && (
                  <div className="p-2 rounded bg-bg border border-border">
                    <div className="text-[10px] text-muted uppercase">Typical dose</div>
                    <div className="text-text text-sm mt-0.5">{safety.typical_dose}</div>
                  </div>
                )}
              </div>
              {safety.notes && (
                <div className="text-xs text-muted mt-3 p-2 bg-bg2 rounded">
                  {safety.notes}
                </div>
              )}
            </Card>
          )}

          {faers && (
            <Card>
              <h2 className="text-sm font-semibold text-text mb-3">
                FAERS (live OpenFDA)
                {faers.total_reports > 0 && (
                  <span className="text-muted text-xs font-normal ml-2">
                    {faers.total_reports.toLocaleString()} total reports
                  </span>
                )}
              </h2>
              {faers.error ? (
                <div className="text-xs text-amber">OpenFDA error: {faers.error}</div>
              ) : topReactions.length === 0 ? (
                <div className="text-xs text-muted">No reactions reported.</div>
              ) : (
                <div className="space-y-1">
                  {topReactions.slice(0, 10).map((r: any, i: number) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-text w-56 truncate" title={r.reaction}>
                        {r.reaction.toLowerCase().replace(/\b\w/g, (c: string) => c.toUpperCase())}
                      </span>
                      <div className="flex-1 bg-bg2 rounded h-2 overflow-hidden">
                        <div className="bg-rose h-2"
                             style={{ width: `${(r.count / topReactions[0].count) * 100}%` }} />
                      </div>
                      <span className="text-muted font-mono w-20 text-right">
                        {r.count.toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
