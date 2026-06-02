"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox, Live } from "@/components/Panels";

const EMPTY = {
  key: "",
  name: "",
  description: "",
  target_proteins: [] as string[],
  current_treatments: "",
  unmet_need: "high",
  clinical_trials: 0,
};

export default function RegistryPage() {
  const [list, setList] = useState<any[] | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [editing, setEditing] = useState<any | null>(null);
  const [err, setErr] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [ts, setTs] = useState<number>(0);

  async function refresh() {
    setLoading(true); setErr(null);
    try {
      const [l, s]: any[] = await Promise.all([api.registryList(), api.registrySummary()]);
      setList(l.diseases);
      setSummary(s);
      setTs(Date.now());
    } catch (e: any) { setErr(e); }
    finally { setLoading(false); }
  }
  useEffect(() => { refresh(); }, []);

  async function save() {
    if (!editing) return;
    try {
      if (editing._existing) {
        await api.registryUpdate(editing.key, editing);
      } else {
        await api.registryCreate(editing);
      }
      setEditing(null);
      await refresh();
    } catch (e: any) { setErr(e); }
  }
  async function remove(key: string) {
    if (!confirm(`Delete "${key}"?`)) return;
    try { await api.registryDelete(key); await refresh(); }
    catch (e: any) { setErr(e); }
  }

  function startEdit(d: any) {
    setEditing({ ...d, _existing: true });
  }
  function startCreate() {
    setEditing({ ...EMPTY, _existing: false });
  }

  function update(field: string, val: any) {
    setEditing((e: any) => ({ ...e, [field]: val }));
  }
  function updateArr(field: string, csv: string) {
    setEditing((e: any) => ({ ...e, [field]: csv.split(",").map(s => s.trim()).filter(Boolean) }));
  }

  const unmetColor = (u: string) => {
    if (u === "critical") return "bg-rose/20 border-rose/40 text-rose";
    if (u === "high") return "bg-amber/20 border-amber/40 text-amber";
    return "bg-yellow/10 border-yellow/30 text-yellow";
  };

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text">Disease registry</h1>
          <p className="text-sm text-muted mt-1">
            Postgres-backed catalog of disease entries, target proteins, current
            treatments, and clinical-trial counts. Seeded with upstream-disease
            data, open for custom additions.
          </p>
        </div>
        {ts > 0 && <Live ts={ts} />}
      </div>

      {err && <ErrorBox err={err} />}

      {summary && (
        <Card>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Total" v={summary.n_diseases} />
            <Stat k="Target proteins" v={summary.n_target_proteins} />
            <Stat k="Clinical trials" v={summary.total_clinical_trials.toLocaleString()} />
            <Stat k="By unmet need" v={
              <div className="flex gap-1 text-[10px]">
                {Object.entries(summary.by_unmet_need).map(([k, v]: any) => (
                  <span key={k} className="px-1.5 py-0.5 rounded bg-bg2 text-muted">
                    {k}: <span className="text-text font-mono">{v}</span>
                  </span>
                ))}
              </div>
            } />
          </div>
        </Card>
      )}

      <Card>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-text">Entries</h2>
          <button onClick={startCreate}
                  className="text-xs px-3 py-1.5 bg-teal text-bg rounded-md font-medium hover:bg-teal/80">
            + New disease
          </button>
        </div>
        {loading && <div className="text-xs text-muted">Loading...</div>}
        {list && list.length === 0 && (
          <div className="text-xs text-muted">No diseases registered yet.</div>
        )}
        <div className="space-y-2">
          {list?.map((d: any) => (
            <div key={d.key} className="p-3 rounded border border-border bg-bg">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-sm text-text">{d.key}</span>
                <span className={"chip border " + unmetColor(d.unmet_need)}>
                  {d.unmet_need}
                </span>
                {d.added_by === "system" && <span className="chip chip-info">seeded</span>}
                <span className="text-xs text-muted ml-auto">
                  {d.clinical_trials?.toLocaleString()} trials
                </span>
                <button onClick={() => startEdit(d)}
                        className="text-xs text-teal hover:underline">edit</button>
                {d.added_by !== "system" && (
                  <button onClick={() => remove(d.key)}
                          className="text-xs text-rose hover:underline">delete</button>
                )}
              </div>
              <div className="text-sm text-text mt-1 font-medium">{d.name}</div>
              <div className="text-xs text-muted mt-0.5">{d.description}</div>
              {d.target_proteins?.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-2">
                  {d.target_proteins.map((p: string) => (
                    <span key={p} className="text-[10px] font-mono text-teal bg-teal/10 px-1.5 py-0.5 rounded">{p}</span>
                  ))}
                </div>
              )}
              {d.current_treatments && (
                <div className="text-xs text-muted mt-1">
                  <span className="text-muted">treatments:</span>{" "}
                  <span className="text-text">{d.current_treatments}</span>
                </div>
              )}
              {d.added_at && (
                <div className="text-[10px] text-muted mt-1">
                  added {new Date(d.added_at).toLocaleString()}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {editing && (
        <Card>
          <h2 className="text-sm font-semibold text-text mb-3">
            {editing._existing ? `Edit ${editing.key}` : "New disease"}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <Field label="Key (slug)" v={editing.key} onChange={v => update("key", v)} disabled={editing._existing} />
            <Field label="Name" v={editing.name} onChange={v => update("name", v)} />
            <label className="text-muted">Unmet need
              <select value={editing.unmet_need} onChange={e => update("unmet_need", e.target.value)}
                      className="block mt-1 px-2 py-1.5 bg-bg border border-border rounded text-text w-full">
                <option value="critical">critical</option>
                <option value="high">high</option>
                <option value="medium">medium</option>
                <option value="low">low</option>
              </select>
            </label>
            <Field label="Clinical trials" type="number" v={editing.clinical_trials} onChange={v => update("clinical_trials", +v || 0)} />
            <Field label="Description" v={editing.description} onChange={v => update("description", v)} full />
            <Field label="Target proteins (CSV)" v={(editing.target_proteins || []).join(", ")} onChange={v => updateArr("target_proteins", v)} full />
            <Field label="Current treatments" v={editing.current_treatments} onChange={v => update("current_treatments", v)} full />
          </div>
          <div className="flex gap-2 mt-4">
            <button onClick={save}
                    className="px-3 py-1.5 bg-teal text-bg rounded-md text-sm font-medium hover:bg-teal/80">
              {editing._existing ? "Save changes" : "Create"}
            </button>
            <button onClick={() => setEditing(null)}
                    className="px-3 py-1.5 bg-bg2 text-muted rounded-md text-sm hover:text-text">
              Cancel
            </button>
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
function Field({ label, v, onChange, type = "text", full, disabled }: {
  label: string; v: any; onChange: (s: string) => void; type?: string; full?: boolean; disabled?: boolean;
}) {
  return (
    <label className={"text-muted " + (full ? "md:col-span-2" : "")}>
      {label}
      <input type={type} value={v ?? ""} disabled={disabled} onChange={e => onChange(e.target.value)}
             className="block mt-1 px-2 py-1.5 bg-bg border border-border
                        rounded text-text w-full disabled:opacity-60" />
    </label>
  );
}
