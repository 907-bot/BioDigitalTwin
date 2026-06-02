"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBox } from "@/components/Panels";

const EMPTY = {
  key: "",
  name: "",
  category: "metabolic",
  description: "",
  prevalence_per_100k: 0,
  genes_involved: [] as string[],
  biomarkers: [] as string[],
  drugs_used: [] as string[],
  notes: "",
};

export default function RegistryPage() {
  const [list, setList] = useState<any[] | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [editing, setEditing] = useState<any | null>(null);
  const [err, setErr] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true); setErr(null);
    try {
      const [l, s] = await Promise.all([api.registryList(), api.registrySummary()]);
      setList(l.diseases);
      setSummary(s);
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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text">Disease registry</h1>
        <p className="text-sm text-muted mt-1">
          Postgres-backed CRUD catalog of disease entries, gene panels,
          biomarkers, and standard-of-care drug regimens. Seeded with 8 upstream diseases.
        </p>
      </div>

      {err && <ErrorBox err={err} />}

      {summary && (
        <Card>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <Stat k="Total" v={summary.total} />
            <Stat k="Upstream-seeded" v={summary.upstream_seeded} />
            <Stat k="Custom" v={summary.custom} />
            <Stat k="Categories" v={Object.keys(summary.by_category || {}).length} />
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
                <span className="chip chip-info">{d.category}</span>
                {d.upstream_seeded && <span className="chip chip-ok">seeded</span>}
                <span className="text-xs text-muted ml-auto">
                  prev {d.prevalence_per_100k}/100k
                </span>
                <button onClick={() => startEdit(d)}
                        className="text-xs text-teal hover:underline">edit</button>
                {!d.upstream_seeded && (
                  <button onClick={() => remove(d.key)}
                          className="text-xs text-rose hover:underline">delete</button>
                )}
              </div>
              <div className="text-sm text-text mt-1 font-medium">{d.name}</div>
              <div className="text-xs text-muted mt-0.5">{d.description}</div>
              {d.genes_involved?.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-2">
                  {d.genes_involved.map((g: string) => (
                    <span key={g} className="text-[10px] font-mono text-teal bg-teal/10 px-1.5 py-0.5 rounded">{g}</span>
                  ))}
                </div>
              )}
              {d.drugs_used?.length > 0 && (
                <div className="flex gap-1 flex-wrap mt-1">
                  {d.drugs_used.map((dr: string) => (
                    <span key={dr} className="text-[10px] font-mono text-amber bg-amber/10 px-1.5 py-0.5 rounded">{dr}</span>
                  ))}
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
            <Field label="Category" v={editing.category} onChange={v => update("category", v)} />
            <Field label="Prevalence per 100k" type="number" v={editing.prevalence_per_100k} onChange={v => update("prevalence_per_100k", +v || 0)} />
            <Field label="Description" v={editing.description} onChange={v => update("description", v)} full />
            <Field label="Genes (CSV)" v={(editing.genes_involved || []).join(", ")} onChange={v => updateArr("genes_involved", v)} full />
            <Field label="Biomarkers (CSV)" v={(editing.biomarkers || []).join(", ")} onChange={v => updateArr("biomarkers", v)} full />
            <Field label="Drugs (CSV)" v={(editing.drugs_used || []).join(", ")} onChange={v => updateArr("drugs_used", v)} full />
            <Field label="Notes" v={editing.notes} onChange={v => update("notes", v)} full />
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
