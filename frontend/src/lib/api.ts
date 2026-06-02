/**
 * Thin client for the FastAPI backend.
 * In dev, calls go through Next's rewrite -> http://api:8000.
 * In a browser without rewrites (e.g. local node), set NEXT_PUBLIC_API_URL.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}/api${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}

async function post<T>(path: string, body: any): Promise<T> {
  const r = await fetch(`${BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}

async function put<T>(path: string, body: any): Promise<T> {
  const r = await fetch(`${BASE}/api${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}/api${path}`, {
    method: "DELETE",
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}

// --- types ---
export type Patient = {
  patient_id: string; age: number; gender: string; bmi: number;
  hr: number; hrv: number; spo2: number; glucose: number;
  systolic_bp: number; diastolic_bp: number;
  risk_score?: number; risk_label?: string;
};

export type Simulation = {
  disease: string; horizon_days: number; steps: number;
  disease_state: string; final_risk: number; initial_risk: number;
  risk_evolution: { day: number; risk: number }[];
  biomarkers: { name: string; label: string; unit: string;
                healthy_lo: number; healthy_hi: number;
                baseline: number;
                trajectory: { day: number; value: number }[] }[];
  spike_view: { dominant_biomarker: string; spike_count: number; spike_rate_hz: number };
  intervention_applied: Record<string, number> | null;
};

export type Counterfactual = {
  disease: string; horizon_days: number;
  intervention_applied: Record<string, number> | null;
  control: { final_risk: number; disease_state: string; final_state: Record<string, number> };
  treated: { final_risk: number; disease_state: string; final_state: Record<string, number> };
  counterfactual_effect: {
    absolute_risk_reduction: number; relative_risk_reduction: number;
    state_changed: boolean; from_state: string; to_state: string;
  };
};

export type CausalNode = { id: string; kind: string; name: string };
export type CausalEdge = { src: string; dst: string; rel: string; weight: number };
export type CausalGraph = { n_nodes: number; n_edges: number;
                            nodes: CausalNode[]; edges: CausalEdge[] };

export type ChatReply = {
  session_id: string; user_message: string; reply: string;
  backend: string; tool_calls: { tool: string; args: any }[];
  elapsed_s: number; turn: number;
};

export type PatientCounterfactual = {
  patient_id: string; treatment: string; outcome: string;
  factual: number; counterfactual: number; effect: number;
  effect_direction: string;
};

// --- endpoints ---
export const api = {
  health:        () => get<{ status: string; phase: string }>("/health"),
  generate:      (n = 500) => post<any>("/generate-patients", { n }),
  buildGraph:    (use_neo4j = false) =>
                  post<any>(`/phase2/build-graph?threshold=0.80&use_neo4j=${use_neo4j}`, {}),
  trainGnn:      (epochs = 50) => post<any>(`/phase2/train-gnn?epochs=${epochs}`, {}),
  embedding:     (pid: string) => get<{ embedding: number[] }>(`/phase2/embedding/${pid}`),
  graphStats:    () => get<any>("/phase2/graph-stats"),
  clusterSummary:(k = 4) => get<any>(`/phase2/cluster-summary?n_clusters=${k}`),

  diseases:      () => get<any>("/phase3/diseases"),
  interventions: () => get<any>("/phase3/interventions"),
  attractors:    () => get<any>("/phase3/attractors"),
  simulate:      (body: {
    initial_state: Record<string, number>;
    disease: string; horizon_days?: number; dt_hours?: number;
    intervention_name?: string; intervention?: Record<string, number>;
    sample_every_hours?: number; rng_seed?: number;
  }) => post<Simulation>("/phase3/simulate", body),
  simulatePatient: (pid: string, disease: string, horizon = 365,
                    intervention_name?: string) =>
    get<any>(`/phase3/patients/${pid}/simulate?disease=${disease}`
             + `&horizon_days=${horizon}`
             + (intervention_name ? `&intervention_name=${intervention_name}` : "")),
  counterfactual: (body: {
    initial_state: Record<string, number>;
    disease: string; horizon_days?: number; intervention_name?: string;
    intervention?: Record<string, number>;
  }) => post<Counterfactual>("/phase3/counterfactual", body),

  causalGraph:   () => get<CausalGraph>("/phase4/causal-graph"),
  buildScm:      (force = true) => post<any>(`/phase4/build-scm?force=${force}`, {}),
  patientCounterfactual: (body: PatientCounterfactual) =>
    post<PatientCounterfactual>("/phase4/patient-counterfactual", body),

  chatTools:     () => get<{ tools: { name: string; description: string; args: any }[] }>("/phase5/tools"),
  chat:          (session_id: string, message: string) =>
                  post<ChatReply>("/phase5/chat", { session_id, message }),
  chatHistory:   (session_id: string) =>
                  get<{ turns: number; messages: { role: string; content: string }[] }>(
                    `/phase5/history?session_id=${session_id}`),
  chatReset:     (session_id: string) =>
                  post<{ status: string }>(`/phase5/reset?session_id=${session_id}`, {}),

  // ---- Phase 8 — Pharmacogenomics ----
  pgxGenes:      () => get<{ genes: string[] }>("/phase8/genes"),
  pgxRegistry:   (drug?: string) => get<any>(`/phase8/registry${drug ? `?drug=${drug}` : ""}`),
  pgxProfile:    (pid: string) => get<any>(`/phase8/patients/${pid}/pgx`),
  pgxCheck:      (patient_id: string, drugs: string[]) =>
                  post<any>("/phase8/patients/pgx-check", { patient_id, drugs }),

  // ---- Phase 9 — DDI ----
  ddiCheck:      (drugs: string[]) => post<any>("/phase9/check", { drugs }),
  ddiPair:       (drug_a: string, drug_b: string) => post<any>("/phase9/pair", { drug_a, drug_b }),
  ddiRules:      () => get<any>("/phase9/rules"),
  ddiGraph:      () => get<any>("/phase9/graph"),

  // ---- Phase 10 — PK/PD ----
  pkpdDrugs:     () => get<any>("/phase10/drugs"),
  pkpdDrug:      (name: string) => get<any>(`/phase10/drugs/${name}`),
  pkpdSimulate:  (body: any) => post<any>("/phase10/pk/simulate", body),
  pkpdSim:       (body: any) => post<any>("/phase10/pd/simulate", body),
  pkpdPop:       (body: any) => post<any>("/phase10/population", body),

  // ---- Phase 11 — UQ ----
  uqCounterfactual: (body: any) => post<any>("/phase11/patient-counterfactual", body),
  uqAte:           (treatment: string, outcome: string,
                    common_causes?: string[], n_bootstrap = 50) =>
                    post<any>(`/phase11/ate?treatment=${treatment}&outcome=${outcome}`
                              + `&n_bootstrap=${n_bootstrap}`
                              + (common_causes?.length ? `&common_causes=${common_causes.join(",")}` : ""),
                              {}),

  // ---- Phase 12 — Trials ----
  trialsSearch:  (q: string, by: "condition" | "drug" = "condition", max = 10) =>
                  get<any>(`/phase12/trials/search?q=${encodeURIComponent(q)}&by=${by}&max=${max}`),
  trialDetail:   (nct_id: string) => get<any>(`/phase12/trials/${nct_id}`),

  // ---- Phase 13 — Regulatory ----
  regProfile:    (drug: string) => get<any>(`/phase13/drugs/${drug}/regulatory`),
  regBlackBox:   (drug: string) => get<any>(`/phase13/drugs/${drug}/black-box`),
  regFaers:      (drug: string) => get<any>(`/phase13/drugs/${drug}/faers`),
  regApproval:   (drug: string) => get<any>(`/phase13/drugs/${drug}/approval`),
  regRxNorm:     (name: string) => get<any>(`/phase13/rxnorm/normalize?name=${encodeURIComponent(name)}`),
  regSnapshot:   () => get<any>("/phase13/registry/snapshot"),

  // ---- Phase 14 — Wet-Lab ----
  wetlabValidate:    (smiles: string) => post<any>("/phase14/validate", { smiles }),
  wetlabRDKit:       () => get<any>("/phase14/rdkit-version"),

  // ---- Phase 15 — Registry ----
  registryList:      () => get<any>("/phase15/registry/diseases"),
  registryGet:       (key: string) => get<any>(`/phase15/registry/diseases/${key}`),
  registryCreate:    (body: any) => post<any>("/phase15/registry/diseases", body),
  registryUpdate:    (key: string, body: any) => put<any>(`/phase15/registry/diseases/${key}`, body),
  registryDelete:    (key: string) => del<any>(`/phase15/registry/diseases/${key}`),
  registrySummary:   () => get<any>("/phase15/registry/summary"),

  // ---- Phase 16 — Explainable AI ----
  xaiMethods:             () => get<any>("/phase16/methods"),
  xaiCounterfactual:      (body: any) => post<any>("/phase16/explain/counterfactual", body),
  xaiDDI:                 (drug_a: string, drug_b: string) =>
                          post<any>("/phase16/explain/ddi", { drug_a, drug_b }),
  xaiPK:                  (body: any) => post<any>("/phase16/explain/pk", body),
  xaiPGx:                 (patient_id: string, drug: string) =>
                          post<any>("/phase16/explain/pgx", { patient_id, drug }),
  xaiPatient:             (body: any) => post<any>("/phase16/explain/patient", body),
};
