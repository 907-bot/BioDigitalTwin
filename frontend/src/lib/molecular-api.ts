/**
 * frontend/src/lib/molecular-api.ts
 * Client for the Spatial Pharmacology & Molecular Target Engagement API.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}/api/molecular${path}`, { cache: "no-store" });
  if (!r.ok) {
    const errorText = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${errorText}`);
  }
  return r.json();
}

async function post<T>(path: string, body: any): Promise<T> {
  const r = await fetch(`${BASE}/api/molecular${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!r.ok) {
    const errorText = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${errorText}`);
  }
  return r.json();
}

export interface MolecularTargetSummary {
  target_id: string;
  name: string;
  gene_symbol: string;
  uniprot_id: string;
  target_class: string;
  receptor_family: string | null;
  known_drugs: string[];
  expressed_organs: string[];
}

export interface TargetExpressionRecord {
  organ: string;
  region: string;
  cell_type_id: string;
  subcellular_location: string;
  status: string;
  density_pmol_mg_protein: number | null;
  copies_per_cell: number | null;
  relative_expression_tpm: number | null;
  evidence_database: string;
  confidence_score: number;
  notes: string;
}

export interface MolecularTargetDetails extends MolecularTargetSummary {
  description: string;
  binding_parameters: {
    endogenous_ligand: string | null;
    Kd_nM: number;
    k_on: number;
    k_off: number;
    residence_time_min: number;
    hill_coefficient: number;
  };
  functional_parameters: {
    mechanism_type: string;
    E_max: number;
    tau: number;
    IC50_nM: number | null;
    EC50_nM: number | null;
    Km_uM: number | null;
  };
  spatial_expressions: TargetExpressionRecord[];
}

export interface SupportedDrug {
  drug_name: string;
  target_id: string;
  target_class: string;
  mechanism: string;
  is_agonist: boolean;
  is_blocker: boolean;
  Kd_nM: number;
  IC50_nM: number | null;
  approved_indications: string[];
  evidence: {
    pubmed_id: string;
    chembl_id?: string;
    source: string;
    confidence: number;
  };
}

export interface AnatomyHierarchy {
  hierarchy: Record<
    string,
    Record<
      string,
      Array<{
        cell_type_id: string;
        name: string;
        description: string;
        markers: string[];
      }>
    >
  >;
}

export interface SpatialSimulateRequest {
  organ: string;
  cell_type_id: string;
  target_id: string;
  drug_name: string;
  plasma_concentration_mg_L?: number;
  molecular_weight_g_mol?: number;
  fu_plasma?: number;
  partition_coefficient_Kp?: number;
  is_agonist?: boolean;
  is_blocker?: boolean;
  current_plasma_glucose_mg_dL?: number;
  current_sbp_mmHg?: number;
  current_hr_bpm?: number;
  patient_gfr_mL_min?: number;
}

export interface SpatialSimulateResponse {
  status: string;
  simulation: {
    organ: string;
    region: string;
    cell_type_id: string;
    cell_type_name: string;
    subcellular_location: string;
    target_id: string;
    target_name: string;
    target_class: string;
    gene_symbol: string;
    uniprot_id: string;
    expression_status: string;
    target_density_pmol_g: number;
    copies_per_cell: number | null;
    evidence_source: string;
    evidence_confidence: number;
    engagement_status: string;
    drug_name: string;
    plasma_total_mg_L: number;
    plasma_free_nM: number;
    tissue_total_mg_L: number;
    interstitial_free_nM: number;
    luminal_free_nM: number;
    intracellular_free_nM: number;
    target_accessible_conc_nM: number;
    route_of_target_access: string;
    fractional_occupancy: number;
    modulation_type: string;
    functional_modulation_pct: number;
    effective_Kd_nM: number;
    bound_target_density_pmol_g: number;
    residence_time_min: number;
    second_messengers: {
      cAMP_uM: number;
      cytosolic_Ca2_uM: number;
      IP3_uM: number;
      PKA_activity_fraction?: number;
      Akt_phosphorylation_fraction?: number;
      MLCK_activity_fraction?: number;
    };
    primary_cellular_effect: string;
    quantitative_change_fold: number;
    affected_cellular_pathways: string[];
    cellular_description: string;
    organ_endpoints: Record<string, any>;
    whole_body_consequences: Record<string, any>;
    potential_adverse_effects: Array<{
      adverse_effect: string;
      mechanism: string;
      severity: string;
    }>;
    summary: string;
  };
}

export const molecularApi = {
  listTargets: (targetClass?: string, organ?: string) => {
    const params = new URLSearchParams();
    if (targetClass) params.set("target_class", targetClass);
    if (organ) params.set("organ", organ);
    const qs = params.toString() ? `?${params.toString()}` : "";
    return get<{ count: number; targets: MolecularTargetSummary[] }>(`/targets${qs}`);
  },

  getTarget: (targetId: string) => {
    return get<MolecularTargetDetails>(`/targets/${targetId}`);
  },

  getAnatomyHierarchy: () => {
    return get<AnatomyHierarchy>("/anatomy/hierarchy");
  },

  listSupportedDrugs: () => {
    return get<{ count: number; drugs: SupportedDrug[] }>("/drugs");
  },

  simulateSpatial: (req: SpatialSimulateRequest) => {
    return post<SpatialSimulateResponse>("/spatial-simulate", req);
  },
};
