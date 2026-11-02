"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  molecularApi,
  MolecularTargetSummary,
  MolecularTargetDetails,
  SupportedDrug,
  SpatialSimulateResponse,
} from "@/lib/molecular-api";

export default function SpatialPharmacologyPage() {
  const [targets, setTargets] = useState<MolecularTargetSummary[]>([]);
  const [drugs, setDrugs] = useState<SupportedDrug[]>([]);
  const [selectedTargetId, setSelectedTargetId] = useState<string>("SGLT2");
  const [targetDetails, setTargetDetails] = useState<MolecularTargetDetails | null>(null);

  // Form state
  const [selectedOrgan, setSelectedOrgan] = useState<string>("kidney");
  const [selectedCellType, setSelectedCellType] = useState<string>("kidney_proximal_tubule_epithelial");
  const [selectedDrugName, setSelectedDrugName] = useState<string>("Empagliflozin");
  const [plasmaConc, setPlasmaConc] = useState<number>(0.25);
  const [patientGfr, setPatientGfr] = useState<number>(100.0);
  const [bloodGlucose, setBloodGlucose] = useState<number>(160.0);
  const [systolicBp, setSystolicBp] = useState<number>(130.0);
  const [heartRate, setHeartRate] = useState<number>(75.0);

  // Simulation results
  const [loading, setLoading] = useState<boolean>(false);
  const [simResult, setSimResult] = useState<SpatialSimulateResponse["simulation"] | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Initial data load
  useEffect(() => {
    molecularApi.listTargets().then((res) => {
      setTargets(res.targets);
    }).catch(console.error);

    molecularApi.listSupportedDrugs().then((res) => {
      setDrugs(res.drugs);
    }).catch(console.error);
  }, []);

  // Target detail loader
  useEffect(() => {
    if (selectedTargetId) {
      molecularApi.getTarget(selectedTargetId).then((details) => {
        setTargetDetails(details);
        if (details.spatial_expressions.length > 0) {
          const firstExp = details.spatial_expressions[0];
          setSelectedOrgan(firstExp.organ);
          setSelectedCellType(firstExp.cell_type_id);
        }
      }).catch(console.error);
    }
  }, [selectedTargetId]);

  // Handle drug selection change to synchronize target if applicable
  const handleDrugChange = (drugName: string) => {
    setSelectedDrugName(drugName);
    const matched = drugs.find((d) => d.drug_name.toLowerCase() === drugName.toLowerCase());
    if (matched) {
      setSelectedTargetId(matched.target_id);
    }
  };

  const handleSimulate = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await molecularApi.simulateSpatial({
        organ: selectedOrgan,
        cell_type_id: selectedCellType,
        target_id: selectedTargetId,
        drug_name: selectedDrugName,
        plasma_concentration_mg_L: plasmaConc,
        patient_gfr_mL_min: patientGfr,
        current_plasma_glucose_mg_dL: bloodGlucose,
        current_sbp_mmHg: systolicBp,
        current_hr_bpm: heartRate,
      });
      setSimResult(res.simulation);
    } catch (err: any) {
      setErrorMsg(err.message || "Spatial simulation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 p-6 font-sans">
      {/* Top Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between border-b border-cyan-900/50 pb-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 mb-1">
              <Link href="/" className="hover:underline text-cyan-500">COMMAND CENTER</Link>
              <span>/</span>
              <span>MOLECULAR LAB</span>
              <span>/</span>
              <span className="text-slate-400">SPATIAL PHARMACOLOGY V2</span>
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
              <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
              Spatial Pharmacology & Molecular Target Engagement Engine
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Physiologically resolved drug-target interaction across organs, tissues, cell populations, and subcellular compartments.
            </p>
          </div>
          <div className="flex gap-2 text-xs font-mono">
            <span className="px-2.5 py-1 rounded bg-cyan-950/80 border border-cyan-800/80 text-cyan-300">
              TARGETS: {targets.length}
            </span>
            <span className="px-2.5 py-1 rounded bg-blue-950/80 border border-blue-800/80 text-blue-300">
              DRUGS: {drugs.length}
            </span>
            <span className="px-2.5 py-1 rounded bg-emerald-950/80 border border-emerald-800/80 text-emerald-300">
              MODE: IN SILICO VERIFIED
            </span>
          </div>
        </div>
      </div>

      {/* Main Workspace Grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Configuration & Target Profile */}
        <div className="lg:col-span-5 space-y-6">
          {/* Target & Drug Selection Card */}
          <div className="bg-[#0c1322] border border-cyan-900/40 rounded-xl p-5 shadow-xl">
            <h2 className="text-sm font-semibold text-cyan-400 font-mono uppercase tracking-wider mb-4 flex items-center gap-2">
              <span>01.</span> Molecular Target & Drug Selection
            </h2>

            <div className="space-y-4 text-xs">
              <div>
                <label className="block text-slate-400 mb-1 font-mono">TARGET CLASS / SYMBOL</label>
                <select
                  value={selectedTargetId}
                  onChange={(e) => setSelectedTargetId(e.target.value)}
                  className="w-full bg-[#131d31] border border-slate-700 rounded-lg p-2.5 text-white font-mono focus:border-cyan-400 focus:outline-none"
                >
                  {targets.map((t) => (
                    <option key={t.target_id} value={t.target_id}>
                      {t.target_id} — {t.name} ({t.target_class.toUpperCase()})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1 font-mono">VALIDATED THERAPEUTIC AGENT</label>
                <select
                  value={selectedDrugName}
                  onChange={(e) => handleDrugChange(e.target.value)}
                  className="w-full bg-[#131d31] border border-slate-700 rounded-lg p-2.5 text-white font-mono focus:border-cyan-400 focus:outline-none"
                >
                  {drugs.map((d) => (
                    <option key={`${d.drug_name}-${d.target_id}`} value={d.drug_name}>
                      {d.drug_name} → {d.target_id} ({d.mechanism})
                    </option>
                  ))}
                </select>
              </div>

              {/* Anatomical Selection */}
              {targetDetails && (
                <div className="p-3 bg-[#0a0f1d] border border-cyan-900/30 rounded-lg space-y-3">
                  <div className="text-[11px] font-mono text-cyan-300 font-semibold uppercase">
                    Anatomical Expression Compartment
                  </div>
                  <div>
                    <label className="block text-[10px] text-slate-400 font-mono">CELL POPULATION & TISSUE</label>
                    <select
                      value={selectedCellType}
                      onChange={(e) => {
                        setSelectedCellType(e.target.value);
                        const exp = targetDetails.spatial_expressions.find((x) => x.cell_type_id === e.target.value);
                        if (exp) setSelectedOrgan(exp.organ);
                      }}
                      className="w-full bg-[#131d31] border border-slate-700 rounded p-2 text-white text-xs font-mono"
                    >
                      {targetDetails.spatial_expressions.map((exp) => (
                        <option key={exp.cell_type_id} value={exp.cell_type_id}>
                          {exp.organ.toUpperCase()} · {exp.cell_type_id} ({exp.status})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {/* Patient State Baseline */}
              <div className="grid grid-cols-2 gap-3 pt-2">
                <div>
                  <label className="block text-[10px] text-slate-400 font-mono">PLASMA CONC (mg/L)</label>
                  <input
                    type="number"
                    step="0.05"
                    value={plasmaConc}
                    onChange={(e) => setPlasmaConc(parseFloat(e.target.value) || 0.1)}
                    className="w-full bg-[#131d31] border border-slate-700 rounded p-2 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 font-mono">PATIENT GFR (mL/min)</label>
                  <input
                    type="number"
                    value={patientGfr}
                    onChange={(e) => setPatientGfr(parseFloat(e.target.value) || 90)}
                    className="w-full bg-[#131d31] border border-slate-700 rounded p-2 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 font-mono">GLUCOSE (mg/dL)</label>
                  <input
                    type="number"
                    value={bloodGlucose}
                    onChange={(e) => setBloodGlucose(parseFloat(e.target.value) || 120)}
                    className="w-full bg-[#131d31] border border-slate-700 rounded p-2 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-slate-400 font-mono">SYSTOLIC BP (mmHg)</label>
                  <input
                    type="number"
                    value={systolicBp}
                    onChange={(e) => setSystolicBp(parseFloat(e.target.value) || 120)}
                    className="w-full bg-[#131d31] border border-slate-700 rounded p-2 text-white font-mono"
                  />
                </div>
              </div>

              <button
                onClick={handleSimulate}
                disabled={loading}
                className="w-full py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-mono font-semibold rounded-lg shadow-lg shadow-cyan-900/30 transition-all flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    SIMULATING MOLECULAR ENGAGEMENT...
                  </>
                ) : (
                  "EXECUTE SPATIAL SIMULATION"
                )}
              </button>
            </div>
          </div>

          {/* Target Metadata & Provenance Card */}
          {targetDetails && (
            <div className="bg-[#0c1322] border border-slate-800 rounded-xl p-5 text-xs font-mono space-y-3">
              <h3 className="text-cyan-400 font-semibold uppercase tracking-wider flex items-center justify-between">
                <span>Target Specification</span>
                <span className="text-slate-500 text-[10px]">{targetDetails.uniprot_id}</span>
              </h3>
              <p className="text-slate-300 font-sans text-xs">{targetDetails.description}</p>
              <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                <div className="bg-[#131d31] p-2 rounded border border-slate-800">
                  <span className="text-slate-400 block">GENE SYMBOL</span>
                  <span className="text-cyan-300 font-bold">{targetDetails.gene_symbol}</span>
                </div>
                <div className="bg-[#131d31] p-2 rounded border border-slate-800">
                  <span className="text-slate-400 block">DISSOCIATION CONSTANT (Kd)</span>
                  <span className="text-emerald-300 font-bold">{targetDetails.binding_parameters.Kd_nM} nM</span>
                </div>
                <div className="bg-[#131d31] p-2 rounded border border-slate-800">
                  <span className="text-slate-400 block">MECHANISM</span>
                  <span className="text-blue-300">{targetDetails.functional_parameters.mechanism_type}</span>
                </div>
                <div className="bg-[#131d31] p-2 rounded border border-slate-800">
                  <span className="text-slate-400 block">RESIDENCE TIME</span>
                  <span className="text-amber-300">{targetDetails.binding_parameters.residence_time_min.toFixed(1)} min</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Simulation Output & Downstream Effects */}
        <div className="lg:col-span-7 space-y-6">
          {errorMsg && (
            <div className="p-4 bg-red-950/80 border border-red-800 rounded-xl text-red-200 text-xs font-mono">
              ERROR: {errorMsg}
            </div>
          )}

          {!simResult && !errorMsg && (
            <div className="bg-[#0c1322] border border-dashed border-cyan-900/40 rounded-xl p-12 text-center text-slate-500 font-mono text-xs">
              Select molecular target parameters and click <span className="text-cyan-400 font-semibold">EXECUTE SPATIAL SIMULATION</span> to generate micro-exposure and whole-body clinical response.
            </div>
          )}

          {simResult && (
            <div className="space-y-6">
              {/* Engagement Status Banner */}
              <div className={`p-4 rounded-xl border flex items-center justify-between font-mono text-xs ${
                simResult.engagement_status === "confirmed_engagement"
                  ? "bg-emerald-950/40 border-emerald-800/80 text-emerald-200"
                  : simResult.engagement_status === "target_not_expressed"
                  ? "bg-amber-950/40 border-amber-800/80 text-amber-200"
                  : "bg-red-950/40 border-red-800/80 text-red-200"
              }`}>
                <div className="flex items-center gap-3">
                  <span className={`w-3 h-3 rounded-full ${
                    simResult.engagement_status === "confirmed_engagement" ? "bg-emerald-400 animate-ping" : "bg-amber-400"
                  }`} />
                  <div>
                    <span className="font-bold uppercase tracking-wider">{simResult.engagement_status.replace(/_/g, " ")}</span>
                    <p className="text-[11px] opacity-80 font-sans mt-0.5">{simResult.summary}</p>
                  </div>
                </div>
                <div className="text-right text-[10px]">
                  <span>CONFIDENCE: {(simResult.evidence_confidence * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Micro-Exposure & Occupancy Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-[#0c1322] border border-cyan-900/40 rounded-xl p-4">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">LOCAL EXPOSURE ROUTE</span>
                  <div className="text-xs font-mono text-cyan-300 font-semibold truncate">
                    {simResult.route_of_target_access}
                  </div>
                  <div className="mt-3 text-2xl font-bold font-mono text-white">
                    {simResult.target_accessible_conc_nM.toFixed(1)} <span className="text-xs text-slate-400 font-normal">nM</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">Accessible Free Conc</span>
                </div>

                <div className="bg-[#0c1322] border border-cyan-900/40 rounded-xl p-4">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">TARGET OCCUPANCY (ρ)</span>
                  <div className="text-xs font-mono text-blue-300 font-semibold">
                    {(simResult.fractional_occupancy * 100).toFixed(1)}% Bound
                  </div>
                  <div className="mt-3 text-2xl font-bold font-mono text-blue-400">
                    {simResult.functional_modulation_pct.toFixed(1)}%
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">Functional Modulation</span>
                </div>

                <div className="bg-[#0c1322] border border-cyan-900/40 rounded-xl p-4">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">CELLULAR PHENOTYPE</span>
                  <div className="text-xs font-mono text-emerald-300 font-semibold truncate">
                    {simResult.primary_cellular_effect}
                  </div>
                  <div className="mt-3 text-2xl font-bold font-mono text-emerald-400">
                    {simResult.quantitative_change_fold.toFixed(2)}x
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">Change Fold</span>
                </div>
              </div>

              {/* Second Messengers & Pathways */}
              <div className="bg-[#0c1322] border border-slate-800 rounded-xl p-5">
                <h3 className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider mb-3">
                  Transduced Second Messengers & Intracellular Cascades
                </h3>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-2 text-center font-mono">
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">cAMP</span>
                    <span className="text-sm font-bold text-cyan-300">{simResult.second_messengers.cAMP_uM} µM</span>
                  </div>
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">CYTOSOLIC Ca2+</span>
                    <span className="text-sm font-bold text-amber-300">{simResult.second_messengers.cytosolic_Ca2_uM} µM</span>
                  </div>
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">IP3</span>
                    <span className="text-sm font-bold text-blue-300">{simResult.second_messengers.IP3_uM} µM</span>
                  </div>
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">PKA ACTIVATION</span>
                    <span className="text-sm font-bold text-emerald-300">
                      {((simResult.second_messengers.PKA_activity_fraction || 0.1) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Akt PHOSPHO</span>
                    <span className="text-sm font-bold text-purple-300">
                      {((simResult.second_messengers.Akt_phosphorylation_fraction || 0.15) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="bg-[#131d31] p-2.5 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">MLCK TONE</span>
                    <span className="text-sm font-bold text-rose-300">
                      {((simResult.second_messengers.MLCK_activity_fraction || 0.2) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800/80">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">ENGAGED CELLULAR PATHWAYS:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {simResult.affected_cellular_pathways.map((p, i) => (
                      <span key={i} className="px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 text-[11px] font-mono">
                        {p}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Organ Endpoints & Whole-Body Consequences */}
              <div className="bg-[#0c1322] border border-slate-800 rounded-xl p-5 space-y-4">
                <h3 className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
                  Organ-Level Clinical Endpoints & Whole-Body Systemic Outcomes
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                  <div className="bg-[#10192b] p-4 rounded-lg border border-slate-800">
                    <span className="text-cyan-400 font-semibold block mb-2 uppercase">Primary Organ Endpoints</span>
                    <pre className="text-slate-300 whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
                      {JSON.stringify(simResult.organ_endpoints, null, 2)}
                    </pre>
                  </div>
                  <div className="bg-[#10192b] p-4 rounded-lg border border-slate-800">
                    <span className="text-blue-400 font-semibold block mb-2 uppercase">Whole-Body Consequences</span>
                    <pre className="text-slate-300 whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
                      {JSON.stringify(simResult.whole_body_consequences, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>

              {/* FDA Safety Warnings & Adverse Effects */}
              {simResult.potential_adverse_effects.length > 0 && (
                <div className="bg-[#1a1215] border border-rose-900/60 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-mono font-semibold text-rose-400 uppercase tracking-wider flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                    Predicted Safety Flags & Clinical Adverse Reactions
                  </h3>
                  <div className="space-y-2">
                    {simResult.potential_adverse_effects.map((ae, i) => (
                      <div key={i} className="p-3 bg-[#24151b] border border-rose-900/40 rounded-lg text-xs font-mono">
                        <div className="flex items-center justify-between text-rose-300 font-semibold">
                          <span>{ae.adverse_effect}</span>
                          <span className="text-[10px] text-amber-400">{ae.severity}</span>
                        </div>
                        <p className="text-[11px] text-slate-300 font-sans mt-1">{ae.mechanism}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

