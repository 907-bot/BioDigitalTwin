"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { HumanTwinCanvas, ANATOMICAL_ORGANS } from "@/components/HumanTwinCanvas";
import { api } from "@/lib/api";

export default function CommandCenter() {
  const [selectedOrganKey, setSelectedOrganKey] = useState<string>("kidney");
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [activeScenario, setActiveScenario] = useState<string>("baseline");

  // Telemetry vitals
  const [vitals, setVitals] = useState({
    hr: 72,
    sbp: 126,
    dbp: 82,
    glucose: 145,
    sao2: 98.2,
    ph: 7.395,
    gfr: 95.0,
  });

  useEffect(() => {
    api.health()
      .then(setHealthStatus)
      .catch(() => setHealthStatus({ status: "connected_local" }));
  }, []);

  const selectedOrgan = ANATOMICAL_ORGANS[selectedOrganKey] || ANATOMICAL_ORGANS["kidney"];

  const handleScenarioTrigger = (scenario: string) => {
    setActiveScenario(scenario);
    if (scenario === "empagliflozin") {
      setSelectedOrganKey("kidney");
      setVitals({ hr: 72, sbp: 121, dbp: 79, glucose: 95, sao2: 98.4, ph: 7.391, gfr: 98.0 });
    } else if (scenario === "semaglutide") {
      setSelectedOrganKey("pancreas");
      setVitals({ hr: 74, sbp: 124, dbp: 80, glucose: 90, sao2: 98.5, ph: 7.402, gfr: 95.0 });
    } else if (scenario === "atenolol") {
      setSelectedOrganKey("heart");
      setVitals({ hr: 58, sbp: 116, dbp: 75, glucose: 140, sao2: 98.0, ph: 7.398, gfr: 94.0 });
    } else if (scenario === "atorvastatin") {
      setSelectedOrganKey("liver");
      setVitals({ hr: 71, sbp: 125, dbp: 81, glucose: 142, sao2: 98.2, ph: 7.396, gfr: 95.0 });
    } else {
      setSelectedOrganKey("kidney");
      setVitals({ hr: 72, sbp: 126, dbp: 82, glucose: 145, sao2: 98.2, ph: 7.395, gfr: 95.0 });
    }
  };

  return (
    <div className="h-screen w-full bg-[#050811] text-slate-100 flex flex-col overflow-hidden font-sans select-none">
      {/* ─── Top Command Bar ────────────────────────────────────────────── */}
      <header className="h-14 border-b border-cyan-900/40 bg-[#070c18]/90 backdrop-blur px-6 flex items-center justify-between z-30">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="font-mono font-bold tracking-widest text-sm text-white">
              BIODIGITAL<span className="text-cyan-400">TWIN</span>
            </span>
          </div>
          <span className="text-slate-600 font-mono text-xs">|</span>
          <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
            <span className="text-cyan-400/80">SUBJECT:</span>
            <span className="text-white font-semibold">P000001</span>
            <span className="text-[10px] text-slate-500">(MALE, 70 KG, METABOLIC / CARDIOVASCULAR TWIN)</span>
          </div>
        </div>

        {/* Real-time vitals HUD */}
        <div className="hidden md:flex items-center gap-5 text-xs font-mono">
          <div className="flex items-center gap-1.5 bg-[#0d1627] px-3 py-1 rounded border border-slate-800">
            <span className="text-slate-400">HR:</span>
            <span className="text-rose-400 font-bold">{vitals.hr}</span>
            <span className="text-[10px] text-slate-500">bpm</span>
          </div>
          <div className="flex items-center gap-1.5 bg-[#0d1627] px-3 py-1 rounded border border-slate-800">
            <span className="text-slate-400">BP:</span>
            <span className="text-cyan-400 font-bold">{vitals.sbp}/{vitals.dbp}</span>
            <span className="text-[10px] text-slate-500">mmHg</span>
          </div>
          <div className="flex items-center gap-1.5 bg-[#0d1627] px-3 py-1 rounded border border-slate-800">
            <span className="text-slate-400">GLU:</span>
            <span className="text-amber-400 font-bold">{vitals.glucose}</span>
            <span className="text-[10px] text-slate-500">mg/dL</span>
          </div>
          <div className="flex items-center gap-1.5 bg-[#0d1627] px-3 py-1 rounded border border-slate-800">
            <span className="text-slate-400">pH:</span>
            <span className="text-emerald-400 font-bold">{vitals.ph}</span>
          </div>
          <div className="flex items-center gap-1.5 bg-[#0d1627] px-3 py-1 rounded border border-slate-800">
            <span className="text-slate-400">GFR:</span>
            <span className="text-blue-400 font-bold">{vitals.gfr}</span>
            <span className="text-[10px] text-slate-500">mL/min</span>
          </div>
        </div>

        {/* Engine status indicator */}
        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>ENGINE ONLINE</span>
          </div>
          <Link
            href="/molecular"
            className="px-3 py-1 rounded bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-semibold transition-all shadow-md shadow-cyan-900/30"
          >
            MOLECULAR LAB →
          </Link>
        </div>
      </header>

      {/* ─── Main Central Command Area ───────────────────────────────────── */}
      <div className="flex-1 relative flex overflow-hidden">
        {/* Left Navigation Quick Dock */}
        <div className="w-16 border-r border-cyan-900/30 bg-[#070c18]/80 backdrop-blur flex flex-col items-center py-4 gap-4 z-20">
          <Link
            href="/"
            title="Command Center"
            className="w-10 h-10 rounded-lg bg-cyan-950/80 border border-cyan-600 text-cyan-300 flex items-center justify-center text-lg hover:scale-105 transition-all shadow-lg shadow-cyan-950"
          >
            ⌘
          </Link>
          <Link
            href="/molecular"
            title="Spatial Pharmacology Lab"
            className="w-10 h-10 rounded-lg bg-[#0e172a] border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-700 flex items-center justify-center text-lg hover:scale-105 transition-all"
          >
            🔬
          </Link>
          <Link
            href="/simulate"
            title="Whole-Body Dynamics"
            className="w-10 h-10 rounded-lg bg-[#0e172a] border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-700 flex items-center justify-center text-lg hover:scale-105 transition-all"
          >
            ⚡
          </Link>
          <Link
            href="/personal-twin"
            title="Personal Twin Studio"
            className="w-10 h-10 rounded-lg bg-[#0e172a] border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-700 flex items-center justify-center text-lg hover:scale-105 transition-all"
          >
            🧬
          </Link>
          <Link
            href="/samsung-twin"
            title="Real-World Watch Sync"
            className="w-10 h-10 rounded-lg bg-[#0e172a] border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-700 flex items-center justify-center text-lg hover:scale-105 transition-all"
          >
            ⌚
          </Link>
          <Link
            href="/pharmacogenomics"
            title="Pharmacogenomics"
            className="w-10 h-10 rounded-lg bg-[#0e172a] border border-slate-800 text-slate-400 hover:text-cyan-300 hover:border-cyan-700 flex items-center justify-center text-lg hover:scale-105 transition-all"
          >
            ⬟
          </Link>
        </div>

        {/* Center Viewport: Holographic 3D Anatomy */}
        <div className="flex-1 relative h-full">
          <HumanTwinCanvas
            selectedOrgan={selectedOrganKey}
            onSelectOrgan={(organId) => setSelectedOrganKey(organId)}
          />

          {/* Organ Quick Selector Overlay */}
          <div className="absolute top-4 left-4 z-20 flex flex-wrap gap-2 max-w-xl">
            {Object.values(ANATOMICAL_ORGANS).map((org) => {
              const active = selectedOrganKey === org.id;
              return (
                <button
                  key={org.id}
                  onClick={() => setSelectedOrganKey(org.id)}
                  className={`px-3 py-1 rounded text-xs font-mono transition-all border ${
                    active
                      ? "bg-cyan-500/20 border-cyan-400 text-white shadow-md shadow-cyan-900/50"
                      : "bg-[#0b1220]/80 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                  }`}
                >
                  {org.name.split(" ")[0].toUpperCase()}
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Inspector HUD */}
        <div className="w-96 border-l border-cyan-900/30 bg-[#070c18]/90 backdrop-blur p-5 flex flex-col justify-between z-20 overflow-y-auto">
          <div className="space-y-5">
            {/* Header */}
            <div className="border-b border-slate-800 pb-3">
              <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-widest block">
                ORGAN INSPECTOR · {selectedOrgan.system.toUpperCase()}
              </span>
              <h2 className="text-xl font-bold font-mono text-white mt-1 flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: selectedOrgan.color }} />
                {selectedOrgan.name}
              </h2>
            </div>

            {/* Organ State Telemetry */}
            <div className="space-y-3 font-mono text-xs">
              <div className="p-3 bg-[#0d1627] rounded-lg border border-slate-800 space-y-2">
                <span className="text-slate-400 text-[10px] block">ACTIVE MOLECULAR TARGETS</span>
                <div className="flex flex-wrap gap-1.5">
                  {selectedOrgan.targets.map((tgt) => (
                    <span
                      key={tgt}
                      className="px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-700 text-cyan-300 text-xs font-bold"
                    >
                      {tgt}
                    </span>
                  ))}
                </div>
              </div>

              {selectedOrgan.id === "kidney" && (
                <div className="p-3 bg-[#0d1627] rounded-lg border border-slate-800 space-y-1.5 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">GFR:</span>
                    <span className="text-white font-bold">{vitals.gfr} mL/min</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Filtration Load:</span>
                    <span className="text-white font-bold">{((vitals.gfr * vitals.glucose) / 100).toFixed(1)} mg/min</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">SGLT2 Status:</span>
                    <span className="text-emerald-400 font-bold">
                      {activeScenario === "empagliflozin" ? "INHIBITED (96.2%)" : "ACTIVE (100%)"}
                    </span>
                  </div>
                </div>
              )}

              {selectedOrgan.id === "heart" && (
                <div className="p-3 bg-[#0d1627] rounded-lg border border-slate-800 space-y-1.5 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Heart Rate:</span>
                    <span className="text-white font-bold">{vitals.hr} bpm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Cardiac Output:</span>
                    <span className="text-white font-bold">{((vitals.hr * 75) / 1000).toFixed(2)} L/min</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">β1 Adrenergic Tone:</span>
                    <span className="text-rose-400 font-bold">
                      {activeScenario === "atenolol" ? "BLOCKED (Atenolol)" : "PHYSIOLOGICAL"}
                    </span>
                  </div>
                </div>
              )}

              {selectedOrgan.id === "pancreas" && (
                <div className="p-3 bg-[#0d1627] rounded-lg border border-slate-800 space-y-1.5 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Beta Cell Incretin Tone:</span>
                    <span className="text-amber-400 font-bold">
                      {activeScenario === "semaglutide" ? "POTENTIATED (2.4x)" : "NORMAL"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Glucagon Output:</span>
                    <span className="text-white font-bold">{activeScenario === "semaglutide" ? "-45%" : "BASE"}</span>
                  </div>
                </div>
              )}

              {selectedOrgan.id === "liver" && (
                <div className="p-3 bg-[#0d1627] rounded-lg border border-slate-800 space-y-1.5 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">HMG-CoA Reductase:</span>
                    <span className="text-white font-bold">
                      {activeScenario === "atorvastatin" ? "INHIBITED (72%)" : "100% CAPACITY"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">LDL Clearance Rate:</span>
                    <span className="text-emerald-400 font-bold">
                      {activeScenario === "atorvastatin" ? "+48% Upregulation" : "NORMAL"}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Quick Action */}
            <Link
              href={`/molecular`}
              className="block w-full py-2.5 px-3 bg-cyan-950/70 border border-cyan-600/70 hover:bg-cyan-900/80 text-cyan-300 text-xs font-mono font-semibold text-center rounded-lg transition-all"
            >
              LAUNCH SPATIAL SIMULATION FOR {selectedOrgan.id.toUpperCase()} →
            </Link>
          </div>

          {/* System Footer info */}
          <div className="pt-4 border-t border-slate-800/80 text-[10px] font-mono text-slate-500 space-y-1">
            <div>RESOLUTION: 36 PHYSIOLOGICAL STATES</div>
            <div>SOLVER: RK45 EMBEDDED INTEGRATOR</div>
            <div>GOAL: IN SILICO HYPOTHESIS TESTING</div>
          </div>
        </div>
      </div>

      {/* ─── Bottom Scenario Quick-Replay Bar ────────────────────────────── */}
      <footer className="h-14 border-t border-cyan-900/40 bg-[#070c18] px-6 flex items-center justify-between z-30 font-mono text-xs">
        <div className="flex items-center gap-3">
          <span className="text-cyan-400 text-xs font-semibold">INTERVENTION SCENARIOS:</span>
          <button
            onClick={() => handleScenarioTrigger("baseline")}
            className={`px-3 py-1 rounded border text-xs transition-all ${
              activeScenario === "baseline"
                ? "bg-cyan-500/20 border-cyan-400 text-white"
                : "bg-[#0b1220] border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            RESTING BASELINE
          </button>
          <button
            onClick={() => handleScenarioTrigger("empagliflozin")}
            className={`px-3 py-1 rounded border text-xs transition-all ${
              activeScenario === "empagliflozin"
                ? "bg-emerald-500/20 border-emerald-400 text-white"
                : "bg-[#0b1220] border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            + EMPAGLIFLOZIN (SGLT2)
          </button>
          <button
            onClick={() => handleScenarioTrigger("semaglutide")}
            className={`px-3 py-1 rounded border text-xs transition-all ${
              activeScenario === "semaglutide"
                ? "bg-amber-500/20 border-amber-400 text-white"
                : "bg-[#0b1220] border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            + SEMAGLUTIDE (GLP-1R)
          </button>
          <button
            onClick={() => handleScenarioTrigger("atenolol")}
            className={`px-3 py-1 rounded border text-xs transition-all ${
              activeScenario === "atenolol"
                ? "bg-rose-500/20 border-rose-400 text-white"
                : "bg-[#0b1220] border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            + ATENOLOL (ADRB1)
          </button>
          <button
            onClick={() => handleScenarioTrigger("atorvastatin")}
            className={`px-3 py-1 rounded border text-xs transition-all ${
              activeScenario === "atorvastatin"
                ? "bg-purple-500/20 border-purple-400 text-white"
                : "bg-[#0b1220] border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            + ATORVASTATIN (HMGCR)
          </button>
        </div>

        <div className="flex items-center gap-4 text-slate-500 text-[11px]">
          <span>TIMESTEP: dt=1.0 MIN</span>
          <span>RECEPTOR RESERVE: OPERATIONAL</span>
        </div>
      </footer>
    </div>
  );
}
