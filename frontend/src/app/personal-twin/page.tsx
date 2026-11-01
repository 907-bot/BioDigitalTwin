"use client";

import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { Card, Stat } from "@/components/Panels";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function PersonalTwinPage() {
  const [activeTab, setActiveTab] = useState<"intake" | "workout" | "otc" | "cellular" | "samsung">("intake");
  
  // Real Intake Form State
  const [intake, setIntake] = useState({
    patient_id: "P_USER_ME",
    age: 28,
    sex: "male",
    height_cm: 178,
    weight_kg: 76,
    resting_hr: 64,
    resting_sbp: 118,
    resting_dbp: 76,
    body_fat_pct: 15.5,
    skeletal_muscle_mass_kg: 35.2,
    bone_mineral_density_g_cm2: 1.25,
    vo2_max: 48,
    fasting_glucose: 90,
    crp: 0.8,
    serum_creatinine: 0.88,
  });
  const [seededTwin, setSeededTwin] = useState<any>(null);
  const [seedingLoading, setSeedingLoading] = useState(false);

  // Workout Simulator State
  const [workoutForm, setWorkoutForm] = useState({
    workout_type: "hypertrophy",
    intensity_pct: 80,
    duration_mins: 60,
    sets: 4,
    reps: 10,
    frequency_per_week: 4,
    protein_intake_g_day: 160,
    horizon_days: 30,
  });
  const [workoutResult, setWorkoutResult] = useState<any>(null);
  const [workoutLoading, setWorkoutLoading] = useState(false);

  // OTC Drug Mismatch State
  const [otcForm, setOtcForm] = useState({
    patient_condition: "viral_cold_no_fever",
    administered_drug: "paracetamol",
    dose_mg: 500,
    repeat_doses_per_day: 3,
    duration_days: 3,
  });
  const [otcResult, setOtcResult] = useState<any>(null);
  const [otcLoading, setOtcLoading] = useState(false);

  // Cellular Simulation State
  const [cellularForm, setCellularForm] = useState({
    simulation_type: "fever_thermoregulation",
    pyrogen_load: 1.2,
    drug_name: "paracetamol",
    drug_dose_mg: 1000,
    duration_hours: 24,
  });
  const [cellularResult, setCellularResult] = useState<any>(null);
  const [cellularLoading, setCellularLoading] = useState(false);

  // Samsung Health Payload State
  const [samsungJson, setSamsungJson] = useState(
    JSON.stringify(
      {
        heart_rate: [{ value: 68, date_time: new Date().toISOString() }],
        blood_pressure: [{ systolic: 118, diastolic: 76, date_time: new Date().toISOString() }],
        body_composition: [
          { body_fat_pct: 15.5, skeletal_muscle_mass_kg: 35.2, bone_mass_kg: 3.1, date_time: new Date().toISOString() },
        ],
        steps: [{ value: 8450, date_time: new Date().toISOString() }],
        sleep: [{ duration: 7.5, quality: "good", date_time: new Date().toISOString() }],
      },
      null,
      2
    )
  );
  const [samsungResult, setSamsungResult] = useState<any>(null);
  const [samsungLoading, setSamsungLoading] = useState(false);

  // Handlers
  const handleSeedTwin = async () => {
    setSeedingLoading(true);
    try {
      const res = await fetch(`${API_BASE}/personalization/v2/seed-real-twin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(intake),
      });
      const data = await res.json();
      setSeededTwin(data);
    } catch (e) {
      console.error(e);
    } finally {
      setSeedingLoading(false);
    }
  };

  const handleRunWorkout = async () => {
    setWorkoutLoading(true);
    try {
      const res = await fetch(`${API_BASE}/personalization/v2/workout/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: intake.patient_id, ...workoutForm }),
      });
      const data = await res.json();
      setWorkoutResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setWorkoutLoading(false);
    }
  };

  const handleRunOtc = async () => {
    setOtcLoading(true);
    try {
      const res = await fetch(`${API_BASE}/phase10/otc-simulation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(otcForm),
      });
      const data = await res.json();
      setOtcResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setOtcLoading(false);
    }
  };

  const handleRunCellular = async () => {
    setCellularLoading(true);
    try {
      const res = await fetch(`${API_BASE}/personalization/v2/cellular/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cellularForm),
      });
      const data = await res.json();
      setCellularResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setCellularLoading(false);
    }
  };

  const handleSyncSamsung = async () => {
    setSamsungLoading(true);
    try {
      const parsed = JSON.parse(samsungJson);
      const res = await fetch(`${API_BASE}/phase17/samsung-health/sync-to-twin?patient_id=${intake.patient_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      const data = await res.json();
      setSamsungResult(data);
    } catch (e) {
      console.error(e);
      alert("Invalid JSON format");
    } finally {
      setSamsungLoading(false);
    }
  };

  return (
    <div>
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Header Banner */}
        <div className="rounded-2xl p-6 bg-gradient-to-r from-[#0d1726] via-[#111e33] to-[#15233c] border border-[#1e2d42] shadow-xl">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-[#00d4a0]/10 text-[#00d4a0] border border-[#00d4a0]/20 mb-2">
                <span>🧬 Real-World Digital Twin & Experimentation Engine</span>
              </div>
              <h1 className="text-2xl font-bold text-[#e2eaf5]">Personalized Bio-Twin Studio</h1>
              <p className="text-sm text-[#7a93b0] mt-1 max-w-3xl">
                Seed your digital twin from your real body metrics, run progressive overload workout simulations, evaluate OTC drug mismatch hazards, simulate cellular cytokine cascades & hepatocyte toxicity, and sync Samsung Health wearable streams.
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex flex-wrap gap-2 mt-6 pt-4 border-t border-[#1e2d42]">
            {[
              { id: "intake", label: "👤 Real Metrics Intake" },
              { id: "workout", label: "🏋️ Workout & Muscle/Bone Growth" },
              { id: "otc", label: "💊 OTC Cold/Fever Mismatch Lab" },
              { id: "cellular", label: "🔬 Cellular & Cytokine Cascade" },
              { id: "samsung", label: "⌚ Samsung Health App Sync" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-2 rounded-xl text-xs font-medium transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-[#00d4a0] text-[#070c14] font-bold shadow-md shadow-[#00d4a0]/20"
                    : "bg-[#152035] text-[#7a93b0] hover:text-[#e2eaf5] hover:bg-[#1c2b47]"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab 1: Real Metrics Intake */}
        {activeTab === "intake" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card title="Enter Real Body Metrics" className="lg:col-span-1">
              <div className="space-y-3 text-xs text-[#e2eaf5]">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[#7a93b0] font-mono">Patient ID</label>
                    <input
                      type="text"
                      value={intake.patient_id}
                      onChange={(e) => setIntake({ ...intake, patient_id: e.target.value })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                  <div>
                    <label className="text-[#7a93b0] font-mono">Age / Sex</label>
                    <div className="flex gap-1 mt-1">
                      <input
                        type="number"
                        value={intake.age}
                        onChange={(e) => setIntake({ ...intake, age: Number(e.target.value) })}
                        className="w-1/2 px-2 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                      />
                      <select
                        value={intake.sex}
                        onChange={(e) => setIntake({ ...intake, sex: e.target.value })}
                        className="w-1/2 px-2 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                      >
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[#7a93b0] font-mono">Height (cm)</label>
                    <input
                      type="number"
                      value={intake.height_cm}
                      onChange={(e) => setIntake({ ...intake, height_cm: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                  <div>
                    <label className="text-[#7a93b0] font-mono">Weight (kg)</label>
                    <input
                      type="number"
                      value={intake.weight_kg}
                      onChange={(e) => setIntake({ ...intake, weight_kg: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[#7a93b0] font-mono">Body Fat %</label>
                    <input
                      type="number"
                      step="0.5"
                      value={intake.body_fat_pct}
                      onChange={(e) => setIntake({ ...intake, body_fat_pct: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                  <div>
                    <label className="text-[#7a93b0] font-mono">Skeletal Muscle (kg)</label>
                    <input
                      type="number"
                      step="0.5"
                      value={intake.skeletal_muscle_mass_kg}
                      onChange={(e) => setIntake({ ...intake, skeletal_muscle_mass_kg: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[#7a93b0] font-mono">BMD (g/cm²)</label>
                    <input
                      type="number"
                      step="0.05"
                      value={intake.bone_mineral_density_g_cm2}
                      onChange={(e) => setIntake({ ...intake, bone_mineral_density_g_cm2: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                  <div>
                    <label className="text-[#7a93b0] font-mono">Resting HR (bpm)</label>
                    <input
                      type="number"
                      value={intake.resting_hr}
                      onChange={(e) => setIntake({ ...intake, resting_hr: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[#7a93b0] font-mono">Resting SBP / DBP</label>
                    <div className="flex gap-1 mt-1">
                      <input
                        type="number"
                        value={intake.resting_sbp}
                        onChange={(e) => setIntake({ ...intake, resting_sbp: Number(e.target.value) })}
                        className="w-1/2 px-2 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                      />
                      <input
                        type="number"
                        value={intake.resting_dbp}
                        onChange={(e) => setIntake({ ...intake, resting_dbp: Number(e.target.value) })}
                        className="w-1/2 px-2 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-[#7a93b0] font-mono">Fasting Glucose (mg/dL)</label>
                    <input
                      type="number"
                      value={intake.fasting_glucose}
                      onChange={(e) => setIntake({ ...intake, fasting_glucose: Number(e.target.value) })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                    />
                  </div>
                </div>

                <button
                  onClick={handleSeedTwin}
                  disabled={seedingLoading}
                  className="w-full mt-4 py-2.5 rounded-xl bg-[#00d4a0] text-[#070c14] font-bold text-xs hover:bg-[#00e6ae] transition-all shadow-lg shadow-[#00d4a0]/20"
                >
                  {seedingLoading ? "Seeding Twin..." : "⚡ Seed Personal Twin from Metrics"}
                </button>
              </div>
            </Card>

            <Card title="Seeded Twin Subsystem State" className="lg:col-span-2">
              {seededTwin ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <Stat label="Muscle Mass" value={`${seededTwin.subsystems?.musculoskeletal?.muscle_mass?.toFixed(1)} kg`} hint="Musculoskeletal" />
                    <Stat label="Bone Density" value={`${seededTwin.subsystems?.musculoskeletal?.bone_mineral_density?.toFixed(2)} g/cm²`} hint="Trabecular BMD" />
                    <Stat label="Strength Capacity" value={`${seededTwin.subsystems?.musculoskeletal?.muscle_strength_index?.toFixed(0)} / 100`} hint="Relative 1RM" />
                    <Stat label="Resting HR" value={`${seededTwin.subsystems?.cardiovascular?.HR?.toFixed(0)} bpm`} hint="Cardiovascular" />
                  </div>

                  <div className="p-4 rounded-xl bg-[#0d1422] border border-[#1e2d42] text-xs space-y-2">
                    <div className="text-[#00d4a0] font-bold font-mono">✓ Twin Initialized & Calibrated</div>
                    <pre className="text-[11px] font-mono text-[#a78bfa] overflow-x-auto p-2 bg-[#070c14] rounded-lg">
                      {JSON.stringify(seededTwin.subsystems, null, 2)}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="p-12 text-center text-xs text-[#7a93b0]">
                  Enter your physical metrics on the left and click <strong className="text-[#00d4a0]">Seed Personal Twin</strong> to generate your calibrated 36-state digital twin.
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Tab 2: Workout & Progressive Overload */}
        {activeTab === "workout" && (
          <div className="space-y-6">
            <Card title="Configure Workout Program">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div>
                  <label className="text-[#7a93b0] font-mono">Workout Type</label>
                  <select
                    value={workoutForm.workout_type}
                    onChange={(e) => setWorkoutForm({ ...workoutForm, workout_type: e.target.value })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  >
                    <option value="hypertrophy">Resistance (Hypertrophy)</option>
                    <option value="strength">Heavy Strength (1RM Focus)</option>
                    <option value="endurance">Endurance Cardio</option>
                    <option value="plyometric">High-Impact Plyometrics</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Intensity (% 1RM / HRmax)</label>
                  <input
                    type="number"
                    value={workoutForm.intensity_pct}
                    onChange={(e) => setWorkoutForm({ ...workoutForm, intensity_pct: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Frequency (Days / Week)</label>
                  <input
                    type="number"
                    value={workoutForm.frequency_per_week}
                    onChange={(e) => setWorkoutForm({ ...workoutForm, frequency_per_week: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Daily Protein (g)</label>
                  <input
                    type="number"
                    value={workoutForm.protein_intake_g_day}
                    onChange={(e) => setWorkoutForm({ ...workoutForm, protein_intake_g_day: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>
              </div>

              <button
                onClick={handleRunWorkout}
                disabled={workoutLoading}
                className="mt-4 px-6 py-2 rounded-xl bg-[#00d4a0] text-[#070c14] font-bold text-xs hover:bg-[#00e6ae] transition-all"
              >
                {workoutLoading ? "Simulating Adaptations..." : "▶ Simulate Progressive Overload Trajectory"}
              </button>
            </Card>

            {workoutResult && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Stat label="Muscle Mass Gain" value={`+${workoutResult.summary?.muscle_mass_delta_kg} kg`} hint={`Final: ${workoutResult.summary?.final_muscle_mass_kg} kg`} />
                  <Stat label="Strength Delta" value={`+${workoutResult.summary?.strength_delta}`} hint={`Final: ${workoutResult.summary?.final_strength_index} / 100`} />
                  <Stat label="Bone Density Accretion" value={`+${workoutResult.summary?.bmd_gain_pct}%`} hint="Wolff's Law Strain" />
                  <Stat label="Resting HR Reduction" value={`-${workoutResult.summary?.resting_hr_reduction_bpm} bpm`} hint="Cardiovascular Output" />
                </div>

                <Card title="30-Day Muscle Hypertrophy & Strength Trajectory">
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={workoutResult.trajectory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e2d42" />
                        <XAxis dataKey="day" stroke="#7a93b0" fontSize={11} label={{ value: "Day", position: "insideBottom", offset: -5 }} />
                        <YAxis yAxisId="left" stroke="#00d4a0" fontSize={11} domain={["dataMin - 0.5", "dataMax + 0.5"]} />
                        <YAxis yAxisId="right" orientation="right" stroke="#a78bfa" fontSize={11} />
                        <Tooltip contentStyle={{ backgroundColor: "#0d1422", borderColor: "#1e2d42", fontSize: 11 }} />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="muscle_mass_kg" name="Muscle Mass (kg)" stroke="#00d4a0" strokeWidth={2} dot={false} />
                        <Line yAxisId="right" type="monotone" dataKey="strength_index" name="Strength Index" stroke="#a78bfa" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: OTC Cold/Fever Mismatch Lab */}
        {activeTab === "otc" && (
          <div className="space-y-6">
            <Card title="OTC Cold/Fever Drug Mismatch Simulator">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div>
                  <label className="text-[#7a93b0] font-mono">Patient Clinical Condition</label>
                  <select
                    value={otcForm.patient_condition}
                    onChange={(e) => setOtcForm({ ...otcForm, patient_condition: e.target.value })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  >
                    <option value="viral_cold_no_fever">Viral Cold (No Fever, Nasal Congestion)</option>
                    <option value="fever_39C">High Fever (39.1 °C Hyperthermia)</option>
                    <option value="cold_with_hypertension">Viral Cold + Baseline Hypertension</option>
                    <option value="fever_with_ckd">Fever + Stage 3 CKD</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Administered OTC Medication</label>
                  <select
                    value={otcForm.administered_drug}
                    onChange={(e) => setOtcForm({ ...otcForm, administered_drug: e.target.value })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  >
                    <option value="paracetamol">Paracetamol (Acetaminophen)</option>
                    <option value="ibuprofen">Ibuprofen (NSAID)</option>
                    <option value="pseudoephedrine">Pseudoephedrine (Decongestant)</option>
                    <option value="diphenhydramine">Diphenhydramine (Antihistamine)</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Unit Dose (mg)</label>
                  <input
                    type="number"
                    value={otcForm.dose_mg}
                    onChange={(e) => setOtcForm({ ...otcForm, dose_mg: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>
              </div>

              <button
                onClick={handleRunOtc}
                disabled={otcLoading}
                className="mt-4 px-6 py-2 rounded-xl bg-[#00d4a0] text-[#070c14] font-bold text-xs hover:bg-[#00e6ae] transition-all"
              >
                {otcLoading ? "Evaluating Mismatch..." : "🧪 Run OTC Drug Mismatch Simulation"}
              </button>
            </Card>

            {otcResult && (
              <div className="space-y-6">
                {/* Clinical Warnings */}
                {otcResult.warnings?.map((warn: string, idx: number) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-xl text-xs border ${
                      warn.includes("CRITICAL") || warn.includes("MISMATCH")
                        ? "bg-[#ff4757]/10 border-[#ff4757]/30 text-[#ff4757]"
                        : "bg-[#ffa502]/10 border-[#ffa502]/30 text-[#ffa502]"
                    }`}
                  >
                    <div className="font-bold">{warn}</div>
                  </div>
                ))}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Stat label="Verdict" value={otcResult.clinical_verdict} hint="Safety Assessment" />
                  <Stat label="Initial Body Temp" value={`${otcResult.outcomes?.initial_temp_C} °C`} hint="Baseline" />
                  <Stat label="Final Body Temp" value={`${otcResult.outcomes?.final_temp_C} °C`} hint="Post-Treatment" />
                  <Stat label="Peak SBP" value={`${otcResult.outcomes?.peak_sbp_mmHg} mmHg`} hint="Vascular Load" />
                </div>

                <Card title="Temperature & Systolic BP Simulation Trajectory">
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={otcResult.trajectories?.temperature}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e2d42" />
                        <XAxis dataKey="hour" stroke="#7a93b0" fontSize={11} />
                        <YAxis stroke="#ff4757" fontSize={11} domain={[36, 40]} />
                        <Tooltip contentStyle={{ backgroundColor: "#0d1422", borderColor: "#1e2d42", fontSize: 11 }} />
                        <Line type="monotone" dataKey="body_temp_C" name="Body Temp (°C)" stroke="#ff4757" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Cellular Simulation */}
        {activeTab === "cellular" && (
          <div className="space-y-6">
            <Card title="Cellular Level & Sub-Organ Simulator">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div>
                  <label className="text-[#7a93b0] font-mono">Simulation Mode</label>
                  <select
                    value={cellularForm.simulation_type}
                    onChange={(e) => setCellularForm({ ...cellularForm, simulation_type: e.target.value })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  >
                    <option value="fever_thermoregulation">Hypothalamic Thermoregulation & PGE2</option>
                    <option value="hepatocyte_toxicity">Hepatocyte CYP2E1 & Glutathione (GSH) Depletion</option>
                    <option value="cytokine_cascade">Macrophage M1/M2 Cytokine Cascade</option>
                  </select>
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Pyrogen / Infection Intensity</label>
                  <input
                    type="number"
                    step="0.2"
                    value={cellularForm.pyrogen_load}
                    onChange={(e) => setCellularForm({ ...cellularForm, pyrogen_load: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>

                <div>
                  <label className="text-[#7a93b0] font-mono">Paracetamol Dose (mg)</label>
                  <input
                    type="number"
                    value={cellularForm.drug_dose_mg}
                    onChange={(e) => setCellularForm({ ...cellularForm, drug_dose_mg: Number(e.target.value) })}
                    className="w-full mt-1 px-3 py-1.5 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-xs text-[#e2eaf5]"
                  />
                </div>
              </div>

              <button
                onClick={handleRunCellular}
                disabled={cellularLoading}
                className="mt-4 px-6 py-2 rounded-xl bg-[#00d4a0] text-[#070c14] font-bold text-xs hover:bg-[#00e6ae] transition-all"
              >
                {cellularLoading ? "Simulating Sub-Organ Dynamics..." : "🔬 Run Sub-Organ Cellular Simulation"}
              </button>
            </Card>

            {cellularResult && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Stat label="Peak Core Temp" value={`${cellularResult.summary?.peak_core_temp_C} °C`} hint="Core Thermostat" />
                  <Stat label="Peak IL-6" value={`${cellularResult.summary?.peak_il6}`} hint="Pro-inflammatory" />
                  <Stat label="Lowest GSH Reserve" value={`${cellularResult.summary?.min_gsh_pct}%`} hint="Glutathione" />
                  <Stat label="Hepatocyte Viability" value={`${cellularResult.summary?.final_hepatocyte_viability_pct}%`} hint="ALT/AST Leakage" />
                </div>

                <Card title="Intracellular Glutathione (GSH) & Cytokine Dynamics">
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cellularResult.trajectory}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e2d42" />
                        <XAxis dataKey="hour" stroke="#7a93b0" fontSize={11} label={{ value: "Hour", position: "insideBottom", offset: -5 }} />
                        <YAxis stroke="#00d4a0" fontSize={11} />
                        <Tooltip contentStyle={{ backgroundColor: "#0d1422", borderColor: "#1e2d42", fontSize: 11 }} />
                        <Legend />
                        <Line type="monotone" dataKey="glutathione_gsh_pct" name="Glutathione GSH (%)" stroke="#00d4a0" strokeWidth={2} />
                        <Line type="monotone" dataKey="il_6" name="IL-6 Cytokine" stroke="#ff4757" strokeWidth={2} />
                        <Line type="monotone" dataKey="tnf_alpha" name="TNF-alpha Cytokine" stroke="#ffa502" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>
            )}
          </div>
        )}

        {/* Tab 5: Samsung Health Sync */}
        {activeTab === "samsung" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Samsung Health Export JSON Payload">
              <textarea
                value={samsungJson}
                onChange={(e) => setSamsungJson(e.target.value)}
                rows={12}
                className="w-full font-mono text-xs p-3 rounded-lg bg-[#0d1422] border border-[#1e2d42] text-[#00d4a0]"
              />
              <button
                onClick={handleSyncSamsung}
                disabled={samsungLoading}
                className="w-full mt-4 py-2.5 rounded-xl bg-[#00d4a0] text-[#070c14] font-bold text-xs hover:bg-[#00e6ae] transition-all"
              >
                {samsungLoading ? "Syncing Watch Stream..." : "⌚ Sync Samsung Health Data to Digital Twin"}
              </button>
            </Card>

            <Card title="Sync Status & Twin Update">
              {samsungResult ? (
                <div className="space-y-4">
                  <div className="p-4 rounded-xl bg-[#00d4a0]/10 border border-[#00d4a0]/30 text-xs text-[#00d4a0]">
                    <strong>✓ Samsung Health Data Successfully Synced to Twin!</strong>
                    <div className="mt-1 text-[#e2eaf5]">
                      Synced {samsungResult.synced_metrics_count} health parameters into patient twin state.
                    </div>
                  </div>

                  <pre className="text-[11px] font-mono text-[#a78bfa] overflow-x-auto p-4 bg-[#0d1422] rounded-xl border border-[#1e2d42]">
                    {JSON.stringify(samsungResult, null, 2)}
                  </pre>
                </div>
              ) : (
                <div className="p-12 text-center text-xs text-[#7a93b0]">
                  Paste Samsung Health export JSON or Bluetooth stream data and click <strong className="text-[#00d4a0]">Sync Samsung Health Data</strong> to update your twin.
                </div>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
