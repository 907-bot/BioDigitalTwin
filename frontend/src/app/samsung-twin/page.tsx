"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function apiGet(path: string) {
  const r = await fetch(`${BASE}/api${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function apiPost(path: string, body?: any) {
  const r = await fetch(`${BASE}/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ─── Mini chart component ────────────────────────────────────────────────────
function Sparkline({
  data,
  color = "#00d4a0",
  height = 40,
}: {
  data: number[];
  color?: string;
  height?: number;
}) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 120;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = height - ((v - min) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${height}`} style={{ width: "100%", height }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.85"
      />
      <polyline
        points={`0,${height} ${pts} ${w},${height}`}
        fill={color}
        opacity="0.08"
      />
    </svg>
  );
}

// ─── Gauge arc ───────────────────────────────────────────────────────────────
function GaugeArc({
  value,
  max = 100,
  color = "#00d4a0",
  size = 80,
}: {
  value: number;
  max?: number;
  color?: string;
  size?: number;
}) {
  const pct = Math.min(value / max, 1);
  const r = size / 2 - 6;
  const circ = 2 * Math.PI * r;
  const dash = pct * circ * 0.75;
  const gap = circ - dash;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(135deg)" }}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="rgba(255,255,255,0.08)"
        strokeWidth="6"
        strokeDasharray={`${circ * 0.75} ${circ * 0.25}`}
        strokeLinecap="round"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeDasharray={`${dash} ${gap + circ * 0.25}`}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 0.6s ease" }}
      />
    </svg>
  );
}

// ─── Metric card ─────────────────────────────────────────────────────────────
function MetricCard({
  label,
  value,
  unit,
  color,
  sparkData,
  badge,
}: {
  label: string;
  value: string | number;
  unit: string;
  color: string;
  sparkData?: number[];
  badge?: { text: string; ok: boolean };
}) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.04)",
        border: `1px solid ${color}33`,
        borderRadius: 14,
        padding: "16px 18px",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          background: `linear-gradient(90deg, ${color}00, ${color}, ${color}00)`,
        }}
      />
      <div style={{ fontSize: 11, color: "#888", marginBottom: 4, letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ fontSize: 26, fontWeight: 700, color }}>{value}</span>
        <span style={{ fontSize: 12, color: "#666" }}>{unit}</span>
      </div>
      {badge && (
        <div
          style={{
            marginTop: 6,
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 20,
            fontSize: 10,
            fontWeight: 600,
            background: badge.ok ? "#00d4a022" : "#ff6b5b22",
            color: badge.ok ? "#00d4a0" : "#ff6b5b",
            border: `1px solid ${badge.ok ? "#00d4a044" : "#ff6b5b44"}`,
          }}
        >
          {badge.text}
        </div>
      )}
      {sparkData && sparkData.length > 2 && (
        <div style={{ marginTop: 8, opacity: 0.7 }}>
          <Sparkline data={sparkData} color={color} height={36} />
        </div>
      )}
    </div>
  );
}

// ─── SHAP bar ────────────────────────────────────────────────────────────────
function ShapBar({
  feature,
  importance,
  value,
  maxImportance,
  color,
}: {
  feature: string;
  importance: number;
  value: number;
  maxImportance: number;
  color: string;
}) {
  const pct = (importance / maxImportance) * 100;
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          marginBottom: 3,
        }}
      >
        <span style={{ color: "#ccc", fontFamily: "JetBrains Mono, monospace" }}>
          {feature}
        </span>
        <span style={{ color: "#888" }}>
          {value.toFixed(2)} (imp: {(importance * 100).toFixed(2)}%)
        </span>
      </div>
      <div
        style={{
          height: 6,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}, ${color}88)`,
            borderRadius: 3,
            transition: "width 0.5s ease",
          }}
        />
      </div>
    </div>
  );
}

// ─── Sleep stage waterfall ───────────────────────────────────────────────────
function SleepStageBar({ stages }: { stages: Record<string, number> }) {
  const colors: Record<string, string> = {
    deep: "#a78bfa",
    rem: "#00bcd4",
    light: "#60a5fa",
    awake: "#ff6b5b",
  };
  const total =
    (stages.deep || 0) + (stages.rem || 0) + (stages.light || 0) + (stages.awake || 0);
  if (!total) return null;
  return (
    <div>
      <div
        style={{
          display: "flex",
          height: 24,
          borderRadius: 6,
          overflow: "hidden",
          gap: 1,
        }}
      >
        {Object.entries(stages).map(([k, v]) => (
          <div
            key={k}
            style={{
              width: `${((v || 0) / total) * 100}%`,
              background: colors[k] || "#666",
            }}
            title={`${k}: ${((v / total) * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <div
        style={{
          display: "flex",
          gap: 12,
          marginTop: 8,
          flexWrap: "wrap",
        }}
      >
        {Object.entries(stages).map(([k, v]) => (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: 2,
                background: colors[k] || "#666",
              }}
            />
            <span style={{ fontSize: 10, color: "#888" }}>
              {k}: {((v / total) * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Pipeline step ───────────────────────────────────────────────────────────
const PIPELINE_STEPS = [
  { id: "idle", label: "Idle", icon: "○" },
  { id: "parsing", label: "Parsing CSVs", icon: "📂" },
  { id: "feature_engineering", label: "Feature Engineering", icon: "⚙️" },
  { id: "scaling", label: "Scaling Features", icon: "📏" },
  { id: "training_cardio", label: "Training CardioRisk", icon: "❤️" },
  { id: "training_sleep", label: "Training SleepQuality", icon: "🌙" },
  { id: "training_stress", label: "Training StressModel", icon: "🧠" },
  { id: "training_anomaly", label: "Training AnomalyDetector", icon: "🔍" },
  { id: "saving", label: "Saving Models", icon: "💾" },
  { id: "complete", label: "Complete", icon: "✅" },
  { id: "error", label: "Error", icon: "❌" },
];

function PipelineStatus({ status }: { status: any }) {
  if (!status || status.state === "idle") return null;
  const currentIdx = PIPELINE_STEPS.findIndex((s) => s.id === status.state);
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          marginBottom: 12,
        }}
      >
        {PIPELINE_STEPS.filter((s) => !["idle", "error"].includes(s.id)).map((step, i) => {
          const stepIdx = PIPELINE_STEPS.findIndex((s2) => s2.id === step.id);
          const done = currentIdx > stepIdx;
          const active = currentIdx === stepIdx;
          return (
            <div key={step.id} style={{ display: "flex", alignItems: "center" }}>
              <div
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  background: done
                    ? "#00d4a0"
                    : active
                    ? "#a78bfa"
                    : "rgba(255,255,255,0.06)",
                  border: `2px solid ${done ? "#00d4a0" : active ? "#a78bfa" : "transparent"}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  transition: "all 0.3s",
                  boxShadow: active ? "0 0 12px #a78bfa66" : "none",
                }}
                title={step.label}
              >
                {done ? "✓" : active ? "⋯" : i + 1}
              </div>
              {i < PIPELINE_STEPS.filter((s) => !["idle", "error"].includes(s.id)).length - 1 && (
                <div
                  style={{
                    height: 2,
                    width: 20,
                    background: done ? "#00d4a0" : "rgba(255,255,255,0.08)",
                    transition: "background 0.3s",
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
      {/* Progress bar */}
      <div
        style={{
          height: 6,
          background: "rgba(255,255,255,0.06)",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${status.progress || 0}%`,
            background: status.state === "error"
              ? "linear-gradient(90deg, #ff6b5b, #ff9f43)"
              : "linear-gradient(90deg, #a78bfa, #00d4a0)",
            borderRadius: 3,
            transition: "width 0.4s ease",
          }}
        />
      </div>
      <div style={{ fontSize: 11, color: "#888", marginTop: 6 }}>
        {status.state === "complete"
          ? "✅ Training complete"
          : status.state === "error"
          ? `❌ Error: ${status.error}`
          : `${status.progress || 0}% — ${PIPELINE_STEPS.find((s) => s.id === status.state)?.label || status.state}`}
      </div>
      {status.log && status.log.length > 0 && (
        <div
          style={{
            marginTop: 10,
            background: "rgba(0,0,0,0.3)",
            borderRadius: 8,
            padding: "10px 12px",
            maxHeight: 120,
            overflowY: "auto",
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 10,
            color: "#888",
          }}
        >
          {status.log.slice(-8).map((line: string, i: number) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────
export default function SamsungTwinPage() {
  const [ingestData, setIngestData] = useState<any>(null);
  const [patientRecord, setPatientRecord] = useState<any>(null);
  const [features, setFeatures] = useState<any>(null);
  const [trainStatus, setTrainStatus] = useState<any>(null);
  const [trainMetrics, setTrainMetrics] = useState<any>(null);
  const [predictions, setPredictions] = useState<any>(null);
  const [timeline, setTimeline] = useState<any>(null);
  const [fuseResult, setFuseResult] = useState<any>(null);

  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [hitlDecision, setHitlDecision] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState("overview");

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const setLoad = (key: string, v: boolean) =>
    setLoading((p) => ({ ...p, [key]: v }));
  const setErr = (key: string, v: string) =>
    setErrors((p) => ({ ...p, [key]: v }));

  // ── Auto-load existing data on mount
  useEffect(() => {
    apiGet("/phase18/patient-record")
      .then((d) => setPatientRecord(d))
      .catch(() => {});
    apiGet("/phase18/training-status")
      .then((d) => setTrainStatus(d))
      .catch(() => {});
    apiGet("/phase18/training-metrics")
      .then((d) => setTrainMetrics(d))
      .catch(() => {});
  }, []);

  // ── Ingest
  const handleIngest = useCallback(async () => {
    setLoad("ingest", true);
    setErr("ingest", "");
    try {
      const d = await apiPost("/phase18/ingest");
      setIngestData(d);
      const rec = await apiGet("/phase18/patient-record");
      setPatientRecord(rec);
      const feat = await apiGet("/phase18/features?age_est=25");
      setFeatures(feat);
      const tl = await apiGet("/phase18/health-timeline?limit=300");
      setTimeline(tl);
    } catch (e: any) {
      setErr("ingest", e.message);
    } finally {
      setLoad("ingest", false);
    }
  }, []);

  // ── Train
  const handleTrain = useCallback(async () => {
    setLoad("train", true);
    setErr("train", "");
    try {
      await apiPost("/phase18/train?async_train=true", {
        n_synthetic: 600,
        age_est: 25,
      });
      // Poll training status
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const s = await apiGet("/phase18/training-status");
          setTrainStatus(s);
          if (s.state === "complete" || s.state === "error") {
            clearInterval(pollRef.current!);
            setLoad("train", false);
            if (s.state === "complete") {
              const m = await apiGet("/phase18/training-metrics");
              setTrainMetrics(m);
            }
          }
        } catch {}
      }, 1500);
    } catch (e: any) {
      setErr("train", e.message);
      setLoad("train", false);
    }
  }, []);

  // ── Predict
  const handlePredict = useCallback(async () => {
    setLoad("predict", true);
    setErr("predict", "");
    try {
      const d = await apiPost("/phase18/predict", { age_est: 25, top_n_explain: 10 });
      setPredictions(d);
    } catch (e: any) {
      setErr("predict", e.message);
    } finally {
      setLoad("predict", false);
    }
  }, []);

  // ── Fuse with cohort
  const handleFuse = useCallback(async () => {
    setLoad("fuse", true);
    setErr("fuse", "");
    try {
      const d = await apiPost("/phase18/fuse-with-cohort", {
        patient_id: "P_REAL_001",
        age: 25,
        gender: "Male",
      });
      setFuseResult(d);
    } catch (e: any) {
      setErr("fuse", e.message);
    } finally {
      setLoad("fuse", false);
    }
  }, []);

  // ── Timeline HR sparkline data
  const hrSparkData =
    timeline?.events
      ?.filter((e: any) => e.type === "heart_rate")
      ?.slice(-60)
      ?.map((e: any) => e.value) || [];

  const stressSparkData =
    timeline?.events
      ?.filter((e: any) => e.type === "stress")
      ?.slice(-60)
      ?.map((e: any) => e.value) || [];

  const vitals = patientRecord?.vitals_snapshot || {};
  const agg = patientRecord?.aggregates || {};

  const sleepStages = {
    deep: agg.sleep_deep_pct || 0,
    rem: agg.sleep_rem_pct || 0,
    light: agg.sleep_light_pct || 0,
    awake: agg.sleep_awake_pct || 0,
  };

  return (
    <div style={{ padding: "24px 28px", minHeight: "100vh", color: "#e0e0e0" }}>
      {/* ── Header ── */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 14,
              background: "linear-gradient(135deg, #00d4a0, #00bcd4)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 24,
              boxShadow: "0 0 24px #00d4a044",
            }}
          >
            ⌚
          </div>
          <div>
            <h1
              style={{
                fontSize: 26,
                fontWeight: 800,
                margin: 0,
                background: "linear-gradient(90deg, #00d4a0, #00bcd4, #a78bfa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Real-World Digital Twin
            </h1>
            <div style={{ fontSize: 13, color: "#666", marginTop: 2 }}>
              Phase 18 — Samsung Health Data Integration · ML Training · Predictions · HITL Review
            </div>
          </div>
          {patientRecord && (
            <div
              style={{
                marginLeft: "auto",
                padding: "6px 14px",
                background: "#00d4a018",
                border: "1px solid #00d4a044",
                borderRadius: 20,
                fontSize: 12,
                color: "#00d4a0",
                fontWeight: 600,
              }}
            >
              ● P_REAL_001 Active
            </div>
          )}
        </div>

        {/* Action bar */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 16 }}>
          {[
            {
              key: "ingest",
              label: "📂 Ingest Samsung Data",
              onClick: handleIngest,
              color: "#00d4a0",
            },
            {
              key: "train",
              label: "🧠 Train ML Models",
              onClick: handleTrain,
              color: "#a78bfa",
            },
            {
              key: "predict",
              label: "⚡ Run Predictions",
              onClick: handlePredict,
              color: "#60a5fa",
            },
            {
              key: "fuse",
              label: "🔗 Fuse with Cohort",
              onClick: handleFuse,
              color: "#ffa94d",
            },
          ].map((btn) => (
            <button
              key={btn.key}
              id={`btn-${btn.key}`}
              onClick={btn.onClick}
              disabled={loading[btn.key]}
              style={{
                padding: "10px 20px",
                borderRadius: 10,
                border: `1px solid ${btn.color}44`,
                background: loading[btn.key]
                  ? `${btn.color}11`
                  : `${btn.color}18`,
                color: btn.color,
                fontWeight: 600,
                fontSize: 13,
                cursor: loading[btn.key] ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                opacity: loading[btn.key] ? 0.6 : 1,
              }}
            >
              {loading[btn.key] ? "⋯ Working..." : btn.label}
            </button>
          ))}
        </div>

        {/* Errors */}
        {Object.entries(errors).filter(([, v]) => v).map(([k, v]) => (
          <div
            key={k}
            style={{
              marginTop: 10,
              padding: "8px 14px",
              background: "#ff6b5b18",
              border: "1px solid #ff6b5b44",
              borderRadius: 8,
              fontSize: 12,
              color: "#ff6b5b",
            }}
          >
            ⚠ {k}: {v}
          </div>
        ))}
      </div>

      {/* ── Tabs ── */}
      <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: 0 }}>
        {[
          { id: "overview", label: "Overview" },
          { id: "training", label: "ML Pipeline" },
          { id: "predictions", label: "Predictions" },
          { id: "timeline", label: "Timeline" },
          { id: "xai", label: "Explainability" },
          { id: "hitl", label: "HITL Review" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "8px 18px",
              border: "none",
              borderBottom: activeTab === tab.id
                ? "2px solid #00d4a0"
                : "2px solid transparent",
              background: "transparent",
              color: activeTab === tab.id ? "#00d4a0" : "#666",
              fontWeight: activeTab === tab.id ? 600 : 400,
              fontSize: 13,
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════════ OVERVIEW ══════════════════════════════ */}
      {activeTab === "overview" && (
        <div>
          {!patientRecord ? (
            <div
              style={{
                textAlign: "center",
                padding: "60px 20px",
                color: "#555",
                border: "1px dashed rgba(255,255,255,0.1)",
                borderRadius: 16,
              }}
            >
              <div style={{ fontSize: 48, marginBottom: 16 }}>⌚</div>
              <div style={{ fontSize: 16, marginBottom: 8 }}>No Samsung data loaded yet</div>
              <div style={{ fontSize: 13 }}>
                Click <strong style={{ color: "#00d4a0" }}>Ingest Samsung Data</strong> to parse your health export
              </div>
            </div>
          ) : (
            <>
              {/* Ingest summary */}
              {ingestData && (
                <div
                  style={{
                    marginBottom: 24,
                    padding: "14px 18px",
                    background: "#00d4a012",
                    border: "1px solid #00d4a033",
                    borderRadius: 12,
                  }}
                >
                  <div style={{ fontSize: 12, color: "#00d4a0", fontWeight: 600, marginBottom: 8 }}>
                    ✅ Samsung Health Data Ingested
                  </div>
                  <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                    {Object.entries(ingestData.data_counts || {}).map(([k, v]) => (
                      <div key={k} style={{ fontSize: 11, color: "#888" }}>
                        <span style={{ color: "#ccc" }}>{String(v).padStart(4)}</span>{" "}
                        {k.replace(/_/g, " ")}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Vitals grid */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
                  gap: 14,
                  marginBottom: 24,
                }}
              >
                <MetricCard
                  label="Resting Heart Rate"
                  value={vitals.resting_hr_bpm?.toFixed(0) || "—"}
                  unit="bpm"
                  color={
                    (vitals.resting_hr_bpm || 0) > 80
                      ? "#ff6b5b"
                      : "#00d4a0"
                  }
                  sparkData={hrSparkData}
                  badge={{
                    text:
                      (vitals.resting_hr_bpm || 0) > 80
                        ? "Elevated"
                        : "Normal",
                    ok: (vitals.resting_hr_bpm || 0) <= 80,
                  }}
                />
                <MetricCard
                  label="Mean Stress Score"
                  value={agg.stress_mean?.toFixed(1) || "—"}
                  unit="/ 100"
                  color={
                    (agg.stress_mean || 0) > 60
                      ? "#ff6b5b"
                      : (agg.stress_mean || 0) > 35
                      ? "#ffa94d"
                      : "#00d4a0"
                  }
                  sparkData={stressSparkData}
                  badge={{
                    text:
                      (agg.stress_mean || 0) > 60
                        ? "High Stress"
                        : (agg.stress_mean || 0) > 35
                        ? "Medium"
                        : "Low",
                    ok: (agg.stress_mean || 0) <= 35,
                  }}
                />
                <MetricCard
                  label="Daily Steps"
                  value={
                    agg.steps_daily_mean
                      ? Math.round(agg.steps_daily_mean).toLocaleString()
                      : "—"
                  }
                  unit="steps"
                  color="#60a5fa"
                  badge={{
                    text:
                      (agg.steps_daily_mean || 0) >= 10000
                        ? "Goal Met"
                        : "Below Goal",
                    ok: (agg.steps_daily_mean || 0) >= 10000,
                  }}
                />
                <MetricCard
                  label="Sleep / Night"
                  value={vitals.sleep_hours_per_night?.toFixed(1) || "—"}
                  unit="hrs"
                  color="#a78bfa"
                  badge={{
                    text:
                      (vitals.sleep_hours_per_night || 0) >= 7 &&
                      (vitals.sleep_hours_per_night || 0) <= 9
                        ? "Optimal"
                        : "Suboptimal",
                    ok:
                      (vitals.sleep_hours_per_night || 0) >= 7 &&
                      (vitals.sleep_hours_per_night || 0) <= 9,
                  }}
                />
                <MetricCard
                  label="Body Weight"
                  value={vitals.weight_kg?.toFixed(1) || "—"}
                  unit="kg"
                  color="#ffa94d"
                />
                <MetricCard
                  label="BMI"
                  value={vitals.bmi?.toFixed(1) || "—"}
                  unit=""
                  color={
                    (vitals.bmi || 0) >= 25 ? "#ff6b5b" : "#00d4a0"
                  }
                  badge={{
                    text:
                      (vitals.bmi || 0) < 18.5
                        ? "Underweight"
                        : (vitals.bmi || 0) < 25
                        ? "Normal"
                        : (vitals.bmi || 0) < 30
                        ? "Overweight"
                        : "Obese",
                    ok: (vitals.bmi || 0) >= 18.5 && (vitals.bmi || 0) < 25,
                  }}
                />
                <MetricCard
                  label="Body Fat %"
                  value={vitals.body_fat_pct?.toFixed(1) || "—"}
                  unit="%"
                  color="#00bcd4"
                />
                <MetricCard
                  label="SpO₂ Mean"
                  value={vitals.spo2_mean?.toFixed(1) || "—"}
                  unit="%"
                  color={
                    (vitals.spo2_mean || 100) < 95 ? "#ff6b5b" : "#4ade80"
                  }
                  badge={{
                    text:
                      (vitals.spo2_mean || 100) >= 95
                        ? "Normal"
                        : "Low",
                    ok: (vitals.spo2_mean || 100) >= 95,
                  }}
                />
                <MetricCard
                  label="VO₂ Max"
                  value={vitals.vo2_max?.toFixed(1) || "—"}
                  unit="ml/kg/min"
                  color="#4ade80"
                  badge={{
                    text:
                      (vitals.vo2_max || 0) > 45
                        ? "Excellent"
                        : (vitals.vo2_max || 0) > 35
                        ? "Good"
                        : (vitals.vo2_max || 0) > 0
                        ? "Average"
                        : "—",
                    ok: (vitals.vo2_max || 0) > 35,
                  }}
                />
              </div>

              {/* Sleep architecture */}
              {(agg.sleep_deep_pct || 0) > 0 && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 14,
                    padding: "18px 20px",
                    marginBottom: 20,
                  }}
                >
                  <div style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>
                    SLEEP ARCHITECTURE
                  </div>
                  <SleepStageBar stages={sleepStages} />
                </div>
              )}

              {/* Exercise stats */}
              {(agg.exercise_session_count || 0) > 0 && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 14,
                  }}
                >
                  <MetricCard
                    label="Exercise Sessions"
                    value={agg.exercise_session_count || 0}
                    unit="total"
                    color="#ffa94d"
                  />
                  <MetricCard
                    label="Mean Exercise HR"
                    value={agg.exercise_mean_hr?.toFixed(0) || "—"}
                    unit="bpm"
                    color="#60a5fa"
                  />
                  <MetricCard
                    label="Total Ex. Calories"
                    value={
                      agg.exercise_total_calories
                        ? Math.round(agg.exercise_total_calories).toLocaleString()
                        : "—"
                    }
                    unit="kcal"
                    color="#4ade80"
                  />
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ═════════════════════════════ ML PIPELINE ══════════════════════════════ */}
      {activeTab === "training" && (
        <div>
          <div
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 16,
              padding: "22px 24px",
              marginBottom: 20,
            }}
          >
            <div style={{ fontSize: 13, color: "#888", marginBottom: 16 }}>
              TRAINING PIPELINE STATUS
            </div>
            <PipelineStatus status={trainStatus} />
            {!trainStatus || trainStatus.state === "idle" ? (
              <div style={{ fontSize: 13, color: "#555", padding: "20px 0" }}>
                Click <strong style={{ color: "#a78bfa" }}>Train ML Models</strong> to start the pipeline.
                <br />
                <span style={{ fontSize: 11, color: "#444" }}>
                  Uses real Samsung data to build a 600-sample augmented cohort → trains 4 models
                </span>
              </div>
            ) : null}
          </div>

          {/* Model metrics */}
          {trainMetrics?.models && (
            <div>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 14 }}>
                MODEL PERFORMANCE METRICS
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14 }}>
                {trainMetrics.models.map((m: any) => (
                  <div
                    key={m.model_name}
                    style={{
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: 14,
                      padding: "18px 20px",
                    }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 600, color: "#e0e0e0", marginBottom: 4 }}>
                      {m.model_name}
                    </div>
                    <div style={{ fontSize: 10, color: "#555", marginBottom: 12, fontFamily: "JetBrains Mono, monospace" }}>
                      {m.model_type}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {Object.entries(m.metrics || {})
                        .filter(([k]) => typeof m.metrics[k] === "number")
                        .map(([k, v]) => (
                          <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                            <span style={{ color: "#666" }}>{k}</span>
                            <span style={{ color: "#00d4a0", fontWeight: 600, fontFamily: "JetBrains Mono, monospace" }}>
                              {(v as number).toFixed(4)}
                            </span>
                          </div>
                        ))}
                    </div>
                    {m.top_features?.length > 0 && (
                      <div style={{ marginTop: 12, fontSize: 10, color: "#555" }}>
                        Top features: {m.top_features.slice(0, 3).join(", ")}
                      </div>
                    )}
                    <div style={{ marginTop: 8, fontSize: 10, color: "#444" }}>
                      {m.n_samples} samples · {m.n_features} features
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ═════════════════════════════ PREDICTIONS ══════════════════════════════ */}
      {activeTab === "predictions" && (
        <div>
          {!predictions ? (
            <div
              style={{
                textAlign: "center",
                padding: "60px 20px",
                color: "#555",
                border: "1px dashed rgba(255,255,255,0.1)",
                borderRadius: 16,
              }}
            >
              <div style={{ fontSize: 48, marginBottom: 16 }}>⚡</div>
              <div style={{ fontSize: 16, marginBottom: 8 }}>No predictions yet</div>
              <div style={{ fontSize: 13 }}>
                Ingest data and train models, then click{" "}
                <strong style={{ color: "#60a5fa" }}>Run Predictions</strong>
              </div>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
              {/* Cardiovascular Risk */}
              {predictions.cardiovascular_risk && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${predictions.cardiovascular_risk.risk_label === "elevated" ? "#ff6b5b44" : "#00d4a044"}`,
                    borderRadius: 16,
                    padding: "22px 24px",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      background: predictions.cardiovascular_risk.risk_label === "elevated"
                        ? "linear-gradient(90deg, #ff6b5b00, #ff6b5b, #ff6b5b00)"
                        : "linear-gradient(90deg, #00d4a000, #00d4a0, #00d4a000)",
                      borderRadius: "16px 16px 0 0",
                    }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>CARDIOVASCULAR RISK</div>
                      <div style={{ fontSize: 24, fontWeight: 800, color: predictions.cardiovascular_risk.risk_label === "elevated" ? "#ff6b5b" : "#00d4a0" }}>
                        {(predictions.cardiovascular_risk.risk_score * 100).toFixed(1)}%
                      </div>
                    </div>
                    <GaugeArc
                      value={predictions.cardiovascular_risk.risk_score * 100}
                      max={100}
                      color={predictions.cardiovascular_risk.risk_label === "elevated" ? "#ff6b5b" : "#00d4a0"}
                    />
                  </div>
                  <div
                    style={{
                      display: "inline-block",
                      padding: "4px 12px",
                      borderRadius: 20,
                      fontSize: 12,
                      fontWeight: 700,
                      background: predictions.cardiovascular_risk.risk_label === "elevated" ? "#ff6b5b22" : "#00d4a022",
                      color: predictions.cardiovascular_risk.risk_label === "elevated" ? "#ff6b5b" : "#00d4a0",
                    }}
                  >
                    {predictions.cardiovascular_risk.risk_label.toUpperCase()}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 11, color: "#555" }}>
                    Confidence: {(predictions.cardiovascular_risk.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}

              {/* Sleep Quality */}
              {predictions.sleep_quality && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid #a78bfa44",
                    borderRadius: 16,
                    padding: "22px 24px",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      background: "linear-gradient(90deg, #a78bfa00, #a78bfa, #a78bfa00)",
                      borderRadius: "16px 16px 0 0",
                    }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>SLEEP QUALITY INDEX</div>
                      <div style={{ fontSize: 24, fontWeight: 800, color: "#a78bfa" }}>
                        {predictions.sleep_quality.quality_index.toFixed(1)}
                        <span style={{ fontSize: 14, color: "#666" }}> / 100</span>
                      </div>
                    </div>
                    <GaugeArc
                      value={predictions.sleep_quality.quality_index}
                      max={100}
                      color="#a78bfa"
                    />
                  </div>
                  <div
                    style={{
                      display: "inline-block",
                      padding: "4px 12px",
                      borderRadius: 20,
                      fontSize: 12,
                      fontWeight: 700,
                      background: "#a78bfa22",
                      color: "#a78bfa",
                    }}
                  >
                    {predictions.sleep_quality.quality_label.toUpperCase()}
                  </div>
                </div>
              )}

              {/* Stress Pattern */}
              {predictions.stress_pattern && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid #ffa94d44",
                    borderRadius: 16,
                    padding: "22px 24px",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      background: "linear-gradient(90deg, #ffa94d00, #ffa94d, #ffa94d00)",
                      borderRadius: "16px 16px 0 0",
                    }}
                  />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>STRESS PATTERN</div>
                    <div style={{ fontSize: 24, fontWeight: 800, color: "#ffa94d" }}>
                      {predictions.stress_pattern.stress_label.toUpperCase()}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {Object.entries(predictions.stress_pattern.probabilities).map(
                      ([label, prob]) => (
                        <div key={label} style={{ flex: 1 }}>
                          <div style={{ fontSize: 10, color: "#666", marginBottom: 4 }}>{label}</div>
                          <div
                            style={{
                              height: 40,
                              background: "rgba(255,255,255,0.04)",
                              borderRadius: 4,
                              display: "flex",
                              alignItems: "flex-end",
                              padding: "0 2px",
                            }}
                          >
                            <div
                              style={{
                                width: "100%",
                                height: `${(prob as number) * 100}%`,
                                background: label === "low" ? "#00d4a0" : label === "medium" ? "#ffa94d" : "#ff6b5b",
                                borderRadius: 3,
                                transition: "height 0.5s ease",
                              }}
                            />
                          </div>
                          <div style={{ fontSize: 10, color: "#888", textAlign: "center", marginTop: 2 }}>
                            {((prob as number) * 100).toFixed(0)}%
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {/* Anomaly Detection */}
              {predictions.anomaly && (
                <div
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${predictions.anomaly.is_anomaly ? "#ff6b5b44" : "#4ade8044"}`,
                    borderRadius: 16,
                    padding: "22px 24px",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      height: 3,
                      background: predictions.anomaly.is_anomaly
                        ? "linear-gradient(90deg, #ff6b5b00, #ff6b5b, #ff6b5b00)"
                        : "linear-gradient(90deg, #4ade8000, #4ade80, #4ade8000)",
                      borderRadius: "16px 16px 0 0",
                    }}
                  />
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>ANOMALY DETECTION</div>
                    <div
                      style={{
                        fontSize: 42,
                        textAlign: "center",
                        padding: "12px 0",
                      }}
                    >
                      {predictions.anomaly.is_anomaly ? "🚨" : "✅"}
                    </div>
                    <div
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: predictions.anomaly.is_anomaly ? "#ff6b5b" : "#4ade80",
                        textAlign: "center",
                      }}
                    >
                      {predictions.anomaly.interpretation}
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: "#555", textAlign: "center" }}>
                    Anomaly score: {predictions.anomaly.anomaly_score.toFixed(4)}
                    <br />
                    <span style={{ fontSize: 10 }}>(negative = more anomalous)</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ═════════════════════════════ TIMELINE ═════════════════════════════════ */}
      {activeTab === "timeline" && (
        <div>
          {!timeline ? (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "#555" }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>📅</div>
              <div>Ingest data first to see the health timeline</div>
            </div>
          ) : (
            <>
              <div style={{ fontSize: 12, color: "#888", marginBottom: 16 }}>
                {timeline.returned} of {timeline.total_events} events shown (most recent first)
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {timeline.events?.slice(0, 100).map((evt: any, i: number) => {
                  const colors: Record<string, string> = {
                    heart_rate: "#ff6b5b",
                    stress: "#ffa94d",
                    sleep_stage: "#a78bfa",
                    exercise: "#00d4a0",
                    weight: "#60a5fa",
                  };
                  const color = colors[evt.type] || "#888";
                  return (
                    <div
                      key={i}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 14,
                        padding: "8px 14px",
                        background: "rgba(255,255,255,0.02)",
                        borderRadius: 8,
                        borderLeft: `3px solid ${color}`,
                      }}
                    >
                      <div
                        style={{
                          fontSize: 10,
                          color: "#555",
                          fontFamily: "JetBrains Mono, monospace",
                          minWidth: 140,
                        }}
                      >
                        {evt.timestamp?.slice(0, 19).replace("T", " ")}
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          color: color,
                          fontWeight: 600,
                          minWidth: 90,
                          textTransform: "uppercase",
                          letterSpacing: 0.5,
                        }}
                      >
                        {evt.type.replace("_", " ")}
                      </div>
                      <div style={{ fontSize: 12, color: "#ccc" }}>
                        {evt.type === "heart_rate" && `${evt.value} bpm`}
                        {evt.type === "stress" && `Score: ${evt.value} (${evt.label})`}
                        {evt.type === "sleep_stage" && `${evt.label} · ${evt.duration_min}min`}
                        {evt.type === "exercise" &&
                          `${evt.exercise_type} · ${evt.duration_min}min · HR ${evt.mean_hr} bpm · ${evt.calories} kcal`}
                        {evt.type === "weight" &&
                          `${evt.weight_kg}kg · BMI ${evt.bmi} · Fat ${evt.body_fat_pct}%`}
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}

      {/* ═════════════════════════════ XAI ══════════════════════════════════════ */}
      {activeTab === "xai" && (
        <div>
          {!predictions?.explanation ? (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "#555" }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>◬</div>
              <div>Run predictions first to see feature explanations</div>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 20 }}>
              {Object.entries(predictions.explanation).map(([modelKey, feats]: [string, any]) => {
                const colors: Record<string, string> = {
                  cardiovascular_risk: "#ff6b5b",
                  sleep_quality: "#a78bfa",
                  stress_pattern: "#ffa94d",
                };
                const color = colors[modelKey] || "#60a5fa";
                const maxImp = Math.max(...feats.map((f: any) => f.importance));
                return (
                  <div
                    key={modelKey}
                    style={{
                      background: "rgba(255,255,255,0.03)",
                      border: `1px solid ${color}33`,
                      borderRadius: 14,
                      padding: "18px 20px",
                    }}
                  >
                    <div style={{ fontSize: 12, color, fontWeight: 700, marginBottom: 16 }}>
                      {modelKey.replace(/_/g, " ").toUpperCase()} — TOP FEATURES
                    </div>
                    {feats.map((f: any) => (
                      <ShapBar
                        key={f.feature}
                        feature={f.feature}
                        importance={f.importance}
                        value={f.patient_value}
                        maxImportance={maxImp}
                        color={color}
                      />
                    ))}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ═════════════════════════════ HITL ══════════════════════════════════════ */}
      {activeTab === "hitl" && (
        <div>
          <div
            style={{
              marginBottom: 20,
              padding: "14px 18px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 12,
              fontSize: 13,
              color: "#888",
            }}
          >
            🧑‍⚕️ <strong style={{ color: "#e0e0e0" }}>Human-in-the-Loop Review</strong>
            <br />
            <span style={{ fontSize: 11 }}>
              Review AI predictions and provide clinical oversight. Accept, flag, or override model outputs.
            </span>
          </div>

          {!predictions ? (
            <div style={{ textAlign: "center", padding: "40px 20px", color: "#555" }}>
              Run predictions first to enable HITL review
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {[
                {
                  id: "cardio",
                  label: "Cardiovascular Risk Assessment",
                  summary: predictions.cardiovascular_risk
                    ? `${(predictions.cardiovascular_risk.risk_score * 100).toFixed(1)}% risk — ${predictions.cardiovascular_risk.risk_label}`
                    : "—",
                  severity: predictions.cardiovascular_risk?.risk_label === "elevated" ? "high" : "low",
                },
                {
                  id: "sleep",
                  label: "Sleep Quality Prediction",
                  summary: predictions.sleep_quality
                    ? `Quality index ${predictions.sleep_quality.quality_index.toFixed(1)}/100 — ${predictions.sleep_quality.quality_label}`
                    : "—",
                  severity: predictions.sleep_quality?.quality_label === "poor" ? "high" : "low",
                },
                {
                  id: "stress",
                  label: "Stress Pattern Classification",
                  summary: predictions.stress_pattern
                    ? `${predictions.stress_pattern.stress_label} stress (confidence: ${(predictions.stress_pattern.confidence * 100).toFixed(0)}%)`
                    : "—",
                  severity: predictions.stress_pattern?.stress_label === "high" ? "high" : "low",
                },
                {
                  id: "anomaly",
                  label: "Anomaly Detection",
                  summary: predictions.anomaly
                    ? predictions.anomaly.interpretation
                    : "—",
                  severity: predictions.anomaly?.is_anomaly ? "high" : "low",
                },
              ].map((item) => (
                <div
                  key={item.id}
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${item.severity === "high" ? "#ff6b5b33" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: 14,
                    padding: "18px 20px",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#e0e0e0", marginBottom: 4 }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: 12, color: "#888" }}>{item.summary}</div>
                    </div>
                    {item.severity === "high" && (
                      <div
                        style={{
                          padding: "3px 10px",
                          background: "#ff6b5b22",
                          color: "#ff6b5b",
                          borderRadius: 20,
                          fontSize: 10,
                          fontWeight: 700,
                          border: "1px solid #ff6b5b44",
                        }}
                      >
                        REVIEW REQUIRED
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
                    {["accept", "flag", "override"].map((action) => {
                      const selected = hitlDecision[item.id] === action;
                      const colors: Record<string, string> = {
                        accept: "#00d4a0",
                        flag: "#ffa94d",
                        override: "#ff6b5b",
                      };
                      return (
                        <button
                          key={action}
                          id={`hitl-${item.id}-${action}`}
                          onClick={() =>
                            setHitlDecision((p) => ({ ...p, [item.id]: action }))
                          }
                          style={{
                            padding: "7px 16px",
                            borderRadius: 8,
                            border: `1px solid ${colors[action]}${selected ? "88" : "33"}`,
                            background: selected ? `${colors[action]}22` : "transparent",
                            color: colors[action],
                            fontSize: 12,
                            fontWeight: selected ? 700 : 400,
                            cursor: "pointer",
                            transition: "all 0.2s",
                          }}
                        >
                          {action === "accept" && "✓ Accept"}
                          {action === "flag" && "⚑ Flag for Review"}
                          {action === "override" && "✎ Override"}
                        </button>
                      );
                    })}
                  </div>
                  {hitlDecision[item.id] && (
                    <div
                      style={{
                        marginTop: 10,
                        fontSize: 11,
                        color: "#666",
                        fontFamily: "JetBrains Mono, monospace",
                      }}
                    >
                      Decision: {hitlDecision[item.id].toUpperCase()} — logged at{" "}
                      {new Date().toISOString().slice(0, 19).replace("T", " ")} UTC
                    </div>
                  )}
                </div>
              ))}

              {/* Fuse result */}
              {fuseResult && (
                <div
                  style={{
                    padding: "16px 20px",
                    background: "#00d4a012",
                    border: "1px solid #00d4a033",
                    borderRadius: 14,
                    marginTop: 8,
                  }}
                >
                  <div style={{ fontSize: 12, color: "#00d4a0", fontWeight: 700, marginBottom: 8 }}>
                    ✅ {fuseResult.message}
                  </div>
                  <div style={{ fontSize: 11, color: "#888" }}>
                    {fuseResult.next_steps?.map((s: string) => (
                      <div key={s}>→ {s}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
