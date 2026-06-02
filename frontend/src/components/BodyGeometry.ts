"use client";
import { useMemo } from "react";

/**
 * Anatomical positions for organs in a stylized 3D human body.
 * Coordinates are roughly in meters; y=0 is feet, y=1.8 is head.
 * x = left(-) / right(+), z = back(-) / front(+)
 *
 * Each entry is an optional list of additional node IDs that share
 * the same anatomical location (so multiple causal nodes can
 * map to the same organ).
 */
export const ORGAN_POSITIONS: Record<string, [number, number, number]> = {
  // organs
  heart:        [-0.08, 1.42, 0.10],
  vasculature:  [ 0.00, 1.20, 0.05],
  lungs:        [ 0.00, 1.40, 0.06],
  pancreas:     [-0.04, 1.10, 0.08],
  liver:        [ 0.10, 1.18, 0.10],
  kidney:       [ 0.00, 1.05, -0.08],

  // biomarkers sit on/inside the organ that produces them
  hr:           [-0.10, 1.46, 0.10],   // heart
  hrv:          [-0.06, 1.40, 0.10],   // heart
  spo2:         [ 0.00, 1.42, 0.08],   // lungs
  glucose:      [-0.04, 1.08, 0.08],   // pancreas
  systolic_bp:  [ 0.00, 1.20, 0.05],   // vasculature
  diastolic_bp: [ 0.00, 1.18, 0.05],   // vasculature
  bmi:          [ 0.00, 0.95, 0.20],   // torso / fat

  // diseases are body-wide — show on chest for visibility
  t2d:          [-0.04, 1.06, 0.08],   // near pancreas
  hypertension: [ 0.00, 1.22, 0.04],   // heart/vasculature
  cvd:          [-0.08, 1.40, 0.10],   // heart
  copd:         [ 0.00, 1.40, 0.10],   // lungs

  // demographic — head
  age:          [ 0.00, 1.70, 0.06],
};

const ORGAN_LABEL: Record<string, string> = {
  heart: "Heart", vasculature: "Vasculature", lungs: "Lungs",
  pancreas: "Pancreas", liver: "Liver", kidney: "Kidney",
  hr: "HR", hrv: "HRV", spo2: "SpO₂",
  glucose: "Glucose", systolic_bp: "SBP", diastolic_bp: "DBP", bmi: "BMI",
  t2d: "T2D", hypertension: "HTN", cvd: "CVD", copd: "COPD",
  age: "Age",
};

const ORGAN_COLOR: Record<string, string> = {
  heart:        "#ef4444",
  vasculature:  "#f97316",
  lungs:        "#06b6d4",
  pancreas:     "#a855f7",
  liver:        "#eab308",
  kidney:       "#ec4899",
  hr:           "#fb7185", hrv: "#fb7185",
  spo2:         "#22d3ee",
  glucose:      "#c084fc",
  systolic_bp:  "#fb923c", diastolic_bp: "#fb923c",
  bmi:          "#94a3b8",
  t2d:          "#f43f5e", hypertension: "#f59e0b",
  cvd:          "#dc2626", copd: "#0ea5e9",
  age:          "#a3a3a3",
};

export function organPosition(id: string): [number, number, number] | null {
  return ORGAN_POSITIONS[id] || null;
}

export function organColor(id: string, fallback: string): string {
  return ORGAN_COLOR[id] || fallback;
}

export function organLabel(id: string): string {
  return ORGAN_LABEL[id] || id;
}

export function useBodyGeometry() {
  return useMemo(() => ({
    // height ≈ 1.8m, simplified primitives
    head:        { r: 0.10, y: 1.70 },
    neck:        { rTop: 0.05, rBot: 0.06, h: 0.10, y: 1.55 },
    torso:       { rTop: 0.20, rBot: 0.22, h: 0.65, y: 1.20 },
    shoulderL:   { x: -0.30, y: 1.50, len: 0.30 },
    shoulderR:   { x:  0.30, y: 1.50, len: 0.30 },
    armL:        { x: -0.34, y: 1.05, len: 0.40 },
    armR:        { x:  0.34, y: 1.05, len: 0.40 },
    hip:         { rTop: 0.18, rBot: 0.16, h: 0.10, y: 0.85 },
    legL:        { x: -0.10, y: 0.45, len: 0.75 },
    legR:        { x:  0.10, y: 0.45, len: 0.75 },
  }), []);
}
