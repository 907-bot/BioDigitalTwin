"""
Biological graph ontology — shared across all microservices.

Defines the static backbone that every patient graph hangs off:
  - 7 biomarker nodes (HR, HRV, SpO2, glucose, BP-systolic, BP-diastolic, BMI)
  - 6 organ nodes   (heart, vasculature, lungs, pancreas, liver, kidney)
  - 4 disease nodes (T2D, hypertension, CVD, COPD)
  - typed edges
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal


NodeKind = Literal["biomarker", "organ", "disease"]


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    name: str
    unit: str = ""
    healthy_lo: Optional[float] = None
    healthy_hi: Optional[float] = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: str
    weight: float = 1.0


BIOMARKERS: List[Node] = [
    Node("hr",           "biomarker", "Heart rate",          "bpm",   60, 100),
    Node("hrv",          "biomarker", "HR variability",      "ms",    20, 70),
    Node("spo2",         "biomarker", "Blood oxygen",        "%",     95, 100),
    Node("glucose",      "biomarker", "Fasting glucose",     "mg/dL", 70, 110),
    Node("systolic_bp",  "biomarker", "Systolic BP",         "mmHg",  90, 130),
    Node("diastolic_bp", "biomarker", "Diastolic BP",        "mmHg",  60, 85),
    Node("bmi",          "biomarker", "Body mass index",     "kg/m2", 18.5, 25),
]

ORGANS: list[Node] = [
    Node("heart",      "organ", "Heart"),
    Node("vasculature","organ", "Vasculature"),
    Node("lungs",      "organ", "Lungs"),
    Node("pancreas",   "organ", "Pancreas"),
    Node("liver",      "organ", "Liver"),
    Node("kidney",     "organ", "Kidney"),
]

DISEASES: list[Node] = [
    Node("t2d",          "disease", "Type 2 diabetes"),
    Node("hypertension", "disease", "Hypertension"),
    Node("cvd",          "disease", "Cardiovascular disease"),
    Node("copd",         "disease", "COPD"),
]

ALL_NODES: list[Node] = BIOMARKERS + ORGANS + DISEASES
NODE_INDEX: dict[str, Node] = {n.id: n for n in ALL_NODES}

EDGES: list[Edge] = [
    Edge("hr",           "heart",       "REGULATED_BY", 1.0),
    Edge("hrv",          "heart",       "REGULATED_BY", 1.0),
    Edge("spo2",         "lungs",       "REGULATED_BY", 1.0),
    Edge("glucose",      "pancreas",    "REGULATED_BY", 1.0),
    Edge("glucose",      "liver",       "REGULATED_BY", 0.7),
    Edge("systolic_bp",  "heart",       "REGULATED_BY", 0.8),
    Edge("systolic_bp",  "vasculature", "REGULATED_BY", 1.0),
    Edge("systolic_bp",  "kidney",      "REGULATED_BY", 0.7),
    Edge("diastolic_bp", "vasculature", "REGULATED_BY", 1.0),
    Edge("diastolic_bp", "kidney",      "REGULATED_BY", 0.6),
    Edge("bmi",          "liver",       "REGULATED_BY", 0.5),
    Edge("bmi",          "pancreas",    "REGULATED_BY", 0.5),
    Edge("glucose",      "t2d",          "ELEVATED_IN",  1.0),
    Edge("systolic_bp",  "hypertension", "ELEVATED_IN",  1.0),
    Edge("diastolic_bp", "hypertension", "ELEVATED_IN",  1.0),
    Edge("systolic_bp",  "cvd",          "ELEVATED_IN",  0.7),
    Edge("diastolic_bp", "cvd",          "ELEVATED_IN",  0.6),
    Edge("bmi",          "t2d",          "ELEVATED_IN",  0.7),
    Edge("bmi",          "hypertension", "ELEVATED_IN",  0.6),
    Edge("bmi",          "cvd",          "ELEVATED_IN",  0.5),
    Edge("hrv",          "cvd",          "DEPRESSED_IN", 0.8),
    Edge("spo2",         "copd",         "DEPRESSED_IN", 1.0),
    Edge("heart",        "cvd",          "AFFECTED_BY",  1.0),
    Edge("vasculature",  "cvd",          "AFFECTED_BY",  1.0),
    Edge("vasculature",  "hypertension", "AFFECTED_BY",  1.0),
    Edge("pancreas",     "t2d",          "AFFECTED_BY",  1.0),
    Edge("liver",        "t2d",          "AFFECTED_BY",  0.6),
    Edge("lungs",        "copd",         "AFFECTED_BY",  1.0),
    Edge("t2d",          "cvd",          "COMORBID_WITH", 0.7),
    Edge("t2d",          "hypertension", "COMORBID_WITH", 0.8),
    Edge("hypertension", "cvd",          "COMORBID_WITH", 0.9),
    Edge("copd",         "cvd",          "COMORBID_WITH", 0.5),
]


def neighbors(node_id: str) -> list[Edge]:
    return [e for e in EDGES if e.src == node_id or e.dst == node_id]
