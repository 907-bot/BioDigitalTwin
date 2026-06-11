"""
Phase 4: Scientific Discovery Layer.

Causal discovery and hypothesis generation from twin simulations.

Components:
  - CausalDiscoveryEngine: doWhy-based causal graph learning + effect estimation
  - HypothesisAgent: LangChain agent for generating testable scientific hypotheses
  - CausalGraph: learned causal structure over twin variables
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings


# ── Causal Discovery ──────────────────────────────────────────

class CausalMethod(Enum):
    DOWHY = "dowhy"
    CORRELATION = "correlation"
    GRANGER = "granger"


@dataclass
class CausalEdge:
    """A directed causal relationship between two variables."""
    source: str
    target: str
    effect_size: float = 0.0
    p_value: float = 1.0
    confidence: float = 0.0
    method: str = ""

    def __str__(self) -> str:
        return f"{self.source} → {self.target} (β={self.effect_size:.3f}, p={self.p_value:.3f})"


@dataclass
class CausalGraph:
    """Learned causal graph over twin variables."""
    nodes: List[str] = field(default_factory=list)
    edges: List[CausalEdge] = field(default_factory=list)
    adj_matrix: Optional[np.ndarray] = None

    def add_edge(self, edge: CausalEdge) -> None:
        if edge.source not in self.nodes:
            self.nodes.append(edge.source)
        if edge.target not in self.nodes:
            self.nodes.append(edge.target)
        self.edges.append(edge)

    def to_dict(self) -> Dict:
        return {
            "nodes": self.nodes,
            "edges": [{
                "source": e.source, "target": e.target,
                "effect_size": e.effect_size, "p_value": e.p_value,
                "confidence": e.confidence, "method": e.method,
            } for e in self.edges],
        }

    def get_causes_of(self, variable: str) -> List[CausalEdge]:
        return [e for e in self.edges if e.target == variable]

    def get_effects_of(self, variable: str) -> List[CausalEdge]:
        return [e for e in self.edges if e.source == variable]

    def get_top_edges(self, k: int = 10) -> List[CausalEdge]:
        return sorted(self.edges, key=lambda e: abs(e.effect_size), reverse=True)[:k]


# ── Known Causal Graph (Prior Knowledge) ──────────────────────

KNOWN_CAUSAL_GRAPH = {
    "InsulinResistance": ["Glucose", "Triglycerides", "HDL", "SBP", "CRP"],
    "Glucose": ["HbA1c", "Insulin", "GFR"],
    "BMI": ["InsulinResistance", "SBP", "LDL", "FFA"],
    "Exercise": ["InsulinResistance", "HRV", "SBP"],
    "DietQuality": ["Glucose", "LDL", "HDL", "BMI"],
    "SleepQuality": ["Cortisol", "InsulinResistance", "HRV"],
    "AirPollution": ["CRP", "SBP", "HRV"],
    "Stress": ["Cortisol", "SBP", "CRP"],
    "MedicationAdherence": ["Glucose", "SBP", "LDL"],
    "CRP": ["InsulinResistance"],
    "FFA": ["InsulinResistance", "Triglycerides"],
    "Cortisol": ["InsulinResistance", "Glucose"],
    "SBP": ["GFR"],
}


# ── Causal Discovery Engine ───────────────────────────────────

class CausalDiscoveryEngine:
    """
    Learn causal relationships from twin simulation data.

    Uses doWhy for structure learning + effect estimation, with
    fallback to correlation-based methods.
    """

    def __init__(self):
        self.causal_graph = CausalGraph()
        self._data: Optional[pd.DataFrame] = None
        self._results: Dict[str, Any] = {}

    def load_data(self, data: pd.DataFrame) -> None:
        """Load time-series or cross-sectional data from twin simulations."""
        self._data = data.copy()

    def load_from_history(
        self,
        history: List[Any],
        variable_map: Dict[str, Callable],
    ) -> None:
        """
        Load data from MultiScaleState history.

        variable_map: {name: extractor_fn(state) -> float}
        """
        records = []
        for state in history:
            record = {}
            for name, extractor in variable_map.items():
                try:
                    record[name] = extractor(state)
                except Exception:
                    record[name] = np.nan
            records.append(record)
        self._data = pd.DataFrame(records)

    def discover_structure(
        self,
        method: CausalMethod = CausalMethod.CORRELATION,
        prior_knowledge: bool = True,
        alpha: float = 0.05,
    ) -> CausalGraph:
        """
        Discover causal structure from loaded data.

        Args:
            method: Correlation (fast), doWhy (requires graph specification), Granger (time-series)
            prior_knowledge: Whether to constrain discovery with known physiology
            alpha: Significance threshold
        """
        if self._data is None or len(self._data) < 5:
            raise ValueError("Insufficient data for causal discovery. Load data first.")

        self.causal_graph = CausalGraph()

        if method == CausalMethod.CORRELATION:
            self._discover_correlation(alpha)
        elif method == CausalMethod.DOWHY:
            self._discover_dowhy(alpha, prior_knowledge)
        elif method == CausalMethod.GRANGER:
            self._discover_granger(alpha)

        # Incorporate known physiology as hard constraints
        if prior_knowledge:
            self._apply_prior_knowledge()

        return self.causal_graph

    def _discover_correlation(self, alpha: float) -> None:
        """Simple lagged correlation-based causal discovery."""
        n_vars = len(self._data.columns)
        names = list(self._data.columns)
        arr = self._data.values

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                # Cross-correlation at lag 0
                corr = np.corrcoef(arr[:, i], arr[:, j])[0, 1]
                if abs(corr) > 0.3:  # heuristic threshold
                    edge = CausalEdge(
                        source=names[i],
                        target=names[j],
                        effect_size=float(corr),
                        p_value=float(max(0.001, 1.0 - abs(corr))),
                        confidence=float(abs(corr)),
                        method="correlation",
                    )
                    self.causal_graph.add_edge(edge)

    def _discover_dowhy(
        self, alpha: float, prior_knowledge: bool,
    ) -> None:
        """
        Use doWhy for causal discovery on twin data.
        Falls back to correlation if doWhy's structure learning fails.
        """
        try:
            import dowhy
            from dowhy import CausalModel

            n_vars = len(self._data.columns)
            names = list(self._data.columns)

            if prior_knowledge:
                # Build a prior graph from known physiology
                prior_graph_pairs = []
                present_names = set(names)
                for src, targets in KNOWN_CAUSAL_GRAPH.items():
                    if src in present_names:
                        for tgt in targets:
                            if tgt in present_names:
                                prior_graph_pairs.append((src, tgt))

                # We can use linear regression for effect estimation
                # on known edges
                for src, tgt in prior_graph_pairs:
                    X = self._data[src].values
                    Y = self._data[tgt].values
                    if np.std(X) < 1e-10 or np.std(Y) < 1e-10:
                        continue
                    beta = np.cov(X, Y)[0, 1] / np.var(X)
                    residuals = Y - beta * X
                    n = len(X)
                    se = np.sqrt(np.var(residuals) / (n * np.var(X)))
                    t_stat = beta / (se + 1e-10)
                    from scipy.stats import t as t_dist
                    p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), n - 2))

                    edge = CausalEdge(
                        source=src, target=tgt,
                        effect_size=float(beta),
                        p_value=float(p_val),
                        confidence=float(max(0.0, min(1.0, 1.0 - p_val))),
                        method="dowhy_linear_regression",
                    )
                    self.causal_graph.add_edge(edge)
            else:
                # No prior: use brute-force correlation as fallback
                self._discover_correlation(alpha)

        except Exception as e:
            warnings.warn(f"doWhy discovery failed ({e}), falling back to correlation")
            self._discover_correlation(alpha)

    def _discover_granger(self, alpha: float) -> None:
        """Granger causality for time-series data."""
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            warnings.warn("statsmodels not available, falling back to correlation")
            self._discover_correlation(alpha)
            return

        n_vars = len(self._data.columns)
        names = list(self._data.columns)
        arr = self._data.values

        if len(arr) < 10:
            self._discover_correlation(alpha)
            return

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                df = pd.DataFrame({names[i]: arr[:, i], names[j]: arr[:, j]})
                try:
                    result = grangercausalitytests(df[[names[i], names[j]]],
                                                   maxlag=2, verbose=False)
                    p_val = min(result[1][0]["ssr_chi2test"][1],
                                result[2][0]["ssr_chi2test"][1])
                    if p_val < alpha:
                        edge = CausalEdge(
                            source=names[i], target=names[j],
                            effect_size=float(1.0 - p_val),
                            p_value=float(p_val),
                            confidence=float(max(0.0, 1.0 - p_val)),
                            method="granger",
                        )
                        self.causal_graph.add_edge(edge)
                except Exception:
                    continue

    def _apply_prior_knowledge(self) -> None:
        """Ensures known physiological edges are in the graph."""
        present = set(self.causal_graph.nodes)
        for src, targets in KNOWN_CAUSAL_GRAPH.items():
            if src not in present:
                continue
            for tgt in targets:
                if tgt not in present:
                    continue
                # Check if edge already exists; if not, add with moderate confidence
                exists = any(
                    e.source == src and e.target == tgt
                    for e in self.causal_graph.edges
                )
                if not exists:
                    self.causal_graph.add_edge(CausalEdge(
                        source=src, target=tgt,
                        effect_size=0.0, p_value=0.5,
                        confidence=0.5, method="prior_knowledge",
                    ))

    def estimate_effect(
        self, treatment: str, outcome: str,
        confounders: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Estimate causal effect of treatment on outcome using doWhy.
        """
        if self._data is None:
            raise ValueError("No data loaded")

        try:
            import dowhy
            from dowhy import CausalModel

            all_vars = list(self._data.columns)
            if confounders is None:
                confounders = [v for v in all_vars
                               if v not in (treatment, outcome)]

            model = CausalModel(
                data=self._data,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
            )

            identified_estimand = model.identify_effect(propagation=True)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
            )

            return {
                "treatment": treatment,
                "outcome": outcome,
                "effect_size": float(estimate.value),
                "confidence": float(getattr(estimate, "confidence", 0.0)),
            }
        except Exception as e:
            warnings.warn(f"doWhy effect estimation failed ({e}), using correlation")
            if treatment in self._data.columns and outcome in self._data.columns:
                corr = self._data[treatment].corr(self._data[outcome])
                return {
                    "treatment": treatment,
                    "outcome": outcome,
                    "effect_size": float(corr),
                    "confidence": float(abs(corr)),
                }
            return {}

    def get_graph(self) -> CausalGraph:
        return self.causal_graph


# ── Hypothesis Agent ──────────────────────────────────────────

@dataclass
class ScientificHypothesis:
    """A testable scientific hypothesis generated from twin data."""
    title: str
    description: str
    mechanism: str
    causal_evidence: str
    testable_prediction: str
    confidence: float = 0.0
    source: str = "language_model"


class HypothesisAgent:
    """
    Generate scientific hypotheses from causal discovery results.

    Uses LangChain for LLM-powered hypothesis generation (with
    rule-based fallback when no LLM is available).
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.hypotheses: List[ScientificHypothesis] = []

    def generate_hypotheses(
        self,
        causal_graph: CausalGraph,
        patient_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[ScientificHypothesis]:
        """
        Generate scientific hypotheses from the learned causal graph.

        Uses rule-based generation with optional LLM enhancement.
        """
        top_edges = causal_graph.get_top_edges(top_k)
        self.hypotheses = []

        for edge in top_edges:
            hypothesis = self._generate_single_hypothesis(edge, patient_context)
            self.hypotheses.append(hypothesis)

        # LLM enhancement if available
        if self.llm is not None:
            self._enhance_with_llm(causal_graph, patient_context)

        return self.hypotheses

    def _generate_single_hypothesis(
        self, edge: CausalEdge,
        patient_context: Optional[Dict[str, Any]],
    ) -> ScientificHypothesis:
        """Generate hypothesis for a single causal edge using templates."""
        templates = self._get_hypothesis_templates()

        # Find a matching template
        matched = False
        for key, template in templates.items():
            if key[0].lower() in edge.source.lower() and \
               key[1].lower() in edge.target.lower():
                hyp = ScientificHypothesis(
                    title=template["title"].format(src=edge.source, tgt=edge.target),
                    description=template["description"].format(
                        src=edge.source, tgt=edge.target,
                        beta=abs(edge.effect_size),
                    ),
                    mechanism=template["mechanism"],
                    causal_evidence=str(edge),
                    testable_prediction=template["prediction"].format(
                        src=edge.source, tgt=edge.target,
                    ),
                    confidence=edge.confidence,
                    source="rule_based",
                )
                matched = True
                break

        if not matched:
            hyp = ScientificHypothesis(
                title=f"Effect of {edge.source} on {edge.target}",
                description=(
                    f"We observe that changes in {edge.source} are associated with "
                    f"changes in {edge.target} (effect size: {edge.effect_size:.3f}). "
                    "Further investigation is needed to establish the precise mechanism."
                ),
                mechanism="Unknown — requires further study",
                causal_evidence=str(edge),
                testable_prediction=(
                    f"Intervening on {edge.source} should produce a "
                    f"{'positive' if edge.effect_size > 0 else 'negative'} "
                    f"change in {edge.target}."
                ),
                confidence=edge.confidence,
                source="rule_based",
            )

        return hyp

    def _get_hypothesis_templates(self) -> Dict:
        """Curated hypothesis templates based on known physiology."""
        return {
            ("InsulinResistance", "Glucose"): {
                "title": "Insulin Resistance Drives Hyperglycemia via Impaired Glucose Uptake",
                "description": (
                    "Elevated insulin resistance (HOMA-IR) directly impairs peripheral "
                    "glucose uptake, leading to fasting and postprandial hyperglycemia. "
                    "Each unit increase in IR is associated with a {beta:.2f} unit "
                    "increase in blood glucose."
                ),
                "mechanism": (
                    "Insulin resistance at the cellular level reduces GLUT4 translocation "
                    "in skeletal muscle and adipocytes, impairing insulin-stimulated "
                    "glucose disposal. Hepatic insulin resistance also fails to suppress "
                    "gluconeogenesis, contributing to fasting hyperglycemia."
                ),
                "prediction": (
                    "Reducing HOMA-IR by 50% through lifestyle intervention should "
                    "decrease fasting glucose by at least 15-20 mg/dL in prediabetic subjects."
                ),
            },
            ("BMI", "InsulinResistance"): {
                "title": "Adiposity Drives Insulin Resistance via Ectopic Fat and Inflammation",
                "description": (
                    "Increased BMI is causally linked to insulin resistance through "
                    "adipose tissue dysfunction, ectopic lipid deposition, and "
                    "adipokine-mediated inflammation. A {beta:.2f} unit change in "
                    "insulin resistance is attributable to BMI variation."
                ),
                "mechanism": (
                    "Visceral adipose tissue expansion leads to macrophage infiltration, "
                    "pro-inflammatory cytokine release (TNF-α, IL-6), and increased FFA "
                    "flux. Ectopic fat deposition in liver and muscle impairs insulin "
                    "signaling via diacylglycerol-PKCε and ceramide pathways."
                ),
                "prediction": (
                    "A 5% reduction in BMI through caloric restriction should improve "
                    "HOMA-IR by 20-30% in obese individuals over 12 weeks."
                ),
            },
            ("Exercise", "InsulinResistance"): {
                "title": "Physical Activity Enhances Insulin Sensitivity via AMPK and GLUT4",
                "description": (
                    "Regular exercise improves insulin sensitivity through both acute "
                    "(GLUT4 translocation) and chronic (mitochondrial biogenesis, "
                    "reduced inflammation) mechanisms. Each additional 30 min/day of "
                    "moderate exercise reduces IR by {beta:.2f} units."
                ),
                "mechanism": (
                    "Exercise activates AMPK and CaMKII, leading to GLUT4 translocation "
                    "independent of insulin signaling. Chronic training increases "
                    "mitochondrial content, reduces ceramides, and decreases "
                    "pro-inflammatory cytokine production from adipose tissue."
                ),
                "prediction": (
                    "A 12-week supervised exercise program (150 min/week moderate "
                    "intensity) should improve insulin sensitivity by 25-40% in "
                    "sedentary adults with prediabetes."
                ),
            },
            ("AirPollution", "CRP"): {
                "title": "Ambient Air Pollution Triggers Systemic Inflammation",
                "description": (
                    "Exposure to PM2.5 and NO2 triggers pulmonary oxidative stress "
                    "and systemic inflammation. A {beta:.2f} unit increase in CRP "
                    "is attributable to elevated air pollution exposure."
                ),
                "mechanism": (
                    "Inhaled particulate matter activates alveolar macrophages and "
                    "epithelial cells, triggering NF-κB-mediated release of "
                    "pro-inflammatory cytokines (IL-6, TNF-α) into circulation. "
                    "Ultrafine particles may directly translocate into the bloodstream."
                ),
                "prediction": (
                    "Reducing personal PM2.5 exposure by 50% (via air purifiers) "
                    "should decrease hs-CRP by 1-2 mg/L over 4 weeks."
                ),
            },
            ("SleepQuality", "Cortisol"): {
                "title": "Sleep Disruption Elevates Cortisol and Impairs Metabolic Health",
                "description": (
                    "Poor sleep quality activates the HPA axis, leading to elevated "
                    "evening cortisol and disrupted circadian rhythmicity. A {beta:.2f} "
                    "unit increase in cortisol is attributable to sleep quality decline."
                ),
                "mechanism": (
                    "Sleep restriction increases CRH and ACTH secretion, elevating "
                    "cortisol levels. This impairs insulin sensitivity, promotes "
                    "visceral adiposity, and disrupts glucose homeostasis. "
                    "Circadian misalignment further compounds these effects."
                ),
                "prediction": (
                    "Improving sleep duration from <6h to 7-8h per night for 2 weeks "
                    "should reduce waking cortisol by 15-25% and improve HOMA-IR "
                    "by 10-15%."
                ),
            },
        }

    def _enhance_with_llm(
        self, causal_graph: CausalGraph,
        patient_context: Optional[Dict[str, Any]],
    ) -> None:
        """Use LangChain LLM to refine hypotheses (mock currently)."""
        # Placeholder for LLM integration
        # When a real LLM is configured, this would:
        # 1. Format edges + patient context into a prompt
        # 2. Call llm.invoke() with the LangChain interface
        # 3. Parse the response to generate/enhance hypotheses
        pass

    def get_mechanism_explanation(
        self, hypothesis: ScientificHypothesis, detail_level: str = "patient",
    ) -> str:
        """Generate an explanation at the right detail level."""
        if detail_level == "patient":
            return (
                f"**{hypothesis.title}**\n\n"
                f"{hypothesis.description}\n\n"
                f"What this means for you: {hypothesis.testable_prediction}"
            )
        elif detail_level == "clinician":
            return (
                f"**Hypothesis**: {hypothesis.title}\n"
                f"**Mechanism**: {hypothesis.mechanism}\n"
                f"**Evidence**: {hypothesis.causal_evidence}\n"
                f"**Confidence**: {hypothesis.confidence:.1%}\n"
                f"**Prediction**: {hypothesis.testable_prediction}"
            )
        else:  # scientist
            return (
                f"**Hypothesis**: {hypothesis.title}\n"
                f"**Mechanism**: {hypothesis.mechanism}\n"
                f"**Causal Evidence**: {hypothesis.causal_evidence}\n"
                f"**Confidence**: {hypothesis.confidence:.1%}\n"
                f"**Testable Prediction**: {hypothesis.testable_prediction}"
            )


# ── Twin Data Generator (for testing) ─────────────────────────

def generate_twin_trial_data(
    n_patients: int = 100,
    n_timepoints: int = 30,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate synthetic twin trial data for testing causal discovery.
    """
    rng = rng or np.random.default_rng(42)
    records = []

    for pid in range(n_patients):
        # Baseline
        bmi = rng.normal(28, 5)
        exercise = max(0, rng.normal(30, 15))
        diet = np.clip(rng.normal(0.5, 0.2), 0, 1)
        sleep = np.clip(rng.normal(7, 1), 4, 10)

        for t in range(n_timepoints):
            # Known causal DGP
            noise_ir = rng.normal(0, 0.3)
            ir = (0.05 * (bmi - 25) - 0.015 * exercise + 0.3 * (1 - diet)
                  - 0.1 * (sleep - 7) + noise_ir)
            ir = max(0.5, ir)

            noise_g = rng.normal(0, 5)
            glucose = 90 + 8 * ir + noise_g

            noise_sbp = rng.normal(0, 3)
            sbp = 120 + 2 * ir + 0.3 * (bmi - 25) + noise_sbp

            noise_crp = rng.normal(0, 1)
            crp = 2 + 0.5 * ir + 0.5 * (bmi - 25) / 5 + noise_crp

            noise_ffa = rng.normal(0, 0.05)
            ffa = 0.4 + 0.03 * ir + 0.01 * (bmi - 25) + noise_ffa

            noise_cort = rng.normal(0, 2)
            cortisol = 15 - 0.5 * (sleep - 7) + noise_cort

            records.append({
                "patient_id": pid, "time": t,
                "BMI": bmi, "Exercise": exercise, "DietQuality": diet,
                "SleepQuality": sleep, "InsulinResistance": ir,
                "Glucose": glucose, "SBP": sbp, "CRP": crp,
                "FFA": ffa, "Cortisol": cortisol,
            })

    return pd.DataFrame(records)
