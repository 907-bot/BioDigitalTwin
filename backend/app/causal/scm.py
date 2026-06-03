"""
Phase 4 — Structural Causal Model built from the biological ontology.

We construct a DAG where:
  - nodes = (per-patient) biomarkers + demographics + (global) disease states
  - edges = the typed edges in app.graph.ontology (ELEVATED_IN, REGULATED_BY, ...)

The SCM is a deterministic, time-invariant DAG for the whole cohort;
treatment / outcome / confounders are selected per query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from app.graph.ontology import (
    ALL_NODES, BIOMARKERS, DISEASES, EDGES, NODE_INDEX, ORGANS,
)


# Treatment/outcome registry — which biomarker is a valid treatment for
# which disease. Used by the ATE / CATE endpoints.
TREATMENT_TARGETS: dict[str, list[str]] = {
    "metformin":     ["t2d"],
    "losartan":      ["hypertension"],
    "statin":        ["cvd", "hypertension"],
    "exercise_30m":  ["t2d", "hypertension", "cvd"],
    "weight_loss":   ["t2d", "hypertension", "cvd"],
    "smoking_cessation": ["copd", "cvd"],
}

OUTCOMES_FOR_DISEASE: dict[str, list[str]] = {
    "t2d":          ["glucose", "bmi", "hrv"],
    "hypertension": ["systolic_bp", "diastolic_bp"],
    "cvd":          ["systolic_bp", "hrv", "spo2"],
    "copd":         ["spo2"],
}


def build_causal_dag() -> nx.DiGraph:
    """Translate the ontology into a NetworkX DiGraph.
    Edge direction follows the semantic direction in EDGES (src -> dst).

    We also inject well-established clinical causal edges (age/gender/BMI
    drive downstream biomarkers) so the SCM has actual structure to fit.
    """
    g = nx.DiGraph()
    for n in ALL_NODES:
        g.add_node(n.id, kind=n.kind, name=n.name)
    for e in EDGES:
        g.add_edge(e.src, e.dst, rel=e.rel, weight=e.weight)
    # clinical drivers (these are the exogenous root causes)
    for biomarker, weight, rel in [
        ("glucose",      0.8,  "DRIVES"),
        ("systolic_bp",  0.6,  "DRIVES"),
        ("diastolic_bp", 0.3,  "DRIVES"),
        ("hrv",         -0.5,  "DRIVES"),
        ("spo2",        -0.2,  "DRIVES"),
    ]:
        if "bmi" not in g:
            g.add_node("bmi", kind="biomarker", name="Body mass index")
        g.add_edge("bmi", biomarker, rel=rel, weight=weight)
    for biomarker, weight, rel in [
        ("systolic_bp",  0.5, "AGE_DRIVES"),
        ("diastolic_bp", 0.2, "AGE_DRIVES"),
        ("hrv",         -0.6, "AGE_DRIVES"),
        ("glucose",      0.2, "AGE_DRIVES"),
    ]:
        if "age" not in g:
            g.add_node("age", kind="demographic", name="Age")
        g.add_edge("age", biomarker, rel=rel, weight=weight)
    return g


def per_patient_dag(patient_row: dict, dag: nx.DiGraph) -> nx.DiGraph:
    """Add per-patient demographic nodes (age, gender, bmi) and link them
    into the cohort DAG. Returns a new graph."""
    g = dag.copy()
    pid = patient_row["patient_id"]
    g.add_node(pid, kind="patient")
    # connect patient to its demographics
    g.add_edge(pid, "bmi", rel="HAS")
    # age and gender are exogenous — they influence the biomarkers directly
    g.add_edge(pid, "systolic_bp", rel="AGE_DRIVES")
    g.add_edge(pid, "hrv",         rel="AGE_DRIVES")
    return g


# --- minimal linear SCM: x_i := sum_j w_ji x_j + beta_i + N(0, sigma_i) --
@dataclass
class LinearSCM:
    """A linear-Gaussian SCM fitted by OLS on the cohort dataframe.

    x_i = sum_j w_ji * x_j + beta_i + eps_i,   eps_i ~ N(0, sigma_i^2)
    """
    graph: nx.DiGraph
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    intercepts: dict[str, float] = field(default_factory=dict)
    sigmas: dict[str, float] = field(default_factory=dict)
    fitted: bool = False
    fit_metrics: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, dag: nx.DiGraph, df: pd.DataFrame) -> "LinearSCM":
        scm = cls(graph=dag.copy())
        for node in dag.nodes:
            parents = list(dag.predecessors(node))
            if not parents or node not in df.columns:
                continue
            y = df[node].astype(float).values
            valid_parents = [p for p in parents if p in df.columns]
            if not valid_parents:
                continue
            X = df[valid_parents].astype(float).values
            # add intercept
            X1 = np.column_stack([np.ones(len(X)), X])
            try:
                beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            pred = X1 @ beta
            resid = y - pred
            sigma = float(np.std(resid)) + 1e-6
            scm.intercepts[node] = float(beta[0])
            scm.coefficients[node] = {p: float(b) for p, b in zip(valid_parents, beta[1:])}
            scm.sigmas[node] = sigma
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
            r2 = 1.0 - ss_res / ss_tot
            scm.fit_metrics[node] = {
                "n_parents": len(valid_parents),
                "sigma": round(sigma, 4),
                "r2": round(r2, 3),
            }
        scm.fitted = True
        return scm

    # --- interventions & counterfactuals (linear case) --------------------
    def intervene(self, do: dict[str, float]) -> "LinearSCM":
        """Return a new SCM with the do-operator applied: incoming edges
        to the intervened nodes are severed (Pearl's do-calculus, step 1)."""
        new = LinearSCM(
            graph=self.graph.copy(),
            coefficients={k: dict(v) for k, v in self.coefficients.items()},
            intercepts=dict(self.intercepts),
            sigmas=dict(self.sigmas),
        )
        for node, value in do.items():
            if node not in new.graph:
                new.graph.add_node(node)
            # sever all incoming edges to the intervened node
            for parent in list(new.graph.predecessors(node)):
                new.graph.remove_edge(parent, node)
                if node in new.coefficients and parent in new.coefficients[node]:
                    del new.coefficients[node][parent]
            # set intercept = do-value (so the structural equation yields it)
            new.intercepts[node] = float(value)
            new.sigmas[node] = 0.0  # no noise on a do-ed variable
        return new

    def query(self, target: str, do: dict[str, float] | None = None,
              given: dict[str, float] | None = None) -> dict:
        """Compute E[target | do(.), given(.)] in the linear SCM.

        Algorithm (topological order):
          - sever incoming edges for do-ed vars
          - fix observed vars to given values
          - propagate values through the DAG
        """
        scm = self.intervene(do) if do else self
        observed = dict(given or {})
        # topological order
        try:
            order = list(nx.topological_sort(scm.graph))
        except nx.NetworkXUnfeasible as e:
            return {"error": f"SCM has cycles: {e}"}

        values: dict[str, float] = {}
        for node in order:
            if node in observed:
                values[node] = float(observed[node])
                continue
            if node in scm.intercepts and node in scm.coefficients.get(node, {}):
                pass
            coefs = scm.coefficients.get(node, {})
            if not coefs and node not in scm.intercepts:
                values[node] = 0.0
                continue
            v = scm.intercepts.get(node, 0.0)
            for parent, w in coefs.items():
                v += w * values.get(parent, 0.0)
            values[node] = float(v)
        return {"target": target, "value": round(values.get(target, 0.0), 4),
                "do": do, "given": given, "graph_nodes": len(order)}


# --- DoWhy-style ATE estimation ---------------------------------------
def ate_estimate(df: pd.DataFrame, treatment: str, outcome: str,
                 common_causes: list[str] | None = None) -> dict:
    """Estimate ATE via OLS adjustment (equivalent to DoWhy's
    backdoor.linear_regression with method='iv' off)."""
    from sklearn.linear_model import LinearRegression
    if treatment not in df.columns or outcome not in df.columns:
        return {"error": f"need columns '{treatment}' and '{outcome}' in cohort"}
    common_causes = [c for c in (common_causes or []) if c in df.columns]
    covariates = common_causes
    if not covariates:
        # fall back to a proxy: all other numeric columns
        covariates = [c for c in df.columns
                      if c not in (treatment, outcome, "patient_id", "gender", "created_at")
                      and df[c].dtype.kind in "fi"]
    X = df[covariates + [treatment]].astype(float).values
    y = df[outcome].astype(float).values
    model = LinearRegression().fit(X, y)
    ate = float(model.coef_[-1])
    return {
        "treatment": treatment,
        "outcome": outcome,
        "ate": round(ate, 4),
        "ate_interpretation": _ate_interpretation(treatment, outcome, ate),
        "covariates_used": covariates,
        "n_samples": int(len(df)),
        "r2": round(float(model.score(X, y)), 3),
    }


def cate_estimate(df: pd.DataFrame, treatment: str, outcome: str,
                  effect_modifiers: list[str], common_causes: list[str] | None = None) -> dict:
    """Conditional ATE per subgroup using EconML's CausalForestDML.

    Falls back to a simple per-subgroup regression if EconML is unavailable.
    """
    common_causes = [c for c in (common_causes or []) if c in df.columns]
    effect_modifiers = [c for c in effect_modifiers if c in df.columns]
    if not effect_modifiers:
        return {"error": "no valid effect_modifiers in cohort columns"}
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        Y = df[outcome].astype(float).values
        T = df[treatment].astype(float).values
        X = df[effect_modifiers].astype(float).values
        W = df[common_causes].astype(float).values if common_causes else None
        est = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0),
            model_t=GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0),
            n_estimators=200, random_state=0,
        )
        est.fit(Y, T, X=X, W=W)
        cate = est.effect(X).flatten()
        return {
            "method": "CausalForestDML",
            "treatment": treatment,
            "outcome": outcome,
            "effect_modifiers": effect_modifiers,
            "mean_cate": round(float(cate.mean()), 4),
            "std_cate":  round(float(cate.std()), 4),
            "min_cate":  round(float(cate.min()), 4),
            "max_cate":  round(float(cate.max()), 4),
            "n_samples": int(len(df)),
            "ate": round(float(cate.mean()), 4),
        }
    except Exception as e:
        # fallback: per-bin ATE
        return _cate_fallback(df, treatment, outcome, effect_modifiers, str(e))


def _cate_fallback(df: pd.DataFrame, treatment: str, outcome: str,
                   modifiers: list[str], reason: str) -> dict:
    """Discretise the first effect modifier into 4 quantile bins and
    estimate ATE per bin via OLS adjustment."""
    from sklearn.linear_model import LinearRegression
    m0 = modifiers[0]
    bins = pd.qcut(df[m0], 4, duplicates="drop")
    out = []
    for b, sub in df.groupby(bins, observed=True):
        if sub[treatment].nunique() < 2:
            continue
        covariates = [c for c in ["age", "bmi", "hr", "hrv"] if c in sub.columns]
        X = sub[covariates + [treatment]].astype(float).values
        y = sub[outcome].astype(float).values
        m = LinearRegression().fit(X, y)
        out.append({
            "bin": str(b),
            "ate": round(float(m.coef_[-1]), 4),
            "n":  int(len(sub)),
        })
    return {
        "method": "OLS-per-quantile-bin",
        "fallback_reason": reason,
        "modifier": m0,
        "treatment": treatment,
        "outcome": outcome,
        "mean_cate": round(float(np.mean([r["ate"] for r in out])) if out else 0.0, 4),
        "bins": out,
        "ate": round(float(np.mean([r["ate"] for r in out])) if out else 0.0, 4),
    }


def _ate_interpretation(treatment: str, outcome: str, ate: float) -> str:
    direction = "increases" if ate > 0 else "decreases"
    return (f"Setting '{treatment}' to its high value {direction} '{outcome}' "
            f"by {abs(ate):.3f} units on average across the cohort.")


# --- refutation tests -------------------------------------------------
def refute_ate(df: pd.DataFrame, treatment: str, outcome: str,
               common_causes: list[str] | None = None,
               method: str = "random_common_cause") -> dict:
    """Light refutation suite: random common cause + placebo treatment.

    We do not require DoWhy's full refuters — we just check that adding
    random noise or a placebo treatment drives the ATE to ~0, which is
    the standard sanity check.
    """
    common_causes = [c for c in (common_causes or []) if c in df.columns]
    base = ate_estimate(df, treatment, outcome, common_causes).get("ate")
    if base is None:
        return {"error": "base ATE failed"}

    rng = np.random.default_rng(0)

    # 1) random common cause: add a column of pure noise, ATE should be ~unchanged
    df2 = df.copy()
    df2["_rand"] = rng.normal(0, 1, len(df2))
    ate_with_noise = ate_estimate(df2, treatment, outcome,
                                  common_causes + ["_rand"]).get("ate")

    # 2) placebo treatment: replace treatment with random values, ATE should be ~0
    df3 = df.copy()
    df3[treatment] = rng.permutation(df3[treatment].values)
    ate_placebo = ate_estimate(df3, treatment, outcome, common_causes).get("ate")

    return {
        "method": method,
        "ate_original": round(float(base), 4),
        "ate_with_random_cause": round(float(ate_with_noise), 4),
        "ate_placebo_treatment":  round(float(ate_placebo), 4),
        "passes_random_cause": abs(float(ate_with_noise) - float(base)) < abs(float(base)) * 0.5 + 0.05,
        "passes_placebo": abs(float(ate_placebo)) < abs(float(base)) * 0.5 + 0.05,
    }


# --- 3-step patient counterfactual ----------------------------------
def patient_counterfactual(scm: LinearSCM, observed: dict[str, float],
                            treatment: str, value: float,
                            outcome: str) -> dict:
    """
    Pearl's 3-step counterfactual:
      1. Abduction   — infer exogenous noise U from observed evidence
      2. Action      — set X := x via do-operator
      3. Prediction  — compute Y under the modified SCM, with U preserved
    """
    if not scm.fitted:
        return {"error": "SCM not fitted"}

    # ---- step 1: abduction -------------------------------------------
    # For linear-Gaussian SCM, exogenous noise U_i is simply the residual
    # of the structural equation at observed values.
    try:
        order = list(nx.topological_sort(scm.graph))
    except nx.NetworkXUnfeasible as e:
        return {"error": f"SCM has cycles: {e}"}

    values: dict[str, float] = {}
    residuals: dict[str, float] = {}
    for node in order:
        if node in observed:
            v_obs = float(observed[node])
        else:
            v_obs = scm.intercepts.get(node, 0.0)
        coefs = scm.coefficients.get(node, {})
        v_pred = scm.intercepts.get(node, 0.0)
        for parent, w in coefs.items():
            v_pred += w * values.get(parent, 0.0)
        values[node] = v_obs
        residuals[node] = v_obs - v_pred

    factual = float(values.get(outcome, 0.0))
    outcome_residual = float(residuals.get(outcome, 0.0))

    # ---- step 2: action ---------------------------------------------
    scm_do = scm.intervene({treatment: value})

    # ---- step 3: prediction -----------------------------------------
    # Replay the modified SCM with the do-ed value, then ADD the
    # abducted noise term to the outcome. This is the key step of the
    # counterfactual: preserve the patient's idiosyncrasies (U) while
    # intervening on X.
    try:
        do_order = list(nx.topological_sort(scm_do.graph))
    except nx.NetworkXUnfeasible as e:
        return {"error": f"SCM has cycles: {e}"}

    do_values: dict[str, float] = {}
    # set the intervened variable explicitly
    do_values[treatment] = float(value)
    for node in do_order:
        if node in do_values:
            continue
        if node in observed and node != treatment:
            do_values[node] = float(observed[node])
            continue
        coefs = scm_do.coefficients.get(node, {})
        v = scm_do.intercepts.get(node, 0.0)
        for parent, w in coefs.items():
            v += w * do_values.get(parent, 0.0)
        do_values[node] = float(v)

    structural = float(do_values.get(outcome, 0.0))
    counterfactual = structural + outcome_residual

    return {
        "treatment": treatment,
        "treatment_value": float(value),
        "outcome": outcome,
        "factual": round(factual, 4),
        "counterfactual": round(counterfactual, 4),
        "counterfactual_structural": round(structural, 4),
        "abducted_noise": round(outcome_residual, 4),
        "effect": round(counterfactual - factual, 4),
        "effect_direction": ("increase" if counterfactual > factual
                             else "decrease" if counterfactual < factual
                             else "no change"),
        "abducted_residuals_norm": round(float(np.linalg.norm(list(residuals.values()))), 4),
        "n_nodes_in_dag": len(order),
    }


# Module-level cache
_SCM: LinearSCM | None = None
_DAG: nx.DiGraph | None = None


def get_dag() -> nx.DiGraph:
    global _DAG
    if _DAG is None:
        _DAG = build_causal_dag()
    return _DAG


def fit_cohort_scm(df: pd.DataFrame, force: bool = False) -> LinearSCM:
    global _SCM
    if _SCM is None or force:
        _SCM = LinearSCM.fit(get_dag(), df)
    return _SCM


def reset_scm() -> None:
    global _SCM, _DAG
    _SCM = None
    _DAG = None
