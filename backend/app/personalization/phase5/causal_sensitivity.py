"""
Causal Graph Sensitivity Analysis.

Quantifies how causal conclusions change under perturbations
to the causal graph structure, data resampling, and parameter variation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Callable
from itertools import combinations


@dataclass
class EdgePerturbationResult:
    edge: Tuple[str, str]
    perturbation_type: str
    original_strength: float
    perturbed_strength: float
    strength_change: float
    direction_flipped: bool
    p_value_original: float
    p_value_perturbed: float


@dataclass
class BootstrapResult:
    edge: Tuple[str, str]
    mean_strength: float
    std_strength: float
    stability_index: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    inclusion_frequency: float


@dataclass
class SensitivityReport:
    n_edges_tested: int
    edge_perturbations: List[EdgePerturbationResult]
    bootstrap_results: List[BootstrapResult]
    stable_edges: List[Tuple[str, str]]
    unstable_edges: List[Tuple[str, str]]
    mean_stability: float
    graph_edit_distance: float


class CausalGraphSensitivity:
    """
    Sensitivity analysis for causal discovery results.

    Tests:
      1. Edge perturbation: add/remove/reverse each edge
      2. Bootstrap stability: resample data and re-run discovery
      3. Confounder sensitivity: add unobserved confounder with varying strength
    """

    def __init__(self, n_bootstrap: int = 100, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

    def edge_perturbation_analysis(self, data: pd.DataFrame,
                                   discover_fn: Callable,
                                   edges_to_test: Optional[List[Tuple[str, str]]] = None,
                                   ) -> List[EdgePerturbationResult]:
        base_graph = discover_fn(data)
        base_edges = {}
        for m in (base_graph.mechanisms if hasattr(base_graph, 'mechanisms') else []):
            ms = getattr(m, 'source', getattr(m, 'cause', ''))
            mt = getattr(m, 'target', getattr(m, 'effect', ''))
            base_edges[(ms, mt)] = m
        if edges_to_test is None:
            edges_to_test = list(base_edges.keys())[:20]
        results = []
        all_vars = list(data.columns)
        for edge in edges_to_test:
            if edge not in base_edges:
                continue
            base_m = base_edges[edge]
            src = getattr(base_m, 'source', getattr(base_m, 'cause', edge[0]))
            tgt = getattr(base_m, 'target', getattr(base_m, 'effect', edge[1]))
            if src not in all_vars or tgt not in all_vars:
                continue
            try:
                remove_data = data.drop(columns=[src])
            except KeyError:
                remove_data = data
            remove_graph = discover_fn(remove_data)
            remove_strength = 0.0
            for m in (remove_graph.mechanisms if hasattr(remove_graph, 'mechanisms') else []):
                ms = getattr(m, 'source', getattr(m, 'cause', ''))
                mt = getattr(m, 'target', getattr(m, 'effect', ''))
                if ms == tgt or mt == tgt:
                    remove_strength = max(remove_strength, getattr(m, 'strength', 0))
            results.append(EdgePerturbationResult(
                edge=edge, perturbation_type="remove_cause",
                original_strength=getattr(base_m, 'strength', getattr(base_m, 'weight', 0)),
                perturbed_strength=remove_strength,
                strength_change=getattr(base_m, 'strength', getattr(base_m, 'weight', 0)) - remove_strength,
                direction_flipped=False,
                p_value_original=getattr(base_m, 'p_value', getattr(base_m, 'pvalue', 1.0)),
                p_value_perturbed=0.0,
            ))
            try:
                reverse_data = data.rename(
                    columns={src: f"__{src}", tgt: src, f"__{src}": tgt}
                )
            except KeyError:
                reverse_data = data
            reverse_graph = discover_fn(reverse_data)
            rev_strength = 0.0
            for m in (reverse_graph.mechanisms if hasattr(reverse_graph, 'mechanisms') else []):
                ms = getattr(m, 'source', getattr(m, 'cause', ''))
                mt = getattr(m, 'target', getattr(m, 'effect', ''))
                if ms == tgt and mt == src:
                    rev_strength = getattr(m, 'strength', 0)
            results.append(EdgePerturbationResult(
                edge=edge, perturbation_type="reverse_edge",
                original_strength=getattr(base_m, 'strength', getattr(base_m, 'weight', 0)),
                perturbed_strength=rev_strength,
                strength_change=getattr(base_m, 'strength', getattr(base_m, 'weight', 0)) - rev_strength,
                direction_flipped=rev_strength > getattr(base_m, 'strength', getattr(base_m, 'weight', 0)) * 0.5,
                p_value_original=getattr(base_m, 'p_value', getattr(base_m, 'pvalue', 1.0)),
                p_value_perturbed=0.0,
            ))
        return results

    def bootstrap_stability_analysis(self, data: pd.DataFrame,
                                      discover_fn: Callable,
                                      ) -> List[BootstrapResult]:
        n = len(data)
        edge_strengths: Dict[Tuple[str, str], List[float]] = {}
        edge_inclusions: Dict[Tuple[str, str], int] = {}
        for _ in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, n)
            sample = data.iloc[idx]
            try:
                graph = discover_fn(sample)
                for m in (graph.mechanisms if hasattr(graph, 'mechanisms') else []):
                    ms = getattr(m, 'source', getattr(m, 'cause', ''))
                    mt = getattr(m, 'target', getattr(m, 'effect', ''))
                    key = (ms, mt)
                    if key not in edge_strengths:
                        edge_strengths[key] = []
                        edge_inclusions[key] = 0
                    edge_strengths[key].append(getattr(m, 'strength', getattr(m, 'weight', 0)))
                    edge_inclusions[key] += 1
            except Exception:
                continue
        results = []
        for edge, strengths in edge_strengths.items():
            strengths = np.array(strengths)
            inclusion_freq = edge_inclusions[edge] / self.n_bootstrap
            if len(strengths) < 5:
                continue
            mean_s = float(np.mean(strengths))
            std_s = float(np.std(strengths))
            ci = float(1.96 * std_s / max(np.sqrt(len(strengths)), 1))
            stability = float(1.0 / (1.0 + std_s / max(abs(mean_s), 0.01)))
            results.append(BootstrapResult(
                edge=edge, mean_strength=mean_s, std_strength=std_s,
                stability_index=stability,
                bootstrap_ci_lower=mean_s - ci,
                bootstrap_ci_upper=mean_s + ci,
                inclusion_frequency=inclusion_freq,
            ))
        return results

    def confounder_sensitivity(self, data: pd.DataFrame,
                                treatment: str, outcome: str,
                                effect_estimator: Callable,
                                confounder_strengths: Optional[List[float]] = None,
                                ) -> Dict:
        if confounder_strengths is None:
            confounder_strengths = np.linspace(0, 1, 11).tolist()
        effects = []
        for strength in confounder_strengths:
            data_copy = data.copy()
            data_copy[f"_unobserved_confounder_{treatment}_{outcome}"] = (
                strength * data[treatment] + (1 - strength) * self.rng.normal(0, 1, len(data))
            )
            try:
                effect = effect_estimator(data_copy, treatment, outcome)
                effects.append({"confounder_strength": strength, "estimated_effect": effect})
            except Exception:
                effects.append({"confounder_strength": strength, "estimated_effect": None})
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounder_sensitivity": effects,
            "max_effect_change": max(
                (abs(e["estimated_effect"]) for e in effects if e["estimated_effect"] is not None),
                default=0.0,
            ),
        }

    def full_sensitivity_report(self, data: pd.DataFrame,
                                 discover_fn: Callable,
                                 effect_estimator: Optional[Callable] = None,
                                 ) -> SensitivityReport:
        edge_pert = self.edge_perturbation_analysis(data, discover_fn)
        boot_stab = self.bootstrap_stability_analysis(data, discover_fn)
        stable = [r.edge for r in boot_stab if r.stability_index > 0.7 and r.inclusion_frequency > 0.5]
        unstable = [r.edge for r in boot_stab if r.stability_index < 0.3 or r.inclusion_frequency < 0.3]
        mean_stab = float(np.mean([r.stability_index for r in boot_stab])) if boot_stab else 0.0
        graph_edit = float(np.mean([abs(p.strength_change) for p in edge_pert])) if edge_pert else 0.0
        return SensitivityReport(
            n_edges_tested=len(edge_pert),
            edge_perturbations=edge_pert,
            bootstrap_results=boot_stab,
            stable_edges=stable,
            unstable_edges=unstable,
            mean_stability=mean_stab,
            graph_edit_distance=graph_edit,
        )


def run_causal_sensitivity(data: pd.DataFrame,
                            discover_fn: Callable,
                            n_bootstrap: int = 50,
                            effect_estimator: Optional[Callable] = None) -> SensitivityReport:
    analyzer = CausalGraphSensitivity(n_bootstrap=n_bootstrap)
    return analyzer.full_sensitivity_report(data, discover_fn, effect_estimator)
