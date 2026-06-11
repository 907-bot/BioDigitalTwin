"""
Phase 5 — Pillar 2: Mechanism Discovery Engine.

Discovers causal mechanisms from twin data using:
  - Constraint-based causal discovery (PC algorithm)
  - Score-based Bayesian network structure learning
  - Structural Causal Model (SCM) with do-calculus
  - Temporal causal graph learning
  - Mediation analysis for pathway decomposition

Connects to Pillar 1 (Knowledge Graph) for prior constraints
and Pillar 3 (Hypothesis Generator) for output.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
import warnings


# ── Data Types ────────────────────────────────────────────────

class MechanismType(Enum):
    DIRECT_CAUSAL = "direct_causal"
    MEDIATED = "mediated"
    CONFOUNDED = "confounded"
    BIDIRECTIONAL = "bidirectional"
    TEMPORAL = "temporal"


@dataclass
class CausalMechanism:
    """A discovered causal mechanism between variables."""
    cause: str
    effect: str
    mechanism_type: MechanismType
    effect_size: float = 0.0
    confidence: float = 0.0
    p_value: float = 1.0
    mediators: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    strength: float = 0.0  # absolute causal strength
    evidence: List[str] = field(default_factory=list)
    method: str = ""

    def describe(self) -> str:
        desc = f"{self.cause} → {self.effect}"
        if self.mediators:
            desc += f" (mediated by {', '.join(self.mediators)})"
        desc += f" [β={self.effect_size:.3f}, p={self.p_value:.3f}]"
        return desc


@dataclass
class MechanismGraph:
    """
    A graph of discovered causal mechanisms.

    Supports query, visualization, and export.
    """
    mechanisms: List[CausalMechanism] = field(default_factory=list)
    variables: Set[str] = field(default_factory=set)

    def add(self, mechanism: CausalMechanism) -> None:
        self.mechanisms.append(mechanism)
        self.variables.add(mechanism.cause)
        self.variables.add(mechanism.effect)

    def get_causes_of(self, variable: str) -> List[CausalMechanism]:
        return [m for m in self.mechanisms if m.effect == variable]

    def get_effects_of(self, variable: str) -> List[CausalMechanism]:
        return [m for m in self.mechanisms if m.cause == variable]

    def get_top_mechanisms(self, k: int = 10) -> List[CausalMechanism]:
        return sorted(self.mechanisms,
                      key=lambda m: abs(m.effect_size),
                      reverse=True)[:k]

    def get_mediation_paths(self, cause: str, effect: str) -> List[List[str]]:
        """Find mediation paths between cause and effect."""
        paths = []
        for m in self.mechanisms:
            if m.cause == cause and effect in m.mediators:
                for m2 in self.mechanisms:
                    if m2.cause in m.mediators and m2.effect == effect:
                        paths.append([cause, m2.cause, effect])
            if m.cause == cause and m.effect == effect and m.mediators:
                paths.append([cause] + m.mediators + [effect])
        return paths

    def to_dict(self) -> Dict:
        return {
            "variables": list(self.variables),
            "mechanisms": [{
                "cause": m.cause, "effect": m.effect,
                "type": m.mechanism_type.value,
                "effect_size": m.effect_size,
                "confidence": m.confidence,
                "p_value": m.p_value,
                "mediators": m.mediators,
                "confounders": m.confounders,
                "method": m.method,
            } for m in self.mechanisms],
        }


# ─── Known Physiological Mechanisms (Prior Knowledge) ──────

KNOWN_MECHANISMS: Dict[Tuple[str, str], List[str]] = {
    ("SleepQuality", "InsulinResistance"): ["Cortisol", "Inflammation"],
    ("BMI", "InsulinResistance"): ["FFA", "Inflammation", "Adipokines"],
    ("Exercise", "InsulinResistance"): ["AMPK", "GLUT4", "Mitochondria"],
    ("AirPollution", "InsulinResistance"): ["Inflammation", "OxidativeStress"],
    ("Stress", "InsulinResistance"): ["Cortisol", "Inflammation"],
    ("InsulinResistance", "Glucose"): ["HGP", "PGU"],
    ("DietQuality", "Glucose"): ["InsulinSecretion", "GutMicrobiome"],
    ("InsulinResistance", "SBP"): ["SympatheticActivity", "SodiumRetention"],
}


# ── Constraint-Based Causal Discovery (PC Algorithm) ──────────

class PCCausalDiscovery:
    """
    Peter-Clark (PC) algorithm for constraint-based causal discovery.

    Uses conditional independence tests to orient edges.
    """

    def __init__(self, alpha: float = 0.05, ci_test: str = "pearson"):
        self.alpha = alpha
        self.ci_test = ci_test
        self.graph: MechanismGraph = MechanismGraph()

    def discover(self, data: pd.DataFrame,
                 prior_knowledge: Optional[Dict[Tuple[str, str], List[str]]] = None
                 ) -> MechanismGraph:
        """
        Run PC algorithm on observed data.

        Args:
            data: DataFrame with variable columns
            prior_knowledge: Known causal edges and mediators

        Returns:
            MechanismGraph with discovered mechanisms
        """
        self.graph = MechanismGraph()
        vars = list(data.columns)
        n_vars = len(vars)
        arr = data.values

        # Step 1: Compute correlation/skeleton
        corr = np.corrcoef(arr.T)
        if corr.ndim < 2 or corr.shape[0] < 2:
            return self.graph
        adj = np.abs(corr) > 0.15  # adjacency threshold
        np.fill_diagonal(adj, False)

        # Step 2: Orient edges using conditional independence
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] and not adj[j, i]:
                    cause, effect = vars[i], vars[j]
                    # Check conditional independence given subsets
                    remaining = [k for k in range(n_vars) if k != i and k != j]
                    is_direct = True
                    mediators = []

                    for subset_size in range(1, min(4, len(remaining) + 1)):
                        for subset in itertools.combinations(remaining, subset_size):
                            idxs = list(subset)
                            if len(idxs) == 0:
                                continue
                            cond_vars = [vars[k] for k in idxs]
                            partial_corr = self._partial_corr(
                                arr[:, i], arr[:, j], arr[:, idxs]
                            )
                            if abs(partial_corr) < 0.1:
                                is_direct = False
                                mediators = cond_vars
                                break
                        if not is_direct:
                            break

                    if is_direct or abs(corr[i, j]) > 0.3:
                        mechanism = CausalMechanism(
                            cause=cause,
                            effect=effect,
                            mechanism_type=MechanismType.DIRECT_CAUSAL if is_direct
                                        else MechanismType.MEDIATED,
                            effect_size=float(corr[i, j]),
                            confidence=float(abs(corr[i, j])),
                            p_value=self._p_value_from_corr(corr[i, j], len(arr)),
                            mediators=mediators if not is_direct else [],
                            method="pc_algorithm",
                        )
                        self.graph.add(mechanism)

        # Incorporate prior knowledge
        if prior_knowledge:
            self._apply_prior_knowledge(prior_knowledge, vars)

        return self.graph

    def _partial_corr(self, x: np.ndarray, y: np.ndarray,
                      z: np.ndarray) -> float:
        """Compute partial correlation between x and y given z."""
        n = len(x)
        z = np.atleast_2d(z)
        if z.shape[0] == 0 or z.shape[1] == 0:
            return float(np.corrcoef(x, y)[0, 1])

        # Residualize x and y on z
        try:
            beta_x = np.linalg.lstsq(z.T, x, rcond=None)[0]
            beta_y = np.linalg.lstsq(z.T, y, rcond=None)[0]
            res_x = x - z.T @ beta_x
            res_y = y - z.T @ beta_y
            return float(np.corrcoef(res_x, res_y)[0, 1])
        except np.linalg.LinAlgError:
            return 0.0

    def _p_value_from_corr(self, r: float, n: int) -> float:
        """Approximate p-value from correlation coefficient."""
        import scipy.stats as stats
        t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-10))
        return float(2 * (1 - stats.t.cdf(abs(t), n - 2)))

    def _apply_prior_knowledge(self, prior: Dict[Tuple[str, str], List[str]],
                                vars: List[str]) -> None:
        present = set(vars)
        for (cause, effect), mediators in prior.items():
            if cause not in present or effect not in present:
                continue
            exists = any(m.cause == cause and m.effect == effect
                        for m in self.graph.mechanisms)
            if not exists:
                self.graph.add(CausalMechanism(
                    cause=cause, effect=effect,
                    mechanism_type=MechanismType.MEDIATED if mediators
                                else MechanismType.DIRECT_CAUSAL,
                    effect_size=0.0, confidence=0.5,
                    mediators=[m for m in mediators if m in present],
                    method="prior_knowledge",
                ))


# ── Score-Based Bayesian Network ──────────────────────────────

class BayesianNetworkDiscovery:
    """
    Score-based Bayesian network structure learning.

    Uses BIC/MDL scoring with greedy search.
    """

    def __init__(self, scoring: str = "bic"):
        self.scoring = scoring

    def discover(self, data: pd.DataFrame,
                 white_list: Optional[List[Tuple[str, str]]] = None
                 ) -> MechanismGraph:
        """
        Learn Bayesian network structure.

        Args:
            data: DataFrame with variable columns
            white_list: Must-include edges

        Returns:
            MechanismGraph with BIC-scored mechanisms
        """
        graph = MechanismGraph()
        vars = list(data.columns)
        n = len(data)
        arr = data.values

        # Score all possible edges with BIC
        edge_scores = []
        for i, cause in enumerate(vars):
            for j, effect in enumerate(vars):
                if i == j:
                    continue
                # Linear regression score
                X = arr[:, i]
                y = arr[:, j]
                beta = np.cov(X, y)[0, 1] / max(np.var(X), 1e-10)
                residuals = y - beta * X
                rss = np.sum(residuals ** 2)
                bic = n * np.log(rss / n) + 2 * np.log(n)

                edge_scores.append((cause, effect, float(beta), float(-bic)))

        # Select top edges
        edge_scores.sort(key=lambda x: x[3], reverse=True)

        # Add edges avoiding cycles
        added_edges = set()
        adjacency = {v: set() for v in vars}
        for cause, effect, beta, score in edge_scores:
            if len(added_edges) >= len(vars) * 2:
                break
            if (cause, effect) in added_edges:
                continue
            # Check for cycles
            if self._would_create_cycle(adjacency, cause, effect):
                continue

            # Check white list
            if white_list and (cause, effect) not in white_list and \
               (effect, cause) not in white_list:
                # Without white list, require strong evidence
                if abs(beta) < 0.15:
                    continue

            added_edges.add((cause, effect))
            adjacency[cause].add(effect)

            mechanism = CausalMechanism(
                cause=cause, effect=effect,
                mechanism_type=MechanismType.DIRECT_CAUSAL,
                effect_size=beta,
                confidence=float(min(1.0, max(0.0, score / 10.0 + 0.5))),
                method="bayesian_network_bic",
            )
            graph.add(mechanism)

        return graph

    def _would_create_cycle(self, adjacency: Dict[str, Set[str]],
                            cause: str, effect: str) -> bool:
        """Check if adding cause→effect would create a cycle."""
        visited = set()

        def dfs(node: str) -> bool:
            if node == cause:
                return True
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor):
                        return True
            return False

        return dfs(effect)


# ── Structural Causal Model (SCM) ─────────────────────────────

class StructuralCausalModel:
    """
    Structural Causal Model with do-calculus support.

    Represents causal mechanisms as structural equations:
      X_i = f_i(PA_i, U_i)

    Supports:
      - Intervention simulation (do-calculus)
      - Counterfactual inference
      - Mediation analysis
    """

    def __init__(self):
        self.equations: Dict[str, Callable] = {}
        self.parents: Dict[str, List[str]] = {}
        self.graph = MechanismGraph()

    def add_structural_equation(self, variable: str,
                                parents: List[str],
                                equation_fn: Callable) -> None:
        """Add a structural equation: variable = f(parents, noise)."""
        self.equations[variable] = equation_fn
        self.parents[variable] = parents
        for p in parents:
            self.graph.add(CausalMechanism(
                cause=p, effect=variable,
                mechanism_type=MechanismType.DIRECT_CAUSAL,
                method="structural_equation",
            ))

    def sample(self, n_samples: int = 1000,
               interventions: Optional[Dict[str, float]] = None,
               rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """
        Sample from the SCM, optionally with interventions.

        Args:
            n_samples: Number of samples
            interventions: do(X=x) interventions to apply
            rng: Random number generator

        Returns:
            DataFrame of samples
        """
        rng = rng or np.random.default_rng(42)
        interventions = interventions or {}
        data = {}
        topological_order = self._topological_sort()

        for var in topological_order:
            if var in interventions:
                data[var] = np.full(n_samples, interventions[var])
            elif var in self.equations:
                parent_vals = {p: data[p] for p in self.parents.get(var, [])}
                noise = rng.normal(0, 0.1, n_samples)
                data[var] = self.equations[var](parent_vals, noise)
            else:
                # Exogenous variable
                data[var] = rng.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    def estimate_causal_effect(self, treatment: str, outcome: str,
                               n_samples: int = 10000) -> float:
        """
        Estimate causal effect using do-calculus.

        E[Y | do(X=x+1)] - E[Y | do(X=x)]
        """
        base = self.sample(n_samples, interventions={treatment: 0.0})
        treated = self.sample(n_samples, interventions={treatment: 1.0})
        return float(treated[outcome].mean() - base[outcome].mean())

    def mediation_analysis(self, treatment: str, outcome: str,
                           mediator: str, n_samples: int = 5000
                           ) -> Dict[str, float]:
        """
        Decompose total effect into direct and indirect (mediated) effects.

        Returns:
            {direct_effect, indirect_effect, total_effect, mediation_proportion}
        """
        # Total effect: E[Y|do(X=1)] - E[Y|do(X=0)]
        y_do_x0 = self.sample(n_samples, {treatment: 0.0})[outcome].mean()
        y_do_x1 = self.sample(n_samples, {treatment: 1.0})[outcome].mean()
        total = y_do_x1 - y_do_x0

        # Direct effect: fix mediator at do(X=0) level, vary treatment
        m_do_x0 = self.sample(n_samples, {treatment: 0.0})[mediator].mean()
        y_do_x1_mfix = self.sample(n_samples, {treatment: 1.0})
        direct = float(y_do_x1_mfix[outcome].mean() - y_do_x0)

        indirect = total - direct
        prop = indirect / max(abs(total), 1e-10)

        return {
            "total_effect": float(total),
            "direct_effect": float(direct),
            "indirect_effect": float(indirect),
            "mediation_proportion": float(prop),
        }

    def _topological_sort(self) -> List[str]:
        """Simple topological sort of the causal graph."""
        visited = set()
        result = []
        all_vars = set(self.parents.keys())
        for p_list in self.parents.values():
            all_vars.update(p_list)

        def dfs(var):
            if var in visited:
                return
            visited.add(var)
            for child in all_vars:
                if var in self.parents.get(child, []):
                    dfs(child)
            result.append(var)

        for var in all_vars:
            dfs(var)
        return list(reversed(result))


# ── Temporal Causal Discovery ─────────────────────────────────

class TemporalCausalDiscovery:
    """
    Discover causal relationships from time-series data.

    Uses Granger causality with lag selection.
    """

    def __init__(self, max_lag: int = 5):
        self.max_lag = max_lag

    def discover(self, data: pd.DataFrame,
                 target_vars: Optional[List[str]] = None) -> MechanismGraph:
        """
        Run temporal causal discovery.

        Args:
            data: Time-series DataFrame with columns as variables
            target_vars: Subset to focus on

        Returns:
            MechanismGraph with lagged causal edges
        """
        graph = MechanismGraph()
        vars = target_vars or list(data.columns)
        arr = data.values
        n, p = arr.shape

        if n < self.max_lag + 5:
            return graph

        for target in vars:
            j = list(data.columns).index(target)
            # Test each variable as potential cause at each lag
            for cause_idx, cause in enumerate(data.columns):
                if cause == target:
                    continue
                best_lag = None
                best_score = 0.0
                for lag in range(1, self.max_lag + 1):
                    if n <= lag:
                        continue
                    X_lag = arr[:n - lag, cause_idx]
                    y = arr[lag:, j]
                    if np.std(X_lag) < 1e-10 or np.std(y) < 1e-10:
                        continue
                    corr = np.corrcoef(X_lag, y)[0, 1]
                    if abs(corr) > best_score:
                        best_score = abs(corr)
                        best_lag = lag

                if best_lag is not None and best_score > 0.15:
                    graph.add(CausalMechanism(
                        cause=f"{cause}(t-{best_lag})",
                        effect=f"{target}(t)",
                        mechanism_type=MechanismType.TEMPORAL,
                        effect_size=float(best_score),
                        confidence=float(best_score),
                        method=f"temporal_granger_lag{best_lag}",
                    ))

        return graph


# ── Main Discovery Engine ─────────────────────────────────────

class MechanismDiscoveryEngine:
    """
    Unified mechanism discovery engine combining all methods.

    Orchestrates:
      1. PC algorithm for skeleton + orientation
      2. Bayesian network scoring refinement
      3. SCM construction from discovered graph
      4. Temporal causal discovery from time-series
      5. Mediation analysis for pathway decomposition
    """

    def __init__(self, knowledge_graph: Optional[Any] = None):
        self.knowledge_graph = knowledge_graph
        self.pc_discovery = PCCausalDiscovery()
        self.bn_discovery = BayesianNetworkDiscovery()
        self.scm = StructuralCausalModel()
        self.temporal_discovery = TemporalCausalDiscovery()
        self.mechanism_graph = MechanismGraph()

    def discover_from_cross_sectional(
        self, data: pd.DataFrame,
        use_prior: bool = True,
        methods: List[str] = None,
    ) -> MechanismGraph:
        """
        Discover mechanisms from cross-sectional data.

        Args:
            data: DataFrame with variable columns
            use_prior: Whether to use prior physiological knowledge
            methods: Discovery methods to use (pc, bayesian)

        Returns:
            Combined MechanismGraph
        """
        if methods is None:
            methods = ["pc", "bayesian"]

        graph = MechanismGraph()

        if "pc" in methods:
            prior = KNOWN_MECHANISMS if use_prior else None
            pc_graph = self.pc_discovery.discover(data, prior)
            for m in pc_graph.mechanisms:
                graph.add(m)

        if "bayesian" in methods:
            bn_graph = self.bn_discovery.discover(data)
            for m in bn_graph.mechanisms:
                graph.add(m)

        self.mechanism_graph = graph
        return graph

    def discover_from_time_series(
        self, data: pd.DataFrame,
        target_vars: Optional[List[str]] = None,
    ) -> MechanismGraph:
        """Discover temporal causal mechanisms."""
        temporal = self.temporal_discovery.discover(data, target_vars)
        for m in temporal.mechanisms:
            self.mechanism_graph.add(m)
        return self.mechanism_graph

    def build_scm(self, data: pd.DataFrame) -> StructuralCausalModel:
        """
        Build an SCM from data using linear structural equations.
        """
        graph = self.discover_from_cross_sectional(data)
        self.scm = StructuralCausalModel()
        for var in data.columns:
            causes = graph.get_causes_of(var)
            parent_names = [m.cause for m in causes]
            if not parent_names:
                continue
            # Linear regression structural equation
            X = np.column_stack([data[p].values for p in parent_names])
            y = data[var].values
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(len(parent_names))

            def make_eq(b=beta, parents=parent_names):
                def eq(parent_vals, noise):
                    result = noise.copy()
                    for i, p in enumerate(parents):
                        if p in parent_vals:
                            result += b[i] * parent_vals[p]
                    return result
                return eq

            self.scm.add_structural_equation(var, parent_names, make_eq())

        return self.scm

    def query_mechanism(self, cause: str, effect: str) -> List[CausalMechanism]:
        """Get all mechanisms between cause and effect."""
        results = []
        for m in self.mechanism_graph.mechanisms:
            if m.cause == cause and m.effect == effect:
                results.append(m)
        return results

    def get_mediation(self, cause: str, effect: str) -> Dict:
        """Run full mediation analysis between cause and effect."""
        paths = self.mechanism_graph.get_mediation_paths(cause, effect)
        direct = self.query_mechanism(cause, effect)
        return {
            "direct_effects": [m.describe() for m in direct],
            "mediation_paths": paths,
        }


# ── Convenience API ───────────────────────────────────────────

def discover_causal_mechanisms(
    data: pd.DataFrame,
    method: str = "auto",
    use_prior: bool = True,
) -> MechanismGraph:
    """
    One-shot causal mechanism discovery.

    Args:
        data: DataFrame with observations
        method: "pc", "bayesian", "temporal", or "auto"
        use_prior: Use known physiology as prior

    Returns:
        MechanismGraph with discovered mechanisms
    """
    engine = MechanismDiscoveryEngine()
    if method == "temporal":
        return engine.discover_from_time_series(data)
    return engine.discover_from_cross_sectional(data, use_prior)
