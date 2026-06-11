"""
Parameter Heritage Database — The Data Moat.

Every patient personalized by the twin contributes their posterior
parameter distribution to a shared population prior. This prior:
  - Improves initialization speed for new patients (fewer updates needed)
  - Enables patient subgroup discovery (clustering on learned params)
  - Creates a data flywheel: more patients → better prior → faster adaptation → more patients

This is the defensible asset. A competitor can replicate the ODEs,
but they cannot replicate 10,000 patient-specific parameter sets.

Architecture:
  - Each patient stores (mean_param, cov_param, demographics)
  - Population prior = empirical mean and covariance of all patient means
  - Subgroup priors = cluster-specific distributions
  - Federated: only aggregate statistics leave the hospital, not raw data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import cluster as sp_cluster
import json
import os
from datetime import datetime

from app.personalization.state import PARAM_DIM
from app.personalization.dynamics import DEFAULT_PARAMS
from app.personalization.priors import PARAMETER_NAMES


@dataclass
class PatientRecord:
    patient_id: str
    param_mean: np.ndarray
    param_cov: np.ndarray
    n_observations: int
    demographics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class PopulationPrior:
    mean: np.ndarray
    cov: np.ndarray
    n_patients: int
    subgroups: List[Dict] = field(default_factory=list)


class HeritageDatabase:
    """
    Population-level parameter database that improves with each patient.

    Each personalized patient contributes their posterior parameter
    distribution to the population prior. The prior is then used to
    initialize the UKF for new patients, reducing convergence time.

    Usage:
        db = HeritageDatabase()

        # After personalizing a patient:
        db.record_patient("P001", param_mean, param_cov, n_obs=500)

        # Get the population prior for a new patient:
        prior = db.get_population_prior()
        new_patient_initial_params = prior.mean  # better than DEFAULT_PARAMS

        # Find similar patients:
        similar = db.find_similar_patients(demographics={"age": 55, "bmi": 30})
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._records: Dict[str, PatientRecord] = {}
        self._prior = PopulationPrior(
            mean=DEFAULT_PARAMS[:PARAM_DIM].copy(),
            cov=np.eye(PARAM_DIM) * 0.1,
            n_patients=0,
        )
        self._storage_path = storage_path
        if storage_path and os.path.exists(storage_path):
            self._load()

    def record_patient(
        self,
        patient_id: str,
        param_mean: np.ndarray,
        param_cov: np.ndarray,
        n_observations: int,
        demographics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a patient's personalized parameter distribution."""
        record = PatientRecord(
            patient_id=patient_id,
            param_mean=param_mean.copy(),
            param_cov=param_cov.copy(),
            n_observations=n_observations,
            demographics=demographics or {},
            timestamp=datetime.utcnow().isoformat(),
        )
        self._records[patient_id] = record
        self._update_population_prior()

    def _update_population_prior(self) -> None:
        """Recompute population prior from all patient records."""
        if len(self._records) < 1:
            return

        all_means = np.array([r.param_mean[:PARAM_DIM] for r in self._records.values()])
        n = len(all_means)

        if n >= 2:
            pop_mean = np.mean(all_means, axis=0)
            pop_cov = np.cov(all_means, rowvar=False)
            # Regularize: shrink toward default
            default = DEFAULT_PARAMS[:PARAM_DIM]
            shrinkage = 10.0 / (n + 10.0)
            pop_mean = (1 - shrinkage) * pop_mean + shrinkage * default
            pop_cov = (1 - shrinkage) * pop_cov + shrinkage * np.eye(PARAM_DIM) * 0.05
        else:
            pop_mean = all_means[0]
            pop_cov = np.eye(PARAM_DIM) * 0.05

        self._prior = PopulationPrior(
            mean=pop_mean, cov=pop_cov, n_patients=n
        )

    def get_population_prior(self, n_min: int = 1) -> PopulationPrior:
        """Get the current population prior. Returns default if < n_min patients."""
        if self._prior.n_patients < n_min:
            return PopulationPrior(
                mean=DEFAULT_PARAMS[:PARAM_DIM].copy(),
                cov=np.eye(PARAM_DIM) * 0.1,
                n_patients=0,
            )
        return self._prior

    def get_patient_count(self) -> int:
        return len(self._records)

    def get_subgroup_priors(self, n_clusters: int = 3) -> List[PopulationPrior]:
        """Cluster patients into subgroups and return per-cluster priors."""
        if len(self._records) < n_clusters * 2:
            return [self._prior]

        all_means = np.array([r.param_mean[:PARAM_DIM] for r in self._records.values()])
        if len(all_means) < n_clusters * 2:
            return [self._prior]

        try:
            centroids, labels = sp_cluster.vq.kmeans2(all_means, n_clusters, minit='points')
            subgroup_priors = []
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() >= 2:
                    sg_mean = np.mean(all_means[mask], axis=0)
                    sg_cov = np.cov(all_means[mask], rowvar=False)
                else:
                    sg_mean = all_means[mask[0]] if mask.any() else self._prior.mean
                    sg_cov = np.eye(PARAM_DIM) * 0.05
                subgroup_priors.append(PopulationPrior(
                    mean=sg_mean, cov=sg_cov, n_patients=int(mask.sum())
                ))
            return subgroup_priors
        except Exception:
            return [self._prior]

    def find_similar_patients(
        self,
        demographics: Dict[str, Any],
        n_neighbors: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find patients with similar demographics."""
        if not self._records or not demographics:
            return []

        scored = []
        for pid, rec in self._records.items():
            score = 0.0
            n_matched = 0
            for key, val in demographics.items():
                if key in rec.demographics:
                    if isinstance(val, (int, float)) and isinstance(rec.demographics[key], (int, float)):
                        diff = abs(val - rec.demographics[key])
                        score -= diff / max(abs(val), 1.0)
                        n_matched += 1
                    elif val == rec.demographics[key]:
                        score += 1.0
                        n_matched += 1
            if n_matched > 0:
                scored.append((pid, score / n_matched))

        scored.sort(key=lambda x: -x[1])
        return scored[:n_neighbors]

    def get_identifiability_report(self) -> Dict[str, float]:
        """Report which parameters are well-identified by the population."""
        if self._prior.n_patients < 2:
            return {}

        default = DEFAULT_PARAMS[:PARAM_DIM]
        pop_var = np.diag(self._prior.cov)
        default_var = (default * 0.1) ** 2

        report = {}
        for i, name in enumerate(PARAMETER_NAMES):
            if i < len(pop_var):
                # Contraction = 1 - posterior_var / prior_var
                contraction = 1.0 - pop_var[i] / max(default_var, 1e-8)
                report[name] = float(np.clip(contraction, 0, 1))
        return report

    def to_dict(self) -> Dict:
        """Serialize to dict for saving/transmission."""
        return {
            "n_patients": len(self._records),
            "pop_mean": self._prior.mean.tolist(),
            "pop_cov_diag": np.diag(self._prior.cov).tolist(),
            "patient_ids": list(self._records.keys()),
            "identifiability": self.get_identifiability_report(),
        }

    def _load(self) -> None:
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            for pid, rec in data.get("records", {}).items():
                self._records[pid] = PatientRecord(
                    patient_id=pid,
                    param_mean=np.array(rec["mean"]),
                    param_cov=np.array(rec["cov"]),
                    n_observations=rec.get("n_obs", 0),
                    demographics=rec.get("demographics", {}),
                    timestamp=rec.get("ts", ""),
                )
            self._update_population_prior()
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        if self._storage_path is None:
            return
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        data = {
            "records": {
                pid: {
                    "mean": rec.param_mean.tolist(),
                    "cov": rec.param_cov.tolist(),
                    "n_obs": rec.n_observations,
                    "demographics": rec.demographics,
                    "ts": rec.timestamp,
                }
                for pid, rec in self._records.items()
            }
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)
