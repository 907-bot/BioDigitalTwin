"""
Samsung Health Training Module — Phase 18
Trains 4 ML models from Samsung Health data:
  1. CardiovascularRiskModel  (RandomForest binary classifier, calibrated)
  2. SleepQualityModel        (GradientBoosting regressor)
  3. StressPatternModel       (GradientBoosting multi-class classifier)
  4. AnomalyDetector          (IsolationForest unsupervised)

Optionally trains an LSTM sequence model for next-day risk prediction.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (heavy deps)
# ---------------------------------------------------------------------------
def _sklearn():
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        GradientBoostingRegressor, IsolationForest,
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, f1_score, mean_absolute_error, r2_score,
        classification_report,
    )
    return (
        RandomForestClassifier, GradientBoostingClassifier,
        GradientBoostingRegressor, IsolationForest,
        CalibratedClassifierCV,
        StratifiedKFold, cross_val_score, cross_val_predict,
        StandardScaler,
        roc_auc_score, f1_score, mean_absolute_error, r2_score,
        classification_report,
    )


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    model_name: str
    model_type: str
    n_samples: int
    n_features: int
    trained_at: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SamsungHealthTrainer:
    """
    Full training pipeline for Samsung Health ML models.
    Call: trainer.run(data_dir, model_dir)
    """

    MODEL_FILES = {
        "cardio_risk": "samsung_cardio_risk.pkl",
        "sleep_quality": "samsung_sleep_quality.pkl",
        "stress_classifier": "samsung_stress_classifier.pkl",
        "anomaly_detector": "samsung_anomaly_detector.pkl",
        "scaler": "samsung_scaler.pkl",
        "feature_names": "samsung_feature_names.json",
        "metrics": "samsung_training_metrics.json",
    }

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.metrics: List[ModelMetrics] = []
        self._status: Dict[str, Any] = {"state": "idle", "progress": 0, "log": []}

    def _log(self, msg: str):
        logger.info(msg)
        self._status["log"].append(msg)

    def status(self) -> Dict[str, Any]:
        return dict(self._status)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self, data_dir: str, n_synthetic: int = 600, age_est: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse Samsung CSVs and generate augmented training set.
        Returns X, y_cardio, y_sleep, y_stress arrays.
        """
        from .csv_parser import SamsungCSVParser
        from .feature_engineering import generate_augmented_cohort, FEATURE_NAMES

        self._log(f"Parsing Samsung Health CSVs from {data_dir} ...")
        self._status["progress"] = 5
        self._status["state"] = "parsing"

        parser = SamsungCSVParser(data_dir)
        record = parser.parse_all()

        self._log(
            f"Parsed: {len(record.heart_rate)} HR | {len(record.stress)} stress | "
            f"{len(record.sleep_stages)} sleep stages | {len(record.pedometer)} ped days | "
            f"{len(record.exercise)} exercise sessions"
        )
        self._status["progress"] = 20
        self._status["state"] = "feature_engineering"

        X, y_cardio, y_sleep, y_stress = generate_augmented_cohort(
            record, n_synthetic=n_synthetic
        )
        self._log(f"Feature matrix: {X.shape} | labels: cardio={y_cardio.sum()}/{len(y_cardio)}")
        self._status["progress"] = 35
        return X, y_cardio, y_sleep, y_stress

    # ------------------------------------------------------------------
    # Individual model training
    # ------------------------------------------------------------------
    def _train_cardio_risk(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> ModelMetrics:
        (RFC, GBC, GBR, IF, CalibCV,
         SKF, cv_score, cv_pred,
         Scaler, roc_auc, f1, mae, r2, clf_report) = _sklearn()

        self._log("Training CardiovascularRiskModel (RandomForest + Platt calibration)...")
        base_clf = RFC(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        # Calibrate with sigmoid (Platt scaling)
        clf = CalibCV(base_clf, cv=3, method="sigmoid")
        clf.fit(X_scaled, y)

        # CV evaluation
        skf = SKF(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cv_score(base_clf, X_scaled, y, cv=skf, scoring="roc_auc").mean()
        cv_f1 = cv_score(base_clf, X_scaled, y, cv=skf, scoring="f1").mean()

        # Feature importances from base estimator (after fitting on full data)
        base_clf_full = RFC(n_estimators=300, max_depth=8, min_samples_leaf=5,
                            class_weight="balanced", random_state=42, n_jobs=-1)
        base_clf_full.fit(X_scaled, y)
        importances = dict(zip(feature_names, base_clf_full.feature_importances_))
        top = sorted(importances, key=importances.get, reverse=True)[:10]

        self.models["cardio_risk"] = clf
        metrics = ModelMetrics(
            model_name="CardiovascularRiskModel",
            model_type="RandomForestClassifier + CalibratedClassifierCV (sigmoid)",
            n_samples=len(y), n_features=X_scaled.shape[1],
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics={"cv_roc_auc": round(float(cv_auc), 4),
                     "cv_f1": round(float(cv_f1), 4),
                     "prevalence": round(float(y.mean()), 4)},
            feature_importances={k: round(float(v), 6) for k, v in importances.items()},
            top_features=top,
        )
        self._log(f"  → AUC={cv_auc:.3f} F1={cv_f1:.3f}")
        return metrics

    def _train_sleep_quality(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> ModelMetrics:
        (RFC, GBC, GBR, IF, CalibCV,
         SKF, cv_score, cv_pred,
         Scaler, roc_auc, f1, mae, r2, clf_report) = _sklearn()

        self._log("Training SleepQualityModel (GradientBoosting regressor)...")
        reg = GBR(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )
        from sklearn.model_selection import cross_val_score as cvs
        cv_mae = -cvs(reg, X_scaled, y, cv=5, scoring="neg_mean_absolute_error").mean()
        cv_r2 = cvs(reg, X_scaled, y, cv=5, scoring="r2").mean()
        reg.fit(X_scaled, y)

        importances = dict(zip(feature_names, reg.feature_importances_))
        top = sorted(importances, key=importances.get, reverse=True)[:10]

        self.models["sleep_quality"] = reg
        metrics = ModelMetrics(
            model_name="SleepQualityModel",
            model_type="GradientBoostingRegressor",
            n_samples=len(y), n_features=X_scaled.shape[1],
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics={"cv_mae": round(float(cv_mae), 3),
                     "cv_r2": round(float(cv_r2), 4),
                     "target_mean": round(float(y.mean()), 2),
                     "target_std": round(float(y.std()), 2)},
            feature_importances={k: round(float(v), 6) for k, v in importances.items()},
            top_features=top,
        )
        self._log(f"  → MAE={cv_mae:.2f} R²={cv_r2:.3f}")
        return metrics

    def _train_stress_classifier(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> ModelMetrics:
        (RFC, GBC, GBR, IF, CalibCV,
         SKF, cv_score, cv_pred,
         Scaler, roc_auc, f1, mae, r2, clf_report) = _sklearn()

        self._log("Training StressPatternModel (GradientBoosting multi-class)...")
        clf = GBC(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )
        cv_f1_macro = cv_score(clf, X_scaled, y, cv=5, scoring="f1_macro").mean()
        clf.fit(X_scaled, y)

        importances = dict(zip(feature_names, clf.feature_importances_))
        top = sorted(importances, key=importances.get, reverse=True)[:10]

        self.models["stress_classifier"] = clf
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
        metrics = ModelMetrics(
            model_name="StressPatternModel",
            model_type="GradientBoostingClassifier (3-class: low/med/high)",
            n_samples=len(y), n_features=X_scaled.shape[1],
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics={"cv_f1_macro": round(float(cv_f1_macro), 4),
                     "class_distribution": class_dist},
            feature_importances={k: round(float(v), 6) for k, v in importances.items()},
            top_features=top,
        )
        self._log(f"  → F1_macro={cv_f1_macro:.3f}")
        return metrics

    def _train_anomaly_detector(
        self,
        X_scaled: np.ndarray,
        feature_names: List[str],
    ) -> ModelMetrics:
        (RFC, GBC, GBR, IF, CalibCV,
         SKF, cv_score, cv_pred,
         Scaler, roc_auc, f1, mae, r2, clf_report) = _sklearn()

        self._log("Training AnomalyDetector (IsolationForest)...")
        iso = IF(
            n_estimators=200, contamination=0.05,
            random_state=42, n_jobs=-1
        )
        iso.fit(X_scaled)
        scores = iso.decision_function(X_scaled)
        anomaly_rate = (iso.predict(X_scaled) == -1).mean()

        self.models["anomaly_detector"] = iso
        metrics = ModelMetrics(
            model_name="AnomalyDetector",
            model_type="IsolationForest (unsupervised)",
            n_samples=len(X_scaled), n_features=X_scaled.shape[1],
            trained_at=datetime.now(timezone.utc).isoformat(),
            metrics={
                "anomaly_rate_train": round(float(anomaly_rate), 4),
                "decision_score_mean": round(float(scores.mean()), 4),
                "decision_score_std": round(float(scores.std()), 4),
            },
            feature_importances={},
            top_features=[],
        )
        self._log(f"  → Anomaly rate (train): {anomaly_rate:.2%}")
        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def train_all(
        self,
        data_dir: str,
        model_dir: str = "models",
        n_synthetic: int = 600,
        age_est: int = 30,
    ) -> Dict[str, Any]:
        """Run complete training pipeline. Returns metrics dict."""
        from .feature_engineering import FEATURE_NAMES
        from sklearn.preprocessing import StandardScaler

        self._status = {"state": "starting", "progress": 0, "log": []}
        t0 = time.time()
        os.makedirs(model_dir, exist_ok=True)

        # 1. Data preparation
        X, y_cardio, y_sleep, y_stress = self.prepare_data(data_dir, n_synthetic, age_est)

        # 2. Scale features
        self._log("Fitting StandardScaler...")
        self._status["state"] = "scaling"
        self._status["progress"] = 40
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        self.scaler = scaler

        # 3. Train all models
        all_metrics = []

        self._status["state"] = "training_cardio"
        self._status["progress"] = 45
        m1 = self._train_cardio_risk(X_scaled, y_cardio, FEATURE_NAMES)
        all_metrics.append(m1)

        self._status["state"] = "training_sleep"
        self._status["progress"] = 60
        m2 = self._train_sleep_quality(X_scaled, y_sleep, FEATURE_NAMES)
        all_metrics.append(m2)

        self._status["state"] = "training_stress"
        self._status["progress"] = 75
        m3 = self._train_stress_classifier(X_scaled, y_stress, FEATURE_NAMES)
        all_metrics.append(m3)

        self._status["state"] = "training_anomaly"
        self._status["progress"] = 88
        m4 = self._train_anomaly_detector(X_scaled, FEATURE_NAMES)
        all_metrics.append(m4)
        self.metrics = all_metrics

        # 4. Save models
        self._status["state"] = "saving"
        self._status["progress"] = 95
        self._save_models(model_dir, FEATURE_NAMES)

        elapsed = round(time.time() - t0, 1)
        self._log(f"Training complete in {elapsed}s")
        self._status["state"] = "complete"
        self._status["progress"] = 100

        result = {
            "status": "success",
            "elapsed_seconds": elapsed,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "models_trained": [m.model_name for m in all_metrics],
            "model_dir": model_dir,
            "metrics": {m.model_name: m.metrics for m in all_metrics},
        }
        return result

    def _save_models(self, model_dir: str, feature_names: List[str]):
        model_dir = Path(model_dir)
        # Save each model
        for key, filename in self.MODEL_FILES.items():
            path = model_dir / filename
            if key == "cardio_risk" and "cardio_risk" in self.models:
                with open(path, "wb") as f:
                    pickle.dump(self.models["cardio_risk"], f)
            elif key == "sleep_quality" and "sleep_quality" in self.models:
                with open(path, "wb") as f:
                    pickle.dump(self.models["sleep_quality"], f)
            elif key == "stress_classifier" and "stress_classifier" in self.models:
                with open(path, "wb") as f:
                    pickle.dump(self.models["stress_classifier"], f)
            elif key == "anomaly_detector" and "anomaly_detector" in self.models:
                with open(path, "wb") as f:
                    pickle.dump(self.models["anomaly_detector"], f)
            elif key == "scaler" and self.scaler is not None:
                with open(path, "wb") as f:
                    pickle.dump(self.scaler, f)
            elif key == "feature_names":
                with open(path, "w") as f:
                    json.dump(feature_names, f)
            elif key == "metrics":
                with open(path, "w") as f:
                    json.dump([m.to_dict() for m in self.metrics], f, indent=2)
        self._log(f"Models saved to {model_dir}")


# ---------------------------------------------------------------------------
# Predictor (inference)
# ---------------------------------------------------------------------------
class SamsungHealthPredictor:
    """
    Loads trained Samsung Health models and runs predictions.
    """

    STRESS_LABELS = {0: "low", 1: "medium", 2: "high"}

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_names: List[str] = []
        self._loaded = False

    def load_models(self) -> bool:
        """Load all serialized models from model_dir. Returns True if all found."""
        try:
            for key in ["cardio_risk", "sleep_quality", "stress_classifier", "anomaly_detector"]:
                fname = SamsungHealthTrainer.MODEL_FILES[key]
                path = self.model_dir / fname
                if not path.exists():
                    logger.warning(f"Model not found: {path}")
                    return False
                with open(path, "rb") as f:
                    self.models[key] = pickle.load(f)

            scaler_path = self.model_dir / SamsungHealthTrainer.MODEL_FILES["scaler"]
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            fn_path = self.model_dir / SamsungHealthTrainer.MODEL_FILES["feature_names"]
            if fn_path.exists():
                with open(fn_path) as f:
                    self.feature_names = json.load(f)

            self._loaded = True
            logger.info("Samsung Health models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def models_available(self) -> bool:
        return self._loaded and bool(self.models)

    def predict(
        self,
        record=None,
        feature_vector: Optional[np.ndarray] = None,
        age_est: int = 30,
    ) -> Dict[str, Any]:
        """
        Run all predictions for a patient.
        Either pass a SamsungPatientRecord or a pre-computed feature vector.
        """
        from .feature_engineering import engineer_features, feature_vector_to_dict, FEATURE_NAMES

        if feature_vector is None:
            if record is None:
                raise ValueError("Provide either record or feature_vector")
            feature_vector = engineer_features(record, age_est=age_est)

        X = feature_vector.reshape(1, -1)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X).astype(np.float32)
        else:
            X_scaled = X

        results: Dict[str, Any] = {
            "patient_id": getattr(record, "patient_id", "unknown") if record else "unknown",
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "feature_count": int(X.shape[1]),
        }

        # Cardiovascular risk
        if "cardio_risk" in self.models:
            clf = self.models["cardio_risk"]
            prob = clf.predict_proba(X_scaled)[0]
            pred = int(prob[1] >= 0.5)
            results["cardiovascular_risk"] = {
                "risk_score": round(float(prob[1]), 4),
                "risk_label": "elevated" if pred else "normal",
                "confidence": round(float(max(prob)), 4),
                "probability_elevated": round(float(prob[1]), 4),
                "probability_normal": round(float(prob[0]), 4),
            }

        # Sleep quality
        if "sleep_quality" in self.models:
            reg = self.models["sleep_quality"]
            score = float(np.clip(reg.predict(X_scaled)[0], 0, 100))
            results["sleep_quality"] = {
                "quality_index": round(score, 2),
                "quality_label": (
                    "excellent" if score >= 80 else
                    "good" if score >= 65 else
                    "fair" if score >= 50 else
                    "poor"
                ),
            }

        # Stress pattern
        if "stress_classifier" in self.models:
            clf = self.models["stress_classifier"]
            raw_probs = clf.predict_proba(X_scaled)[0]
            pred_class = int(clf.predict(X_scaled)[0])
            classes = list(getattr(clf, "classes_", [0, 1, 2]))
            prob_map = {0: 0.0, 1: 0.0, 2: 0.0}
            for cls_idx, cls_val in enumerate(classes):
                if int(cls_val) in prob_map:
                    prob_map[int(cls_val)] = round(float(raw_probs[cls_idx]), 4)
            results["stress_pattern"] = {
                "stress_class": pred_class,
                "stress_label": self.STRESS_LABELS.get(pred_class, "medium"),
                "probabilities": {
                    "low": prob_map[0],
                    "medium": prob_map[1],
                    "high": prob_map[2],
                },
                "confidence": round(float(raw_probs.max()), 4),
            }

        # Anomaly detection
        if "anomaly_detector" in self.models:
            iso = self.models["anomaly_detector"]
            anomaly_pred = int(iso.predict(X_scaled)[0])  # -1 = anomaly, 1 = normal
            anomaly_score = float(iso.decision_function(X_scaled)[0])
            results["anomaly"] = {
                "is_anomaly": anomaly_pred == -1,
                "anomaly_score": round(anomaly_score, 4),  # negative = more anomalous
                "interpretation": "anomalous health pattern" if anomaly_pred == -1 else "normal range",
            }

        # Feature values (for explainability)
        results["feature_values"] = {
            name: round(float(val), 4)
            for name, val in zip(FEATURE_NAMES, feature_vector)
        }

        return results

    def explain(
        self,
        record=None,
        feature_vector: Optional[np.ndarray] = None,
        top_n: int = 10,
        age_est: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate SHAP-style feature importance explanation.
        Uses tree-model feature importances as surrogate for SHAP values.
        """
        from .feature_engineering import engineer_features, FEATURE_NAMES

        if feature_vector is None:
            if record is None:
                raise ValueError("Provide either record or feature_vector")
            feature_vector = engineer_features(record, age_est=age_est)

        X = feature_vector.reshape(1, -1)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        explanation: Dict[str, Any] = {}

        for model_key, display_name in [
            ("cardio_risk", "cardiovascular_risk"),
            ("sleep_quality", "sleep_quality"),
            ("stress_classifier", "stress_pattern"),
        ]:
            model = self.models.get(model_key)
            if model is None:
                continue
            # Get base estimator importances
            base = getattr(model, "base_estimator_", model)
            # For CalibratedClassifierCV, extract base estimator
            if hasattr(base, "calibrated_classifiers_"):
                base = base.calibrated_classifiers_[0].estimator
            if hasattr(base, "feature_importances_"):
                importances = base.feature_importances_
            else:
                continue
            # Weight importances by feature values (approximation of SHAP directionality)
            feature_vals = feature_vector
            if self.scaler is not None:
                feature_vals_scaled = X_scaled[0]
            else:
                feature_vals_scaled = feature_vector
            # Top N features
            ranked_idx = np.argsort(importances)[::-1][:top_n]
            explanation[display_name] = [
                {
                    "feature": FEATURE_NAMES[i],
                    "importance": round(float(importances[i]), 6),
                    "patient_value": round(float(feature_vector[i]), 3),
                    "normalized_value": round(float(feature_vals_scaled[i]), 3),
                }
                for i in ranked_idx
            ]

        return explanation

    def load_training_metrics(self) -> List[Dict]:
        """Load saved training metrics."""
        path = self.model_dir / SamsungHealthTrainer.MODEL_FILES["metrics"]
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
_trainer: Optional[SamsungHealthTrainer] = None
_predictor: Optional[SamsungHealthPredictor] = None

def get_trainer() -> SamsungHealthTrainer:
    global _trainer
    if _trainer is None:
        _trainer = SamsungHealthTrainer()
    return _trainer

def get_predictor(model_dir: str = "models") -> SamsungHealthPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SamsungHealthPredictor(model_dir=model_dir)
        _predictor.load_models()
    return _predictor
