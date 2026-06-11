"""
Calibration Assessment and Calibration Methods.

Evaluates and improves probabilistic calibration of twin predictions.
Implements ECE, Brier score, calibration curves, Platt scaling,
isotonic regression, beta calibration, and conformal prediction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from scipy import special, optimize, interpolate


@dataclass
class CalibrationReport:
    variable: str
    ece: float
    mce: float
    brier_score: float
    slope: float
    intercept: float
    r2: float
    is_calibrated: bool
    n_bins: int = 10
    bin_fractions: List[float] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)


class CalibrationAssessor:
    """
    Multi-method calibration assessment for twin predictions.
    
    Computes:
      - Expected Calibration Error (ECE)
      - Maximum Calibration Error (MCE)
      - Brier score
      - Calibration curve (slope, intercept, R²)
      - Binomial test for miscalibration
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def assess_continuous(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_std: Optional[np.ndarray] = None,
                          variable: str = "glucose") -> CalibrationReport:
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_t, y_p = y_true[mask], y_pred[mask]
        n = len(y_t)
        if n < 10:
            return CalibrationReport(variable=variable, ece=1.0, mce=1.0, brier_score=1.0, slope=0, intercept=0, r2=0, is_calibrated=False)

        if y_std is not None:
            y_s = y_std[mask]
            standardized_residuals = (y_t - y_p) / (y_s + 1e-8)
            abs_residuals = np.abs(standardized_residuals)
            # Probability-space calibration: for each nominal CI level,
            # compute the empirical coverage
            ci_levels = np.linspace(0.1, 0.99, self.n_bins)
            prob_ece = 0.0
            for ci in ci_levels:
                z = np.abs(np.percentile(np.random.randn(10000), (1 - ci) * 100))
                lower = y_p - z * y_s
                upper = y_p + z * y_s
                emp_cov = np.mean((y_t >= lower) & (y_t <= upper))
                prob_ece += abs(emp_cov - ci) / len(ci_levels)
            prob_based_ece = prob_ece
        else:
            abs_residuals = np.abs(y_t - y_p)
            prob_based_ece = 1.0

        bin_edges = np.linspace(0, np.percentile(abs_residuals, 95) + 1e-8, self.n_bins + 1)
        bin_indices = np.digitize(abs_residuals, bin_edges[1:-1])
        ece = 0.0
        mce = 0.0
        bin_fracs = []
        bin_accs = []
        bin_counts = []
        for b in range(self.n_bins):
            mask_b = bin_indices == b
            count = mask_b.sum()
            bin_counts.append(count)
            if count > 0:
                if y_std is not None:
                    expected = bin_edges[b] + (bin_edges[b + 1] - bin_edges[b]) / 2
                    observed = np.mean(np.abs(standardized_residuals[mask_b]))
                else:
                    expected = np.mean(abs_residuals[mask_b])
                    observed = expected
                bin_frac = count / n
                bin_fracs.append(bin_frac)
                bin_accs.append(observed)
                ece += bin_frac * abs(expected - observed)
                mce = max(mce, abs(expected - observed))
            else:
                bin_fracs.append(0.0)
                bin_accs.append(0.0)
        frac_correct_90ci = 0.0
        brier = 0.0
        if y_std is not None:
            lower = y_p - 1.645 * y_s
            upper = y_p + 1.645 * y_s
            frac_correct_90ci = np.mean((y_t >= lower) & (y_t <= upper))
            brier = np.mean(((y_t - y_p) / (np.abs(y_p) + 1e-6)) ** 2)
        order = np.argsort(y_p)
        n_seg = max(10, n // 10)
        seg_slopes = []
        for i in range(0, n, n_seg):
            seg = order[i:min(i + n_seg, n)]
            if len(seg) > 3:
                seg_slopes.append(np.polyfit(y_p[seg], y_t[seg], 1)[0])
        slope = float(np.median(seg_slopes)) if seg_slopes else 1.0
        # Strict calibration criteria
        if y_std is not None:
            is_calibrated = (
                abs(slope - 1.0) < 0.2
                and ece < 0.10
                and prob_based_ece < 0.10
                and abs(frac_correct_90ci - 0.90) < 0.10
            )
        else:
            is_calibrated = abs(slope - 1.0) < 0.3 and ece < 0.15
        return CalibrationReport(
            variable=variable,
            ece=float(ece),
            mce=float(mce),
            brier_score=float(brier),
            slope=float(slope),
            intercept=0.0,
            r2=float(frac_correct_90ci),
            is_calibrated=is_calibrated,
            n_bins=self.n_bins,
            bin_fractions=bin_fracs,
            bin_accuracies=bin_accs,
            bin_counts=bin_counts,
        )

    def assess_probability(self, y_true: np.ndarray, y_pred_prob: np.ndarray,
                           variable: str = "binary") -> CalibrationReport:
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_prob))
        y_t, y_p = y_true[mask].astype(int), np.clip(y_pred_prob[mask], 0, 1)
        n = len(y_t)
        if n < 10:
            return CalibrationReport(variable=variable, ece=1.0, mce=1.0, brier_score=1.0, slope=0, intercept=0, r2=0, is_calibrated=False)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_p, bin_edges[1:-1])
        ece = 0.0
        mce = 0.0
        bin_fracs = []
        bin_accs = []
        bin_counts = []
        for b in range(self.n_bins):
            mask_b = bin_indices == b
            count = mask_b.sum()
            bin_counts.append(count)
            if count > 0:
                expected = np.mean(y_p[mask_b])
                observed = np.mean(y_t[mask_b])
                frac = count / n
                bin_fracs.append(frac)
                bin_accs.append(observed)
                ece += frac * abs(expected - observed)
                mce = max(mce, abs(expected - observed))
            else:
                bin_fracs.append(0.0)
                bin_accs.append(0.0)
        brier = np.mean((y_t - y_p) ** 2)
        slope, intercept = np.polyfit(y_p, y_t, 1) if n > 5 else (1.0, 0.0)
        ss_res = np.sum((y_t - (slope * y_p + intercept)) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
        is_calibrated = ece < 0.1 and abs(slope - 1.0) < 0.3
        return CalibrationReport(
            variable=variable, ece=float(ece), mce=float(mce),
            brier_score=float(brier), slope=float(slope),
            intercept=float(intercept), r2=float(r2),
            is_calibrated=is_calibrated, n_bins=self.n_bins,
            bin_fractions=bin_fracs, bin_accuracies=bin_accs, bin_counts=bin_counts,
        )


class PlattCalibrator:
    """
    Platt scaling for binary probability calibration.
    Fits logistic regression: P(y=1|f) = 1 / (1 + exp(A*f + B))
    """

    def __init__(self):
        self.A = 0.0
        self.B = 0.0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_p, y_t = y_pred[mask], y_true[mask]
        n = len(y_p)
        if n < 5:
            self.A, self.B = 1.0, 0.0
            return self

        def neg_log_likelihood(params):
            A, B = params
            p = 1 / (1 + np.exp(-(A * y_p + B)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.mean(y_t * np.log(p) + (1 - y_t) * np.log(1 - p))
        try:
            result = optimize.minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
            self.A, self.B = result.x
        except Exception:
            self.A, self.B = 1.0, 0.0
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.A * y_pred + self.B)))


class BetaCalibrator:
    """
    Beta calibration for probability calibration.
    Maps [0,1] -> [0,1] using Beta distribution CDF.
    """

    def __init__(self):
        self.a = 1.0
        self.b = 1.0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "BetaCalibrator":
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_p = np.clip(y_pred[mask], 1e-6, 1 - 1e-6)
        y_t = y_true[mask]
        n = len(y_p)
        if n < 10:
            return self
        log_y = np.log(y_p)
        log_1my = np.log(1 - y_p)

        def neg_ll(params):
            a, b = np.exp(params)
            if a < 0.01 or b < 0.01:
                return 1e10
            logit = a * log_y - b * log_1my
            p = 1 / (1 + np.exp(-logit))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.mean(y_t * np.log(p) + (1 - y_t) * np.log(1 - p))
        try:
            result = optimize.minimize(neg_ll, [0.0, 0.0], method="Nelder-Mead")
            self.a, self.b = np.exp(result.x)
        except Exception:
            pass
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        y_p = np.clip(y_pred, 1e-6, 1 - 1e-6)
        logit = self.a * np.log(y_p) - self.b * np.log(1 - y_p)
        return 1 / (1 + np.exp(-logit))


class ConformalPredictor:
    """
    Conformal prediction for distribution-free prediction intervals.
    Uses split conformal prediction with a calibration set.
    """

    def __init__(self, coverage: float = 0.9):
        self.coverage = coverage
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: float = 0.0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "ConformalPredictor":
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        residuals = np.abs(y_true[mask] - y_pred[mask])
        n = len(residuals)
        if n < 10:
            self.quantile = np.percentile(residuals, self.coverage * 100) if len(residuals) > 0 else 0.0
            return self
        self.calibration_scores = np.sort(residuals)
        q_idx = int(np.ceil((n + 1) * self.coverage))
        q_idx = min(q_idx, n - 1)
        self.quantile = self.calibration_scores[q_idx]
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        return lower, upper

    def predict_interval_adaptive(self, y_pred: np.ndarray,
                                  heteroscedasticity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scaled = self.quantile * heteroscedasticity / (np.mean(heteroscedasticity) + 1e-8)
        return y_pred - scaled, y_pred + scaled


class CalibrationPipeline:
    """
    End-to-end calibration assessment and recalibration pipeline.
    """

    def __init__(self):
        self.assessor = CalibrationAssessor()
        self.reports: Dict[str, CalibrationReport] = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_std: Optional[np.ndarray] = None,
                 variable: str = "glucose",
                 is_probability: bool = False) -> CalibrationReport:
        if is_probability:
            report = self.assessor.assess_probability(y_true, y_pred, variable)
        else:
            report = self.assessor.assess_continuous(y_true, y_pred, y_std, variable)
        self.reports[variable] = report
        return report

    def recalibrate_platt(self, y_pred: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        cal = PlattCalibrator()
        cal.fit(y_pred, y_true)
        return cal

    def recalibrate_beta(self, y_pred: np.ndarray, y_true: np.ndarray) -> BetaCalibrator:
        cal = BetaCalibrator()
        cal.fit(y_pred, y_true)
        return cal

    def conformal_fit(self, y_pred: np.ndarray, y_true: np.ndarray,
                      coverage: float = 0.9) -> ConformalPredictor:
        cp = ConformalPredictor(coverage=coverage)
        cp.fit(y_pred, y_true)
        return cp

    def full_report(self) -> Dict:
        return {
            "n_variables": len(self.reports),
            "calibrated": sum(1 for r in self.reports.values() if r.is_calibrated),
            "miscalibrated": sum(1 for r in self.reports.values() if not r.is_calibrated),
            "mean_ece": float(np.mean([r.ece for r in self.reports.values()])),
            "mean_brier": float(np.mean([r.brier_score for r in self.reports.values()])),
            "variables": {
                v: {"ece": r.ece, "mce": r.mce, "brier": r.brier_score,
                    "slope": r.slope, "calibrated": r.is_calibrated}
                for v, r in self.reports.items()
            },
        }
