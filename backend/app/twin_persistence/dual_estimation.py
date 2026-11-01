"""
Dual Estimation Engine
Combines UKF (Unscented Kalman Filter) with MAP/MCMC for improved identifiability
Reduces glucose RMSE through Bayesian parameter estimation
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    pm = None
    az = None
    HAS_PYMC = False
try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    class MerweScaledSigmaPoints:
        def __init__(self, *args, **kwargs): pass
    class UnscentedKalmanFilter:
        def __init__(self, dim_x=3, dim_z=1, dt=5.0, fx=None, hx=None, points=None):
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.dt = dt
            self.fx = fx
            self.hx = hx
            self.x = np.array([100.0, 0.0, 0.0])
            self.P = np.diag([10.0, 1.0, 0.1])
            self.Q = np.eye(dim_x) * 0.1
            self.R = np.eye(dim_z) * 0.5
        def predict(self, dt=None, **kwargs):
            if self.fx:
                self.x = self.fx(self.x, dt or self.dt)
        def update(self, z, **kwargs):
            if self.hx:
                z_pred = self.hx(self.x)
                y = z - z_pred
                # Simple innovation update
                self.x[0] += 0.5 * y[0]
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlucoseModel(BaseModel):
    """Glucose-insulin model parameters for dual estimation"""
    # State vector: [glucose, insulin_on_board, glucose_rate]
    state_dim: int = 3
    
    # UKF parameters
    process_noise: float = 0.1
    measurement_noise: float = 0.5
    alpha: float = 0.001  # UKF sigma point parameter
    beta: float = 2.0    # UKF sigma point parameter
    kappa: float = 0.0   # UKF sigma point parameter
    
    # Physiological parameters (to be estimated via MCMC)
    insulin_sensitivity: float = 0.5  # SI
    carb_absorption_rate: float = 0.1  # k_abs
    glucose_effectiveness: float = 0.01  # EG
    endogenous_glucose_production: float = 1.0  # EGP
    
    # MCMC parameters
    mcmc_samples: int = 2000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4


class DualEstimationEngine:
    """
    Dual estimation combining:
    1. UKF for real-time state estimation (fast, online)
    2. MAP/MCMC for parameter identification (offline, batch)
    
    This hybrid approach addresses identifiability issues in glucose modeling
    by separating state estimation from parameter learning.
    """
    
    def __init__(self, model: GlucoseModel):
        self.model = model
        self.ukf = self._init_ukf()
        self.parameter_estimates = {
            "insulin_sensitivity": model.insulin_sensitivity,
            "carb_absorption_rate": model.carb_absorption_rate,
            "glucose_effectiveness": model.glucose_effectiveness,
            "endogenous_glucose_production": model.endogenous_glucose_production
        }
        self.parameter_uncertainty = {
            "insulin_sensitivity": 0.1,
            "carb_absorption_rate": 0.05,
            "glucose_effectiveness": 0.005,
            "endogenous_glucose_production": 0.2
        }
        self.observation_history = []
        self.rmse_history = []
        
    def _init_ukf(self) -> UnscentedKalmanFilter:
        """Initialize Unscented Kalman Filter"""
        points = MerweScaledSigmaPoints(
            self.model.state_dim,
            alpha=self.model.alpha,
            beta=self.model.beta,
            kappa=self.model.kappa
        )
        
        ukf = UnscentedKalmanFilter(
            dim_x=self.model.state_dim,
            dim_z=1,  # glucose measurement
            dt=5.0,  # 5-minute time step
            fx=self._state_transition,
            hx=self._measurement_function,
            points=points
        )
        
        # Initialize state
        ukf.x = np.array([100.0, 0.0, 0.0])  # [glucose, IOB, glucose_rate]
        ukf.P = np.diag([10.0, 1.0, 0.1])  # Initial covariance
        ukf.Q = np.eye(self.model.state_dim) * self.model.process_noise
        ukf.R = np.eye(1) * self.model.measurement_noise
        
        return ukf
    
    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition function for glucose-insulin model
        x = [glucose, insulin_on_board, glucose_rate]
        """
        glucose, iob, glucose_rate = x
        
        # Get current parameter estimates
        SI = self.parameter_estimates["insulin_sensitivity"]
        k_abs = self.parameter_estimates["carb_absorption_rate"]
        EG = self.parameter_estimates["glucose_effectiveness"]
        EGP = self.parameter_estimates["endogenous_glucose_production"]
        
        # Minimal model equations
        d_glucose = -EG * glucose + EGP - SI * iob * glucose
        d_iob = -0.1 * iob  # Insulin decay
        d_rate = -0.05 * glucose_rate  # Rate decay
        
        return np.array([
            glucose + d_glucose * dt,
            iob + d_iob * dt,
            glucose_rate + d_rate * dt
        ])
    
    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: observe glucose"""
        return np.array([x[0]])  # Return glucose
    
    def update_ukf(self, glucose_measurement: float, dt: float = 5.0) -> Dict[str, Any]:
        """
        Update UKF with new glucose measurement
        Returns state estimate and covariance
        """
        try:
            # Predict step
            self.ukf.predict(dt=dt)
            
            # Update step
            self.ukf.update(np.array([glucose_measurement]))
            
            # Store observation
            self.observation_history.append({
                "timestamp": datetime.now(),
                "glucose": glucose_measurement,
                "estimated_glucose": self.ukf.x[0],
                "estimated_iob": self.ukf.x[1],
                "covariance": self.ukf.P.copy()
            })
            
            # Calculate RMSE
            if len(self.observation_history) > 1:
                errors = [
                    abs(obs["glucose"] - obs["estimated_glucose"])
                    for obs in self.observation_history[-100:]  # Last 100 observations
                ]
                rmse = np.sqrt(np.mean(np.array(errors) ** 2))
                self.rmse_history.append(rmse)
            
            return {
                "state": self.ukf.x.tolist(),
                "covariance": self.ukf.P.tolist(),
                "timestamp": datetime.now().isoformat(),
                "rmse": self.rmse_history[-1] if self.rmse_history else None
            }
        except Exception as e:
            logger.error(f"UKF update error: {e}")
            return {"error": str(e)}
    
    async def estimate_parameters_mcmc(
        self,
        glucose_data: np.ndarray,
        insulin_data: np.ndarray,
        carb_data: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate physiological parameters using MCMC
        This addresses identifiability by exploring parameter space
        """
        if not HAS_PYMC:
            logger.warning("PyMC not installed. Falling back to prior parameters.")
            return {
                "parameters": self.parameter_estimates,
                "note": "PyMC unavailable; returning current parameter estimates"
            }
        try:
            logger.info("Starting MCMC parameter estimation...")
            
            # Prepare data
            dt = np.diff(timestamps).mean() if len(timestamps) > 1 else 5.0
            
            with pm.Model() as mcmc_model:
                # Priors for physiological parameters
                insulin_sensitivity = pm.Normal(
                    "insulin_sensitivity",
                    mu=0.5,
                    sigma=0.2,
                    testval=self.parameter_estimates["insulin_sensitivity"]
                )
                carb_absorption_rate = pm.Normal(
                    "carb_absorption_rate",
                    mu=0.1,
                    sigma=0.05,
                    testval=self.parameter_estimates["carb_absorption_rate"]
                )
                glucose_effectiveness = pm.Normal(
                    "glucose_effectiveness",
                    mu=0.01,
                    sigma=0.005,
                    testval=self.parameter_estimates["glucose_effectiveness"]
                )
                endogenous_glucose_production = pm.Normal(
                    "endogenous_glucose_production",
                    mu=1.0,
                    sigma=0.3,
                    testval=self.parameter_estimates["endogenous_glucose_production"]
                )
                
                # Measurement noise
                sigma = pm.HalfNormal("sigma", sigma=1.0)
                
                # Likelihood
                def glucose_likelihood(params):
                    SI, k_abs, EG, EGP = params
                    predicted = self._simulate_glucose(
                        glucose_data[0],
                        insulin_data,
                        carb_data,
                        SI, k_abs, EG, EGP,
                        dt
                    )
                    return predicted
                
                # Simplified likelihood for computational efficiency
                expected_glucose = self._simulate_glucose(
                    glucose_data[0],
                    insulin_data,
                    carb_data,
                    insulin_sensitivity,
                    carb_absorption_rate,
                    glucose_effectiveness,
                    endogenous_glucose_production,
                    dt
                )
                
                likelihood = pm.Normal(
                    "likelihood",
                    mu=expected_glucose,
                    sigma=sigma,
                    observed=glucose_data
                )
                
                # Sample
                trace = pm.sample(
                    draws=self.model.mcmc_samples,
                    tune=self.model.mcmc_tune,
                    chains=self.model.mcmc_chains,
                    cores=1,  # Single core for compatibility
                    return_inferencedata=False,
                    progressbar=True
                )
                
                # Extract parameter estimates
                estimates = {
                    "insulin_sensitivity": float(np.mean(trace["insulin_sensitivity"])),
                    "carb_absorption_rate": float(np.mean(trace["carb_absorption_rate"])),
                    "glucose_effectiveness": float(np.mean(trace["glucose_effectiveness"])),
                    "endogenous_glucose_production": float(np.mean(trace["endogenous_glucose_production"]))
                }
                
                uncertainties = {
                    "insulin_sensitivity": float(np.std(trace["insulin_sensitivity"])),
                    "carb_absorption_rate": float(np.std(trace["carb_absorption_rate"])),
                    "glucose_effectiveness": float(np.std(trace["glucose_effectiveness"])),
                    "endogenous_glucose_production": float(np.std(trace["endogenous_glucose_production"]))
                }
                
                # Update model parameters
                self.parameter_estimates = estimates
                self.parameter_uncertainty = uncertainties
                
                # Re-initialize UKF with new parameters
                self.ukf = self._init_ukf()
                
                logger.info(f"MCMC estimation complete. New parameters: {estimates}")
                
                return {
                    "estimates": estimates,
                    "uncertainties": uncertainties,
                    "trace_stats": {
                        "insulin_sensitivity_mean": float(np.mean(trace["insulin_sensitivity"])),
                        "carb_absorption_rate_mean": float(np.mean(trace["carb_absorption_rate"])),
                        "r_hat": float(pm.stats.rhat(trace)),
                        "effective_n": int(pm.stats.effective_n(trace).mean())
                    }
                }
        except Exception as e:
            logger.error(f"MCMC estimation error: {e}")
            return {"error": str(e)}
    
    def _simulate_glucose(
        self,
        initial_glucose: float,
        insulin_data: np.ndarray,
        carb_data: np.ndarray,
        SI: float,
        k_abs: float,
        EG: float,
        EGP: float,
        dt: float
    ) -> np.ndarray:
        """
        Simulate glucose trajectory given parameters
        Simplified minimal model for MCMC likelihood
        """
        n = len(insulin_data)
        glucose = np.zeros(n)
        glucose[0] = initial_glucose
        iob = 0.0
        
        for i in range(1, n):
            # Update insulin on board
            iob = iob * np.exp(-0.1 * dt) + insulin_data[i] * dt
            
            # Update glucose
            d_glucose = -EG * glucose[i-1] + EGP - SI * iob * glucose[i-1] + k_abs * carb_data[i]
            glucose[i] = glucose[i-1] + d_glucose * dt
        
        return glucose
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current UKF state and parameter estimates"""
        return {
            "state": self.ukf.x.tolist() if hasattr(self.ukf, 'x') else None,
            "covariance": self.ukf.P.tolist() if hasattr(self.ukf, 'P') else None,
            "parameter_estimates": self.parameter_estimates,
            "parameter_uncertainty": self.parameter_uncertainty,
            "observation_count": len(self.observation_history),
            "current_rmse": self.rmse_history[-1] if self.rmse_history else None,
            "avg_rmse": np.mean(self.rmse_history) if self.rmse_history else None
        }
    
    def reset(self):
        """Reset the estimation engine"""
        self.ukf = self._init_ukf()
        self.observation_history = []
        self.rmse_history = []


def create_dual_estimation_engine(
    insulin_sensitivity: float = 0.5,
    carb_absorption_rate: float = 0.1,
    mcmc_samples: int = 2000
) -> DualEstimationEngine:
    """Factory function to create dual estimation engine"""
    model = GlucoseModel(
        insulin_sensitivity=insulin_sensitivity,
        carb_absorption_rate=carb_absorption_rate,
        mcmc_samples=mcmc_samples
    )
    return DualEstimationEngine(model)
