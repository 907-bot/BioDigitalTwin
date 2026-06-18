"""
Phase 17 — FDA Q-Submission Documentation
Regulatory consultant + SaMD documentation (IEC 62304, ISO 14971)
"""
from .q_submission import QSubmissionManager
from .iec_62304 import IEC62304Documentation
from .iso_14971 import ISO14971RiskManagement

__all__ = [
    "QSubmissionManager",
    "IEC62304Documentation",
    "ISO14971RiskManagement",
]
