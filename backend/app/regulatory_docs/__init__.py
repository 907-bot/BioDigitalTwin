"""
Phase 17 — FDA Q-Submission Documentation
Regulatory consultant + SaMD documentation (IEC 62304, ISO 14971)
"""
from .q_submission import QSubmissionManager, get_q_submission_manager
from .iec_62304 import IEC62304Documentation, get_iec_62304_manager
from .iso_14971 import ISO14971RiskManagement, get_iso_14971_manager

__all__ = [
    "QSubmissionManager",
    "get_q_submission_manager",
    "IEC62304Documentation",
    "get_iec_62304_manager",
    "ISO14971RiskManagement",
    "get_iso_14971_manager",
]

