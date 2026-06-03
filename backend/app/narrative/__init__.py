"""Two-tier narrative generators for every domain response.

Every endpoint that returns data should include a `narrative` block with
two audiences:

  - `lay`        : plain English, no jargon. 1-3 sentences a non-scientist
                   can act on.
  - `scientist`  : technical detail including mechanism, pharmacokinetic
                   parameters, statistical interpretation, and evidence
                   source. 2-5 sentences aimed at a clinician/researcher.

A `headline` is a single-sentence top-line summary, and a `risk_level` ∈
{low, moderate, high, critical} drives the UI badge color.

All generators are PURE FUNCTIONS of the response payload — they reference
actual values from the data, not hardcoded templates.
"""
from ._utils import risk_from_severity, empty_narrative
from . import pgx, ddi, pkpd, uq, trials, regulatory, wetlab, registry

__all__ = [
    "pgx", "ddi", "pkpd", "uq", "trials", "regulatory", "wetlab", "registry",
    "risk_from_severity", "empty_narrative",
]
