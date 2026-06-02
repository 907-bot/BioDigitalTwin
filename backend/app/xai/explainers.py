"""Core XAI utilities — feature attribution and reasoning chains."""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import math
import statistics


def normalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Convert a raw weight dict to non-negative signed contributions.

    The output preserves the sign (positive = drives effect up,
    negative = drives effect down) but is shifted so all values are
    non-negative for plotting. A `__baseline` key records the shift.
    """
    if not weights:
        return weights
    mn = min(weights.values())
    shift = max(0.0, -mn)
    return {k: v + shift for k, v in weights.items()} | {"__baseline": shift}


def shap_like_attribution(
    base_features: Dict[str, float],
    perturbed: Dict[str, float],
    outcome_fn,
) -> List[Dict[str, Any]]:
    """SHAP-lite: marginal contribution of each feature by leave-one-out.

    For each feature, replace the perturbed value with the base value and
    re-evaluate outcome_fn. The difference is the feature's contribution.
    Signed: positive => this feature *drove* the effect up from base.
    """
    base_outcome = outcome_fn(base_features)
    pert_outcome = outcome_fn(perturbed)
    total = pert_outcome - base_outcome

    contributions: List[Dict[str, Any]] = []
    for f in base_features.keys():
        if f not in perturbed:
            continue
        # set this feature to base value, leave others at perturbed
        mixed = {**perturbed, f: base_features[f]}
        mixed_outcome = outcome_fn(mixed)
        contrib = pert_outcome - mixed_outcome
        contributions.append({
            "feature": f,
            "base_value": base_features[f],
            "perturbed_value": perturbed[f],
            "contribution": contrib,
            "outcome_with_feature": pert_outcome,
            "outcome_without_feature": mixed_outcome,
        })
    # sort by absolute contribution descending
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    # efficiency check (SHAP property)
    summed = sum(c["contribution"] for c in contributions)
    contributions.append({
        "feature": "__total",
        "contribution": summed,
        "efficiency_error": abs(summed - total),
    })
    return contributions


def build_reasoning_chain(
    question: str,
    evidence: List[Dict[str, Any]],
    conclusion: str,
    confidence: str = "medium",
    confidence_score: float = 0.5,
    alternative_hypotheses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """A structured reasoning chain: question -> evidence -> conclusion.

    `evidence` is a list of {fact, source, weight} dicts.
    `confidence` ∈ {high, medium, low}.
    `alternative_hypotheses` lists what else could be true.
    """
    return {
        "question": question,
        "evidence": evidence,
        "conclusion": conclusion,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "n_evidence": len(evidence),
        "alternative_hypotheses": alternative_hypotheses or [],
    }


def format_evidence_table(evidence: List[Dict[str, Any]]) -> str:
    """Render evidence as a markdown-ish table for an LLM prompt."""
    if not evidence:
        return "  (no evidence)"
    lines = ["| # | Fact | Source | Weight |", "|---|------|--------|--------|"]
    for i, e in enumerate(evidence, 1):
        fact = str(e.get("fact", ""))[:80]
        src = str(e.get("source", ""))[:30]
        w = e.get("weight", 1.0)
        if isinstance(w, float):
            w = f"{w:.2f}"
        lines.append(f"| {i} | {fact} | {src} | {w} |")
    return "\n".join(lines)


def confidence_from_ci(ci_lo: float, ci_hi: float, effect: float,
                       direction_stability: float = 1.0) -> Dict[str, Any]:
    """Map a bootstrap CI to a high/medium/low confidence label + reasoning.

    Heuristics:
      - high:  CI excludes 0 AND width < 0.5 * |effect| AND direction_stab > 0.9
      - medium: CI excludes 0 AND width < 1.0 * |effect|
      - low:  otherwise
    """
    if not (math.isfinite(ci_lo) and math.isfinite(ci_hi)):
        return {"label": "low", "score": 0.0, "reasons": ["CI not finite"]}
    excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
    width = ci_hi - ci_lo
    rel = width / max(abs(effect), 1e-9)
    reasons = []
    if excludes_zero:
        reasons.append(f"CI [{ci_lo:.2f}, {ci_hi:.2f}] excludes zero")
    else:
        reasons.append(f"CI [{ci_lo:.2f}, {ci_hi:.2f}] crosses zero")
    reasons.append(f"CI width = {width:.2f} (relative {rel:.2f})")
    reasons.append(f"direction stability = {direction_stability:.2f}")

    if excludes_zero and rel < 0.5 and direction_stability > 0.9:
        label, score = "high", min(1.0, 0.6 + 0.4 * direction_stability)
    elif excludes_zero and rel < 1.0:
        label, score = "medium", 0.5
    else:
        label, score = "low", 0.2
    return {"label": label, "score": score, "reasons": reasons,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "rel_width": rel}


def topk_features(contributions: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k features by absolute contribution, plus the __total row."""
    body = [c for c in contributions if c.get("feature") != "__total"]
    total = [c for c in contributions if c.get("feature") == "__total"]
    return body[:k] + total
