"""Shared utilities for narrative generation."""


def risk_from_severity(severity: str) -> str:
    if severity in ("critical", "contraindicated"):
        return "critical"
    if severity in ("major",):
        return "high"
    if severity in ("moderate",):
        return "moderate"
    return "low"


def empty_narrative(text: str = "No data") -> dict:
    return {
        "headline": text,
        "lay": text,
        "scientist": text,
        "risk_level": "low",
    }
