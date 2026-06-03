"""Clinical trials narrative generators."""
from typing import List, Dict, Any


def narrate_search(query: str, by: str, n_results: int,
                    trials: List[Dict[str, Any]]) -> Dict[str, str]:
    if n_results == 0:
        return {
            "headline": f"No clinical trials found for {query}",
            "lay": (f"We searched ClinicalTrials.gov and found no active or completed "
                    f"trials matching '{query}'. This doesn't mean none exist — it may "
                    f"be worth trying a broader or differently-phrased search."),
            "scientist": (f"ClinicalTrials.gov v2 search returned 0 results for "
                          f"{by}='{query}'. Consider expanding search terms or "
                          f"checking alternative spellings/synonyms."),
            "risk_level": "low",
        }

    n_recruiting = sum(1 for t in trials if t.get("overall_status") == "RECRUITING")
    n_completed = sum(1 for t in trials if t.get("overall_status") == "COMPLETED")
    n_phase4 = sum(1 for t in trials if any("4" in p for p in (t.get("phase") or [])))
    total_enrolled = sum(t.get("enrollment") or 0 for t in trials)

    lay = (f"Found {n_results} clinical trial(s) matching '{query}' on "
           f"ClinicalTrials.gov. Of these, {n_recruiting} are currently recruiting "
           f"patients, {n_completed} have been completed, and the rest are in other "
           f"stages. {n_phase4} trial(s) are in Phase 4 (post-marketing). "
           f"Total enrolled participants across all matching trials: "
           f"{total_enrolled:,}.")

    sci = (f"ClinicalTrials.gov v2 search: {by}='{query}' yielded {n_results} "
           f"trial(s). Status breakdown: {n_recruiting} RECRUITING, "
           f"{n_completed} COMPLETED, remainder in other phases. "
           f"Phase distribution: {n_phase4} in Phase 4 (post-marketing surveillance). "
           f"Total enrollment: {total_enrolled:,} participants. Data fetched live "
           f"with 24h cache.")

    return {"headline": f"{n_results} clinical trial(s) for '{query}'",
            "lay": lay, "scientist": sci, "risk_level": "low"}
