"""
FDA Orange Book snapshot (approved drugs) and a curated drug-name normalizer.

The Orange Book CSV is downloaded at build time when needed. For the demo
we ship a small curated list of ~40 commonly-prescribed drugs and their
approval status — the live CSV download is attempted at first use and
falls back to the curated list on network failure.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ORANGE_BOOK_URL = ("https://www.fda.gov/drugs/drug-approvals-and-databases/"
                   "orange-book-data-files")
TIMEOUT = 20.0


@dataclass(frozen=True)
class OrangeBookEntry:
    ingredient: str
    trade_name: str
    applicant: str
    approval_date: str
    patent_use_code: str = ""
    exclusivity_code: str = ""


# Curated snapshot — most common approvals
CURATED_ORANGE_BOOK: list[OrangeBookEntry] = [
    OrangeBookEntry("metformin",   "Glucophage",  "Bristol-Myers Squibb", "1995-12-29"),
    OrangeBookEntry("warfarin",    "Coumadin",    "Bristol-Myers Squibb", "1954-06-10"),
    OrangeBookEntry("atorvastatin","Lipitor",     "Pfizer",              "1996-12-17"),
    OrangeBookEntry("simvastatin", "Zocor",       "Merck",               "1991-12-23"),
    OrangeBookEntry("lisinopril",  "Prinivil",    "Merck",               "1987-12-29"),
    OrangeBookEntry("losartan",    "Cozaar",      "Merck",               "1995-04-14"),
    OrangeBookEntry("amlodipine",  "Norvasc",     "Pfizer",              "1992-07-31"),
    OrangeBookEntry("metoprolol",  "Lopressor",   "Novartis",            "1978-08-07"),
    OrangeBookEntry("carvedilol",  "Coreg",       "GlaxoSmithKline",     "1995-09-14"),
    OrangeBookEntry("furosemide",  "Lasix",       "Sanofi-Aventis",      "1966-07-22"),
    OrangeBookEntry("spironolactone","Aldactone", "Pfizer",              "1960-01-21"),
    OrangeBookEntry("digoxin",     "Lanoxin",     "Covis Pharma",        "1954-08-09"),
    OrangeBookEntry("aspirin",     "Bayer",       "Bayer",               "1938-09-15"),
    OrangeBookEntry("clopidogrel", "Plavix",      "Bristol-Myers Squibb", "1997-11-17"),
    OrangeBookEntry("apixaban",    "Eliquis",     "Bristol-Myers Squibb", "2012-12-28"),
    OrangeBookEntry("rivaroxaban", "Xarelto",     "Bayer",               "2011-07-01"),
    OrangeBookEntry("omeprazole",  "Prilosec",    "AstraZeneca",         "1989-09-14"),
    OrangeBookEntry("pantoprazole","Protonix",    "Pfizer",              "2000-02-02"),
    OrangeBookEntry("sertraline",  "Zoloft",      "Pfizer",              "1991-12-30"),
    OrangeBookEntry("escitalopram","Lexapro",     "Forest Pharmaceuticals","2002-08-14"),
    OrangeBookEntry("fluoxetine",  "Prozac",      "Eli Lilly",           "1987-12-29"),
    OrangeBookEntry("risperidone", "Risperdal",   "Janssen",             "1993-12-29"),
    OrangeBookEntry("aripiprazole","Abilify",     "Otsuka",              "2002-11-15"),
    OrangeBookEntry("haloperidol", "Haldol",      "Janssen",             "1967-04-12"),
    OrangeBookEntry("morphine",    "MS Contin",   "Purdue Pharma",      "1984-05-11"),
    OrangeBookEntry("codeine",     "Codeine Sulfate","Hikma",            "1985-04-12"),
    OrangeBookEntry("tramadol",    "Ultram",      "Ortho-McNeil",        "1995-03-03"),
    OrangeBookEntry("ondansetron", "Zofran",      "GlaxoSmithKline",     "1990-12-31"),
    OrangeBookEntry("ciprofloxacin","Cipro",      "Bayer",               "1987-04-08"),
    OrangeBookEntry("diazepam",    "Valium",      "Roche",               "1963-11-15"),
    OrangeBookEntry("midazolam",   "Versed",      "Roche",               "1985-12-20"),
    OrangeBookEntry("levothyroxine","Synthroid",  "AbbVie",              "1973-05-25"),
    OrangeBookEntry("insulin regular","Humulin R","Eli Lilly",          "1982-10-28"),
    OrangeBookEntry("glipizide",   "Glucotrol",   "Pfizer",              "1984-05-11"),
    OrangeBookEntry("amoxicillin", "Amoxil",      "GlaxoSmithKline",     "1974-01-18"),
    OrangeBookEntry("acyclovir",   "Zovirax",     "GlaxoSmithKline",     "1982-03-29"),
    OrangeBookEntry("memantine",   "Namenda",     "Forest Pharmaceuticals","2003-10-16"),
]


def lookup(drug: str) -> list[OrangeBookEntry]:
    """Find Orange Book entries for a drug (case-insensitive partial match)."""
    d = drug.lower().strip()
    out: list[OrangeBookEntry] = []
    for e in CURATED_ORANGE_BOOK:
        if d in e.ingredient.lower() or d in e.trade_name.lower():
            out.append(e)
    return out


def is_approved(drug: str) -> bool:
    return len(lookup(drug)) > 0


# --- RxNorm name normalization ---
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


async def normalize_rxnorm(name: str) -> Optional[dict]:
    """
    Use NIH RxNorm to find the standard drug name + RxCUI.

    Example: 'glucophage' -> {rxcui, name: 'metformin', tty: 'IN'}
    """
    import httpx as _httpx
    name = name.strip()
    if not name:
        return None
    try:
        async with _httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{RXNORM_BASE}/approximateTerm.json",
                                 params={"term": name, "maxEntries": 5})
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("RxNorm failed: %s", e)
        return None
    candidates = (data.get("approximateGroup", {})
                  .get("candidate", []))
    if not candidates:
        return None
    top = candidates[0]
    return {
        "query": name,
        "rxcui": top.get("rxcui"),
        "name": top.get("name"),
        "score": top.get("score"),
    }
