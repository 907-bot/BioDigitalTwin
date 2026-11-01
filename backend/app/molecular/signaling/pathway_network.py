"""
backend/app/molecular/signaling/pathway_network.py
Inter-pathway Cross-talk and Network Topology for Molecular Targets.
"""
from __future__ import annotations

from typing import Dict, List, Any


PATHWAY_TOPOLOGY: Dict[str, Dict[str, Any]] = {
    "cAMP_PKA": {
        "name": "cAMP - Protein Kinase A Pathway",
        "upstream_receptors": ["GLP1R", "ADRB1"],
        "second_messenger": "cAMP",
        "effectors": ["PKA", "Epac1/2", "CREB"],
        "cross_talk": ["Inhibits MLCK in vascular smooth muscle (vasodilation)", "Promotes phospholamban phosphorylation in heart"],
    },
    "PLC_IP3_DAG": {
        "name": "Phospholipase C - IP3/DAG - Ca2+ Pathway",
        "upstream_receptors": ["AGTR1"],
        "second_messenger": "IP3, DAG, Ca2+",
        "effectors": ["PKC", "Calmodulin", "MLCK"],
        "cross_talk": ["Activates CaMKII", "Directly stimulates smooth muscle contraction"],
    },
    "PI3K_Akt": {
        "name": "Phosphoinositide 3-Kinase - Akt Signaling",
        "upstream_receptors": ["INSR"],
        "second_messenger": "PIP3",
        "effectors": ["Akt1/2", "AS160", "GSK3beta", "FoxO1", "mTORC1"],
        "cross_talk": ["Inhibits FoxO1 nuclear translocation (shutting down PEPCK)", "Promotes eNOS phosphorylation (endothelial NO release)"],
    },
}
