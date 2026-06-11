"""
Phase 5 — Pillar 5: Multi-Agent Scientific Reasoning.

Specialist agents that debate hypotheses and produce consensus conclusions:
  - Physiology Agent: organ systems, homeostasis, metabolic regulation
  - Pharmacology Agent: drug mechanisms, PK/PD, drug interactions
  - Genomics Agent: genetic variants, transcriptomics, epigenetics
  - Clinical Agent: patient outcomes, trial data, guidelines
  - Research Agent: literature, prior discoveries, experimental evidence

Agents use structured reasoning with evidence citation and
confidence estimation. Debates resolve via weighted voting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import re


# ── Data Types ────────────────────────────────────────────────

class AgentRole(Enum):
    PHYSIOLOGY = "physiology"
    PHARMACOLOGY = "pharmacology"
    GENOMICS = "genomics"
    CLINICAL = "clinical"
    RESEARCH = "research"


class EvidenceLevel(Enum):
    MECHANISTIC = "mechanistic"        # known biological mechanism
    STRONG = "strong"                  # consistent evidence across studies
    MODERATE = "moderate"              # some evidence, not conclusive
    WEAK = "weak"                      # preliminary or single study
    THEORETICAL = "theoretical"        # plausible but untested


@dataclass
class AgentArgument:
    """A single argument made by an agent."""
    agent_role: AgentRole
    claim: str
    evidence: str
    evidence_level: EvidenceLevel
    confidence: float = 0.5
    citations: List[str] = field(default_factory=list)
    counterarguments: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.agent_role.value.upper()}] {self.claim} "
            f"(confidence={self.confidence:.0%}, "
            f"evidence={self.evidence_level.value})"
        )


@dataclass
class AgentDebate:
    """
    A structured debate among multiple specialist agents.

    Tracks arguments for and against a hypothesis.
    """
    topic: str
    hypothesis: str
    arguments: List[AgentArgument] = field(default_factory=list)
    consensus: Optional[str] = None
    confidence: float = 0.0
    resolved_at: Optional[float] = None
    participants: List[AgentRole] = field(default_factory=list)

    def add_argument(self, argument: AgentArgument) -> None:
        self.arguments.append(argument)
        if argument.agent_role not in self.participants:
            self.participants.append(argument.agent_role)

    def get_args_by_role(self, role: AgentRole) -> List[AgentArgument]:
        return [a for a in self.arguments if a.agent_role == role]


@dataclass
class ConsensusResult:
    """
    The final consensus from a multi-agent debate.

    Includes the resolved conclusion, confidence, and
    remaining disagreement.
    """
    topic: str
    conclusion: str
    overall_confidence: float
    supporting_arguments: List[str]
    opposing_arguments: List[str]
    remaining_uncertainties: List[str]
    recommended_actions: List[str]
    agent_votes: Dict[str, float]  # agent_role → vote weight
    debate_summary: str


# ── Base Agent ────────────────────────────────────────────────

class SpecialistAgent:
    """Base class for domain specialist agents."""

    def __init__(self, role: AgentRole, name: str, expertise: str):
        self.role = role
        self.name = name
        self.expertise = expertise
        self._knowledge_base: Dict[str, Any] = {}

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        """
        Generate a structured argument about a hypothesis.

        Subclasses override this with domain-specific reasoning.
        """
        raise NotImplementedError

    def evaluate_evidence(self, claim: str) -> Tuple[EvidenceLevel, float]:
        """
        Evaluate the evidence level for a claim.

        Returns (evidence_level, confidence).
        """
        return EvidenceLevel.MODERATE, 0.5

    def __repr__(self) -> str:
        return f"{self.name} ({self.role.value})"


# ── Domain-Specific Agents ────────────────────────────────────

class PhysiologyAgent(SpecialistAgent):
    """Expert in organ system physiology and metabolic regulation."""

    def __init__(self):
        super().__init__(AgentRole.PHYSIOLOGY, "Physiology Agent",
                         "Organ systems, homeostasis, metabolism")
        self._physiology_knowledge = {
            "glucose_regulation": (
                "Glucose homeostasis is maintained by insulin-mediated "
                "GLUT4 translocation in muscle and adipose, hepatic "
                "glucose production suppression, and incretin effects."
            ),
            "insulin_resistance": (
                "Insulin resistance arises from impaired IRS-1/PI3K/Akt "
                "signaling due to serine phosphorylation by JNK and "
                "PKCθ, driven by lipid intermediates and inflammation."
            ),
            "circadian_metabolism": (
                "Circadian clock genes (CLOCK/BMAL1) regulate ~15% of "
                "transcriptome including metabolic genes. Disruption "
                "impairs glucose tolerance and lipid metabolism."
            ),
            "adipose_dysfunction": (
                "Adipose tissue dysfunction in obesity includes "
                "macrophage infiltration, adipokine dysregulation, "
                "and ectopic lipid deposition driving insulin resistance."
            ),
        }

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        context = context or {}
        hypothesis_lower = hypothesis.lower()

        # Match hypothesis to known physiology
        claim = ""
        evidence = ""
        level = EvidenceLevel.MODERATE
        confidence = 0.5

        if "insulin" in hypothesis_lower or "glucose" in hypothesis_lower:
            claim = (
                "The proposed mechanism is physiologically plausible. "
                "Insulin signaling integrates with multiple organ systems."
            )
            evidence = self._physiology_knowledge.get("insulin_resistance",
                        "Established physiological mechanism.")
            level = EvidenceLevel.MECHANISTIC
            confidence = 0.85
        elif "sleep" in hypothesis_lower or "circadian" in hypothesis_lower:
            claim = (
                "Circadian disruption has well-documented metabolic effects. "
                "The proposed pathway through cortisol is plausible."
            )
            evidence = self._physiology_knowledge.get("circadian_metabolism", "")
            level = EvidenceLevel.MECHANISTIC
            confidence = 0.8
        elif "inflammation" in hypothesis_lower:
            claim = (
                "Inflammatory pathways are physiologically linked to "
                "metabolic regulation through NF-κB and JNK signaling."
            )
            evidence = (
                "Inflammatory cytokines (TNF-α, IL-6) directly impair "
                "insulin signaling via serine phosphorylation of IRS-1."
            )
            level = EvidenceLevel.MECHANISTIC
            confidence = 0.9
        else:
            claim = (
                "The proposed mechanism requires further investigation. "
                "Current physiological models do not fully explain this pathway."
            )
            evidence = "Insufficient physiological precedent."
            level = EvidenceLevel.THEORETICAL
            confidence = 0.3

        return AgentArgument(
            agent_role=self.role,
            claim=claim,
            evidence=evidence,
            evidence_level=level,
            confidence=confidence,
        )


class PharmacologyAgent(SpecialistAgent):
    """Expert in drug mechanisms, PK/PD, and drug interactions."""

    def __init__(self):
        super().__init__(AgentRole.PHARMACOLOGY, "Pharmacology Agent",
                         "Drug mechanisms, PK/PD, interactions")
        self._drug_knowledge = {
            "metformin": (
                "Metformin activates AMPK, inhibits mitochondrial "
                "complex I, and reduces hepatic gluconeogenesis. "
                "t1/2=6h, primarily renal elimination."
            ),
            "empagliflozin": (
                "SGLT2 inhibitor reduces renal glucose reabsorption. "
                "Off-target effects include ketone body elevation "
                "and uric acid reduction. t1/2=12h."
            ),
            "atorvastatin": (
                "HMG-CoA reductase inhibitor. Reduces LDL by 30-50%. "
                "Pleiotropic effects include anti-inflammatory "
                "and endothelial function improvement."
            ),
        }

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        context = context or {}
        drug = context.get("drug", "").lower()
        hypothesis_lower = hypothesis.lower()

        if drug in self._drug_knowledge:
            return AgentArgument(
                agent_role=self.role,
                claim=f"The pharmacology of {drug} supports the proposed mechanism.",
                evidence=self._drug_knowledge[drug],
                evidence_level=EvidenceLevel.MECHANISTIC,
                confidence=0.85,
                citations=[f"DrugBank: {drug}"],
            )

        if "drug" in hypothesis_lower:
            return AgentArgument(
                agent_role=self.role,
                claim="Pharmacological plausibility depends on specific drug-target interactions.",
                evidence=(
                    "Drug effects are mediated by target engagement, "
                    "downstream signaling, and off-target effects. "
                    "In silico docking and pathway analysis are recommended."
                ),
                evidence_level=EvidenceLevel.MODERATE,
                confidence=0.6,
            )

        return AgentArgument(
            agent_role=self.role,
            claim="No pharmacological evidence directly supports this hypothesis.",
            evidence="No specific drug mechanism is implicated.",
            evidence_level=EvidenceLevel.THEORETICAL,
            confidence=0.3,
        )


class GenomicsAgent(SpecialistAgent):
    """Expert in genetic variants, transcriptomics, and epigenetics."""

    def __init__(self):
        super().__init__(AgentRole.GENOMICS, "Genomics Agent",
                         "Genetics, transcriptomics, epigenetics")
        self._gene_knowledge = {
            "TCF7L2": "Strongest T2D risk variant, affects insulin secretion",
            "PPARG": "Master regulator of adipogenesis, T2D drug target",
            "FTO": "Obesity risk variant, affects IRX3/IRX5 expression",
            "MTNR1B": "Melatonin receptor, links circadian rhythm to glucose",
        }

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        context = context or {}
        hypothesis_lower = hypothesis.lower()

        if "genetic" in hypothesis_lower or "gene" in hypothesis_lower:
            return AgentArgument(
                agent_role=self.role,
                claim="Genetic evidence can support or refute causal mechanisms.",
                evidence=(
                    "Mendelian randomization studies provide evidence for "
                    "causal relationships. eQTL and GWAS data can identify "
                    "causal variants and their target genes."
                ),
                evidence_level=EvidenceLevel.STRONG,
                confidence=0.75,
            )

        if "inflammation" in hypothesis_lower:
            return AgentArgument(
                agent_role=self.role,
                claim="Multiple inflammatory genes support this pathway.",
                evidence=(
                    "GWAS identifies IL6R, TNF, and NFKB1 variants "
                    "associated with inflammatory biomarkers and metabolic traits."
                ),
                evidence_level=EvidenceLevel.STRONG,
                confidence=0.7,
            )

        return AgentArgument(
            agent_role=self.role,
            claim="Genomic evidence for this specific hypothesis is limited.",
            evidence="No strong genetic associations currently reported.",
            evidence_level=EvidenceLevel.WEAK,
            confidence=0.3,
        )


class ClinicalAgent(SpecialistAgent):
    """Expert in patient outcomes, clinical trials, and guidelines."""

    def __init__(self):
        super().__init__(AgentRole.CLINICAL, "Clinical Agent",
                         "Patient outcomes, trials, guidelines")

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        context = context or {}
        hypothesis_lower = hypothesis.lower()

        if "exercise" in hypothesis_lower:
            return AgentArgument(
                agent_role=self.role,
                claim="Exercise interventions have strong clinical evidence.",
                evidence=(
                    "Meta-analyses of exercise trials show 20-40% "
                    "improvement in insulin sensitivity. Look AHEAD "
                    "trial demonstrated diabetes prevention."
                ),
                evidence_level=EvidenceLevel.STRONG,
                confidence=0.85,
            )

        if "diet" in hypothesis_lower:
            return AgentArgument(
                agent_role=self.role,
                claim="Dietary interventions show variable response.",
                evidence=(
                    "Mediterranean diet has strongest evidence for "
                    "cardiometabolic benefit. Individual variability "
                    "is high, suggesting responder subgroups."
                ),
                evidence_level=EvidenceLevel.STRONG,
                confidence=0.75,
            )

        return AgentArgument(
            agent_role=self.role,
            claim="Clinical trial evidence should be reviewed systematically.",
            evidence="No direct clinical trial evidence identified for this specific hypothesis.",
            evidence_level=EvidenceLevel.MODERATE,
            confidence=0.5,
        )


class ResearchAgent(SpecialistAgent):
    """Expert in scientific literature and prior discoveries."""

    def __init__(self):
        super().__init__(AgentRole.RESEARCH, "Research Agent",
                         "Literature, prior discoveries, meta-analysis")

    def reason(self, hypothesis: str, context: Dict[str, Any] = None) -> AgentArgument:
        context = context or {}
        hypothesis_lower = hypothesis.lower()

        # Simulated literature search
        literature_evidence = self._search_literature(hypothesis)

        if literature_evidence:
            return AgentArgument(
                agent_role=self.role,
                claim="Supporting evidence found in published literature.",
                evidence=literature_evidence,
                evidence_level=EvidenceLevel.MODERATE,
                confidence=0.7,
                citations=["Simulated PubMed search"],
            )

        return AgentArgument(
            agent_role=self.role,
            claim="No direct literature support found.",
            evidence="This may represent a novel hypothesis.",
            evidence_level=EvidenceLevel.THEORETICAL,
            confidence=0.3,
        )

    def _search_literature(self, hypothesis: str) -> str:
        """Simulate literature search for supporting evidence."""
        hits = {
            "sleep": (
                "Annals of Internal Medicine (2024): Sleep restriction "
                "reduces insulin sensitivity by 25%. "
                "Diabetes Care (2023): Poor sleep quality predicts "
                "HbA1c elevation independent of diet and exercise."
            ),
            "inflammation": (
                "Nature Immunology (2024): Adipose tissue macrophages "
                "drive systemic insulin resistance via TNF-α secretion. "
                "Cell Metabolism (2023): IL-6 trans-signaling mediates "
                "obesity-induced metabolic dysfunction."
            ),
            "circadian": (
                "Science (2024): Circadian clock disruption promotes "
                "NAFLD through REV-ERBα dysregulation. "
                "Cell (2023): Time-restricted feeding restores "
                "circadian metabolic rhythms."
            ),
        }
        for keyword, evidence in hits.items():
            if keyword in hypothesis.lower():
                return evidence
        return ""


# ── Multi-Agent System ────────────────────────────────────────

class MultiAgentSystem:
    """
    Orchestrates multi-agent reasoning and debate.

    Given a hypothesis or question, each agent produces a structured
    argument. The system then facilitates debate and produces a
    consensus conclusion.
    """

    def __init__(self):
        self.agents: Dict[AgentRole, SpecialistAgent] = {
            AgentRole.PHYSIOLOGY: PhysiologyAgent(),
            AgentRole.PHARMACOLOGY: PharmacologyAgent(),
            AgentRole.GENOMICS: GenomicsAgent(),
            AgentRole.CLINICAL: ClinicalAgent(),
            AgentRole.RESEARCH: ResearchAgent(),
        }
        self._debate_history: List[AgentDebate] = []

    def evaluate_hypothesis(
        self,
        hypothesis: str,
        context: Dict[str, Any] = None,
        participating_agents: Optional[List[AgentRole]] = None,
    ) -> ConsensusResult:
        """
        Evaluate a scientific hypothesis through multi-agent reasoning.

        Args:
            hypothesis: The hypothesis to evaluate
            context: Additional context (drug name, population, etc.)
            participating_agents: Which agents to consult (default: all)

        Returns:
            ConsensusResult with conclusion and confidence
        """
        context = context or {}
        if participating_agents is None:
            participating_agents = list(AgentRole)

        debate = AgentDebate(
            topic="Hypothesis Evaluation",
            hypothesis=hypothesis,
        )

        for role in participating_agents:
            agent = self.agents.get(role)
            if agent:
                argument = agent.reason(hypothesis, context)
                debate.add_argument(argument)

        self._debate_history.append(debate)

        return self._resolve_debate(debate)

    def debate_mechanism(
        self,
        cause: str, effect: str,
        proposed_mechanism: str,
    ) -> ConsensusResult:
        """
        Convenience: debate a specific causal mechanism.

        Args:
            cause: The proposed cause variable
            effect: The proposed effect variable
            proposed_mechanism: Description of the mechanism

        Returns:
            ConsensusResult
        """
        hypothesis = (
            f"Does {cause} cause {effect} through {proposed_mechanism}?"
        )
        return self.evaluate_hypothesis(hypothesis, {
            "cause": cause, "effect": effect, "mechanism": proposed_mechanism,
        })

    def _resolve_debate(self, debate: AgentDebate) -> ConsensusResult:
        """
        Resolve a debate by aggregating agent arguments.

        Uses weighted voting based on confidence and evidence level.
        """
        if not debate.arguments:
            return ConsensusResult(
                topic=debate.topic,
                conclusion="No arguments were generated.",
                overall_confidence=0.0,
                supporting_arguments=[],
                opposing_arguments=[],
                remaining_uncertainties=["Insufficient data"],
                recommended_actions=["Gather more evidence"],
                agent_votes={},
                debate_summary="No agents participated.",
            )

        # Weight by evidence level
        level_weights = {
            EvidenceLevel.MECHANISTIC: 1.0,
            EvidenceLevel.STRONG: 0.8,
            EvidenceLevel.MODERATE: 0.5,
            EvidenceLevel.WEAK: 0.3,
            EvidenceLevel.THEORETICAL: 0.1,
        }

        total_weight = 0.0
        weighted_confidence = 0.0
        supporting = []
        opposing = []
        uncertainties = []
        agent_votes = {}

        for arg in debate.arguments:
            w = level_weights.get(arg.evidence_level, 0.5) * arg.confidence
            total_weight += w
            weighted_confidence += w * arg.confidence
            agent_votes[arg.agent_role.value] = arg.confidence

            if arg.confidence >= 0.5:
                supporting.append(str(arg))
            else:
                opposing.append(str(arg))

            if arg.evidence_level in (EvidenceLevel.WEAK, EvidenceLevel.THEORETICAL):
                uncertainties.append(
                    f"{arg.agent_role.value}: {arg.claim} "
                    f"(limited evidence)"
                )

        overall_confidence = weighted_confidence / max(total_weight, 1e-10)

        # Generate conclusion
        if overall_confidence >= 0.7:
            conclusion = (
                f"The hypothesis is supported by multi-agent consensus "
                f"(confidence={overall_confidence:.0%}). "
                f"{len(supporting)} agents support, "
                f"{len(opposing)} agents express concerns."
            )
        elif overall_confidence >= 0.4:
            conclusion = (
                f"The hypothesis has moderate support "
                f"(confidence={overall_confidence:.0%}). "
                f"Further investigation is warranted."
            )
        else:
            conclusion = (
                f"The hypothesis has limited support "
                f"(confidence={overall_confidence:.0%}). "
                f"Alternative mechanisms should be considered."
            )

        debate.consensus = conclusion
        debate.confidence = overall_confidence
        debate.resolved_at = time.time()

        return ConsensusResult(
            topic=debate.topic,
            conclusion=conclusion,
            overall_confidence=overall_confidence,
            supporting_arguments=supporting,
            opposing_arguments=opposing,
            remaining_uncertainties=uncertainties,
            recommended_actions=self._generate_recommendations(debate),
            agent_votes=agent_votes,
            debate_summary=self._format_debate(debate),
        )

    def _generate_recommendations(self, debate: AgentDebate) -> List[str]:
        """Generate actionable recommendations from a debate."""
        recommendations = []
        uncertainties = [a for a in debate.arguments
                        if a.evidence_level in (EvidenceLevel.WEAK, EvidenceLevel.THEORETICAL)]

        if uncertainties:
            recommendations.append(
                "Design targeted experiments to address evidence gaps "
                f"identified by {len(uncertainties)} agent(s)."
            )
        else:
            recommendations.append(
                "Consider in-silico twin simulation to validate the hypothesis."
            )

        recommendations.append("Review latest literature for emergent evidence.")
        return recommendations

    def _format_debate(self, debate: AgentDebate) -> str:
        lines = [f"=== Debate: {debate.hypothesis} ===\n"]
        for arg in debate.arguments:
            lines.append(f"  {arg}")
            lines.append(f"    Evidence: {arg.evidence[:100]}...")
            lines.append(f"    Level: {arg.evidence_level.value}")
        if debate.consensus:
            lines.append(f"\nConsensus: {debate.consensus}")
            lines.append(f"Confidence: {debate.confidence:.0%}")
        return "\n".join(lines)


# ── Convenience ───────────────────────────────────────────────

def create_default_agent_system() -> MultiAgentSystem:
    """Create a fully configured multi-agent system."""
    return MultiAgentSystem()
