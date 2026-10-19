"""
agent_pipeline.py
Multi-Agent Orchestrator — passes parsed context through all 5 agents.
"""

from typing import Dict, Any, List, Optional
from tools import (
    EmpathyAnalysisTool, ProblemFramingTool, MorphologicalIdeationTool,
    PughMatrixEvaluatorTool, BOMPrototypingTool, deep_parse
)


class InnoAgentOrchestrator:
    def __init__(self):
        self.empathy_tool = EmpathyAnalysisTool()
        self.framing_tool = ProblemFramingTool()
        self.ideation_tool = MorphologicalIdeationTool()
        self.pugh_tool = PughMatrixEvaluatorTool()
        self.bom_tool = BOMPrototypingTool()

    def run_pipeline(self, qualitative_notes: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        agent_trace: List[Dict[str, Any]] = []

        # Parse full context once and pass throughout
        ctx = deep_parse(qualitative_notes)

        # Stage 1: Empathize
        empathy_map = self.empathy_tool.execute(qualitative_notes, api_key=api_key)
        agent_trace.append({
            "stage": "1. Empathize",
            "agent": "User Insight & Empathy Miner",
            "action": "Parsed unstructured notes into 4-quadrant Empathy Map (Says, Thinks, Does, Feels).",
            "key_finding": f"Persona: '{empathy_map.persona_title}' | {len(empathy_map.core_pain_points)} pain points | {len(empathy_map.latent_unmet_needs)} latent needs identified."
        })

        # Stage 2: Define
        problem_def = self.framing_tool.execute(empathy_map, original_text=qualitative_notes, api_key=api_key)
        agent_trace.append({
            "stage": "2. Define",
            "agent": "POV & Problem Framing Agent",
            "action": "Formulated Point-of-View (POV) and 4 How-Might-We opportunity spaces.",
            "key_finding": f"POV: {problem_def.pov_statement}"
        })

        # Stage 3: Ideate — passes full ctx for domain-specific concepts
        concepts = self.ideation_tool.execute(problem_def, ctx=ctx, api_key=api_key)
        agent_trace.append({
            "stage": "3. Ideate",
            "agent": "Morphological & TRIZ Ideator",
            "action": f"Generated 3 distinct architectural variants for domain '{ctx['primary_domain']}'.",
            "key_finding": f"Concepts: {', '.join([c.title for c in concepts])}."
        })

        # Stage 4: Evaluate
        pugh_scores = self.pugh_tool.execute(concepts, ctx=ctx, api_key=api_key)
        winning_score = pugh_scores[0]
        winning_concept = next(c for c in concepts if c.concept_id == winning_score.concept_id)
        agent_trace.append({
            "stage": "4. Evaluate",
            "agent": "Pugh Decision Matrix Agent",
            "action": "Scored concepts across Desirability, Feasibility, Viability, and Maintainability.",
            "key_finding": f"Winner: '{winning_concept.title}' | Total Score: +{winning_score.total_score}."
        })

        # Stage 5: Prototype — passes full ctx for domain-specific BOM
        prototyping_plan = self.bom_tool.execute(winning_concept, ctx=ctx, api_key=api_key)
        agent_trace.append({
            "stage": "5. Prototype",
            "agent": "Subsystem & Prototyping Architect",
            "action": "Synthesized Bill of Materials (BOM) and 6-week Fab Lab prototyping roadmap.",
            "key_finding": f"BOM: INR {prototyping_plan.total_estimated_bom_inr:,.2f} across {len(prototyping_plan.bill_of_materials)} line items."
        })

        return {
            "status": "DESIGN_SPECIFICATION_APPROVED",
            "domain_topic": ctx["primary_domain"],
            "empathy_map": empathy_map.dict(),
            "problem_definition": problem_def.dict(),
            "concepts": [c.dict() for c in concepts],
            "pugh_evaluation": [s.dict() for s in pugh_scores],
            "winning_concept": winning_concept.dict(),
            "prototyping_plan": prototyping_plan.dict(),
            "agent_trace": agent_trace
        }
