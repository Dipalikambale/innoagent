"""
agent_pipeline.py
Multi-Agent Orchestrator for the InnoAgent Design Thinking Pipeline.
Manages the end-to-end execution across 5 specialized subagents:
Empathize -> Define -> Ideate -> Evaluate -> Prototype.
Supports dynamic custom inputs and optional live LLM reasoning.
"""

from typing import Dict, Any, List, Optional
from tools import (
    EmpathyAnalysisTool,
    ProblemFramingTool,
    MorphologicalIdeationTool,
    PughMatrixEvaluatorTool,
    BOMPrototypingTool,
    extract_domain_keywords
)


class InnoAgentOrchestrator:
    """
    Autonomous multi-agent controller executing the Stanford d.school Design Thinking pipeline.
    """
    def __init__(self):
        self.empathy_tool = EmpathyAnalysisTool()
        self.framing_tool = ProblemFramingTool()
        self.ideation_tool = MorphologicalIdeationTool()
        self.pugh_tool = PughMatrixEvaluatorTool()
        self.bom_tool = BOMPrototypingTool()

    def run_pipeline(self, qualitative_notes: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the multi-agent design thinking workflow.
        """
        agent_trace: List[Dict[str, Any]] = []

        # Extract domain context keywords
        info = extract_domain_keywords(qualitative_notes)
        topic_hint = info["topic"]

        # -----------------------------------------------------------------
        # STAGE 1: EMPATHIZE (User Insight Agent)
        # -----------------------------------------------------------------
        empathy_map = self.empathy_tool.execute(qualitative_notes, api_key=api_key)
        agent_trace.append({
            "stage": "1. Empathize",
            "agent": "User Insight & Empathy Miner",
            "action": "Parsed unstructured notes into 4-quadrant Empathy Map (Says, Thinks, Does, Feels).",
            "key_finding": f"Identified persona '{empathy_map.persona_title}' with {len(empathy_map.core_pain_points)} primary pain points and {len(empathy_map.latent_unmet_needs)} latent needs."
        })

        # -----------------------------------------------------------------
        # STAGE 2: DEFINE (Problem Framing Agent)
        # -----------------------------------------------------------------
        problem_def = self.framing_tool.execute(empathy_map, api_key=api_key)
        agent_trace.append({
            "stage": "2. Define",
            "agent": "POV & Problem Framing Agent",
            "action": "Formulated structured Point-of-View (POV) statement and 4 divergent 'How Might We' spaces.",
            "key_finding": f"POV Statement: {problem_def.pov_statement}"
        })

        # -----------------------------------------------------------------
        # STAGE 3: IDEATE (Divergent Architecture Agent)
        # -----------------------------------------------------------------
        concepts = self.ideation_tool.execute(problem_def, topic_hint=topic_hint, api_key=api_key)
        agent_trace.append({
            "stage": "3. Ideate",
            "agent": "Morphological & TRIZ Ideator",
            "action": f"Generated 3 distinct architectural variants tailored for '{topic_hint}'.",
            "key_finding": f"Explored concepts: {', '.join([c.title for c in concepts])}."
        })

        # -----------------------------------------------------------------
        # STAGE 4: EVALUATE (Pugh Decision Matrix Agent)
        # -----------------------------------------------------------------
        pugh_scores = self.pugh_tool.execute(concepts, api_key=api_key)
        winning_score = pugh_scores[0]
        winning_concept = next(c for c in concepts if c.concept_id == winning_score.concept_id)

        agent_trace.append({
            "stage": "4. Evaluate",
            "agent": "Pugh Decision Matrix Agent",
            "action": "Scored concepts across Desirability, Feasibility, Viability, and Maintainability against commercial baseline.",
            "key_finding": f"Winner: '{winning_concept.title}' with total score of +{winning_score.total_score}."
        })

        # -----------------------------------------------------------------
        # STAGE 5: PROTOTYPE (Fabrication Roadmap Agent)
        # -----------------------------------------------------------------
        prototyping_plan = self.bom_tool.execute(winning_concept, topic_hint=topic_hint, api_key=api_key)
        agent_trace.append({
            "stage": "5. Prototype",
            "agent": "Subsystem & Prototyping Architect",
            "action": "Synthesized preliminary Bill of Materials (BOM), functional blocks, and a 6-week rapid prototyping roadmap.",
            "key_finding": f"Preliminary BOM estimated at INR {prototyping_plan.total_estimated_bom_inr:,.2f} across {len(prototyping_plan.bill_of_materials)} line items."
        })

        return {
            "status": "DESIGN_SPECIFICATION_APPROVED",
            "domain_topic": topic_hint,
            "empathy_map": empathy_map.dict(),
            "problem_definition": problem_def.dict(),
            "concepts": [c.dict() for c in concepts],
            "pugh_evaluation": [s.dict() for s in pugh_scores],
            "winning_concept": winning_concept.dict(),
            "prototyping_plan": prototyping_plan.dict(),
            "agent_trace": agent_trace
        }
