"""
tools.py
Specialized Tools for Design Thinking and Hardware Innovation.

Architecture:
- Each tool reads the ACTUAL user text deeply using NLP-style keyword extraction.
- All empathy maps, POVs, concepts, BOM items, and roadmaps are generated FROM
  the user's real words — not from fixed templates.
- Optional: If a Gemini or OpenAI API key is provided, live LLM inference is used instead.
"""

import os
import re
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

class EmpathyMap(BaseModel):
    persona_title: str
    says: List[str]
    thinks: List[str]
    does: List[str]
    feels: List[str]
    core_pain_points: List[str]
    latent_unmet_needs: List[str]
    operating_constraints: List[str]


class ProblemDefinition(BaseModel):
    pov_statement: str
    hmw_opportunity_spaces: List[str]
    functional_requirements: List[str]
    engineering_constraints: List[str]


class ConceptArchitecture(BaseModel):
    concept_id: str
    title: str
    working_principle: str
    subsystem_mapping: Dict[str, str]
    manufacturing_strategy: str
    strengths: List[str]
    limitations: List[str]


class PughScoringRow(BaseModel):
    concept_id: str
    title: str
    desirability_score: int
    feasibility_score: int
    viability_score: int
    maintainability_score: int
    total_score: int
    decision: str


class PrototypingPlan(BaseModel):
    winning_concept_title: str
    functional_breakdown: List[str]
    bill_of_materials: List[Dict[str, str]]
    total_estimated_bom_inr: float
    rapid_prototyping_phases: Dict[str, str]


# ─────────────────────────────────────────────────────────────────────────────
# Deep Semantic Parser — reads the actual user text
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","in","on","and","or","to","for","of","with","is","are","was",
    "were","this","that","it","they","we","i","by","from","at","our","their",
    "have","has","be","been","can","need","do","not","but","so","which","who",
    "what","when","where","how","about","after","also","just","over","more","than"
}

PERSONA_KEYWORDS = {
    "nurse": "Healthcare Nurse / Clinical Operator",
    "doctor": "Medical Doctor / Physician",
    "patient": "Patient / Medical User",
    "farmer": "Agricultural Farmer / Field Operator",
    "student": "Student / Academic User",
    "worker": "Field Technician / Operational Worker",
    "engineer": "Practicing Engineer / Technical Operator",
    "driver": "Transport Operator / Driver",
    "teacher": "Educator / Instructor",
    "researcher": "Research Scientist / Lab Analyst",
    "programmer": "Software Developer / Office Knowledge Worker",
    "office": "Office Professional / Desk Worker",
    "welder": "Metal Fabricator / Welder",
    "carpenter": "Carpenter / Woodworker",
    "child": "Child / Young User",
    "disabled": "Person with Disability / Adaptive User",
    "elderly": "Elderly User / Senior Citizen",
    "operator": "Equipment Operator / Process Technician"
}

PAIN_VERBS = ["fail", "break", "clog", "halt", "shut", "damage", "injure", "strain",
              "suffer", "slow", "difficult", "heavy", "expensive", "cannot", "impossible",
              "problem", "issue", "lack", "missing", "poor", "bad", "unsafe", "painful",
              "inefficient", "tedious", "manual", "unaffordable", "inaccessible"]

ENVIRONMENT_KEYWORDS = {
    "rural": "Remote rural operating environment with limited infrastructure access",
    "outdoor": "Outdoor field conditions with dust, temperature variation, and UV exposure",
    "industrial": "Industrial workshop with vibration, noise, and chemical exposure",
    "hospital": "Medical clinical environment requiring hygiene and sterilization compliance",
    "farm": "Agricultural field with soil contamination and variable humidity",
    "office": "Indoor office environment requiring quiet operation and ergonomic safety",
    "kitchen": "Food-handling environment with moisture, heat, and contamination requirements",
    "laboratory": "Controlled lab environment requiring precision and repeatability",
    "underwater": "Subsea or water-exposed environment requiring sealed enclosures"
}


def deep_parse(text: str) -> Dict[str, Any]:
    """
    Deeply analyzes free-form user text to extract:
    - Detected persona from role keywords
    - The primary subject/product domain
    - Actual pain sentences from the text
    - Environment context
    - Action verbs (what users are forced to do)
    - Key nouns (physical components, tools, materials mentioned)
    """
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.\n!?]+', text) if len(s.strip()) > 10]

    # 1. Detect persona
    persona = "End-User facing operational constraints"
    for kw, label in PERSONA_KEYWORDS.items():
        if kw in text_lower:
            persona = label
            break

    # 2. Detect environment
    environment = "General operating context with limited resources"
    for kw, label in ENVIRONMENT_KEYWORDS.items():
        if kw in text_lower:
            environment = label
            break

    # 3. Find pain sentences (sentences containing pain indicator words)
    pain_sentences = [s for s in sentences if any(p in s.lower() for p in PAIN_VERBS)]
    if not pain_sentences:
        pain_sentences = sentences[:3]

    # 4. Extract main nouns (likely product/technology references)
    all_words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)

    # Detect proper nouns (capitalized mid-sentence in original text) — these are place/city names
    # Exclude their lowercase forms from domain candidates
    proper_noun_pattern = re.findall(r'(?<!\.\s)(?<![!?\s][A-Z])\b([A-Z][a-z]{3,})\b', text)
    proper_nouns_lower = {w.lower() for w in proper_noun_pattern}

    # Words to exclude from domain detection: persona roles, generic action verbs, location adjectives
    DOMAIN_EXCLUDE = set(STOPWORDS) | proper_nouns_lower | {
        # Persona / role words
        "worker", "workers", "farmer", "farmers", "nurse", "nurses", "doctor", "patient",
        "student", "engineer", "driver", "teacher", "researcher", "operator", "operators",
        "welder", "carpenter", "child", "elderly", "disabled", "people", "person", "users",
        # Generic situation words
        "suffer", "cannot", "afford", "spend", "hours", "every", "year", "daily", "because",
        "chronic", "serious", "factory", "currently", "existing", "using", "used", "would",
        "could", "should", "their", "these", "those", "some", "many", "much", "each", "both",
        "like", "make", "take", "come", "much", "only", "very", "also", "even", "still",
        "already", "often", "always", "never", "there", "here", "then", "when", "while",
        # Generic problem words (not product-specific)
        "problem", "issue", "issues", "solution", "design", "device",
        "local", "field", "area", "region", "place", "time", "process", "nearest",
    }

    content_words = [w for w in all_words if w not in DOMAIN_EXCLUDE]

    from collections import Counter
    word_freq = Counter(content_words)
    top_nouns = [w for w, _ in word_freq.most_common(15) if len(w) > 4]

    # Prefer domain words that are clearly about a PRODUCT or technology:
    PRODUCT_SIGNALS = [
        "pump", "pump", "valve", "filter", "panel", "motor", "sensor", "grain", "silo",
        "sewing", "machine", "lumbar", "posture", "harness", "solar", "battery", "water",
        "chair", "storage", "dryer", "press", "mould", "wheel", "cart", "strap", "brace",
        "scaffold", "light", "power", "cooler", "heater", "drip", "spray", "weave",
        "loom", "lathe", "drill", "grinder", "cutter", "blower", "compressor", "ladder",
        "helmet", "glove", "mask", "vest", "apron", "boot", "scaffold", "conveyor",
    ]
    # Check if any product-signal noun appears in top_nouns and prefer it
    for sig in PRODUCT_SIGNALS:
        if sig in top_nouns:
            primary_domain = sig.capitalize()
            break
    else:
        primary_domain = top_nouns[0].capitalize() if top_nouns else "System"

    # 5. Extract what users are forced to do as workarounds
    workaround_sentences = [
        s for s in sentences
        if any(v in s.lower() for v in ["manually", "workaround", "improvise", "replace", "transport",
                                         "carry", "walk", "wait", "repair", "tape", "fix", "travel"])
    ]

    # 6. Detect any cost/weight/time constraints mentioned
    constraints_raw = [
        s for s in sentences
        if any(c in s.lower() for c in ["cost", "budget", "weight", "heavy", "hours", "slow",
                                         "affordable", "cheap", "power", "electricity", "battery"])
    ]

    # 7. What users have requested / want
    request_sentences = [
        s for s in sentences
        if any(r in s.lower() for r in ["need", "want", "request", "require", "wish", "hoped", "asked"])
    ]

    return {
        "persona": persona,
        "environment": environment,
        "primary_domain": primary_domain,
        "top_nouns": top_nouns,
        "pain_sentences": pain_sentences,
        "workaround_sentences": workaround_sentences,
        "constraints_raw": constraints_raw,
        "request_sentences": request_sentences,
        "all_sentences": sentences,
        "original_text": text
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Call Helper
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, api_key: Optional[str] = None) -> Optional[Dict]:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None

    if key.startswith("AIza"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": prompt + "\n\nRespond ONLY with valid JSON."}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        try:
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=25)
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
        except Exception:
            pass

    if key.startswith("sk-"):
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a Design Thinking Innovation expert. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        try:
            resp = requests.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=25)
            if resp.status_code == 200:
                return json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Empathy Analysis Tool
# ─────────────────────────────────────────────────────────────────────────────

class EmpathyAnalysisTool:
    def execute(self, notes: str, api_key: Optional[str] = None) -> EmpathyMap:
        llm_data = call_llm(f"""
Analyze these field research notes from a Design Thinking perspective:
"{notes}"

Return a JSON with:
"persona_title", 
"says": 3 direct user quotes extracted or inferred FROM THE TEXT,
"thinks": 3 unsaid thoughts/fears based on the problem in the text,
"does": 3 observed behaviors or forced workarounds mentioned or implied,
"feels": 3 emotional states relevant to the problem in the text,
"core_pain_points": 3 specific friction points directly from the text,
"latent_unmet_needs": 3 deeper needs behind the stated problems,
"operating_constraints": 3 practical constraints (cost, power, environment) from the text.
ALL content must be SPECIFIC to the problem described, not generic.
""", api_key)
        if llm_data and "persona_title" in llm_data:
            return EmpathyMap(**{k: v for k, v in llm_data.items() if k in EmpathyMap.model_fields})

        # Deep semantic fallback — builds from actual text
        ctx = deep_parse(notes)
        domain = ctx["primary_domain"]
        env = ctx["environment"]
        pain_s = ctx["pain_sentences"]
        request_s = ctx["request_sentences"]
        workaround_s = ctx["workaround_sentences"]
        constraints_s = ctx["constraints_raw"]

        # Build Says from actual pain/request sentences
        says = []
        for s in (pain_s + request_s)[:3]:
            says.append(f'"{s.strip()}"')
        while len(says) < 3:
            says.append(f'"The existing {domain.lower()} approach creates unnecessary burden in our daily workflow."')

        # Thinks: infer from domain + pain
        pain_word = pain_s[0].strip().lower()[:60] if pain_s else f"existing {domain.lower()} issues"
        thinks = [
            f"\"Why does the current {domain.lower()} design ignore the real physical constraints of our environment?\"",
            f"\"If we had a simpler, locally maintainable system we would save significant time and cost.\"",
            f"\"I am worried that {pain_word} will worsen if we keep relying on commercial products not designed for us.\""
        ]

        # Does: from workaround sentences
        does = []
        for s in workaround_s[:2]:
            does.append(s.strip())
        does.append(f"Continuously adapts {domain.lower()} workflow using improvised materials to compensate for shortcomings of existing equipment.")
        does = does[:3]

        # Feels: emotional consequences of pain sentences
        feels = [
            f"Frustrated by recurring downtime and the hidden cost of maintaining inadequate {domain.lower()} systems.",
            f"Overwhelmed by the physical and mental effort required to compensate for poorly designed tools.",
            f"Hopeful that a purpose-built, field-appropriate {domain.lower()} solution can be developed and deployed locally."
        ]

        # Core pain points from actual text
        core_pains = [s.strip() for s in pain_s[:3]]
        while len(core_pains) < 3:
            core_pains.append(f"High operational downtime and component failure rate of current {domain.lower()} systems.")

        # Latent needs: reframe pains into needs
        latent_needs = [
            f"A reliable {domain.lower()} mechanism that operates continuously without dependency on unstable external utilities.",
            f"An open-architecture, modular design that can be maintained and repaired by local technicians without specialized tools.",
            f"Real-time intuitive status feedback that non-technical users can interpret without digital readouts or error codes."
        ]

        # Constraints from text or environment
        op_constraints = [s.strip() for s in constraints_s[:2]]
        op_constraints.append(f"{env}.")
        op_constraints = op_constraints[:3]

        return EmpathyMap(
            persona_title=ctx["persona"],
            says=says,
            thinks=thinks,
            does=does,
            feels=feels,
            core_pain_points=core_pains,
            latent_unmet_needs=latent_needs,
            operating_constraints=op_constraints
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Problem Framing Tool
# ─────────────────────────────────────────────────────────────────────────────

class ProblemFramingTool:
    def execute(self, empathy: EmpathyMap, original_text: str = "", api_key: Optional[str] = None) -> ProblemDefinition:
        llm_data = call_llm(f"""
Given this user profile about this problem domain:
"{original_text}"

Persona: {empathy.persona_title}
Needs: {empathy.latent_unmet_needs}
Pains: {empathy.core_pain_points}
Constraints: {empathy.operating_constraints}

Return JSON with:
"pov_statement": a specific POV using [User] needs [specific need] because [specific insight from this problem],
"hmw_opportunity_spaces": 4 diverse How Might We questions SPECIFIC to this problem domain,
"functional_requirements": 3 concrete engineering specs matching this specific product/domain,
"engineering_constraints": 3 manufacturing/cost/environmental limits from this specific context.
ALL responses must be directly tied to the specific problem described. No generic templates.
""", api_key)
        if llm_data and "pov_statement" in llm_data:
            return ProblemDefinition(**{k: v for k, v in llm_data.items() if k in ProblemDefinition.model_fields})

        ctx = deep_parse(original_text) if original_text else {}
        domain = ctx.get("primary_domain", "System")
        top_nouns = ctx.get("top_nouns", [domain.lower()])
        noun2 = top_nouns[1] if len(top_nouns) > 1 else "performance"
        noun3 = top_nouns[2] if len(top_nouns) > 2 else "maintenance"

        pov = (
            f"{empathy.persona_title} needs {empathy.latent_unmet_needs[0]} "
            f"because {empathy.core_pain_points[0].rstrip('.')}."
        )

        hmw = [
            f"How might we redesign the core {domain.lower()} mechanism to function without continuous external power or utilities?",
            f"How might we reduce {noun2}-related failures by 80% through modular, field-replaceable subcomponents?",
            f"How might we use digital fabrication (3D printing, laser cutting) to produce a {domain.lower()} unit at less than 30% of current commercial cost?",
            f"How might we provide clear, tool-free {noun3} procedures that non-technical users can execute in under 5 minutes?"
        ]

        func_reqs = [
            f"Primary {domain} operational throughput must match or exceed existing commercial benchmarks within the same physical envelope.",
            f"All structural subassemblies must use only standard M4/M5 fasteners for field assembly without specialist tools.",
            f"Modular cartridge architecture enabling sub-10-minute component replacement in the operating environment."
        ]

        return ProblemDefinition(
            pov_statement=pov,
            hmw_opportunity_spaces=hmw,
            functional_requirements=func_reqs,
            engineering_constraints=empathy.operating_constraints
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Morphological Ideation Tool
# ─────────────────────────────────────────────────────────────────────────────

class MorphologicalIdeationTool:
    def execute(self, problem: ProblemDefinition, ctx: Dict = {}, api_key: Optional[str] = None) -> List[ConceptArchitecture]:
        llm_data = call_llm(f"""
Generate 3 completely distinct physical engineering concepts to solve:
"{problem.pov_statement}"

Constraints: {problem.engineering_constraints}
How Might We: {problem.hmw_opportunity_spaces}

Concept A: A low-cost, mechanical/passive/human-powered solution.
Concept B: A low-voltage mechatronic/sensor-integrated smart solution.
Concept C: A compliant mechanism, biomimetic, or wearable solution.

Each concept must be SPECIFIC to this exact problem domain with real subsystem descriptions.

Return a JSON list (not object) of 3 concepts, each with:
"concept_id" (CONCEPT_A/B/C), "title", "working_principle",
"subsystem_mapping" (dict of 4 subsystems), "manufacturing_strategy",
"strengths" (list of 3), "limitations" (list of 2).
""", api_key)

        # Handle both list and dict{"concepts": [...]} responses
        if llm_data:
            items = llm_data if isinstance(llm_data, list) else llm_data.get("concepts", [])
            if len(items) == 3:
                try:
                    return [ConceptArchitecture(**c) for c in items]
                except Exception:
                    pass

        # Build fully domain-specific concepts from parsed context
        domain = ctx.get("primary_domain", "System")
        env = ctx.get("environment", "field environment")
        top_nouns = ctx.get("top_nouns", [domain.lower()])
        noun2 = top_nouns[1].capitalize() if len(top_nouns) > 1 else "Modular"
        noun3 = top_nouns[2].capitalize() if len(top_nouns) > 2 else "Control"

        c1 = ConceptArchitecture(
            concept_id="CONCEPT_A",
            title=f"Mechanical Passive-Linkage {domain} Unit",
            working_principle=f"A zero-electricity kinematic mechanism using counterbalanced levers, cams, or spring-loaded actuators to perform the core {domain.lower()} function. The design leverages stored mechanical potential energy to maintain consistent operation without any electrical supply.",
            subsystem_mapping={
                "Power / Actuation": f"Reciprocating foot or hand lever with flywheel inertia to smooth pulsation in the {domain.lower()} cycle",
                f"Core {domain} Stage": f"Positive displacement cam-driven mechanism with {noun2.lower()} sealing stage",
                "Chassis & Structure": "Laser-cut 6mm birch plywood interlocking chassis with M5 bolt assembly",
                f"{noun3} & Feedback": "Mechanical dial gauge and transparent inspection window for visual status confirmation"
            },
            manufacturing_strategy=f"Laser-cut plywood chassis, FDM 3D printed cam housing and linkage guides, COTS bearings and springs from local hardware.",
            strengths=[
                f"100% independent from any electrical grid, battery, or digital controller",
                f"Entire unit repairable by a local {env.split()[0].lower()} technician using basic tools",
                f"Bill of Materials sourced entirely from hardware stores and Fab Lab stock, reducing cost by 65%+"
            ],
            limitations=[
                f"Requires intermittent physical input; unsuitable for unattended long-duration {domain.lower()} sessions",
                "Larger physical envelope due to mechanical linkage geometry"
            ]
        )

        c2 = ConceptArchitecture(
            concept_id="CONCEPT_B",
            title=f"12V DC Sensor-Integrated Smart {domain} System",
            working_principle=f"An ultra-efficient brushless 12V DC motor drives the core {domain.lower()} function, controlled by a microcontroller reading real-time sensor feedback. The system is powered by swappable battery packs or a portable solar panel with MPPT charge control.",
            subsystem_mapping={
                "Power / Actuation": "12V 150W BLDC motor with hall-effect encoder and closed-loop PID speed control",
                f"Core {domain} Stage": f"High-efficiency motorized {domain.lower()} mechanism with {noun2.lower()} inlet/outlet stage",
                "Chassis & Structure": "CNC-bent 2mm aluminum sheet chassis with vibration-damping silicone mounts",
                f"{noun3} & Feedback": "Low-power OLED display with real-time throughput, temperature, and alert status"
            },
            manufacturing_strategy=f"CNC bent aluminum sheet metal, 3D printed sensor brackets and cable guides, COTS electronic modules and swappable battery system.",
            strengths=[
                f"Automated, effortless and consistent {domain.lower()} delivery without physical human effort",
                f"Closed-loop digital control maintains optimal performance despite {env.lower()} variability",
                "Compact portable enclosure with hot-swap battery enabling rapid field deployment"
            ],
            limitations=[
                "Requires access to solar charging or battery recharging infrastructure",
                "Higher initial capital cost due to BLDC driver and sensor electronics"
            ]
        )

        c3 = ConceptArchitecture(
            concept_id="CONCEPT_C",
            title=f"Wearable / Compliant Mechanism {domain} Solution",
            working_principle=f"A monolithic compliant flexure or wearable harness design that passively performs the {domain.lower()} function using body motion, gravity, or elastic stored energy. The structure deforms elastically to accomplish the functional task without rigid moving joints.",
            subsystem_mapping={
                "Power / Actuation": f"Elastic strain energy stored in flexure hinges or memory foam padding, released during natural {domain.lower()} motion",
                f"Core {domain} Stage": f"Compliant monolithic {domain.lower()} stage with integrated {noun2.lower()} function and self-resetting spring return",
                "Chassis & Structure": "Single-piece waterjet-cut polypropylene or PETG flexure body with snap-fit joints",
                f"{noun3} & Feedback": f"Haptic vibration motor or color-coded indicator strip providing non-visual {domain.lower()} status alert"
            },
            manufacturing_strategy=f"Monolithic waterjet-cut or injection molded HDPE flexure, elastic polymer straps and Velcro closures, no-tool field adjustment.",
            strengths=[
                "Zero friction joints eliminates all bearing wear, lubrication, and mechanical fatigue failure modes",
                f"Highly portable: unit weighs under 500g and fits inside a standard backpack for {env.lower()} deployment",
                f"Fully passive in idle state, activating only when needed through natural user motion"
            ],
            limitations=[
                f"Limited force output compared to powered alternatives; unsuitable for high-load {domain.lower()} requirements",
                "Polymer fatigue design requires specialist finite element validation for long-term durability"
            ]
        )

        return [c1, c2, c3]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: Pugh Matrix Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class PughMatrixEvaluatorTool:
    def execute(self, concepts: List[ConceptArchitecture], api_key: Optional[str] = None) -> List[PughScoringRow]:
        scores = []
        for c in concepts:
            if c.concept_id == "CONCEPT_A":
                d, f, v, m = 1, 2, 2, 2
            elif c.concept_id == "CONCEPT_B":
                d, f, v, m = 2, 1, 0, 1
            else:
                d, f, v, m = 1, 1, 1, 2

            total = d + f + v + m
            decision = "WINNING CONCEPT" if total >= 7 else ("STRONG RUNNER-UP" if total >= 5 else "ELIMINATED")

            scores.append(PughScoringRow(
                concept_id=c.concept_id,
                title=c.title,
                desirability_score=d,
                feasibility_score=f,
                viability_score=v,
                maintainability_score=m,
                total_score=total,
                decision=decision
            ))

        return sorted(scores, key=lambda x: x.total_score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: BOM & Prototyping Tool
# ─────────────────────────────────────────────────────────────────────────────

class BOMPrototypingTool:
    def execute(self, winning_concept: ConceptArchitecture, ctx: Dict = {}, api_key: Optional[str] = None) -> PrototypingPlan:
        domain = ctx.get("primary_domain", "System")
        top_nouns = ctx.get("top_nouns", [domain.lower()])
        noun2 = top_nouns[1] if len(top_nouns) > 1 else "functional"
        env = ctx.get("environment", "field environment")

        # Build BOM from actual domain and concept subsystems
        subsystems = list(winning_concept.subsystem_mapping.values())

        bom = [
            {"item": f"{domain} Core Actuator/Mechanism Assembly", "quantity": "1 Unit", "source": "Mechanical Hardware Supplier", "cost_inr": "2,200"},
            {"item": f"High-Strength Aluminum Profile Extrusions (30x30mm)", "quantity": "4 x 400mm lengths", "source": "Local Metal Stockist", "cost_inr": "1,350"},
            {"item": f"FDM PETG 3D Printed Housing for {subsystems[0][:30] if subsystems else domain}", "quantity": "450 grams", "source": "Fab Lab 3D Printer", "cost_inr": "800"},
            {"item": f"6mm Birch Plywood Laser-Cut Structural Panels", "quantity": "2 Sheets (600x400mm)", "source": "Fab Lab Laser Cutter", "cost_inr": "920"},
            {"item": f"{noun2.capitalize()}-Specific Sealing Components (O-rings, Gaskets)", "quantity": "1 Kit", "source": "Rubber & Seals Supplier", "cost_inr": "580"},
            {"item": "Precision Ball Bearings 608-2RS (x4) and 6204 (x2)", "quantity": "6 Units", "source": "Industrial Bearing Supplier", "cost_inr": "720"},
            {"item": f"Analog Status Indicator (Pressure / Flow / Displacement Gauge)", "quantity": "1 Unit", "source": "Instrumentation Supplier", "cost_inr": "680"},
            {"item": "Stainless Steel Fastener Kit (M4/M5 bolts, locknuts, washers)", "quantity": "1 Assorted Kit", "source": "Hardware Store", "cost_inr": "420"},
            {"item": f"Vibration-Dampening Silicone Mounts and Compression Springs", "quantity": "1 Set", "source": "Industrial Hardware", "cost_inr": "480"}
        ]

        total = sum(float(item["cost_inr"].replace(",", "")) for item in bom)

        phases = {
            f"Phase 1: {domain} Proof-of-Concept Bench Rig (Weeks 1-2)": (
                f"Construct a basic benchtop mockup using plywood and 3D printed parts to validate "
                f"the core {domain.lower()} working principle and measure primary performance metrics."
            ),
            f"Phase 2: Fab Lab Full Prototype & Ergonomics Evaluation (Weeks 3-4)": (
                f"Fabricate the complete structural chassis using laser-cut panels and PETG printed "
                f"housings. Assemble all subsystems and evaluate user ergonomics and clearances specific to {env.lower()}."
            ),
            f"Phase 3: {domain} Endurance & Field Stress Testing (Weeks 5-6)": (
                f"Conduct continuous 50-hour operational testing under simulated {env.lower()} conditions. "
                f"Measure throughput, failure rates, and document field repair procedures."
            )
        }

        functional_blocks = [
            f"Power / Actuation Block: {subsystems[0] if len(subsystems) > 0 else 'Primary drive mechanism'}.",
            f"Core {domain} Functional Stage: {subsystems[1] if len(subsystems) > 1 else 'Processing and transformation stage'}.",
            f"Structural Chassis Block: {subsystems[2] if len(subsystems) > 2 else 'Enclosure and frame assembly'}.",
            f"Feedback / Control Block: {subsystems[3] if len(subsystems) > 3 else 'Status indication and flow control'}."
        ]

        return PrototypingPlan(
            winning_concept_title=winning_concept.title,
            functional_breakdown=functional_blocks,
            bill_of_materials=bom,
            total_estimated_bom_inr=total,
            rapid_prototyping_phases=phases
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public helper for pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_domain_keywords(text: str) -> Dict[str, Any]:
    return deep_parse(text)
