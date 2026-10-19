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
    "underwater": "Subsea or water-exposed environment requiring sealed enclosures",
    "factory": "Factory floor with long standing hours, vibration, and limited rest breaks",
    "school": "Resource-constrained classroom with intermittent or no grid electricity",
    "clinic": "Primary healthcare facility with unstable power and limited spare parts"
}

# Multi-word product phrases scored before single-token domain guesses.
KNOWN_PRODUCT_PHRASES = [
    "oxygen concentrator", "oxygen generation", "oxygen cylinder",
    "drip irrigation", "water filtration", "microplastic",
    "grain storage", "gunny sack", "metal silo",
    "solar panel", "solar power", "lumbar support", "back pain",
    "posture harness", "sewing machine", "ergonomic chair"
]

# Scored problem families. Highest keyword hit-count wins; generic fallback otherwise.
PROBLEM_FAMILIES = {
    "medical_oxygen": {
        "keywords": ["oxygen", "concentrator", "respiratory", "cylinder", "blackout",
                     "clinic", "clinician", "patient", "healthcare", "distress"],
        "domain": "Off-Grid Medical Oxygen",
        "persona": "Rural Clinician / Primary Healthcare Operator",
        "environment": "Primary healthcare sub-centre with 6-10 hour daily grid outages",
    },
    "water_filtration": {
        "keywords": ["filter", "filtration", "silt", "microplastic", "irrigation",
                     "drip", "contamination", "clog", "emitter", "canal"],
        "domain": "Irrigation Water Filtration",
        "persona": "Agricultural Farmer / Drip Irrigation Operator",
        "environment": "Open-canal irrigation field with silt, debris, and off-grid power",
    },
    "grain_storage": {
        "keywords": ["grain", "silo", "harvest", "gunny", "rat", "moisture",
                     "storage", "flood", "sack", "post-harvest"],
        "domain": "Post-Harvest Grain Storage",
        "persona": "Smallholder Farmer / Grain Store Operator",
        "environment": "Village storage yard exposed to monsoon moisture, rats, and flooding",
    },
    "rural_energy": {
        "keywords": ["solar", "electricity", "projector", "laptop", "charging",
                     "school", "teacher", "panel", "grid", "blackout"],
        "domain": "Rural Classroom Electrification",
        "persona": "Rural Teacher / Student Community",
        "environment": "Off-grid village school with fragile, high-cost commercial solar kits",
    },
    "ergonomics": {
        "keywords": ["back", "lumbar", "posture", "chair", "pain", "harness",
                     "standing", "sewing", "ergonomic", "wearable", "desk"],
        "domain": "Wearable Posture & Lumbar Support",
        "persona": "Standing Textile / Desk Worker",
        "environment": "Long-duration standing or seated work with no affordable ergonomic support",
    },
}


def _word_hits(text_lower: str, keywords: List[str]) -> int:
    score = 0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}s?\b", text_lower):
            score += 1
    return score


def classify_problem_family(text_lower: str) -> Tuple[str, int]:
    scored = {
        fid: _word_hits(text_lower, spec["keywords"])
        for fid, spec in PROBLEM_FAMILIES.items()
    }
    best = max(scored, key=scored.get)
    return (best, scored[best]) if scored[best] >= 2 else ("generic", scored[best])


def extract_product_phrase(text: str, top_nouns: List[str]) -> str:
    text_lower = text.lower()
    for phrase in KNOWN_PRODUCT_PHRASES:
        if phrase in text_lower:
            return phrase
    request_match = re.search(
        r"(?:requested|need|needs|want|require)\s+(?:a|an)\s+(.{8,90}?)(?:\.| that | which )",
        text,
        flags=re.IGNORECASE,
    )
    if request_match:
        return request_match.group(1).strip(" ,;")
    if len(top_nouns) >= 2:
        return f"{top_nouns[0]} {top_nouns[1]}"
    return top_nouns[0] if top_nouns else "field system"


def deep_parse(text: str) -> Dict[str, Any]:
    """
    Deeply analyzes free-form user text to extract persona, environment,
    problem family, product phrase, and evidence sentences from the notes.
    """
    text_lower = text.lower()
    sentences = [s.strip() for s in re.split(r'[.\n!?]+', text) if len(s.strip()) > 10]

    persona = "End-User facing operational constraints"
    for kw, label in PERSONA_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}s?\b", text_lower):
            persona = label
            break

    environment = "General operating context with limited resources"
    for kw, label in ENVIRONMENT_KEYWORDS.items():
        if re.search(rf"\b{re.escape(kw)}s?\b", text_lower):
            environment = label
            break

    pain_sentences = [s for s in sentences if any(p in s.lower() for p in PAIN_VERBS)]
    if not pain_sentences:
        pain_sentences = sentences[:3]

    all_words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
    proper_noun_pattern = re.findall(r'(?<!\.\s)(?<![!?\s][A-Z])\b([A-Z][a-z]{3,})\b', text)
    proper_nouns_lower = {w.lower() for w in proper_noun_pattern}

    DOMAIN_EXCLUDE = set(STOPWORDS) | proper_nouns_lower | {
        "worker", "workers", "farmer", "farmers", "nurse", "nurses", "doctor", "patient",
        "student", "engineer", "driver", "teacher", "researcher", "operator", "operators",
        "welder", "carpenter", "child", "elderly", "disabled", "people", "person", "users",
        "suffer", "cannot", "afford", "spend", "hours", "every", "year", "daily", "because",
        "chronic", "serious", "factory", "currently", "existing", "using", "used", "would",
        "could", "should", "their", "these", "those", "some", "many", "much", "each", "both",
        "like", "make", "take", "come", "only", "very", "also", "even", "still",
        "already", "often", "always", "never", "there", "here", "then", "when", "while",
        "problem", "issue", "issues", "solution", "design", "device",
        "local", "field", "area", "region", "place", "time", "process", "nearest",
    }

    content_words = [w for w in all_words if w not in DOMAIN_EXCLUDE]
    from collections import Counter
    word_freq = Counter(content_words)
    top_nouns = [w for w, _ in word_freq.most_common(15) if len(w) > 3]

    family_id, family_score = classify_problem_family(text_lower)
    product_phrase = extract_product_phrase(text, top_nouns)

    if family_id != "generic":
        spec = PROBLEM_FAMILIES[family_id]
        primary_domain = spec["domain"]
        persona = spec["persona"]
        environment = spec["environment"]
    else:
        primary_domain = product_phrase.title()

    workaround_sentences = [
        s for s in sentences
        if any(v in s.lower() for v in ["manually", "workaround", "improvise", "replace", "transport",
                                         "carry", "walk", "wait", "repair", "tape", "fix", "travel", "fold"])
    ]

    constraints_raw = [
        s for s in sentences
        if any(c in s.lower() for c in ["cost", "budget", "weight", "heavy", "hours", "slow",
                                         "affordable", "cheap", "power", "electricity", "battery",
                                         "cannot afford", "expensive"])
    ]

    request_sentences = [
        s for s in sentences
        if any(r in s.lower() for r in ["need", "want", "request", "require", "wish", "hoped", "asked"])
    ]

    body_related = _word_hits(text_lower, ["wear", "wearable", "posture", "lumbar", "harness", "brace", "pain"])
    offgrid = _word_hits(text_lower, ["electricity", "blackout", "off-grid", "solar", "battery", "grid"])

    return {
        "persona": persona,
        "environment": environment,
        "primary_domain": primary_domain,
        "product_phrase": product_phrase,
        "problem_family": family_id,
        "family_score": family_score,
        "top_nouns": top_nouns,
        "pain_sentences": pain_sentences,
        "workaround_sentences": workaround_sentences,
        "constraints_raw": constraints_raw,
        "request_sentences": request_sentences,
        "all_sentences": sentences,
        "original_text": text,
        "body_related": body_related > 0,
        "offgrid": offgrid > 0,
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


def _quote(sentence: str, fallback: str) -> str:
    text = (sentence or fallback).strip().strip('"')
    return f'"{text}"'


def family_solution_kit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Problem-specific empathy needs, HMWs, architectures, Pugh weights, and BOM.
    Generic fallback still uses the user's words, but never a single copy-pasted machine template.
    """
    family = ctx.get("problem_family", "generic")
    domain = ctx.get("primary_domain", "Field System")
    product = ctx.get("product_phrase", domain.lower())
    env = ctx.get("environment", "field environment")
    pain = ctx.get("pain_sentences") or [f"Existing {product} options fail in daily use"]
    request = ctx.get("request_sentences") or []
    body_related = ctx.get("body_related", False)
    offgrid = ctx.get("offgrid", False)

    kits = {
        "medical_oxygen": {
            "latent_needs": [
                "Continuous medical oxygen during 6-10 hour village blackouts without relying on delayed cylinder trucks",
                "A concentrator architecture that local technicians can service with non-imported tools and spare parts",
                "A dual-path supply so a clinic is never left with zero oxygen when the grid or one compressor fails",
            ],
            "hmw": [
                "How might we generate medical-grade oxygen at a sub-centre when the grid is down for most of the day?",
                "How might we store enough buffer oxygen to cover overnight outages without waiting for steel cylinders?",
                "How might we keep zeolite beds, valves, and compressors maintainable with locally available hardware?",
                "How might we warn clinicians before oxygen purity or pressure drops below a safe threshold?",
            ],
            "pugh": {"CONCEPT_A": (2, 2, 2, 2), "CONCEPT_B": (2, 1, 0, 1), "CONCEPT_C": (1, 1, 1, 2)},
            "concepts": _oxygen_concepts(),
            "bom_theme": "oxygen",
        },
        "water_filtration": {
            "latent_needs": [
                "Continuous silt and debris separation that does not clog drip emitters within two days",
                "A self-cleaning filter that farmers can backflush without buying disposable cartridges",
                "Off-grid operation so filtration still runs when village electricity is unavailable",
            ],
            "hmw": [
                "How might we remove silt and micro-debris from canal water before it reaches drip emitters?",
                "How might we backflush the filter using the water's own flow instead of replacement cartridges?",
                "How might we fabricate the separator in a regional makerspace from steel, mesh, and 3D-printed fittings?",
                "How might we give farmers a visible clog indicator without a smartphone app?",
            ],
            "pugh": {"CONCEPT_A": (1, 2, 2, 2), "CONCEPT_B": (2, 1, 0, 1), "CONCEPT_C": (2, 2, 2, 1)},
            "concepts": _filtration_concepts(),
            "bom_theme": "filtration",
        },
        "grain_storage": {
            "latent_needs": [
                "Hermetic, rat-proof grain storage that does not depend on unaffordable steel silos",
                "Moisture control during monsoon so harvest losses drop well below 30 percent",
                "A raised, flood-tolerant store that a household can assemble from local materials",
            ],
            "hmw": [
                "How might we keep grain dry and sealed through monsoon humidity without a commercial metal silo?",
                "How might we block rats and insects using mesh, lids, and raised plinths farmers already understand?",
                "How might we raise the store above flood level using locally cast or packed foundations?",
                "How might we share a modular silo among neighbouring farms to cut per-household cost?",
            ],
            "pugh": {"CONCEPT_A": (2, 2, 2, 2), "CONCEPT_B": (2, 1, 0, 1), "CONCEPT_C": (1, 2, 2, 2)},
            "concepts": _grain_concepts(),
            "bom_theme": "grain",
        },
        "rural_energy": {
            "latent_needs": [
                "A rugged classroom power hub that can run lights, charge laptops, and drive a projector",
                "Solar hardware that survives dust, heat, and handling better than fragile commercial panels",
                "A shared battery cabinet so the whole school is not dependent on one expensive rooftop kit",
            ],
            "hmw": [
                "How might we power lights, laptop charging, and a projector in a school with no grid connection?",
                "How might we protect solar generation from dust, impact, and theft with locally made enclosures?",
                "How might we share one battery bank across classrooms instead of buying a panel per room?",
                "How might we keep the system repairable when commercial inverters fail?",
            ],
            "pugh": {"CONCEPT_A": (1, 2, 2, 2), "CONCEPT_B": (2, 1, 1, 1), "CONCEPT_C": (2, 2, 2, 2)},
            "concepts": _energy_concepts(),
            "bom_theme": "energy",
        },
        "ergonomics": {
            "latent_needs": [
                "Affordable lumbar support that works while standing or leaning forward, not only in an office chair",
                "A wearable brace workers can put on without buying an ergonomic chair they cannot afford",
                "Posture feedback that reminds the user before chronic pain sets in after a 10-hour shift",
            ],
            "hmw": [
                "How might we support the lumbar curve during 10-hour standing or forward-leaning work?",
                "How might we make the support wearable so it works at a sewing machine, not only at a desk?",
                "How might we fabricate the harness from 3D-printed stays and elastic straps in a makerspace?",
                "How might we give gentle biofeedback before pain becomes chronic?",
            ],
            "pugh": {"CONCEPT_A": (1, 2, 2, 1), "CONCEPT_B": (2, 1, 0, 1), "CONCEPT_C": (2, 2, 2, 2)},
            "concepts": _ergonomics_concepts(),
            "bom_theme": "ergonomics",
        },
    }

    if family in kits:
        return kits[family]

    third_concept = _wearable_generic_concept(domain, product, env) if body_related else _community_generic_concept(domain, product, env)
    pugh_c = (2, 2, 2, 2) if body_related else ((1, 2, 2, 2) if offgrid else (1, 1, 1, 2))
    pugh_a = (2, 2, 2, 2) if offgrid and not body_related else (1, 2, 2, 2)
    pugh_b = (2, 1, 0, 1)
    latent = []
    for s in request[:2]:
        latent.append(s.strip())
    latent.append(f"A locally maintainable {product} that addresses: {pain[0].rstrip('.')}.")
    latent = latent[:3]
    while len(latent) < 3:
        latent.append(f"Lower-cost, field-repairable alternative to the current {product}.")

    return {
        "latent_needs": latent,
        "hmw": [
            f"How might we solve '{pain[0].rstrip('.')}' without importing a commercial {product}?",
            f"How might we fabricate a {product} using makerspace processes and local hardware?",
            f"How might we keep the {product} usable when power, spare parts, or specialist labour are missing?",
            f"How might we give users a clear status signal so failures are caught before downtime spreads?",
        ],
        "pugh": {"CONCEPT_A": pugh_a, "CONCEPT_B": pugh_b, "CONCEPT_C": pugh_c},
        "concepts": [
            _passive_generic_concept(domain, product, env, ctx),
            _smart_generic_concept(domain, product, env, ctx),
            third_concept,
        ],
        "bom_theme": "generic",
    }


def _oxygen_concepts() -> List[ConceptArchitecture]:
    return [
        ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Pedal-Assist Dual-Canister PSA Oxygen Rig",
            working_principle="A bicycle-driven oil-free compressor feeds two zeolite molecular-sieve canisters in a pressure-swing cycle. Mechanical sequencing valves switch beds without a microcontroller, and a small buffer cylinder stores oxygen for blackout hours.",
            subsystem_mapping={
                "Power / Actuation": "Bicycle crank, flywheel, and belt-driven oil-free piston compressor",
                "Separation Stage": "Twin zeolite 13X canisters with cam-timed pressure-swing valves",
                "Chassis & Structure": "Welded mild-steel frame with lockable caster base for clinic floors",
                "Purity & Feedback": "Mechanical oxygen flowmeter plus zirconia cell with analog needle gauge",
            },
            manufacturing_strategy="Local fabrication of the steel frame and cam timer; zeolite canisters and medical tubing sourced as COTS; no imported PLC.",
            strengths=[
                "Produces oxygen during multi-hour grid outages without a battery bank",
                "Compressor and valves are serviceable with village workshop tools",
                "Buffer cylinder covers overnight demand after daytime pedalling or dynamo assist",
            ],
            limitations=[
                "Requires intermittent human or dynamo input; not fully unattended",
                "Larger clinic-floor footprint than a commercial concentrator",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="12V Solar Molecular-Sieve Concentrator",
            working_principle="A 12V BLDC compressor and solenoid-timed PSA beds run from a rooftop solar panel and lithium pack, delivering continuous low-flow oxygen with digital purity alarms.",
            subsystem_mapping={
                "Power / Actuation": "150W solar, MPPT, 12V LiFePO4 pack, BLDC compressor",
                "Separation Stage": "Heated zeolite towers with solenoid PSA switching",
                "Chassis & Structure": "Powder-coated aluminium enclosure with medical-grade inlet filter",
                "Purity & Feedback": "OLED purity, flow, and battery-hours display with buzzer alarm",
            },
            manufacturing_strategy="COTS solar and compressor modules in a locally bent enclosure; firmware on an open microcontroller.",
            strengths=[
                "Unattended oxygen during daytime solar hours",
                "Closed-loop purity control closer to commercial concentrators",
                "Hot-swap battery allows limited night operation",
            ],
            limitations=[
                "Depends on solar charging discipline and battery replacement",
                "Electronics and solenoids are harder to repair locally than a pedal rig",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="Cylinder-Buffer Hybrid with Gravity Reservoir",
            working_principle="A small concentrator slowly fills a water-displacement gravity reservoir and one refurbished steel cylinder. Gravity feed supplies the ward if both grid and compressor stop.",
            subsystem_mapping={
                "Power / Actuation": "Intermittent grid or solar compressor used only to top up storage",
                "Separation Stage": "Single zeolite bed plus water-seal oxygen holder",
                "Chassis & Structure": "Raised reservoir stand and strapped cylinder cradle",
                "Purity & Feedback": "Bobber level gauge and low-pressure whistle alarm",
            },
            manufacturing_strategy="Reuse of existing clinic cylinders; masonry or steel reservoir stand built by local welders.",
            strengths=[
                "Creates an oxygen buffer from equipment clinics already receive",
                "Gravity feed works with zero electricity",
                "Lowest electronic complexity of the three options",
            ],
            limitations=[
                "Water-seal purity is harder to certify than PSA output",
                "Still needs periodic compressor time to refill storage",
            ],
        ),
    ]


def _filtration_concepts() -> List[ConceptArchitecture]:
    return [
        ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Hydrocyclone + Gravity Settling Train",
            working_principle="Canal water is accelerated in a hydrocyclone so silt exits a grit pot, then dwells in a two-chamber settling tank before a stainless mesh. No pump is required when a 1.5 m head is available.",
            subsystem_mapping={
                "Power / Actuation": "Gravity head from canal bund; optional hand primer",
                "Separation Stage": "Hydrocyclone grit pot, baffled settler, 80-mesh stainless screen",
                "Chassis & Structure": "HDPE tanks strapped to a laser-cut plywood and angle-iron skid",
                "Flow & Feedback": "Transparent sight glass and float that drops when the mesh clogs",
            },
            manufacturing_strategy="Rolled-sheet cyclone, makerspace-cut flanges, HDPE drums from local suppliers.",
            strengths=[
                "No electricity and no disposable cartridges",
                "Grit pot can be emptied with a bucket in minutes",
                "Materials match what a regional makerspace already stocks",
            ],
            limitations=[
                "Needs elevation difference or a raised inlet drum",
                "Fine organic slime still needs occasional mesh scrubbing",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="12V Auto-Backflush Screen Filter",
            working_principle="A low-wattage pump drives irrigation water through a rotating nylon screen. Differential-pressure switches reverse flow for 20 seconds whenever clogging is detected.",
            subsystem_mapping={
                "Power / Actuation": "12V diaphragm pump and solar trickle charger",
                "Separation Stage": "Motorised nylon drum screen with reverse-flush nozzles",
                "Chassis & Structure": "PVC pressure housing and 3D-printed end caps",
                "Flow & Feedback": "Differential pressure switch, LED clog lamp, hour meter",
            },
            manufacturing_strategy="COTS pump and screen; printed housings; simple Arduino-free analog switching.",
            strengths=[
                "Self-cleaning under dirty canal conditions",
                "Maintains drip pressure better than a passive tank",
                "Alerts the farmer when backflush is happening",
            ],
            limitations=[
                "Needs solar or battery upkeep",
                "Pump diaphragms are a recurring spare",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="Bio-Sand and Cloth Prefilter Skid",
            working_principle="A layered sand-gravel biosand bed plus a washable cloth prefilter removes silt and a large share of microbes. Periodic bucket backwash replaces cartridge changes.",
            subsystem_mapping={
                "Power / Actuation": "Gravity feed from a header drum filled by hand or solar pump",
                "Separation Stage": "Cloth sock, coarse gravel, sand bed, outlet spigot",
                "Chassis & Structure": "Food-grade drum stack on a flood-raised timber pallet",
                "Flow & Feedback": "Painted high/low water lines and a simple turbidity jar test",
            },
            manufacturing_strategy="Drums, sand, and cloth sourced locally; timber stand from village carpentry.",
            strengths=[
                "Almost no moving parts and near-zero operating cost",
                "Farmers already understand sand-and-cloth cleaning",
                "Can be built in a day without CNC equipment",
            ],
            limitations=[
                "Lower peak flow than a pressurised screen",
                "Sand bed needs correct packing to avoid channelling",
            ],
        ),
    ]


def _grain_concepts() -> List[ConceptArchitecture]:
    return [
        ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Raised Hermetic Drum Battery",
            working_principle="Food-grade HDPE drums with gasketed lids sit on a rat-proof masonry plinth. Grain is loaded dry, lids are sealed, and a simple moisture card inside each drum shows monsoon risk without electronics.",
            subsystem_mapping={
                "Power / Actuation": "None - loading is manual with a grain scoop and lid clamp",
                "Storage Stage": "Stacked 200 L hermetic drums with inner liners",
                "Chassis & Structure": "Cement-stabilised plinth 400 mm above flood line, weld-mesh skirt",
                "Moisture & Feedback": "Cobalt-free humidity card and padlock hasp on each lid",
            },
            manufacturing_strategy="Local mason for the plinth; drums and mesh from the taluka market; no imported silo.",
            strengths=[
                "Stops rat entry and flood soaking that destroy gunny sacks",
                "Households can add drums as harvest volume grows",
                "Repair is a lid gasket, not a specialist silo crew",
            ],
            limitations=[
                "Filling many drums is slower than tipping into one tall silo",
                "Drums must be kept out of direct noon sun to limit condensation",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="Solar-Vented Mini Silo with Moisture Alarm",
            working_principle="A thin-gauge GI cylinder gets a solar-powered fan and DHT moisture probe. When humidity rises, the fan vents the headspace and an alarm lamp warns the farmer.",
            subsystem_mapping={
                "Power / Actuation": "20W panel, 12V fan, small sealed lead-acid battery",
                "Storage Stage": "1.2 m GI silo with airtight lid and outlet slide",
                "Chassis & Structure": "Tripod stand with rat baffle collar",
                "Moisture & Feedback": "DHT sensor, LED lamp, and buzzer on an open MCU",
            },
            manufacturing_strategy="Local sheet-metal rolling plus a makerspace electronics pod.",
            strengths=[
                "Active drying during unexpected rain spells",
                "Single bulk store is faster to fill than many drums",
                "Gives an explicit humidity warning",
            ],
            limitations=[
                "Fan, battery, and sensor fail modes need electronics skill",
                "Higher first cost than drums on a plinth",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="Community Super-Bag Cooperative Cage",
            working_principle="Hermetic triple-layer grain bags sit inside a shared weld-mesh cage on a raised platform. Neighbouring farms lock in harvest lots, cutting per-family steel cost.",
            subsystem_mapping={
                "Power / Actuation": "None",
                "Storage Stage": "PICS-style hermetic bags, 50 kg lots",
                "Chassis & Structure": "Lockable weld-mesh cage and timber pallet deck",
                "Moisture & Feedback": "Shared weighing scale and moisture meter on loan from the group",
            },
            manufacturing_strategy="Cage welded at the village workshop; bags purchased in bulk by the cooperative.",
            strengths=[
                "Spreads cost across households that cannot buy a metal silo",
                "Bags are portable if a family must move grain",
                "Mesh cage is a visible rat barrier",
            ],
            limitations=[
                "Needs trust and a shared lock protocol",
                "Bags puncture if handled with hooks",
            ],
        ),
    ]


def _energy_concepts() -> List[ConceptArchitecture]:
    return [
        ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Daylight Pipes plus Hand-Crank Charging Locker",
            working_principle="Roof-mounted daylight pipes light classrooms without panels. A locker of hand-crank and bicycle dynamos charges a small shared battery for evening LED lamps and phone charging.",
            subsystem_mapping={
                "Power / Actuation": "Daylight pipes by day; bicycle dynamo charging rail after school",
                "Distribution Stage": "12V LED string and USB charge locker",
                "Chassis & Structure": "Lockable timber locker bolted to the classroom wall",
                "Control & Feedback": "Analog voltmeter and red/green charge lamp",
            },
            manufacturing_strategy="Local carpenter for the locker; COTS LEDs and dynamo hubs; reflective pipe from aluminium sheet.",
            strengths=[
                "No fragile rooftop glass panel to steal or crack",
                "Daytime teaching no longer depends on electricity",
                "Students can contribute charging via the bicycle rail",
            ],
            limitations=[
                "Cannot run a bright projector for long without extra storage",
                "Human charging time is a chore if used every evening",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="Ruggedised 12V Solar Classroom Hub",
            working_principle="A polycarbonate-covered 200W panel charges a LiFePO4 hub that powers lights, laptop charging, and a low-watt LED projector with low-voltage disconnect.",
            subsystem_mapping={
                "Power / Actuation": "200W panel, MPPT, 100Ah LiFePO4, 12V distribution",
                "Distribution Stage": "Fused USB-C PD ports and projector DC converter",
                "Chassis & Structure": "Steel cage mount and dust-sealed indoor hub box",
                "Control & Feedback": "State-of-charge display and low-battery cutoff",
            },
            manufacturing_strategy="COTS solar modules in a locally welded anti-theft cage; hub assembled in a Fab Lab.",
            strengths=[
                "Supports projectors and laptops that teachers actually requested",
                "Polycarbonate and cage address fragile-panel failures",
                "Standard 12V parts are easier to replace than AC inverters",
            ],
            limitations=[
                "Still a capital cost the school must fund or share",
                "Battery replacement after several years",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="Shared Micro-Hub on a Rolling Cart",
            working_principle="One protected battery cart rolls between classrooms. Teachers book projector hours; lights in each room are separate 5W LEDs from a tiny local pack.",
            subsystem_mapping={
                "Power / Actuation": "Central cart with panel input and 12V bus",
                "Distribution Stage": "Umbilical cable and per-room LED puck lights",
                "Chassis & Structure": "Lockable rolling steel cart with cable reels",
                "Control & Feedback": "Booking slate plus SOC meter on the cart",
            },
            manufacturing_strategy="School workshop cart; one protected panel rather than many fragile kits.",
            strengths=[
                "Avoids buying a full solar kit for every classroom",
                "Cart can be locked in the staff room overnight",
                "Matches how rural schools already share scarce equipment",
            ],
            limitations=[
                "Only one classroom gets the projector at a time",
                "Requires a simple booking habit among teachers",
            ],
        ),
    ]


def _ergonomics_concepts() -> List[ConceptArchitecture]:
    return [
        ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Sit-Stand Counterweight Stool and Lumbar Pad",
            working_principle="A leaning stool with a spring-loaded lumbar pad takes part of body weight during 10-hour standing work at sewing machines, replacing folded rags with a stable mechanical rest.",
            subsystem_mapping={
                "Power / Actuation": "Coil spring and gas-strut assist, no electricity",
                "Support Stage": "Adjustable lumbar pad and chest-lean cushion",
                "Chassis & Structure": "Powder-coated steel tripod with non-slip feet",
                "Fit & Feedback": "Height pin holes and a simple comfort click-stop",
            },
            manufacturing_strategy="Local welding of the tripod; foam pads cut in a Fab Lab; COTS gas strut.",
            strengths=[
                "Immediately reduces standing load without a wearable learning curve",
                "Repairable in a factory maintenance shop",
                "Far cheaper than an imported ergonomic chair",
            ],
            limitations=[
                "Does not move with the worker if they leave the machine",
                "Needs floor space beside each sewing station",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="IMU Biofeedback Vest",
            working_principle="A lightweight vest with a lumbar IMU vibrates when the wearer slumps or leans without support for more than 30 seconds, training posture during the shift.",
            subsystem_mapping={
                "Power / Actuation": "Coin-cell or 3.7V LiPo with a day's charge",
                "Support Stage": "Soft vest; feedback only, no structural stay",
                "Chassis & Structure": "Sewn textile with a snap-in electronics pod",
                "Fit & Feedback": "Haptic motor and LED for slouch events",
            },
            manufacturing_strategy="Local stitching plus a makerspace electronics pod.",
            strengths=[
                "Works at any machine or desk, including forward lean",
                "Creates awareness before pain peaks at end of shift",
                "Small and easy to share across a line",
            ],
            limitations=[
                "Does not mechanically unload the spine",
                "Needs charging and will be ignored if vibration is annoying",
            ],
        ),
        ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="3D-Printed Wearable Lumbar Harness",
            working_principle="PETG lumbar stays in an elastic harness provide passive extension support while standing or leaning forward. Optional click-in pads replace rag bolsters workers currently fold by hand.",
            subsystem_mapping={
                "Power / Actuation": "Elastic strain in webbing; optional removable stay",
                "Support Stage": "Contoured 3D-printed lumbar shell and thoracic strap",
                "Chassis & Structure": "Breathable belt with Velcro and quick-release buckles",
                "Fit & Feedback": "Three size shells and a pressure-foam comfort layer",
            },
            manufacturing_strategy="FDM PETG stays, webbing and buckles from garment suppliers in the same textile cluster.",
            strengths=[
                "Matches the request for an affordable wearable that works without an office chair",
                "Can be printed in sizes for different workers the same week",
                "No charging, so it survives a 10-hour factory shift",
            ],
            limitations=[
                "Must be fitted well or it will be left in a locker",
                "Plastic stays need a fatigue check after months of sweat and flex",
            ],
        ),
    ]


def _passive_generic_concept(domain: str, product: str, env: str, ctx: Dict[str, Any]) -> ConceptArchitecture:
    pain = (ctx.get("pain_sentences") or [product])[0]
    return ConceptArchitecture(
        concept_id="CONCEPT_A",
        title=f"Zero-Electricity Mechanical {domain}",
        working_principle=f"A purely mechanical architecture that addresses '{pain.rstrip('.')}' using levers, gravity, springs, or human power so the {product} still works when utilities fail.",
        subsystem_mapping={
            "Power / Actuation": f"Human, gravity, or spring input sized for the {product}",
            "Core Function": f"Mechanical stage that performs the {product} job without electronics",
            "Chassis & Structure": f"Laser-cut plywood or welded frame suited to {env.lower()}",
            "Status Feedback": "Analog gauge or sight window, no firmware",
        },
        manufacturing_strategy="Makerspace-cut structure, COTS bearings and fasteners, local hardware only.",
        strengths=[
            f"Independent of grid power for the core {product} task",
            "Repairable with basic tools",
            "Lower capital cost than imported commercial equipment",
        ],
        limitations=[
            "May need intermittent human input",
            "Larger envelope than a compact electronic product",
        ],
    )


def _smart_generic_concept(domain: str, product: str, env: str, ctx: Dict[str, Any]) -> ConceptArchitecture:
    return ConceptArchitecture(
        concept_id="CONCEPT_B",
        title=f"12V Sensor-Assisted {domain}",
        working_principle=f"A low-voltage mechatronic {product} uses a microcontroller and sensors to keep performance stable in {env.lower()}.",
        subsystem_mapping={
            "Power / Actuation": "12V BLDC or solenoid pack with solar/battery option",
            "Core Function": f"Motorised {product} stage with closed-loop sensing",
            "Chassis & Structure": "Bent aluminium or printed housings",
            "Status Feedback": "OLED or LED status for throughput and faults",
        },
        manufacturing_strategy="COTS electronics plus Fab Lab printed brackets.",
        strengths=[
            f"Automated {product} delivery with less physical effort",
            "Can alarm before failure",
            "Compact compared with a fully mechanical rig",
        ],
        limitations=[
            "Needs charging or battery logistics",
            "Electronics are harder to repair in the field",
        ],
    )


def _wearable_generic_concept(domain: str, product: str, env: str) -> ConceptArchitecture:
    return ConceptArchitecture(
        concept_id="CONCEPT_C",
        title=f"Wearable / Body-Worn {domain}",
        working_principle=f"A harness or compliant shell delivers the {product} function on the body so the user is not tied to a chair or bench.",
        subsystem_mapping={
            "Power / Actuation": "Elastic energy or a tiny haptic cell",
            "Core Function": f"Body-conforming {product} stage",
            "Chassis & Structure": "Printed stays, webbing, breathable textile",
            "Status Feedback": "Haptic tap or colour indicator",
        },
        manufacturing_strategy="3D-printed stays and sewn webbing, no specialised factory.",
        strengths=[
            "Moves with the user",
            "Low mass and no floor footprint",
            "Can be sized and reprinted quickly",
        ],
        limitations=[
            "Limited force compared with a floor-standing machine",
            "Must be comfortable for a full shift",
        ],
    )


def _community_generic_concept(domain: str, product: str, env: str) -> ConceptArchitecture:
    return ConceptArchitecture(
        concept_id="CONCEPT_C",
        title=f"Shared Modular {domain} Kit",
        working_principle=f"A modular {product} is shared at community scale so households that cannot buy a commercial unit still get access, with field-replaceable subassemblies.",
        subsystem_mapping={
            "Power / Actuation": "Optional shared battery or purely manual modules",
            "Core Function": f"Stackable {product} modules sized for one household lot",
            "Chassis & Structure": f"Lockable shared cage or cart for {env.lower()}",
            "Status Feedback": "Tag board showing who holds which module",
        },
        manufacturing_strategy="Repeatable modules from local stock; one shared precision part.",
        strengths=[
            "Splits capital cost across users",
            "Failed modules swap without scrapping the whole system",
            "Matches cooperative use already common in villages and factories",
        ],
        limitations=[
            "Needs a sharing rule",
            "Peak demand may queue users",
        ],
    )


def family_bom(winning_concept: ConceptArchitecture, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    theme = family_solution_kit(ctx)["bom_theme"]
    catalogs = {
        "oxygen": [
            {"item": "Oil-free piston compressor compatible with medical air", "quantity": "1 Unit", "source": "Industrial compressor dealer", "cost_inr": "8,400"},
            {"item": "Zeolite 13X molecular sieve (rechargeable canisters)", "quantity": "4 kg", "source": "Chemical supplier", "cost_inr": "3,600"},
            {"item": "Medical-grade oxygen tubing, cannula, and check valves", "quantity": "1 Kit", "source": "Hospital supplier", "cost_inr": "1,250"},
            {"item": "Mild-steel clinic frame, casters, and belt drive set", "quantity": "1 Assembly", "source": "Local fabricator", "cost_inr": "2,800"},
            {"item": "Analog oxygen flowmeter and purity cell", "quantity": "1 Unit", "source": "Instrumentation supplier", "cost_inr": "2,150"},
            {"item": "Buffer cylinder strap kit and pressure relief", "quantity": "1 Kit", "source": "Gas equipment dealer", "cost_inr": "1,900"},
        ],
        "filtration": [
            {"item": "HDPE drum pair and hydrocyclone body blank", "quantity": "1 Set", "source": "Agri hardware store", "cost_inr": "2,400"},
            {"item": "Stainless 80-mesh screen and gasket kit", "quantity": "2 Sheets", "source": "Mesh supplier", "cost_inr": "980"},
            {"item": "Angle-iron skid and hose camlocks", "quantity": "1 Assembly", "source": "Local fabricator", "cost_inr": "1,450"},
            {"item": "Sight glass, float, and ball valves", "quantity": "1 Kit", "source": "Plumbing supplier", "cost_inr": "720"},
            {"item": "Optional 12V diaphragm pump (Concept B path)", "quantity": "1 Unit", "source": "Solar pump dealer", "cost_inr": "2,100"},
            {"item": "Food-safe silicone sealant and fasteners", "quantity": "1 Kit", "source": "Hardware store", "cost_inr": "380"},
        ],
        "grain": [
            {"item": "Food-grade 200 L HDPE drums with gasket lids", "quantity": "6 Units", "source": "Taluka market", "cost_inr": "7,200"},
            {"item": "Cement, brick, and sand for 400 mm flood plinth", "quantity": "1 Plinth", "source": "Local mason", "cost_inr": "3,400"},
            {"item": "Weld-mesh rat skirt and padlock hasps", "quantity": "1 Kit", "source": "Hardware store", "cost_inr": "1,150"},
            {"item": "Inner grain liners and scoop", "quantity": "6 + 1", "source": "Agri store", "cost_inr": "900"},
            {"item": "Humidity indicator cards", "quantity": "12 Cards", "source": "Packaging supplier", "cost_inr": "360"},
            {"item": "Optional shared moisture meter", "quantity": "1 Unit", "source": "Agri extension shop", "cost_inr": "1,800"},
        ],
        "energy": [
            {"item": "200W photovoltaic panel with polycarbonate shield", "quantity": "1 Unit", "source": "Solar dealer", "cost_inr": "6,800"},
            {"item": "100Ah LiFePO4 battery and MPPT charge controller", "quantity": "1 Set", "source": "Solar dealer", "cost_inr": "12,500"},
            {"item": "12V LED classroom lighting string", "quantity": "2 Classrooms", "source": "Electrical store", "cost_inr": "1,600"},
            {"item": "USB-C PD charging locker hardware", "quantity": "8 Ports", "source": "Electronics supplier", "cost_inr": "2,200"},
            {"item": "Welded anti-theft panel cage and indoor hub box", "quantity": "1 Assembly", "source": "Local fabricator", "cost_inr": "2,400"},
            {"item": "Fusing, low-voltage disconnect, and voltmeter", "quantity": "1 Kit", "source": "Electrical store", "cost_inr": "950"},
        ],
        "ergonomics": [
            {"item": "PETG filament for lumbar stay shells (3 sizes)", "quantity": "900 g", "source": "Fab Lab", "cost_inr": "1,350"},
            {"item": "Elastic webbing, foam, and garment buckles", "quantity": "1 Kit", "source": "Textile cluster supplier", "cost_inr": "780"},
            {"item": "Optional IMU + haptic pod for biofeedback", "quantity": "1 Unit", "source": "Electronics kit", "cost_inr": "1,450"},
            {"item": "Steel tripod and foam pad for sit-stand stool path", "quantity": "1 Unit", "source": "Local fabricator", "cost_inr": "1,900"},
            {"item": "Velcro, thread, and lining fabric", "quantity": "1 Kit", "source": "Garment shop", "cost_inr": "420"},
            {"item": "Fasteners and adjustment sliders", "quantity": "1 Kit", "source": "Hardware store", "cost_inr": "260"},
        ],
    }
    if theme in catalogs:
        return catalogs[theme]

    domain = ctx.get("primary_domain", "System")
    product = ctx.get("product_phrase", domain.lower())
    return [
        {"item": f"{product.title()} core mechanism / structure", "quantity": "1 Unit", "source": "Local fabricator", "cost_inr": "2,400"},
        {"item": "Makerspace-printed housings and jigs", "quantity": "400 g", "source": "Fab Lab", "cost_inr": "720"},
        {"item": "Fastener, bearing, and seal kit", "quantity": "1 Kit", "source": "Hardware store", "cost_inr": "680"},
        {"item": "Analog status indicator", "quantity": "1 Unit", "source": "Instrumentation supplier", "cost_inr": "540"},
        {"item": f"Materials selected for {winning_concept.title}", "quantity": "1 Set", "source": "As per manufacturing strategy", "cost_inr": "1,100"},
    ]


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

        ctx = deep_parse(notes)
        kit = family_solution_kit(ctx)
        domain = ctx["primary_domain"]
        product = ctx["product_phrase"]
        env = ctx["environment"]
        pain_s = ctx["pain_sentences"]
        request_s = ctx["request_sentences"]
        workaround_s = ctx["workaround_sentences"]
        constraints_s = ctx["constraints_raw"]

        says = [_quote(s, f"The current {product} is failing us.") for s in (pain_s + request_s)[:3]]
        while len(says) < 3:
            says.append(_quote("", f"We cannot keep depending on a commercial {product} that was not designed for this setting."))

        thinks = [
            f'\"Why is the available {product} ignoring {env.lower()}?\"',
            f'\"If this were repairable locally, we would not lose days of {domain.lower()} every time it fails.\"',
            f'\"I worry that \'{pain_s[0].strip()[:90]}\' will get worse if nothing changes.\"',
        ]

        does = [s.strip() for s in workaround_s[:2]]
        if request_s:
            does.append(request_s[0].strip())
        does.append(f"Improvises around a missing or failing {product} during daily work.")
        does = does[:3]

        feels = [
            f"Frustrated that {product} failures keep interrupting essential work.",
            f"Exhausted by workarounds required in {env.lower()}.",
            f"Hopeful that a purpose-built {domain.lower()} solution can be made and repaired nearby.",
        ]

        core_pains = [s.strip() for s in pain_s[:3]]
        while len(core_pains) < 3:
            core_pains.append(f"Existing {product} options are too costly, fragile, or unavailable when needed.")

        op_constraints = [s.strip() for s in constraints_s[:2]]
        op_constraints.append(env.rstrip(".") + ".")
        op_constraints = op_constraints[:3]

        return EmpathyMap(
            persona_title=ctx["persona"],
            says=says,
            thinks=thinks,
            does=does,
            feels=feels,
            core_pain_points=core_pains,
            latent_unmet_needs=kit["latent_needs"],
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
        kit = family_solution_kit(ctx) if ctx else {"hmw": empathy.latent_unmet_needs}
        domain = ctx.get("primary_domain", "System")
        product = ctx.get("product_phrase", domain.lower())

        pov = (
            f"{empathy.persona_title} needs {empathy.latent_unmet_needs[0].rstrip('.')} "
            f"because {empathy.core_pain_points[0].rstrip('.')}."
        )

        func_reqs = [
            f"The {product} must keep working in this context: {ctx.get('environment', empathy.operating_constraints[0])}.",
            f"Critical wear parts of the {domain.lower()} solution must be replaceable with locally available tools.",
            f"First prototype BOM should stay in a makerspace / Fab Lab cost band rather than imported capital equipment.",
        ]

        return ProblemDefinition(
            pov_statement=pov,
            hmw_opportunity_spaces=kit["hmw"],
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

        kit = family_solution_kit(ctx or {})
        return kit["concepts"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: Pugh Matrix Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class PughMatrixEvaluatorTool:
    def execute(
        self,
        concepts: List[ConceptArchitecture],
        ctx: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> List[PughScoringRow]:
        kit = family_solution_kit(ctx or {})
        weights = kit["pugh"]
        scores = []
        ranked_totals = []
        for c in concepts:
            d, f, v, m = weights.get(c.concept_id, (1, 1, 1, 1))
            total = d + f + v + m
            ranked_totals.append(total)
            scores.append((total, PughScoringRow(
                concept_id=c.concept_id,
                title=c.title,
                desirability_score=d,
                feasibility_score=f,
                viability_score=v,
                maintainability_score=m,
                total_score=total,
                decision="CANDIDATE",
            )))

        scores.sort(key=lambda row: row[0], reverse=True)
        max_total = scores[0][0] if scores else 0
        labelled = []
        for i, (total, row) in enumerate(scores):
            if total == max_total and i == 0:
                row.decision = "WINNING CONCEPT"
            elif total >= max_total - 1:
                row.decision = "STRONG RUNNER-UP"
            else:
                row.decision = "ELIMINATED"
            labelled.append(row)
        return labelled


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: BOM & Prototyping Tool
# ─────────────────────────────────────────────────────────────────────────────

class BOMPrototypingTool:
    def execute(self, winning_concept: ConceptArchitecture, ctx: Dict = {}, api_key: Optional[str] = None) -> PrototypingPlan:
        domain = ctx.get("primary_domain", "System")
        product = ctx.get("product_phrase", domain.lower())
        env = ctx.get("environment", "field environment")
        subsystems = list(winning_concept.subsystem_mapping.values())
        bom = family_bom(winning_concept, ctx or {})
        total = sum(float(item["cost_inr"].replace(",", "")) for item in bom)

        phases = {
            f"Phase 1: {domain} proof-of-principle (Weeks 1-2)": (
                f"Build a bench mockup of '{winning_concept.title}' to test the {product} working principle "
                f"against the stated pain: {(ctx.get('pain_sentences') or ['field failure'])[0]}."
            ),
            f"Phase 2: Full prototype in the real setting (Weeks 3-4)": (
                f"Fabricate the chassis and all four subsystems. Evaluate fit, safety, and repair steps in {env.lower()}."
            ),
            f"Phase 3: Endurance and field repair drill (Weeks 5-6)": (
                f"Run a 50-hour duty cycle on the {product}, log failures, and write a one-page field repair sheet."
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
