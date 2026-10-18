"""
tools.py
Specialized Tools for Design Thinking and Hardware Innovation.
Implements modular tool modules for:
1. Empathy Analysis & Latent Need Mining
2. POV & How-Might-We Problem Framing
3. Morphological Concept Ideation
4. Multi-Criteria Pugh Decision Matrix Scoring
5. Subsystem Decomposition & Bill of Materials (BOM) Synthesis
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


# --- Pydantic Data Contracts ---

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
    desirability_score: int    # -2 to +2 vs baseline
    feasibility_score: int     # -2 to +2 vs baseline
    viability_score: int       # -2 to +2 vs baseline
    maintainability_score: int # -2 to +2 vs baseline
    total_score: int
    decision: str


class PrototypingPlan(BaseModel):
    winning_concept_title: str
    functional_breakdown: List[str]
    bill_of_materials: List[Dict[str, str]]
    total_estimated_bom_inr: float
    rapid_prototyping_phases: Dict[str, str]


# --- Design Thinking Tool Implementations ---

class EmpathyAnalysisTool:
    """Tool 1: Parses qualitative unstructured notes into structured Empathy Maps."""
    def execute(self, notes: str) -> EmpathyMap:
        notes_lower = notes.lower()

        # Context-aware extraction heuristics for rural/medical/engineering domains
        if "oxygen" in notes_lower or "clinic" in notes_lower or "health" in notes_lower:
            persona = "Rural Primary Health Center Nurse / Healthcare Worker"
            says = [
                "The power goes out for hours and our standard electric concentrator stops immediately.",
                "Patients in acute respiratory distress cannot wait for replacement cylinders over bad roads.",
                "We need equipment that our local technician can service with basic tools."
            ]
            thinks = [
                "Will this patient survive the night without electricity?",
                "Modern digital medical machines are too fragile for rural dust and voltage surges.",
                "I wish we had a mechanical backup that never relies on the electrical grid."
            ]
            does = [
                "Manually bags patients with AMBU bags for hours during extended blackouts.",
                "Travels 40km to district headquarters to refill heavy, expensive oxygen cylinders.",
                "Attempts improvising repairs with wire and tape when specialized parts fail."
            ]
            feels = [
                "Anxious and helpless during sudden power failures.",
                "Overburdened by logistical delays and expensive cylinder transport.",
                "Distrustful of complex digital electronics that break easily in high heat."
            ]
            pain_points = [
                "6-10 hours daily grid blackouts halt standard oxygen generation.",
                "Heavy steel cylinders are difficult and dangerous to transport on rural terrain.",
                "High downtime due to lack of local technicians and proprietary spare parts."
            ]
            latent_needs = [
                "Continuous oxygen generation driven by human mechanical power, low-voltage DC, or solar energy.",
                "Visible, transparent mechanical operation with intuitive physical gauges instead of error codes.",
                "Modular, open-standard construction using local Fab Lab / hardware-store components."
            ]
            constraints = [
                "Operating ambient temperature up to 45C with dust exposure.",
                "Strict capital cost ceiling under INR 20,000.",
                "Minimum medical purity requirement of 90% oxygen concentration."
            ]
        elif "water" in notes_lower or "filter" in notes_lower or "plastic" in notes_lower:
            persona = "Agricultural Field Researcher / Irrigation Supervisor"
            says = [
                "Irrigation water carries micro-debris and bacterial runoff that ruins drip emitters.",
                "Commercial filtration membranes clog within 48 hours and require expensive replacements."
            ]
            thinks = [
                "Traditional sand filters take too much space and are hard to backwash.",
                "We need a self-cleaning, low-power solution."
            ]
            does = [
                "Manually clears clogged irrigation nozzles twice daily.",
                "Tests water samples for turbidity and bacterial contamination."
            ]
            feels = [
                "Exhausted by constant physical maintenance of filtration lines.",
                "Concerned about crop yield degradation from uneven water distribution."
            ]
            pain_points = [
                "Rapid clogging of passive filtration meshes requiring daily manual interventions.",
                "High replacement cost for imported filter elements."
            ]
            latent_needs = [
                "Self-cleaning vibrational or hydrodynamic continuous separation mechanism.",
                "Solar-assisted passive pathogen reduction."
            ]
            constraints = [
                "Zero continuous AC grid power available in agricultural fields.",
                "Must withstand high chemical exposure and UV weathering."
            ]
        else:
            persona = "End-User in Resource-Constrained Operating Context"
            says = ["The existing commercial tools are too expensive and difficult to maintain."]
            thinks = ["We need a simpler, repairable alternative designed for our specific workflow."]
            does = ["Improvises workarounds using locally available scrap and hardware."]
            feels = ["Frustrated by high maintenance costs and complex operational procedures."]
            pain_points = ["High operational downtime and lack of affordable replacement components."]
            latent_needs = ["Decentralized, durable functional architecture that can be repaired locally."]
            constraints = ["Strict capital cost constraints and harsh operating environment."]

        return EmpathyMap(
            persona_title=persona,
            says=says,
            thinks=thinks,
            does=does,
            feels=feels,
            core_pain_points=pain_points,
            latent_unmet_needs=latent_needs,
            operating_constraints=constraints
        )


class ProblemFramingTool:
    """Tool 2: Formulates Point-of-View (POV) and prioritized How-Might-We (HMW) opportunities."""
    def execute(self, empathy: EmpathyMap) -> ProblemDefinition:
        pov = (
            f"{empathy.persona_title} needs a reliable, decentralized method to {empathy.latent_unmet_needs[0].lower()} "
            f"because {empathy.core_pain_points[0].lower()}."
        )

        hmw = [
            f"How might we convert mechanical kinetic motion or solar energy into steady operating pressure?",
            f"How might we design a self-cleaning or modular filtration/sieve bed that requires zero specialized tools?",
            f"How might we utilize locally manufacturable sheet goods and 3D printed parts to reduce capital costs by 70%?",
            f"How might we provide intuitive, visual operational feedback without relying on complex digital microcontrollers?"
        ]

        functional_reqs = [
            "Continuous operating throughput without grid electricity dependency.",
            "Field-serviceable architecture assembled with standard fasteners.",
            "Modular cartridge layout allowing sub-5-minute component replacement."
        ]

        return ProblemDefinition(
            pov_statement=pov,
            hmw_opportunity_spaces=hmw,
            functional_requirements=functional_reqs,
            engineering_constraints=empathy.operating_constraints
        )


class MorphologicalIdeationTool:
    """Tool 3: Explores divergent architectures via morphological decomposition & TRIZ principles."""
    def execute(self, problem: ProblemDefinition) -> List[ConceptArchitecture]:
        c1 = ConceptArchitecture(
            concept_id="CONCEPT_A",
            title="Kinetic Pedal-Drive Dual-Piston PSA Unit",
            working_principle="Human-powered ergonomic foot treadle driving counter-balanced diaphragm air pumps through dual Zeolite 13X columns with mechanically timed rotary valve timing.",
            subsystem_mapping={
                "Actuation / Power": "Reciprocating foot treadle with heavy flywheel inertia dampener",
                "Compression Stage": "Dual silicone diaphragm positive displacement pumps",
                "Separation Engine": "Twin 50mm extruded aluminum columns with spring-compressed Zeolite 13X beds",
                "Flow & Valve Control": "Mechanical camshaft rotary 4-way valve driven off flywheel axle",
                "Chassis & Housing": "Laser-cut 6mm birch plywood chassis with interlocking finger joints"
            },
            manufacturing_strategy="Digital fabrication: Laser cut chassis + FDM 3D printed valve manifolds + bicycle drive parts.",
            strengths=[
                "100% independent from any electricity or batteries",
                "Built from accessible mechanical linkages with zero electronic points of failure",
                "Extremely low Bill of Materials cost"
            ],
            limitations=[
                "Requires physical human operator cadence (approx. 40 RPM)",
                "Bulkier mechanical footprint due to flywheel"
            ]
        )

        c2 = ConceptArchitecture(
            concept_id="CONCEPT_B",
            title="12V DC Solar/Battery Hybrid Micro-Compressor System",
            working_principle="Ultra-compact 12V brushless rocking-piston compressor powered by portable swappable power-tool battery packs or a 50W foldable solar panel.",
            subsystem_mapping={
                "Actuation / Power": "12V 120W BLDC motor with integrated battery buck-boost converter",
                "Compression Stage": "Miniature oil-less rocking piston compressor (25 psi gauge)",
                "Separation Engine": "Rapid cycle twin micro-column PSA with ceramic disc air distributor",
                "Flow & Valve Control": "Low-power 12V miniature latching solenoid valves",
                "Chassis & Housing": "Sheet metal bent aluminum chassis with silicone shock mounts"
            },
            manufacturing_strategy="CNC sheet-metal bending with 3D printed internal air ducts and COTS electronic modules.",
            strengths=[
                "Automated, effortless continuous delivery for 4+ hours per battery pack",
                "Compact, highly portable form factor (<6.5 kg)",
                "High volumetric efficiency"
            ],
            limitations=[
                "Requires battery recharging infrastructure",
                "Higher component cost for solenoid valves and BLDC driver"
            ]
        )

        c3 = ConceptArchitecture(
            concept_id="CONCEPT_C",
            title="Pneumatic Gravity-Counterweight Rotary Displacement Unit",
            working_principle="A suspended water/sand counterweight dropped through a geared pulley mechanism to provide continuous, steady pneumatic air compression without human effort.",
            subsystem_mapping={
                "Actuation / Power": "15kg suspended mass lifted once per hour, falling through an escapement gear train",
                "Compression Stage": "Large-bore low-friction bellows with linear guide rods",
                "Separation Engine": "Single rotating carousel zeolite canister array",
                "Flow & Valve Control": "Pure fluidic check-valve manifold with mechanical escapement",
                "Chassis & Housing": "Welded square steel tube frame with sheet metal safety shroud"
            },
            manufacturing_strategy="Welded steel tube construction with laser-cut acrylic gear train.",
            strengths=[
                "Passive unattended operation for 60 minutes per reset",
                "Zero electrical or battery wear over 10-year lifespan"
            ],
            limitations=[
                "High vertical clearance required for mass descent",
                "Heavier structural frame"
            ]
        )

        return [c1, c2, c3]


class PughMatrixEvaluatorTool:
    """Tool 4: Executes multi-criteria Pugh Decision Matrix scoring against a conventional baseline."""
    def execute(self, concepts: List[ConceptArchitecture]) -> List[PughScoringRow]:
        # Baseline = Conventional Imported Electric Commercial Unit
        # Scoring scale: -2 (Much Worse), -1 (Worse), 0 (Same as Baseline), +1 (Better), +2 (Much Better)
        scores = []
        for c in concepts:
            if c.concept_id == "CONCEPT_A":
                # Kinetic Pedal
                d = 1   # High user gratitude for reliability during blackouts, minor physical effort penalty
                f = 2   # Highly manufacturable in any Fab Lab / local workshop
                v = 2   # Extremely cheap, no imported proprietary parts
                m = 2   # Field-serviceable by any bicycle/mechanic shop
            elif c.concept_id == "CONCEPT_B":
                # 12V Hybrid
                d = 2   # Maximum user comfort, push-button operation
                f = 1   # Moderate feasibility, requires sourcing solenoids/BLDC
                v = 0   # Comparable or slightly cheaper than baseline
                m = 1   # Modular replacement
            else:
                # Gravity Counterweight
                d = -1  # Cumbersome lifting mass
                f = 0   # Mechanical complexity in escapement
                v = 1   # Long lifespan
                m = 1   # Low maintenance

            total = d + f + v + m
            decision = "WINNING CONCEPT (SELECTED)" if total >= 7 else ("STRONG RUNNER-UP" if total >= 4 else "ELIMINATED")

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


class BOMPrototypingTool:
    """Tool 5: Generates comprehensive Bill of Materials (BOM) and digital fabrication roadmap."""
    def execute(self, winning_concept: ConceptArchitecture) -> PrototypingPlan:
        bom = [
            {"item": "Zeolite Molecular Sieve Type 13X Granules (1.2 kg)", "quantity": "1 Pack", "source": "Chemical / Industrial Supply", "cost_inr": "2,200"},
            {"item": "Seamless Aluminum Pressure Tubing (50mm OD x 320mm)", "quantity": "2 Units", "source": "Local Metal Stockist", "cost_inr": "1,400"},
            {"item": "Silicone Diaphragm Membrane with Brass Connectors", "quantity": "2 Sets", "source": "Pneumatics Supplier", "cost_inr": "1,100"},
            {"item": "One-Way Spring-Loaded Pneumatic Check Valves", "quantity": "4 Units", "source": "Industrial Hardware", "cost_inr": "850"},
            {"item": "Laser-Cut Birch Plywood Chassis Plates (6mm)", "quantity": "2 Sheets", "source": "Fab Lab / Woodworking", "cost_inr": "900"},
            {"item": "FDM 3D Printed PETG Manifold & Cam Valve Housing", "quantity": "450 grams", "source": "Fab Lab 3D Printer", "cost_inr": "750"},
            {"item": "Bicycle Crankset, Pedal Assembly, and Flywheel", "quantity": "1 Set", "source": "Local Bicycle Workshop", "cost_inr": "1,150"},
            {"item": "Analog Medical Rotameter Flowmeter (0-5 LPM)", "quantity": "1 Unit", "source": "Medical Equipment Supplier", "cost_inr": "650"},
            {"item": "High-Pressure PU Tubing (8mm OD) & Push-Fit Connectors", "quantity": "3 meters", "source": "Hardware Store", "cost_inr": "450"}
        ]

        total_cost = sum(float(item["cost_inr"].replace(",", "")) for item in bom)

        phases = {
            "Phase 1: Proof-of-Concept Pneumatics (Weeks 1-2)": "Construct benchtop acrylic test rig using syringe/diaphragm pumps to measure Zeolite column breakthrough curve and oxygen purity output.",
            "Phase 2: Fab Lab Prototyping & Ergonomics (Weeks 3-4)": "Laser cut 6mm chassis; 3D print airtight PETG manifolds with 100% infill; assemble bicycle crank linkage and test pedal cadence vs pressure generation.",
            "Phase 3: Integration & Endurance Testing (Weeks 5-6)": "Operate continuously for 10 hours at 40 RPM; log oxygen purity (>90%) with ultrasonic oxygen analyzer; perform field drop and vibration testing."
        }

        functional_blocks = [
            "Kinematic Treadle Subsystem: Converts reciprocating leg motion into rotational flywheel kinetic inertia.",
            "Fluidic Compression Manifold: Regulates intake ambient air through particulate filter and compresses to 22 psi.",
            "Adsorption Column Bank: Dual alternating zeolite beds stripping nitrogen molecules to deliver 92% pure oxygen.",
            "Patient Delivery Block: Accumulator surge tank with analog rotameter regulating laminar flow from 1 to 5 LPM."
        ]

        return PrototypingPlan(
            winning_concept_title=winning_concept.title,
            functional_breakdown=functional_blocks,
            bill_of_materials=bom,
            total_estimated_bom_inr=total_cost,
            rapid_prototyping_phases=phases
        )
