# InnoAgent: Autonomous Multi-Agent Design Thinking and Concept Synthesis Engine

> An enterprise multi-agent discovery engine mirroring the Stanford d.school Design Thinking framework: transforming unstructured qualitative user research and field notes into structured Empathy Maps, Point-of-View (POV) definitions, Morphological Concept Architectures, multi-criteria Pugh Decision Matrices, and Rapid Prototyping Roadmaps.

---

## The Problem
Product innovation teams, makerspaces, and hardware incubators frequently struggle with the "fuzzy front end" of product design:
* Unstructured user interviews and field observations take weeks to synthesize manually.
* Teams often converge prematurely on the first intuitive concept rather than exploring divergent working principles.
* Failure to balance Desirability (user value), Feasibility (manufacturing constraints), and Viability (cost).

**InnoAgent** automates this front-end discovery workflow through a coordinated team of specialized AI agents that enforce human-centered design rigor.

---

## 5-Stage Multi-Agent Architecture

```
Raw Field Notes / Qualitative Interviews
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│        Stage 1: Empathize (User Insight Miner)            │
│  - Structures 4-quadrant Empathy Map (Says/Thinks/Does/Feels)
│  - Extracts latent unmet needs and operating constraints   │
└───────────────────────────┬───────────────────────────────┘
                            │ Empathy Profile
                            ▼
┌───────────────────────────────────────────────────────────┐
│        Stage 2: Define (Problem Framing Agent)            │
│  - Formulates Point-of-View (POV) Statement               │
│  - Generates prioritized How-Might-We (HMW) spaces        │
└───────────────────────────┬───────────────────────────────┘
                            │ HMW Spaces & Constraints
                            ▼
┌───────────────────────────────────────────────────────────┐
│        Stage 3: Ideate (Divergent Architecture Agent)     │
│  - Performs Morphological Decomposition                   │
│  - Synthesizes 3 divergent mechanical working principles  │
└───────────────────────────┬───────────────────────────────┘
                            │ Concept Architecture Set
                            ▼
┌───────────────────────────────────────────────────────────┐
│        Stage 4: Evaluate (Pugh Decision Matrix Agent)     │
│  - Benchmarks concepts against baseline commercial unit   │
│  - Scores across Desirability, Feasibility, and Viability │
└───────────────────────────┬───────────────────────────────┘
                            │ Selected Concept
                            ▼
┌───────────────────────────────────────────────────────────┐
│        Stage 5: Prototype (Fabrication Roadmap Agent)     │
│  - Synthesizes Preliminary Bill of Materials (BOM)        │
│  - Produces 6-week Fab Lab rapid prototyping roadmap      │
└───────────────────────────────────────────────────────────┘
```

---

## Specialized Tools Implemented
1. **`EmpathyAnalysisTool`**: Analyzes qualitative interview transcripts to extract user personas, core operational pain points, and latent needs.
2. **`ProblemFramingTool`**: Formulates structured POV statements and divergent HMW opportunity spaces.
3. **`MorphologicalIdeationTool`**: Explores subsystem functional alternatives (power/actuation, compression/separation, chassis, flow control) to create distinct concept architectures.
4. **`PughMatrixEvaluatorTool`**: Implements quantitative Pugh Decision Analysis comparing concepts against baseline standards.
5. **`BOMPrototypingTool`**: Synthesizes a preliminary Bill of Materials with estimated unit costs and establishes a 3-phase rapid prototyping strategy.

---

## Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Dipalikambale/innoagent.git
cd innoagent
pip install -r requirements.txt
```

### 2. Run CLI Pipeline
```bash
python main.py
```

### 3. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to:
* Choose between preset real-world scenarios or input custom user research notes.
* Navigate the 5 interactive Design Thinking stages.
* Inspect multi-criteria Pugh matrix bar charts.
* Download generated Product Requirement Documents (PRD) and Bills of Materials.

---

## Repository Structure
```
innoagent/
├── app.py              # Interactive Streamlit Web Dashboard
├── agent_pipeline.py   # Multi-agent orchestrator coordinating the 5 discovery stages
├── tools.py            # Specialized design thinking tools and Pydantic data schemas
├── main.py             # CLI test runner
├── requirements.txt    # Open-source dependencies
└── README.md           # Engineering documentation
```
