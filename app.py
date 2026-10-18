"""
app.py
Interactive Streamlit Dashboard for InnoAgent.
Visualizes the 5 stages of Design Thinking:
Empathize -> Define -> Ideate -> Evaluate -> Prototype.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agent_pipeline import InnoAgentOrchestrator

st.set_page_config(
    page_title="InnoAgent | Design Thinking Engine",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .subtitle { font-size: 1.05rem; color: #4B5563; margin-bottom: 25px; }
    .card-empathy { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .pov-box { background-color: #EFF6FF; border-left: 5px solid #2563EB; padding: 14px; border-radius: 6px; font-size: 1.05rem; }
    .winner-box { background-color: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">InnoAgent: Autonomous Design Thinking Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multi-agent discovery workflow: Raw User Research &rarr; Problem Framing &rarr; Morphological Ideation &rarr; Pugh Matrix Evaluation &rarr; Prototype BOM</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Design Research Input")

PRESETS = {
    "Rural Off-Grid Medical Oxygen Concentrator": """Field visits to primary healthcare sub-centres in rural districts.
Clinicians reported that grid electricity is unavailable for 6 to 10 hours daily.
Conventional commercial oxygen concentrators shut down during blackouts, leaving patients
in severe respiratory distress. Transporting heavy steel cylinders over unpaved rural roads
is expensive and takes days. Clinics urgently requested a robust, low-resource oxygen
generation mechanism that can be maintained locally without specialist imported tools.""",
    
    "Agricultural Microplastic & Silt Water Filtration": """Interviews with agricultural researchers and drip irrigation operators.
Irrigation water from open canals carries severe micro-debris, silt, and bacterial contamination
that rapidly clogs drip emitter lines. Commercial cartridge filters clog within 48 hours and
require costly replacement elements. Farmers need a self-cleaning, low-power continuous
separation system that can operate off-grid and be fabricated in a regional makerspace.""",

    "Custom Problem Statement": ""
}

selected_preset = st.sidebar.selectbox("Choose Problem Scenario", list(PRESETS.keys()), index=0)

if selected_preset == "Custom Problem Statement":
    user_input_notes = st.sidebar.text_area(
        "Enter Raw Field Notes or User Interviews",
        value="Enter unstructured user feedback, pain points, or field observations...",
        height=220
    )
else:
    user_input_notes = st.sidebar.text_area(
        "Field Interview Notes",
        value=PRESETS[selected_preset],
        height=220
    )

if st.sidebar.button("Run Design Thinking Agent Team", type="primary"):
    with st.spinner("Multi-agent team executing Empathize, Define, Ideate, Evaluate, and Prototype workflows..."):
        orchestrator = InnoAgentOrchestrator()
        results = orchestrator.run_pipeline(user_input_notes)
        st.session_state["inno_results"] = results

if "inno_results" in st.session_state:
    res = st.session_state["inno_results"]
    winner = res["winning_concept"]
    proto = res["prototyping_plan"]
    empathy = res["empathy_map"]
    framing = res["problem_definition"]
    pugh = res["pugh_evaluation"]

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Discovery Stage", "5 / 5 Complete")
    with c2:
        st.metric("Primary Persona", empathy["persona_title"].split()[0])
    with c3:
        st.metric("Selected Concept", winner["concept_id"])
    with c4:
        st.metric("Pugh Score", f"+{pugh[0]['total_score']} vs Baseline")
    with c5:
        st.metric("Prototype BOM", f"INR {proto['total_estimated_bom_inr']:,.0f}")

    st.markdown("---")

    # 5 Design Thinking Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Empathize",
        "2. Define",
        "3. Ideate",
        "4. Evaluate (Pugh Matrix)",
        "5. Prototype & PRD"
    ])

    # TAB 1: EMPATHIZE
    with tab1:
        st.subheader("Stage 1: User Empathy Mapping and Latent Needs")
        st.markdown(f"**Identified Target User Persona:** `{empathy['persona_title']}`")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### SAYS (Explicit Statements)")
            for item in empathy["says"]:
                st.write(f"- \"{item}\"")

            st.markdown("#### DOES (Observed Behaviors)")
            for item in empathy["does"]:
                st.write(f"- {item}")

        with col_e2:
            st.markdown("#### THINKS (Internal Doubts & Beliefs)")
            for item in empathy["thinks"]:
                st.write(f"- \"{item}\"")

            st.markdown("#### FEELS (Emotional State)")
            for item in empathy["feels"]:
                st.write(f"- {item}")

        st.markdown("---")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### Core Operational Pain Points")
            for pp in empathy["core_pain_points"]:
                st.write(f"- {pp}")
        with col_p2:
            st.markdown("#### Latent Unmet Needs")
            for ln in empathy["latent_unmet_needs"]:
                st.write(f"- {ln}")

    # TAB 2: DEFINE
    with tab2:
        st.subheader("Stage 2: Point-of-View (POV) and Problem Spaces")
        st.markdown("#### Point-of-View (POV) Statement")
        st.markdown(f'<div class="pov-box">{framing["pov_statement"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Prioritized 'How Might We' (HMW) Opportunity Spaces")
        for i, hmw in enumerate(framing["hmw_opportunity_spaces"], 1):
            st.write(f"**HMW {i}:** {hmw}")

        st.markdown("---")
        st.markdown("#### Engineering & Context Constraints")
        for c in framing["engineering_constraints"]:
            st.write(f"- {c}")

    # TAB 3: IDEATE
    with tab3:
        st.subheader("Stage 3: Divergent Concept Architectures")
        for concept in res["concepts"]:
            is_winner = concept["concept_id"] == winner["concept_id"]
            box_class = "winner-box" if is_winner else "card-empathy"
            badge = " [SELECTED WINNER]" if is_winner else ""

            with st.expander(f"{concept['concept_id']}: {concept['title']}{badge}", expanded=True):
                st.markdown(f"**Working Principle:** {concept['working_principle']}")
                st.markdown(f"**Manufacturing Strategy:** {concept['manufacturing_strategy']}")

                st.markdown("##### Subsystem Functional Decomposition")
                for sub, desc in concept["subsystem_mapping"].items():
                    st.write(f"- **{sub}:** {desc}")

                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("**Key Strengths:**")
                    for p in concept["strengths"]:
                        st.write(f"- {p}")
                with col_c2:
                    st.markdown("**Key Trade-offs:**")
                    for c in concept["limitations"]:
                        st.write(f"- {c}")

    # TAB 4: EVALUATE (PUGH MATRIX)
    with tab4:
        st.subheader("Stage 4: Multi-Criteria Pugh Decision Matrix")
        st.caption("Baseline: Conventional Imported Commercial Unit (Score = 0 across all criteria)")

        # Prepare DataFrame
        df_pugh = pd.DataFrame(pugh)
        df_display = df_pugh[[
            "concept_id", "title", "desirability_score", "feasibility_score",
            "viability_score", "maintainability_score", "total_score", "decision"
        ]].copy()
        df_display.columns = [
            "ID", "Concept Title", "Desirability", "Feasibility",
            "Viability", "Maintainability", "Total Score", "Decision"
        ]
        st.dataframe(df_display, use_container_width=True)

        # Plotly Score Comparison Bar Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[row["title"] for row in pugh],
            y=[row["desirability_score"] for row in pugh],
            name="Desirability (User Value)",
            marker_color="#3B82F6"
        ))
        fig.add_trace(go.Bar(
            x=[row["title"] for row in pugh],
            y=[row["feasibility_score"] for row in pugh],
            name="Feasibility (Local DFM)",
            marker_color="#10B981"
        ))
        fig.add_trace(go.Bar(
            x=[row["title"] for row in pugh],
            y=[row["viability_score"] for row in pugh],
            name="Viability (Capital Cost)",
            marker_color="#F59E0B"
        ))
        fig.add_trace(go.Bar(
            x=[row["title"] for row in pugh],
            y=[row["maintainability_score"] for row in pugh],
            name="Maintainability",
            marker_color="#8B5CF6"
        ))

        fig.update_layout(
            barmode="group",
            title="Pugh Multi-Criteria Comparison Against Baseline",
            yaxis_title="Score Relative to Baseline (-2 to +2)",
            height=400,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 5: PROTOTYPE & PRD
    with tab5:
        st.subheader("Stage 5: Winning Prototype Specification & Bill of Materials")
        st.markdown(f'<div class="winner-box"><strong>Winning Solution:</strong> {proto["winning_concept_title"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Functional Subsystem Breakdown")
        for blk in proto["functional_breakdown"]:
            st.write(f"- {blk}")

        st.markdown("---")
        st.markdown(f"#### Preliminary Bill of Materials (Total: INR {proto['total_estimated_bom_inr']:,.2f})")
        df_bom = pd.DataFrame(proto["bill_of_materials"])
        df_bom.columns = ["Component Item", "Quantity", "Supply Source", "Estimated Cost (INR)"]
        st.dataframe(df_bom, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Rapid Prototyping Roadmap (6-Week Fab Lab Plan)")
        for phase, desc in proto["rapid_prototyping_phases"].items():
            st.write(f"- **{phase}:** {desc}")

        # PRD Download Button
        prd_markdown = f"""# Product Requirement Document (PRD)
Generated by InnoAgent Design Thinking Engine

## 1. Problem Framing
POV Statement: {framing['pov_statement']}

## 2. Selected Concept
Concept: {proto['winning_concept_title']}
Total Pugh Score: +{pugh[0]['total_score']} vs Baseline

## 3. Bill of Materials
Total Estimated BOM: INR {proto['total_estimated_bom_inr']:,.2f}
"""
        st.download_button(
            label="Download Complete Product Requirement Document (PRD)",
            data=prd_markdown,
            file_name="InnoAgent_PRD.md",
            mime="text/markdown"
        )
else:
    st.info("Select a problem scenario in the sidebar and click 'Run Design Thinking Agent Team' to initiate the discovery pipeline.")
