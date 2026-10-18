"""
main.py
CLI test runner for InnoAgent.
Demonstrates the multi-agent design thinking discovery workflow.
"""

from agent_pipeline import InnoAgentOrchestrator

def main():
    print("=================================================================")
    print("     InnoAgent: Autonomous Multi-Agent Design Thinking Engine    ")
    print("=================================================================\n")

    sample_notes = """
    Field visits to primary healthcare sub-centres in rural districts.
    Clinicians reported that grid electricity is unavailable for 6 to 10 hours daily.
    Conventional commercial oxygen concentrators shut down during blackouts, leaving patients
    in severe respiratory distress. Transporting heavy steel cylinders over unpaved rural roads
    is expensive and takes days. Clinics urgently requested a robust, low-resource oxygen
    generation mechanism that can be maintained locally without specialist imported tools.
    """

    orchestrator = InnoAgentOrchestrator()
    print("Executing 5-Stage Design Thinking Pipeline...")
    results = orchestrator.run_pipeline(sample_notes)

    print("\n--- AGENT EXECUTION TRACE ---")
    for step in results["agent_trace"]:
        print(f"[{step['stage']}] {step['agent']}:")
        print(f"  Action: {step['action']}")
        print(f"  Output: {step['key_finding']}\n")

    print("--- WINNING CONCEPT SPECIFICATION ---")
    winner = results["winning_concept"]
    proto = results["prototyping_plan"]
    print(f"Selected Concept: {winner['title']}")
    print(f"Working Principle: {winner['working_principle']}")
    print(f"Estimated Prototype BOM: INR {proto['total_estimated_bom_inr']:,.2f}")
    print(f"Manufacturing Strategy: {winner['manufacturing_strategy']}")
    print("=================================================================")

if __name__ == "__main__":
    main()
