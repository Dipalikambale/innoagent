from agent_pipeline import InnoAgentOrchestrator

orch = InnoAgentOrchestrator()

cases = [
    ("Rural Off-Grid Medical Oxygen Concentrator",
     """Field visits to primary healthcare sub-centres in rural districts.
Clinicians reported that grid electricity is unavailable for 6 to 10 hours daily.
Conventional commercial oxygen concentrators shut down during blackouts, leaving patients
in severe respiratory distress. Transporting heavy steel cylinders over unpaved rural roads
is expensive and takes days. Clinics urgently requested a robust, low-resource oxygen
generation mechanism that can be maintained locally without specialist imported tools."""),
    ("Agricultural Microplastic & Silt Water Filtration",
     """Interviews with agricultural researchers and drip irrigation operators.
Irrigation water from open canals carries severe micro-debris, silt, and bacterial contamination
that rapidly clogs drip emitter lines. Commercial cartridge filters clog within 48 hours and
require costly replacement elements. Farmers need a self-cleaning, low-power continuous
separation system that can operate off-grid and be fabricated in a regional makerspace."""),
    ("Tirupur Textile Back Support",
     "Textile factory workers in Tirupur suffer from chronic back pain because they spend 10 hours standing at sewing machines with no lumbar support. They cannot afford ergonomic chairs. Workers manually fold rags to prop themselves up."),
    ("Vidarbha Grain Storage",
     "Farmers in Vidarbha store grain in gunny sacks that get damaged by moisture and rats. They lose 30 percent of harvest every year. They cannot afford metal silos. Heavy rains flood the storage area."),
    ("Rural School Solar Power",
     "Students in rural Maharashtra walk 4 km to reach the nearest school. The school has no electricity. Teachers cannot run projectors or charge laptops. Solar panels are expensive and fragile."),
]

rows = []
for label, t in cases:
    r = orch.run_pipeline(t)
    rows.append(r)
    print(f"[{label}]")
    print("  Domain  :", r["domain_topic"])
    print("  Winner  :", r["winning_concept"]["title"])
    print("  Concept A:", r["concepts"][0]["title"])
    print("  Concept C:", r["concepts"][2]["title"])
    print("  BOM[0]  :", r["prototyping_plan"]["bill_of_materials"][0]["item"])
    print("  POV      :", r["problem_definition"]["pov_statement"][:140])
    print()

titles = [r["winning_concept"]["title"] for r in rows]
domains = [r["domain_topic"] for r in rows]
boms = [r["prototyping_plan"]["bill_of_materials"][0]["item"] for r in rows]
assert len(set(titles)) == len(titles), f"Winning titles collided: {titles}"
assert len(set(domains)) == len(domains), f"Domains collided: {domains}"
assert len(set(boms)) == len(boms), f"BOM line items collided: {boms}"
print("OK: each problem produced a distinct domain, winner, and BOM.")
