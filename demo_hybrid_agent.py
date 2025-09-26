#!/usr/bin/env python3
"""
Demo: Hybrid LLM + ChemBERT Agent

Demonstrates how LLM reasoning combines with ChemBERT molecular intelligence
for complex chemistry tasks that neither could solve alone effectively.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demonstrate_hybrid_approach():
    """Demonstrate the power of combining LLM + ChemBERT"""

    print("HYBRID LLM + CHEMBERTA AGENT DEMO")
    print("=" * 60)
    print("Showing how LLM reasoning + ChemBERT molecular intelligence")
    print("solves complex chemistry problems neither could handle alone!")
    print()

    # Load our ChemBERT agent
    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        print("[+] Loading ChemBERT Agent...")
        config = AgentConfig(device="cpu", log_level="WARNING")
        chemberta_agent = create_intelligent_chemberta_agent(config)
        print(f"[+] ChemBERT loaded ({chemberta_agent._count_parameters():,} parameters)")
        print(f"   Specialized models: {len(chemberta_agent.specialized_models)}")
        print()

    except Exception as e:
        print(f"[-] Failed to load ChemBERT: {e}")
        return

    # Demo scenarios showing hybrid benefits
    demo_scenarios = [
        {
            "title": "Drug Design with Safety Analysis",
            "user_query": "I need a painkiller similar to ibuprofen but safer for long-term use",
            "hybrid_workflow": [
                ("LLM Analysis", "Understands drug design requirements, safety concerns"),
                ("ChemBERT", "Finds molecules similar to ibuprofen"),
                ("ChemBERT", "Predicts toxicity and bioactivity for candidates"),
                ("LLM Synthesis", "Explains structure-safety relationships"),
                ("LLM Result", "Recommends safer alternatives with reasoning")
            ]
        },
        {
            "title": "Chemistry Education with Examples",
            "user_query": "Why do some drugs have better bioavailability? Show me molecular examples.",
            "hybrid_workflow": [
                ("LLM Analysis", "Explains bioavailability principles (absorption, metabolism)"),
                ("ChemBERT", "Predicts bioactivity and solubility for example drugs"),
                ("LLM Synthesis", "Links molecular properties to bioavailability theory"),
                ("LLM Result", "Educational explanation with computational validation")
            ]
        },
        {
            "title": "Environmental Safety Assessment",
            "user_query": "Is this industrial solvent safe for environmental release?",
            "hybrid_workflow": [
                ("LLM Analysis", "Identifies environmental concerns, regulations"),
                ("ChemBERT", "Predicts toxicity, bioactivity, persistence"),
                ("LLM Synthesis", "Interprets predictions in regulatory context"),
                ("LLM Result", "Complete environmental safety assessment")
            ]
        }
    ]

    for i, scenario in enumerate(demo_scenarios, 1):
        print(f">> DEMO {i}: {scenario['title']}")
        print("-" * 50)
        print(f"[USER] User Query: \"{scenario['user_query']}\"")
        print()

        # Show the hybrid workflow
        print("[WORKFLOW] Hybrid Workflow:")
        for step_name, description in scenario['hybrid_workflow']:
            print(f"   {step_name}: {description}")
        print()

        # Demonstrate actual ChemBERT component
        if i == 1:  # Show real ChemBERT analysis for first demo
            print("[CHEMBERTA] Real ChemBERT Analysis:")
            test_query = "How toxic is ibuprofen compared to aspirin?"

            # Get task detection
            task_type, confidence = chemberta_agent.detect_task_type(test_query)
            print(f"   Task Detection: {task_type} (confidence: {confidence:.2f})")

            # Get molecules
            molecules = chemberta_agent._extract_molecules_from_query(test_query)
            print(f"   Molecules Found: {molecules}")

            # Show routing decision
            if task_type in chemberta_agent.specialized_models and confidence > 0.3:
                print(f"   Routing: Using specialized {task_type} model")
            else:
                print(f"   Routing: Using general ChemBERT")

            print()

        # Show what the final hybrid result would look like
        print("[RESULT] Expected Hybrid Result:")
        if i == 1:  # Drug design example
            print("""   "Based on molecular analysis, I recommend considering naproxen or
   celecoxib as safer long-term alternatives to ibuprofen. ChemBERT
   predictions show naproxen has 23% lower toxicity while maintaining
   85% of ibuprofen's anti-inflammatory activity. The key difference
   is the longer half-life (12-17 hours vs 2-4 hours), reducing
   dosing frequency and cumulative toxicity risk..."

   [Includes both computational predictions AND pharmacological reasoning]""")

        elif i == 2:  # Education example
            print("""   "Bioavailability depends on molecular properties like solubility and
   permeability. For example, ChemBERT predicts aspirin has moderate
   solubility (logS: -2.3) but good bioactivity (0.8), while ibuprofen
   shows lower solubility (-4.1) but similar bioactivity. This explains
   why aspirin works faster (better dissolution) but both are effective..."

   [Combines educational theory with computational validation]""")

        elif i == 3:  # Environmental example
            print("""   "This solvent shows moderate environmental risk. ChemBERT predictions
   indicate low acute toxicity (0.3) but moderate bioactivity (0.6),
   suggesting potential for bioaccumulation. EPA guidelines recommend
   concentrations below 50 ppm for aquatic environments. Consider
   biodegradable alternatives like ethyl lactate..."

   [Merges computational predictions with regulatory knowledge]""")

        print("\n" + "=" * 60 + "\n")

    # Summary of hybrid advantages
    print("[ADVANTAGES] HYBRID APPROACH ADVANTAGES:")
    print("-" * 40)

    advantages = [
        ("[LLM] LLM Strengths", [
            "Natural language understanding",
            "Chemistry domain knowledge",
            "Regulatory and safety context",
            "Educational explanations",
            "Multi-step reasoning"
        ]),
        ("[CHEMBERTA] ChemBERT Strengths", [
            "Molecular structure understanding",
            "Accurate property predictions",
            "Chemical similarity analysis",
            "SMILES processing",
            "Specialized model routing"
        ]),
        ("[SYNERGY] Hybrid Synergy", [
            "Computational validation of concepts",
            "Context-aware molecular analysis",
            "Comprehensive problem solving",
            "Educational content with examples",
            "Regulatory assessment with predictions"
        ])
    ]

    for title, items in advantages:
        print(f"{title}:")
        for item in items:
            print(f"   • {item}")
        print()

    print("[APPLICATIONS] POTENTIAL APPLICATIONS:")
    print("-" * 30)
    applications = [
        "Drug discovery and optimization",
        "Chemistry education and tutoring",
        "Environmental risk assessment",
        "Materials science research",
        "Regulatory compliance analysis",
        "Synthetic route planning",
        "Chemical safety evaluation",
        "Literature synthesis with validation"
    ]

    for app in applications:
        print(f"   • {app}")

    print("\n" + "=" * 60)
    print("[NEXT] Next Steps:")
    print("   1. Add OpenAI API integration")
    print("   2. Implement hybrid workflow orchestration")
    print("   3. Create specialized hybrid agents for different domains")
    print("   4. Develop UI for hybrid interactions")

def show_implementation_roadmap():
    """Show how to implement the hybrid approach"""

    print("\n" + "[ROADMAP] IMPLEMENTATION ROADMAP")
    print("=" * 40)

    roadmap = [
        {
            "phase": "Phase 1: API Integration",
            "tasks": [
                "Add OpenAI API client integration",
                "Create prompt engineering for chemistry queries",
                "Implement async workflow orchestration",
                "Add error handling and fallbacks"
            ]
        },
        {
            "phase": "Phase 2: Hybrid Workflows",
            "tasks": [
                "Build query analysis pipeline",
                "Create ChemBERT result synthesis",
                "Implement reasoning trace integration",
                "Add confidence scoring for hybrid results"
            ]
        },
        {
            "phase": "Phase 3: Specialized Agents",
            "tasks": [
                "DrugDesignAssistant implementation",
                "ChemistryTutor with examples",
                "SafetyAssessmentAgent",
                "EnvironmentalAnalysisAgent"
            ]
        },
        {
            "phase": "Phase 4: User Interface",
            "tasks": [
                "Web UI for hybrid interactions",
                "Visualization of hybrid reasoning",
                "Interactive molecule exploration",
                "Export and sharing capabilities"
            ]
        }
    ]

    for phase_info in roadmap:
        print(f"\n{phase_info['phase']}:")
        for task in phase_info['tasks']:
            print(f"   [ ] {task}")

    print(f"\n[BENEFITS] Expected Benefits:")
    benefits = [
        "10x improvement in chemistry query handling",
        "Educational content creation automation",
        "Comprehensive safety assessments",
        "Research acceleration through AI reasoning"
    ]

    for benefit in benefits:
        print(f"   • {benefit}")

if __name__ == "__main__":
    demonstrate_hybrid_approach()
    show_implementation_roadmap()