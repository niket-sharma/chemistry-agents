#!/usr/bin/env python3
"""
Interactive test script for Intelligent ChemBERTa Agent
Type your own queries to test the intelligent task routing
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Interactive testing of intelligent ChemBERTa agent"""

    print("INTELLIGENT CHEMBERTA AGENT - INTERACTIVE TEST")
    print("=" * 60)
    print("Type your chemistry queries and see the intelligent routing in action!")
    print("Examples:")
    print("- 'Is benzene toxic?'")
    print("- 'How soluble is aspirin in water?'")
    print("- 'What's the bioactivity of caffeine?'")
    print("- 'Find molecules similar to ethanol'")
    print("Type 'quit' to exit")
    print("=" * 60)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        print(f"\n[+] Agent loaded successfully!")
        print(f"[+] Base ChemBERTa: {'Loaded' if agent.is_loaded else 'Not Loaded'}")
        print(f"[+] Specialized Models: {len(agent.specialized_models)} available")

        if not agent.specialized_models:
            print("\n[!] No specialized models found. The agent will use general ChemBERTa.")
            print("   To enable specialized routing, run: python train_task_specific_chemberta.py")

        print("\n" + "=" * 60)

        # Interactive loop
        while True:
            print("\nYour query: ", end="")
            query = input().strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            print("\n" + "-" * 40)

            # Show task detection
            task_type, confidence = agent.detect_task_type(query)
            print(f"[BRAIN] Detected Task: {task_type} (confidence: {confidence:.2f})")

            # Show molecules found
            molecules = agent._extract_molecules_from_query(query)
            if molecules:
                print(f"[CHEM] Molecules Found: {', '.join(molecules)}")

            # Show routing decision
            if task_type in agent.specialized_models and confidence > 0.3:
                print(f"[ROUTE] Using specialized {task_type} ChemBERTa model")
            else:
                print(f"[ROUTE] Using general ChemBERTa analysis")

            print("\n[RESPONSE] Agent Response:")
            print("-" * 20)

            # Get full response
            try:
                response = agent.chat(query)
                print(response)
            except Exception as e:
                print(f"Error: {e}")

            print("\n" + "=" * 60)

    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()