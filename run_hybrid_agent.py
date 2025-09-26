#!/usr/bin/env python3
"""
Working Hybrid LLM + ChemBERT Agent

This script provides a working implementation of the hybrid agent
that you can use once you add your OpenAI API key to the .env file.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add src to path for proper imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables with override
load_dotenv(override=True)

async def run_hybrid_demo():
    """Run a working demo of the hybrid agent"""

    print("WORKING HYBRID LLM + CHEMBERTA AGENT")
    print("=" * 60)

    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[-] OpenAI API key not found!")
        print("\nTo use the hybrid agent:")
        print("1. Edit the .env file")
        print("2. Add your OpenAI API key:")
        print("   OPENAI_API_KEY=your_actual_key_here")
        print("3. Save the file and run this script again")
        return

    print(f"[+] OpenAI API key found")
    print(f"[MODEL] Model: {os.getenv('OPENAI_MODEL', 'gpt-4')}")
    print(f"[TEMP] Temperature: {os.getenv('OPENAI_TEMPERATURE', '0.1')}")
    print()

    # Initialize the hybrid agent
    try:
        from hybrid_agent_concept import HybridChemistryAgent

        print("[INIT] Initializing Hybrid Agent...")
        agent = HybridChemistryAgent()
        await agent.initialize()

        chemberta_models = len(agent.chemberta_agent.specialized_models)
        print(f"[+] ChemBERT Agent: {agent.chemberta_agent._count_parameters():,} parameters")
        print(f"[MODELS] Specialized Models: {chemberta_models} available")
        print(f"[LLM] OpenAI LLM: Ready")
        print()

    except Exception as e:
        print(f"[-] Failed to initialize hybrid agent: {e}")
        print("\nMake sure you have:")
        print("1. OpenAI API key in .env file")
        print("2. Required packages: pip install openai python-dotenv")
        return

    # Demo queries
    demo_queries = [
        "I need a painkiller similar to ibuprofen but with better water solubility. What are my options?",
        "Why is aspirin more effective than ibuprofen for some people? Explain with molecular data.",
        "Is benzene safe for use in consumer products? Include regulatory considerations.",
        "Compare the toxicity profiles of methanol vs ethanol for industrial applications."
    ]

    print("[DEMO] HYBRID AGENT DEMO QUERIES:")
    print("-" * 40)

    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. {query}")

    print(f"\n{'='*60}")
    print("[ANALYSIS] Running Hybrid Analysis...")
    print("(This combines OpenAI reasoning with ChemBERT predictions)")

    # Test with the first query
    test_query = demo_queries[0]
    print(f"\n[QUERY] Query: {test_query}")
    print("-" * 50)

    try:
        # Run hybrid analysis
        result = await agent.analyze_complex_query(test_query)

        print("[RESULTS] HYBRID ANALYSIS RESULTS:")
        print("-" * 30)
        print(f"[LLM] LLM Reasoning:\n{result.llm_reasoning}")
        print(f"\n[CHEMBERTA] ChemBERT Predictions:\n{result.chemberta_predictions}")
        print(f"\n[SYNTHESIS] Final Synthesis:\n{result.synthesis}")
        print(f"\n[STEPS] Reasoning Steps:")
        for i, step in enumerate(result.reasoning_steps, 1):
            print(f"  {i}. {step}")
        print(f"\n[CONFIDENCE] Confidence: {result.confidence:.2f}")

    except Exception as e:
        print(f"[-] Hybrid analysis failed: {e}")
        print("\nThis might be due to:")
        print("1. OpenAI API key issues")
        print("2. Network connectivity")
        print("3. API rate limits")

def run_simple_demo():
    """Run a simpler demo that doesn't require OpenAI"""

    print("SIMPLIFIED HYBRID DEMO (ChemBERT Only)")
    print("=" * 50)
    print("This demo shows the ChemBERT component without OpenAI integration")
    print()

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Initialize ChemBERT agent
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        print(f"[+] ChemBERT Agent Ready")
        print(f"   Parameters: {agent._count_parameters():,}")
        print(f"   Specialized Models: {len(agent.specialized_models)}")
        print()

        # Test queries
        test_queries = [
            "Is benzene toxic?",
            "How soluble is aspirin in water?",
            "What's the bioactivity of caffeine?"
        ]

        for query in test_queries:
            print(f"[Q] Query: {query}")

            # Show detection and routing
            task_type, confidence = agent.detect_task_type(query)
            molecules = agent._extract_molecules_from_query(query)

            print(f"   [DETECTED] Detected: {task_type} (confidence: {confidence:.2f})")
            print(f"   [MOLECULES] Molecules: {molecules}")

            if task_type in agent.specialized_models and confidence > 0.3:
                print(f"   [ROUTE] Routing: Specialized {task_type} model")
            else:
                print(f"   [ROUTE] Routing: General ChemBERT")
            print()

    except Exception as e:
        print(f"[-] ChemBERT demo failed: {e}")

async def main():
    """Main function"""

    # Check if we can run the full hybrid demo
    if os.getenv('OPENAI_API_KEY'):
        await run_hybrid_demo()
    else:
        print("OpenAI API key not found - running ChemBERT-only demo")
        print("Add OPENAI_API_KEY to .env file for full hybrid functionality")
        print()
        run_simple_demo()

if __name__ == "__main__":
    # Check for required packages
    try:
        import openai
        from dotenv import load_dotenv
        asyncio.run(main())
    except ImportError as e:
        print("[-] Missing required packages!")
        print("Install with: pip install openai python-dotenv")
        print(f"Error: {e}")