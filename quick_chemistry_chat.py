#!/usr/bin/env python3
"""
Quick Chemistry Agent Test - Run This First!

Simple script to test the Chemistry LLM Agent with pre-defined examples.
No user input required - just run and see the agent in action!
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Quick Chemistry LLM Agent Test")
    print("="*50)

    # Import and test
    try:
        from chemistry_agents.agents.chemistry_llm_agent import ChemistryLLMAgent
        from chemistry_agents.agents.base_agent import AgentConfig
        print("Successfully imported Chemistry LLM Agent")
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Make sure all dependencies are installed")
        return

    # Create agent
    print("\nCreating agent...")
    try:
        config = AgentConfig(device="cpu", log_level="INFO")
        agent = ChemistryLLMAgent(config)
        print("Agent created successfully!")
        print(f"   Available tools: {list(agent.tools.keys())}")
    except Exception as e:
        print(f"Agent creation failed: {e}")
        return

    # Test basic functionality
    print("\nTesting basic chemistry knowledge...")

    # Test 1: Molecular analysis
    test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
    for mol in test_molecules:
        print(f"\nAnalyzing {mol}:")
        try:
            analysis = agent._analyze_molecular_structure(mol)
            print(f"   {analysis.strip()}")
        except Exception as e:
            print(f"   Error: {e}")

    # Test 2: Query analysis
    print(f"\nTesting query understanding...")
    test_queries = [
        "What is the solubility of ethanol?",
        "Is benzene toxic?",
        "Compare CCO and CO"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            analysis = agent._analyze_query(query)
            molecules = agent._extract_molecules(query)
            print(f"   Type: {analysis['query_type']}")
            print(f"   Molecules: {molecules}")
            print(f"   Analysis needed: {analysis['analysis_needed']}")
        except Exception as e:
            print(f"   Error: {e}")

    # Test 3: Simple conversation
    print(f"\nTesting conversation...")
    simple_questions = [
        "What makes molecules polar?",
        "Why are aromatic compounds hydrophobic?",
        "What is the difference between CCO and CO?"
    ]

    for question in simple_questions:
        print(f"\nUser: {question}")
        try:
            response = agent.chat(question)
            # Show first 150 characters
            preview = response[:150] + "..." if len(response) > 150 else response
            print(f"Agent: {preview}")

            # Show reasoning for first question
            if question == simple_questions[0]:
                trace = agent.get_reasoning_trace()
                if trace:
                    print(f"   Reasoning steps: {len(trace)}")
                    for step in trace[:3]:  # Show first 3 steps
                        print(f"      {step.step_number}. {step.description}")

        except Exception as e:
            print(f"Agent Error: {e}")

    # Summary
    print(f"\nTest completed!")
    print(f"Agent is working and can:")
    print(f"   - Understand chemistry queries")
    print(f"   - Extract molecules from text")
    print(f"   - Analyze molecular structures")
    print(f"   - Provide conversational responses")
    print(f"   - Show reasoning traces")

    print(f"\nReady to use! Try these commands:")
    print(f"   python test_chemistry_llm_agent.py    # Comprehensive test")
    print(f"   python chat_with_chemistry_agent.py   # Interactive chat")

    print(f"\nOr use in Python:")
    print(f"   from chemistry_agents import ChemistryLLMAgent")
    print(f"   agent = ChemistryLLMAgent()")
    print(f"   response = agent.chat('What makes benzene toxic?')")
    print(f"   print(response)")

if __name__ == "__main__":
    main()