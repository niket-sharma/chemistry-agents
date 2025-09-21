#!/usr/bin/env python3
"""
Simple Chemistry LLM Agent Example

This is the simplest way to use the Chemistry LLM Agent for conversation.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Import the agent
    from chemistry_agents.agents.chemistry_llm_agent import ChemistryLLMAgent
    from chemistry_agents.agents.base_agent import AgentConfig

    # Create the agent
    print("Creating Chemistry LLM Agent...")
    config = AgentConfig(device="cpu")
    agent = ChemistryLLMAgent(config)

    print("Agent ready! Here are some example conversations:\n")

    # Example conversations
    questions = [
        "What makes benzene toxic?",
        "Why is ethanol soluble in water?",
        "Compare CCO and c1ccccc1 properties",
        "What functional groups are in CC(=O)O?",
        "Is ibuprofen drug-like?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"Example {i}:")
        print(f"Question: {question}")

        try:
            # Get the response
            response = agent.chat(question)

            # Show first part of response
            print(f"Answer: {response[:200]}...")

            # Show reasoning trace
            trace = agent.get_reasoning_trace()
            print(f"Reasoning steps: {len(trace)}")

        except Exception as e:
            print(f"Error: {e}")

        print("-" * 50)

    print("\nTo use interactively:")
    print("agent = ChemistryLLMAgent()")
    print("response = agent.chat('Your chemistry question here')")
    print("print(response)")

if __name__ == "__main__":
    main()