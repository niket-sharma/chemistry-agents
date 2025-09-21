#!/usr/bin/env python3
"""
ChemBERTa Conversational Agent Demo

This demonstrates a TRUE AI agent that uses ChemBERTa for molecular intelligence
combined with conversational capabilities. This is where ChemBERTa BEATS GPT!

ChemBERTa understands 77 MILLION molecules - GPT understands zero!
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("CHEMBERTA CONVERSATIONAL AGENT DEMO")
    print("=" * 60)
    print("True AI agent: ChemBERTa molecular brain + conversational interface!")

    try:
        from chemistry_agents.agents.chemberta_conversational_agent import ChemBERTaConversationalAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create the agent
        config = AgentConfig(device="cpu", log_level="INFO")
        agent = ChemBERTaConversationalAgent(config)

        print(f"\n1. Loading ChemBERTa conversational agent...")
        agent.load_model()

        if not agent.is_loaded:
            print("ChemBERTa model not available")
            print("Try: python download_huggingface_model.py")
            print("\nBut let's show you what it WOULD do...")
            demo_without_model()
            return

        print("ChemBERTa conversational agent ready!")

        # Demo conversations that showcase ChemBERTa's strengths
        demo_conversations = [
            "Find molecules similar to aspirin",
            "Compare the similarity of benzene and toluene",
            "Cluster these molecules: CCO, c1ccccc1, CCN, CC(=O)O",
            "How diverse are ethanol, benzene, and caffeine?",
            "What's the molecular embedding for ibuprofen?"
        ]

        print(f"\n2. ChemBERTa Conversational Demos:")
        print("-" * 50)

        for i, question in enumerate(demo_conversations, 1):
            print(f"\nDemo {i}: {question}")
            print("ChemBERTa Agent:")

            try:
                response = agent.chat(question)
                # Show first part of response
                lines = response.split('\n')
                for line in lines[:10]:  # First 10 lines
                    print(f"    {line}")

                if len(lines) > 10:
                    print("    ... (response continues)")

                # Show reasoning trace
                trace = agent.get_reasoning_trace()
                print(f"\nReasoning: {len(trace)} steps")
                for step in trace:
                    print(f"    {step.step_number}. {step.description}")

            except Exception as e:
                print(f"    Error: {e}")

            print("-" * 50)

        # Show capabilities
        print(f"\n3. Agent Capabilities:")
        capabilities = agent.explain_capabilities()
        print(capabilities)

    except ImportError as e:
        print(f"Import failed: {e}")
        demo_without_model()
    except Exception as e:
        print(f"Demo failed: {e}")

def demo_without_model():
    """Show what the agent would do without model loaded"""
    print(f"\nWHAT CHEMBERTA CONVERSATIONAL AGENT WOULD DO:")
    print("-" * 60)

    examples = [
        {
            "query": "Find molecules similar to aspirin",
            "chemberta_analysis": "ChemBERTa calculates molecular embeddings for aspirin and compares with database",
            "response": "ChemBERTa found: ibuprofen (0.82 similarity), acetaminophen (0.76 similarity)"
        },
        {
            "query": "Cluster these molecules: CCO, c1ccccc1, CCN",
            "chemberta_analysis": "ChemBERTa groups molecules by learned chemical features in 384D space",
            "response": "Cluster 1: [CCO, CCN] (small organics), Cluster 2: [c1ccccc1] (aromatics)"
        },
        {
            "query": "How diverse are benzene, toluene, and phenol?",
            "chemberta_analysis": "ChemBERTa calculates chemical space diversity metrics",
            "response": "Diversity score: 0.65 - Moderately diverse aromatics with different substituents"
        }
    ]

    for example in examples:
        print(f"\nQuery: {example['query']}")
        print(f"ChemBERTa Analysis: {example['chemberta_analysis']}")
        print(f"Response: {example['response']}")

    print(f"\nWHY THIS BEATS GPT:")
    print("- ChemBERTa was trained on 77 MILLION molecules")
    print("- Understands molecular structure and chemical similarity")
    print("- Calculates true chemical relationships, not text patterns")
    print("- Discovers patterns humans would miss")
    print("- Works in 384-dimensional chemical space")

    print(f"\nWHAT GPT CAN'T DO:")
    print("- GPT doesn't understand molecular structure")
    print("- GPT can't calculate chemical similarity")
    print("- GPT lacks chemical intuition")
    print("- GPT wasn't trained on molecular data")

def interactive_chemberta_chat():
    """Interactive chat with ChemBERTa agent"""
    print(f"\n" + "="*60)
    print("INTERACTIVE CHEMBERTA CHAT")
    print("="*60)
    print("Ask molecular intelligence questions!")

    try:
        from chemistry_agents.agents.chemberta_conversational_agent import ChemBERTaConversationalAgent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = ChemBERTaConversationalAgent(AgentConfig(device="cpu"))
        agent.load_model()

        if not agent.is_loaded:
            print("ChemBERTa model not loaded")
            return

        print("ChemBERTa ready for conversation!")
        print("Try: 'Find molecules similar to caffeine'")
        print("Type 'quit' to exit")

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['quit', 'exit', 'bye']:
                    break

                if query:
                    response = agent.chat(query)
                    print(f"ChemBERTa: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nThanks for chatting with ChemBERTa!")

    except Exception as e:
        print(f"Interactive chat failed: {e}")

if __name__ == "__main__":
    main()

    # Uncomment for interactive mode
    # interactive_chemberta_chat()