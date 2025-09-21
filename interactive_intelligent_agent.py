#!/usr/bin/env python3
"""
Interactive Intelligent ChemBERTa Agent

Chat directly with the intelligent agent and see:
- Real-time task detection
- Model routing decisions
- ChemBERTa molecular analysis
- Reasoning traces
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def interactive_chat():
    """Interactive chat with intelligent ChemBERTa agent"""

    print("INTERACTIVE INTELLIGENT CHEMBERTA AGENT")
    print("=" * 60)
    print("Chat with the intelligent agent that automatically routes queries!")
    print()

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create intelligent agent
        print("🤖 Initializing intelligent ChemBERTa agent...")
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        print(f"✅ Agent ready!")
        print(f"   Base ChemBERTa: {'Loaded' if agent.is_loaded else 'Not Loaded'}")
        print(f"   Model parameters: {agent._count_parameters():,}")
        print(f"   Specialized models: {len(agent.specialized_models)} available")

        if len(agent.specialized_models) > 0:
            print(f"   Available: {', '.join(agent.specialized_models.keys())}")
        else:
            print("   No specialized models (will use general ChemBERTa)")

        print()
        print("=" * 60)
        print("EXAMPLE QUERIES TO TRY:")
        print("📊 Task-Specific Queries:")
        print("   'How toxic is benzene?'")
        print("   'What's the water solubility of aspirin?'")
        print("   'Is ibuprofen bioactive?'")
        print("   'Can caffeine dissolve in water?'")
        print()
        print("🧬 General ChemBERTa Queries:")
        print("   'Find molecules similar to caffeine'")
        print("   'Compare benzene and toluene similarity'")
        print("   'Cluster these molecules: CCO, CCN, c1ccccc1'")
        print()
        print("💡 Mixed Queries:")
        print("   'How soluble and toxic is ethanol?'")
        print("   'What's the bioactivity and solubility of aspirin?'")
        print()
        print("=" * 60)
        print("Type 'help' for more examples, 'debug' for detailed analysis, 'quit' to exit")
        print("=" * 60)

        # Interactive loop
        conversation_count = 0
        while True:
            try:
                # Get user input
                user_input = input(f"\n[{conversation_count + 1}] You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for chatting with the intelligent ChemBERTa agent!")
                    break

                elif user_input.lower() == 'help':
                    show_help()
                    continue

                elif user_input.lower() == 'debug':
                    debug_mode = True
                    print("🔍 Debug mode: Will show detailed analysis")
                    continue

                elif user_input.lower() == 'clear':
                    agent.reset_conversation()
                    print("🗑️ Conversation history cleared")
                    continue

                # Show task detection BEFORE sending to agent
                print(f"\n🔍 ANALYZING QUERY...")
                task_type, confidence = agent.detect_task_type(user_input)
                molecules = agent._extract_molecules_from_query(user_input)

                print(f"   Task detected: {task_type} (confidence: {confidence:.2f})")
                if molecules:
                    print(f"   Molecules found: {', '.join(molecules)}")
                else:
                    print(f"   Molecules found: None")

                # Show routing decision
                if task_type in ['toxicity', 'solubility', 'bioactivity']:
                    if task_type in agent.specialized_models:
                        print(f"   🎯 Routing: Using specialized {task_type} ChemBERTa model")
                    else:
                        print(f"   🔄 Routing: Would use {task_type} model → Fallback to general ChemBERTa")
                else:
                    print(f"   🧬 Routing: Using general ChemBERTa for molecular analysis")

                print(f"\n🤖 Agent Response:")
                print("-" * 40)

                # Get agent response
                response = agent.chat(user_input)
                print(response)

                # Show reasoning trace
                trace = agent.get_reasoning_trace()
                print(f"\n🧠 Reasoning Process ({len(trace)} steps):")
                for step in trace:
                    print(f"   {step.step_number}. {step.description}")

                conversation_count += 1

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ChemBERTa model is downloaded")
        print("2. Check if all dependencies are installed")
        print("3. Try running: python test_chemberta_simple.py")

def show_help():
    """Show help with example queries"""

    print("\n" + "=" * 60)
    print("HELP - EXAMPLE QUERIES TO TEST")
    print("=" * 60)

    examples = [
        {
            "category": "🧪 TOXICITY QUERIES",
            "queries": [
                "Is benzene toxic?",
                "How dangerous is ethanol?",
                "What's the LD50 of aspirin?",
                "Is this compound carcinogenic?"
            ]
        },
        {
            "category": "💧 SOLUBILITY QUERIES",
            "queries": [
                "How soluble is caffeine?",
                "Can aspirin dissolve in water?",
                "What's the aqueous solubility of ibuprofen?",
                "Is this compound hydrophilic?"
            ]
        },
        {
            "category": "⚡ BIOACTIVITY QUERIES",
            "queries": [
                "Is ibuprofen bioactive?",
                "What's the therapeutic efficacy of aspirin?",
                "How pharmacologically active is this drug?",
                "What's the IC50 of this compound?"
            ]
        },
        {
            "category": "🧬 MOLECULAR SIMILARITY",
            "queries": [
                "Find molecules similar to caffeine",
                "Compare benzene and toluene",
                "What's the similarity between ethanol and methanol?",
                "Cluster these: CCO, CCN, c1ccccc1"
            ]
        },
        {
            "category": "🔬 COMPLEX QUERIES",
            "queries": [
                "Is aspirin both soluble and safe?",
                "Compare the toxicity and bioactivity of benzene",
                "How does caffeine's solubility affect its activity?"
            ]
        }
    ]

    for example in examples:
        print(f"\n{example['category']}")
        print("-" * 40)
        for query in example['queries']:
            print(f"   '{query}'")

    print(f"\n📋 SPECIAL COMMANDS:")
    print("-" * 20)
    print("   'help' - Show this help")
    print("   'debug' - Enable detailed analysis")
    print("   'clear' - Clear conversation history")
    print("   'quit' - Exit the chat")
    print("\n" + "=" * 60)

def quick_test():
    """Quick test of specific queries"""

    print("\n" + "=" * 60)
    print("QUICK TEST MODE")
    print("=" * 60)

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        agent = create_intelligent_chemberta_agent(AgentConfig(device="cpu", log_level="WARNING"))

        # Quick test queries
        test_queries = [
            "How toxic is benzene?",
            "What's the water solubility of aspirin?",
            "Find molecules similar to caffeine"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 TEST {i}: {query}")
            print("-" * 30)

            # Show detection
            task_type, confidence = agent.detect_task_type(query)
            molecules = agent._extract_molecules_from_query(query)
            print(f"Task: {task_type} (confidence: {confidence:.2f})")
            print(f"Molecules: {molecules}")

            # Get response
            response = agent.chat(query)
            lines = response.split('\n')
            print("Response preview:")
            for line in lines[:8]:
                print(f"  {line}")
            if len(lines) > 8:
                print("  ... (continues)")

        print(f"\n✅ Quick test complete! Use interactive mode for full conversations.")

    except Exception as e:
        print(f"Quick test failed: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        interactive_chat()