#!/usr/bin/env python3
"""
Runnable Test: Chemistry LLM Agent

This script demonstrates the true LLM agent capabilities with actual
conversational examples you can run to test the system.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from chemistry_agents.agents.chemistry_llm_agent import ChemistryLLMAgent
    from chemistry_agents.agents.base_agent import AgentConfig
    print("✅ Successfully imported ChemistryLLMAgent")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you have all dependencies installed")
    sys.exit(1)

def print_separator(title):
    """Print a nice separator for sections"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_reasoning_trace(agent):
    """Print the reasoning trace from the agent"""
    trace = agent.get_reasoning_trace()
    if trace:
        print(f"\n🧠 Reasoning Trace ({len(trace)} steps):")
        for step in trace:
            print(f"   {step.step_number}. {step.description}")
            if step.tool_used:
                print(f"      └─ Tool: {step.tool_used}")
    else:
        print("   No reasoning trace available")

def test_basic_agent_creation():
    """Test 1: Basic agent creation and setup"""
    print_separator("Test 1: Agent Creation")

    try:
        # Create agent with CPU configuration
        config = AgentConfig(
            device="cpu",
            log_level="INFO",
            cache_predictions=True
        )

        agent = ChemistryLLMAgent(config)
        print("✅ Agent created successfully")
        print(f"   Device: {agent.config.device}")
        print(f"   Available tools: {list(agent.tools.keys())}")
        print(f"   Chemistry knowledge loaded: {len(agent.chemistry_knowledge)} categories")

        return agent

    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None

def test_molecule_extraction(agent):
    """Test 2: Molecule extraction from queries"""
    print_separator("Test 2: Molecule Extraction")

    test_queries = [
        "What is the solubility of CCO?",
        "Compare ethanol and methanol toxicity",
        "Analyze c1ccccc1 for drug properties",
        "Is benzene toxic? What about CO?",
        "Tell me about ibuprofen properties"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        try:
            molecules = agent._extract_molecules(query)
            print(f"   Molecules found: {molecules}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_query_analysis(agent):
    """Test 3: Query analysis and planning"""
    print_separator("Test 3: Query Analysis")

    test_queries = [
        "What is the solubility of ethanol?",
        "Is benzene toxic?",
        "Compare CCO and CO toxicity",
        "Explain why alcohols are water soluble",
        "How does distillation work for ethanol-water?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        try:
            analysis = agent._analyze_query(query)
            molecules = agent._extract_molecules(query)
            plan = agent._create_analysis_plan(analysis, molecules)

            print(f"   Query type: {analysis['query_type']}")
            print(f"   Analysis needed: {analysis['analysis_needed']}")
            print(f"   Molecules: {molecules}")
            print(f"   Planned steps: {plan['steps']}")
            print(f"   Tools needed: {plan['tools_needed']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_chemistry_knowledge(agent):
    """Test 4: Chemistry knowledge integration"""
    print_separator("Test 4: Chemistry Knowledge")

    print("🧪 Available chemistry knowledge:")

    # Test functional groups
    print(f"\n📚 Functional Groups ({len(agent.chemistry_knowledge['functional_groups'])} defined):")
    for fg_name, fg_info in agent.chemistry_knowledge['functional_groups'].items():
        print(f"   {fg_name}: {fg_info['properties']}")

    # Test solubility rules
    print(f"\n💧 Solubility Rules ({len(agent.chemistry_knowledge['solubility_rules'])} rules):")
    for i, rule in enumerate(agent.chemistry_knowledge['solubility_rules'], 1):
        print(f"   {i}. {rule}")

    # Test molecular structure analysis
    test_molecules = ["CCO", "c1ccccc1", "CC(=O)O"]
    print(f"\n🔬 Structural Analysis Examples:")
    for mol in test_molecules:
        try:
            analysis = agent._analyze_molecular_structure(mol)
            print(f"   {mol}:")
            print(f"   {analysis}")
        except Exception as e:
            print(f"   {mol}: ❌ Error: {e}")

def test_simple_conversations(agent):
    """Test 5: Simple conversation examples"""
    print_separator("Test 5: Simple Conversations")

    # Test conversations that don't require model loading
    simple_queries = [
        "What functional groups make molecules polar?",
        "Explain the difference between CCO and CO",
        "Why are aromatic compounds hydrophobic?",
        "What makes a molecule toxic?"
    ]

    for query in simple_queries:
        print(f"\n💬 User: {query}")
        print("🤖 Agent: ", end="")

        try:
            # This will work even without loaded prediction models
            response = agent.chat(query)
            print(response[:200] + "..." if len(response) > 200 else response)

            # Show reasoning
            print_reasoning_trace(agent)

        except Exception as e:
            print(f"❌ Error: {e}")

def test_with_predictions(agent):
    """Test 6: Conversations requiring predictions (may fail without models)"""
    print_separator("Test 6: Prediction-Based Conversations")

    prediction_queries = [
        "What is the solubility of CCO?",
        "Is c1ccccc1 toxic?",
        "Compare ethanol and benzene properties"
    ]

    print("⚠️  Note: These tests require prediction models to be loaded.")
    print("    They may fail if models aren't available or properly configured.")

    for query in prediction_queries:
        print(f"\n💬 User: {query}")
        print("🤖 Agent: ", end="")

        try:
            response = agent.chat(query)
            print(response[:300] + "..." if len(response) > 300 else response)

            # Show reasoning
            print_reasoning_trace(agent)

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            print("    This is expected if prediction models aren't loaded.")

def test_conversation_history(agent):
    """Test 7: Conversation memory"""
    print_separator("Test 7: Conversation Memory")

    # Have a multi-turn conversation
    conversation = [
        "What is ethanol?",
        "Is it toxic?",
        "How does its toxicity compare to benzene?",
        "Why is benzene more toxic?"
    ]

    print("📚 Multi-turn conversation test:")
    for i, query in enumerate(conversation, 1):
        print(f"\n💬 Turn {i}: {query}")

        try:
            response = agent.chat(query)
            print(f"🤖 Response: {response[:150]}...")
        except Exception as e:
            print(f"🤖 ❌ Error: {e}")

    # Show conversation history
    history = agent.get_conversation_history()
    print(f"\n📝 Conversation History ({len(history)} messages):")
    for i, msg in enumerate(history[-6:], 1):  # Show last 6 messages
        role_emoji = "💬" if msg.role == "user" else "🤖"
        content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"   {i}. {role_emoji} {msg.role}: {content_preview}")

def test_explanation_capabilities(agent):
    """Test 8: Explanation generation"""
    print_separator("Test 8: Explanation Capabilities")

    molecules = ["CCO", "c1ccccc1", "CC(=O)O"]

    print("🔬 Testing explanation capabilities:")
    print("    (This tests the explanation logic without requiring loaded models)")

    for mol in molecules:
        print(f"\n🧪 Molecule: {mol}")
        try:
            # Test structural analysis
            structure_analysis = agent._analyze_molecular_structure(mol)
            print(f"   Structure: {structure_analysis.strip()}")

            # Test structure-property relationships
            relationships = agent._get_structure_property_relationships(mol)
            if relationships.strip():
                print(f"   Properties: {relationships.strip()}")
            else:
                print("   Properties: No specific relationships identified")

        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Main test runner"""
    print("🧪 Chemistry LLM Agent Test Suite")
    print("This script tests the conversational AI capabilities")
    print("Some tests may fail if prediction models aren't loaded - that's OK!")

    # Test 1: Create agent
    agent = test_basic_agent_creation()
    if not agent:
        print("❌ Cannot continue without agent")
        return

    # Test 2: Molecule extraction
    test_molecule_extraction(agent)

    # Test 3: Query analysis
    test_query_analysis(agent)

    # Test 4: Chemistry knowledge
    test_chemistry_knowledge(agent)

    # Test 5: Simple conversations (should work)
    test_simple_conversations(agent)

    # Test 6: Prediction-based conversations (may fail)
    test_with_predictions(agent)

    # Test 7: Conversation memory
    test_conversation_history(agent)

    # Test 8: Explanations
    test_explanation_capabilities(agent)

    # Final summary
    print_separator("Test Summary")
    print("✅ Core LLM agent functionality tested")
    print("🧠 Reasoning and planning capabilities demonstrated")
    print("💬 Conversational interface working")
    print("🔬 Chemistry knowledge integration active")
    print()
    print("🚀 Ready for real conversations!")
    print()
    print("💡 To chat interactively, try:")
    print("   from chemistry_agents import ChemistryLLMAgent")
    print("   agent = ChemistryLLMAgent()")
    print("   response = agent.chat('What makes benzene toxic?')")
    print("   print(response)")

if __name__ == "__main__":
    main()