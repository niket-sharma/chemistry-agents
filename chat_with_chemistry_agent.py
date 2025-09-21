#!/usr/bin/env python3
"""
Interactive Chemistry LLM Agent Chat

Run this script to have real conversations with the Chemistry LLM Agent.
Type your chemistry questions and get AI-powered responses with reasoning.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from chemistry_agents.agents.chemistry_llm_agent import ChemistryLLMAgent
    from chemistry_agents.agents.base_agent import AgentConfig
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you have all dependencies installed")
    sys.exit(1)

class ChemistryChatInterface:
    """Interactive chat interface for the Chemistry LLM Agent"""

    def __init__(self):
        self.agent = None
        self.setup_agent()

    def setup_agent(self):
        """Initialize the chemistry agent"""
        print("🔬 Setting up Chemistry LLM Agent...")

        try:
            config = AgentConfig(
                device="cpu",
                log_level="INFO",
                cache_predictions=True
            )

            self.agent = ChemistryLLMAgent(config)
            print("✅ Agent ready for conversation!")

        except Exception as e:
            print(f"❌ Failed to setup agent: {e}")
            sys.exit(1)

    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*70)
        print("  🧪 CHEMISTRY LLM AGENT - Interactive Chat")
        print("="*70)
        print("Ask me anything about chemistry! I can:")
        print("  • Analyze molecular properties and structures")
        print("  • Explain chemical concepts and mechanisms")
        print("  • Compare different molecules")
        print("  • Predict solubility, toxicity, and other properties")
        print("  • Help with chemical engineering questions")
        print()
        print("Example questions:")
        print("  - 'What makes benzene toxic?'")
        print("  - 'Compare the solubility of ethanol and benzene'")
        print("  - 'Why are alcohols water-soluble?'")
        print("  - 'Is CCO drug-like?'")
        print("  - 'Analyze c1ccccc1 for toxicity'")
        print()
        print("Commands:")
        print("  - 'quit' or 'exit' to end conversation")
        print("  - 'trace' to see my reasoning for the last response")
        print("  - 'history' to see conversation history")
        print("  - 'reset' to start a new conversation")
        print("  - 'help' to see this message again")
        print("="*70)

    def print_help(self):
        """Print help message"""
        print("\n📚 Chemistry LLM Agent Help")
        print("-"*40)
        print("I'm an AI agent that can reason about chemistry using multiple tools.")
        print()
        print("🧪 What I can do:")
        print("  • Natural language chemistry conversations")
        print("  • Multi-step reasoning and problem solving")
        print("  • Molecular structure analysis")
        print("  • Property predictions (when models are loaded)")
        print("  • Chemical engineering calculations")
        print("  • Explain chemistry concepts and mechanisms")
        print()
        print("🎯 Tips for better conversations:")
        print("  • Be specific about what you want to know")
        print("  • Use SMILES notation for molecules (e.g., CCO for ethanol)")
        print("  • Ask 'why' questions for detailed explanations")
        print("  • Request comparisons between molecules")
        print("  • Ask me to explain my reasoning")
        print()
        print("⚠️  Note: Some features require prediction models to be loaded.")

    def show_reasoning_trace(self):
        """Show the reasoning trace from the last response"""
        trace = self.agent.get_reasoning_trace()
        if not trace:
            print("No reasoning trace available for the last response.")
            return

        print(f"\n🧠 My Reasoning Process ({len(trace)} steps):")
        print("-"*50)
        for step in trace:
            print(f"{step.step_number}. {step.description}")
            if step.tool_used:
                print(f"   └─ Tool used: {step.tool_used}")
            if step.result and isinstance(step.result, (list, dict)):
                if isinstance(step.result, list) and len(step.result) <= 3:
                    print(f"   └─ Result: {step.result}")
                elif isinstance(step.result, dict) and len(step.result) <= 3:
                    print(f"   └─ Result: {step.result}")
        print("-"*50)

    def show_history(self):
        """Show conversation history"""
        history = self.agent.get_conversation_history()
        if not history:
            print("No conversation history yet.")
            return

        print(f"\n📚 Conversation History ({len(history)} messages):")
        print("-"*50)
        for i, msg in enumerate(history, 1):
            role_emoji = "👤" if msg.role == "user" else "🤖"
            content = msg.content
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"{i}. {role_emoji} {msg.role.title()}: {content}")
        print("-"*50)

    def reset_conversation(self):
        """Reset the conversation"""
        self.agent.reset_conversation()
        print("🔄 Conversation reset. Starting fresh!")

    def process_user_input(self, user_input):
        """Process user input and return response"""
        user_input = user_input.strip()

        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            return None

        elif user_input.lower() == 'trace':
            self.show_reasoning_trace()
            return ""

        elif user_input.lower() == 'history':
            self.show_history()
            return ""

        elif user_input.lower() == 'reset':
            self.reset_conversation()
            return ""

        elif user_input.lower() == 'help':
            self.print_help()
            return ""

        # Process chemistry query
        try:
            print("🤖 Thinking...")
            response = self.agent.chat(user_input)
            return response

        except Exception as e:
            return f"❌ I encountered an error: {e}\n\nThis might happen if prediction models aren't loaded. I can still help with general chemistry questions and structural analysis!"

    def run(self):
        """Run the interactive chat"""
        self.print_welcome()

        while True:
            try:
                # Get user input
                print("\n" + "-"*70)
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # Process input
                response = self.process_user_input(user_input)

                if response is None:  # quit command
                    break

                if response:  # Non-empty response
                    print(f"\n🤖 Agent: {response}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting about chemistry!")
                break

            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Type 'help' for assistance or 'quit' to exit.")

def run_example_conversations():
    """Run some example conversations to demonstrate capabilities"""
    print("🎯 Running Example Conversations")
    print("="*50)

    interface = ChemistryChatInterface()

    example_queries = [
        "What is ethanol?",
        "Why is benzene toxic?",
        "Compare CCO and c1ccccc1",
        "What functional groups are in CC(=O)O?",
        "How do alcohols dissolve in water?"
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n💬 Example {i}: {query}")
        print("🤖 Response: ", end="")

        try:
            response = interface.agent.chat(query)
            # Print first 200 characters
            print(response[:200] + ("..." if len(response) > 200 else ""))

            # Show reasoning trace for first example
            if i == 1:
                print("\n🧠 Reasoning trace for this example:")
                interface.show_reasoning_trace()

        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        run_example_conversations()
    else:
        interface = ChemistryChatInterface()
        interface.run()

if __name__ == "__main__":
    main()