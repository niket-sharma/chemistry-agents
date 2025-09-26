#!/usr/bin/env python3
"""
Web UI for Intelligent ChemBERTa Agent & Hybrid LLM+ChemBERT Agent
A user-friendly interface to test both pure ChemBERT and Hybrid LLM agents
"""

import sys
import os
import gradio as gr
import time
import asyncio
from dotenv import load_dotenv
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
load_dotenv(override=True)

# Global agent variables
chemberta_agent = None
hybrid_agent = None
current_agent_type = "chemberta"

def initialize_chemberta_agent():
    """Initialize the intelligent ChemBERTa agent"""
    global chemberta_agent

    if chemberta_agent is not None:
        return "ChemBERT agent already initialized!"

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent with CPU config
        config = AgentConfig(device="cpu", log_level="WARNING")
        chemberta_agent = create_intelligent_chemberta_agent(config)

        specialized_count = len(chemberta_agent.specialized_models)

        status = f"""[+] **ChemBERT Agent Initialized Successfully!**

**Model Details:**
- Base ChemBERTa: {'Loaded' if chemberta_agent.is_loaded else 'Not Loaded'}
- Specialized Models: {specialized_count} available
- Parameters: {chemberta_agent._count_parameters():,}
- Architecture: RoBERTa-based (3.4M params)
- Training Data: 77M SMILES molecules

**Available Tasks:**
- Toxicity Analysis
- Solubility Prediction
- Bioactivity Assessment
- General Chemistry Analysis

**Status:** Ready for chemistry queries!"""

        if specialized_count == 0:
            status += "\n\n[!] **Note:** No specialized models found. Using general ChemBERTa.\nTo enable task-specific routing: `python train_task_specific_chemberta.py`"

        return status

    except Exception as e:
        return f"[-] **ChemBERT Initialization Failed:** {str(e)}"

def initialize_hybrid_agent(openai_api_key=None):
    """Initialize the hybrid LLM+ChemBERT agent"""
    global hybrid_agent

    if hybrid_agent is not None:
        return "Hybrid agent already initialized!"

    # Prioritize user input over .env, but fallback to .env
    env_key = os.getenv('OPENAI_API_KEY')
    api_key = openai_api_key.strip() if openai_api_key and openai_api_key.strip() else env_key

    if not api_key:
        return "[-] **Initialization Failed:** OpenAI API key required for hybrid agent!\n\nPlease:\n1. Add your API key in the field above, or\n2. Add `OPENAI_API_KEY=your_key` to your .env file\n3. Then restart the application"

    # Show which key source is being used
    key_source = "user input" if (openai_api_key and openai_api_key.strip()) else ".env file"

    try:
        from hybrid_agent_concept import HybridChemistryAgent

        # Create hybrid agent
        hybrid_agent = HybridChemistryAgent(openai_api_key=api_key)

        # Initialize asynchronously (we'll handle this in the UI)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(hybrid_agent.initialize())

        chemberta_models = len(hybrid_agent.chemberta_agent.specialized_models)
        chemberta_params = hybrid_agent.chemberta_agent._count_parameters()

        status = f"""[+] **Hybrid LLM+ChemBERT Agent Initialized Successfully!**

**API Key Source:** {key_source}

**LLM Component:**
- Model: {hybrid_agent.openai_model}
- Temperature: {hybrid_agent.openai_temperature}
- Max Tokens: {hybrid_agent.openai_max_tokens}
- Status: Connected ✓

**ChemBERT Component:**
- Parameters: {chemberta_params:,}
- Specialized Models: {chemberta_models} available
- Architecture: RoBERTa-based molecular transformer

**Hybrid Capabilities:**
- Natural language reasoning with chemistry knowledge
- Molecular property predictions from ChemBERT
- Synthesis of computational results with domain expertise
- Complex drug design and safety assessments

**Status:** Ready for advanced chemistry queries!"""

        return status

    except Exception as e:
        return f"[-] **Hybrid Agent Initialization Failed:** {str(e)}\n\nCommon issues:\n- Invalid OpenAI API key\n- Network connectivity\n- Missing dependencies: `pip install openai python-dotenv`"

def process_chemistry_query(query, show_reasoning=False, agent_type="chemberta"):
    """Process a chemistry query and return results"""
    global chemberta_agent, hybrid_agent

    if not query.strip():
        return "", "Please enter a chemistry question!", ""

    # Select the appropriate agent
    if agent_type == "hybrid":
        current_agent = hybrid_agent
        agent_name = "Hybrid LLM+ChemBERT"
    else:
        current_agent = chemberta_agent
        agent_name = "ChemBERT"

    if current_agent is None:
        return "", f"[-] {agent_name} agent not initialized! Initialize the agent first.", ""

    try:
        if agent_type == "hybrid":
            # Handle hybrid agent query processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run hybrid analysis
            result = loop.run_until_complete(current_agent.analyze_complex_query(query))

            # Create analysis info for hybrid
            analysis_info = f"""**[HYBRID] Hybrid Analysis:**
- **Query Type:** Complex reasoning + molecular predictions
- **LLM Model:** {current_agent.openai_model}
- **ChemBERT Models:** {len(current_agent.chemberta_agent.specialized_models)} specialized available
- **Workflow:** LLM reasoning → ChemBERT predictions → Synthesis
"""

            response = result.synthesis

            # Get reasoning trace for hybrid
            reasoning = ""
            if show_reasoning:
                reasoning = "**[HYBRID WORKFLOW] Reasoning Steps:**\n"
                for i, step in enumerate(result.reasoning_steps, 1):
                    reasoning += f"{i}. {step}\n"
                reasoning += f"\n**[LLM REASONING]**\n{result.llm_reasoning}\n"
                reasoning += f"\n**[CHEMBERTA PREDICTIONS]**\n{result.chemberta_predictions}\n"

        else:
            # Handle ChemBERT agent query processing
            task_type, confidence = current_agent.detect_task_type(query)
            molecules = current_agent._extract_molecules_from_query(query)

            # Create analysis info for ChemBERT
            analysis_info = f"""**[CHEMBERTA] Task Analysis:**
- **Detected Task:** {task_type.title()} (confidence: {confidence:.2f})
- **Molecules Found:** {', '.join(molecules) if molecules else 'None detected'}
- **Routing:** {'Specialized ' + task_type + ' model' if task_type in current_agent.specialized_models and confidence > 0.3 else 'General ChemBERTa'}
"""

            # Get full response
            response = current_agent.chat(query)

            # Get reasoning trace for ChemBERT
            reasoning = ""
            if show_reasoning:
                trace = current_agent.get_reasoning_trace()
                reasoning = "**[CHEMBERTA] Reasoning Steps:**\n"
                for step in trace:
                    reasoning += f"{step.step_number}. {step.description}\n"

        return analysis_info, response, reasoning

    except Exception as e:
        return "", f"[-] **Error processing query:** {str(e)}", ""

def get_example_queries(agent_type="chemberta"):
    """Return example queries for users to try"""

    if agent_type == "hybrid":
        return [
            "I need a painkiller similar to ibuprofen but with better water solubility. What are my options?",
            "Why is aspirin more effective than ibuprofen for some people? Explain with molecular data.",
            "Is benzene safe for use in consumer products? Include regulatory considerations.",
            "Compare the toxicity profiles of methanol vs ethanol for industrial applications.",
            "Design a drug variant of acetaminophen with reduced liver toxicity.",
            "Explain bioavailability differences between oral vs topical drug delivery with examples.",
            "Assess the environmental impact of releasing toluene into water systems.",
            "What makes some antibiotics more effective? Compare molecular structures and bioactivity."
        ]
    else:
        return [
            "Is benzene toxic to humans?",
            "How soluble is aspirin in water?",
            "What's the bioactivity of caffeine?",
            "Find molecules similar to ethanol",
            "Is ibuprofen safe for long-term use?",
            "Can acetaminophen dissolve in water?",
            "What's the toxicity of methanol?",
            "Compare the solubility of ethanol and methanol"
        ]

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(
        title="Chemistry AI: ChemBERT + Hybrid LLM Agents",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .chemistry-header {
            text-align: center;
            background: linear-gradient(45deg, #1e3a8a, #3b82f6, #10b981);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .agent-card {
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .hybrid-card {
            border-color: #10b981;
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        }
        .chemberta-card {
            border-color: #3b82f6;
            background: linear-gradient(135deg, #eff6ff, #dbeafe);
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
        <div class="chemistry-header">
            <h1>🧪 Chemistry AI Agents</h1>
            <p>Choose between ChemBERT-only or Hybrid LLM+ChemBERT • Advanced Molecular Intelligence • Natural Language Reasoning</p>
        </div>
        """)

        # Agent Selection Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>🤖 Select Agent Type</h3>")
                agent_type = gr.Radio(
                    choices=[
                        ("🧠 ChemBERT Agent (Molecular predictions only)", "chemberta"),
                        ("🔥 Hybrid LLM+ChemBERT Agent (Reasoning + predictions)", "hybrid")
                    ],
                    value="chemberta",
                    label="Agent Type",
                    interactive=True
                )

        # Configuration sections for different agents
        with gr.Row():
            with gr.Column():
                # ChemBERT Configuration
                with gr.Group():
                    gr.HTML('<div class="agent-card chemberta-card"><h3>🧠 ChemBERT Agent Setup</h3></div>')
                    chemberta_init_btn = gr.Button("Initialize ChemBERT Agent", variant="primary", size="lg")
                    chemberta_status = gr.Markdown("ChemBERT agent not initialized.")

            with gr.Column():
                # Hybrid Agent Configuration
                with gr.Group():
                    gr.HTML('<div class="agent-card hybrid-card"><h3>🔥 Hybrid Agent Setup</h3></div>')

                    # Check if API key exists in .env and show status
                    env_api_key = os.getenv('OPENAI_API_KEY')
                    if env_api_key:
                        gr.Markdown("✅ **API Key Found in .env file**")
                        hybrid_status_initial = "Ready to initialize hybrid agent with .env API key."
                    else:
                        gr.Markdown("⚠️ **No API Key in .env file**")
                        hybrid_status_initial = "Add your OpenAI API key below or to your .env file."

                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key (Optional if set in .env)",
                        placeholder="sk-... (leave blank to use .env key)",
                        type="password",
                        value="",
                        info="Leave blank to use .env file key, or enter key to override"
                    )

                    hybrid_init_btn = gr.Button("Initialize Hybrid Agent", variant="primary", size="lg")
                    hybrid_status = gr.Markdown(hybrid_status_initial)

        gr.HTML("<hr>")

        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>💬 Ask Chemistry Questions</h3>")

                query_input = gr.Textbox(
                    label="Your Chemistry Question",
                    placeholder="Ask anything about chemistry...",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear")

                show_reasoning = gr.Checkbox(
                    label="Show detailed reasoning steps",
                    value=False
                )

                # Dynamic example queries based on agent type
                gr.HTML("<h4>💡 Example Queries:</h4>")

                # Create example query buttons for ChemBERT
                with gr.Group(visible=True) as chemberta_examples:
                    gr.HTML("<p><em>ChemBERT Agent Examples:</em></p>")
                    chemberta_queries = get_example_queries("chemberta")
                    with gr.Row():
                        with gr.Column():
                            for i in range(0, len(chemberta_queries), 2):
                                if i < len(chemberta_queries):
                                    gr.Button(
                                        chemberta_queries[i],
                                        size="sm"
                                    ).click(
                                        lambda query=chemberta_queries[i]: query,
                                        outputs=query_input
                                    )
                        with gr.Column():
                            for i in range(1, len(chemberta_queries), 2):
                                if i < len(chemberta_queries):
                                    gr.Button(
                                        chemberta_queries[i],
                                        size="sm"
                                    ).click(
                                        lambda query=chemberta_queries[i]: query,
                                        outputs=query_input
                                    )

                # Create example query buttons for Hybrid
                with gr.Group(visible=False) as hybrid_examples:
                    gr.HTML("<p><em>Hybrid Agent Examples:</em></p>")
                    hybrid_queries = get_example_queries("hybrid")
                    with gr.Row():
                        with gr.Column():
                            for i in range(0, len(hybrid_queries), 2):
                                if i < len(hybrid_queries):
                                    gr.Button(
                                        hybrid_queries[i][:80] + ("..." if len(hybrid_queries[i]) > 80 else ""),
                                        size="sm"
                                    ).click(
                                        lambda query=hybrid_queries[i]: query,
                                        outputs=query_input
                                    )
                        with gr.Column():
                            for i in range(1, len(hybrid_queries), 2):
                                if i < len(hybrid_queries):
                                    gr.Button(
                                        hybrid_queries[i][:80] + ("..." if len(hybrid_queries[i]) > 80 else ""),
                                        size="sm"
                                    ).click(
                                        lambda query=hybrid_queries[i]: query,
                                        outputs=query_input
                                    )

            with gr.Column(scale=3):
                gr.HTML("<h3>📊 Analysis Results</h3>")

                analysis_output = gr.Markdown(
                    label="Task Detection & Routing",
                    value="Submit a query to see analysis..."
                )

                response_output = gr.Markdown(
                    label="Agent Response",
                    value="Agent response will appear here..."
                )

                reasoning_output = gr.Markdown(
                    label="Reasoning Steps",
                    value="Enable 'Show reasoning' to see detailed workflow...",
                    visible=False
                )

        # Info section
        gr.HTML("<hr>")
        with gr.Accordion("📖 About These Agents", open=False):
            gr.HTML("""
            <div style="padding: 15px;">
                <h4>🧠 ChemBERT Agent:</h4>
                <ul>
                    <li><strong>Task Detection:</strong> Identifies toxicity, solubility, bioactivity queries</li>
                    <li><strong>Molecule Extraction:</strong> Finds chemical names and SMILES</li>
                    <li><strong>Specialized Routing:</strong> Uses task-specific ChemBERT models</li>
                    <li><strong>Fast Predictions:</strong> 3.4M parameter transformer trained on 77M molecules</li>
                </ul>

                <h4>🔥 Hybrid LLM+ChemBERT Agent:</h4>
                <ul>
                    <li><strong>LLM Reasoning:</strong> Natural language understanding and chemistry knowledge</li>
                    <li><strong>ChemBERT Predictions:</strong> Molecular property predictions</li>
                    <li><strong>Result Synthesis:</strong> Combines computational results with domain expertise</li>
                    <li><strong>Complex Queries:</strong> Drug design, safety assessment, regulatory analysis</li>
                </ul>

                <h4>🏭 Training Specialized Models:</h4>
                <p>To enable task-specific routing: <code>python train_task_specific_chemberta.py</code></p>

                <h4>🔧 Setup:</h4>
                <p>For hybrid agent: Add your OpenAI API key above or in <code>.env</code> file</p>
            </div>
            """)

        # Event handlers

        # Agent initialization handlers
        chemberta_init_btn.click(
            fn=initialize_chemberta_agent,
            outputs=chemberta_status
        )

        hybrid_init_btn.click(
            fn=initialize_hybrid_agent,
            inputs=openai_api_key,
            outputs=hybrid_status
        )

        # Main query processing
        def process_query_with_agent_type(query, show_reasoning, agent_type_value):
            return process_chemistry_query(query, show_reasoning, agent_type_value)

        submit_btn.click(
            fn=process_query_with_agent_type,
            inputs=[query_input, show_reasoning, agent_type],
            outputs=[analysis_output, response_output, reasoning_output]
        )

        # Submit on Enter
        query_input.submit(
            fn=process_query_with_agent_type,
            inputs=[query_input, show_reasoning, agent_type],
            outputs=[analysis_output, response_output, reasoning_output]
        )

        # Clear function
        def clear_all():
            return "", "Submit a query to see analysis...", "Agent response will appear here...", ""

        clear_btn.click(
            fn=clear_all,
            outputs=[query_input, analysis_output, response_output, reasoning_output]
        )

        # Show/hide reasoning based on checkbox
        show_reasoning.change(
            lambda x: gr.update(visible=x),
            inputs=show_reasoning,
            outputs=reasoning_output
        )

        # Show/hide example queries based on agent type
        def toggle_examples(agent_type_value):
            if agent_type_value == "hybrid":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        agent_type.change(
            fn=toggle_examples,
            inputs=agent_type,
            outputs=[chemberta_examples, hybrid_examples]
        )

    return interface

def main():
    """Launch the web interface"""

    print("Starting Chemistry AI Agents UI (ChemBERT + Hybrid)...")
    print("=" * 60)

    # Check if gradio is installed
    try:
        import gradio
        print(f"[+] Gradio version: {gradio.__version__}")
    except ImportError:
        print("[-] Gradio not found. Install with: pip install gradio")
        return

    # Create and launch interface
    interface = create_interface()

    print("[+] Launching web interface...")
    print("[+] Open the URL in your browser to start chatting with the chemistry agent!")

    # Try different ports if 7860 is busy
    ports_to_try = [7860, 7861, 7862, 7863, 7864]

    for port in ports_to_try:
        try:
            print(f"[+] Trying to launch on port {port}...")
            interface.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                show_api=False,
                quiet=False
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"[-] Port {port} is busy, trying next port...")
                continue
            else:
                print(f"[-] Error launching on port {port}: {e}")
                break
    else:
        print("[-] Could not find an available port. Please close other applications using ports 7860-7864.")

if __name__ == "__main__":
    main()