#!/usr/bin/env python3
"""
Web UI for Intelligent ChemBERTa Agent
A user-friendly interface to test the intelligent chemistry agent
"""

import sys
import os
import gradio as gr
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Global agent variable
agent = None

def initialize_agent():
    """Initialize the intelligent ChemBERTa agent"""
    global agent

    if agent is not None:
        return "Agent already initialized!"

    try:
        from chemistry_agents.agents.intelligent_chemberta_agent import create_intelligent_chemberta_agent
        from chemistry_agents.agents.base_agent import AgentConfig

        # Create agent with CPU config
        config = AgentConfig(device="cpu", log_level="WARNING")
        agent = create_intelligent_chemberta_agent(config)

        specialized_count = len(agent.specialized_models)

        status = f"""[+] **Agent Initialized Successfully!**

**Model Details:**
- Base ChemBERTa: {'Loaded' if agent.is_loaded else 'Not Loaded'}
- Specialized Models: {specialized_count} available
- Parameters: {agent._count_parameters():,}
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
        return f"[-] **Initialization Failed:** {str(e)}"

def process_chemistry_query(query, show_reasoning=False):
    """Process a chemistry query and return results"""
    global agent

    if not query.strip():
        return "", "Please enter a chemistry question!", ""

    if agent is None:
        return "", "[-] Agent not initialized! Click 'Initialize Agent' first.", ""

    try:
        # Get task detection info
        task_type, confidence = agent.detect_task_type(query)
        molecules = agent._extract_molecules_from_query(query)

        # Create analysis info
        analysis_info = f"""**[BRAIN] Task Analysis:**
- **Detected Task:** {task_type.title()} (confidence: {confidence:.2f})
- **Molecules Found:** {', '.join(molecules) if molecules else 'None detected'}
- **Routing:** {'Specialized ' + task_type + ' model' if task_type in agent.specialized_models and confidence > 0.3 else 'General ChemBERTa'}
"""

        # Get full response
        response = agent.chat(query)

        # Get reasoning trace if requested
        reasoning = ""
        if show_reasoning:
            trace = agent.get_reasoning_trace()
            reasoning = "**[STEPS] Reasoning Steps:**\n"
            for step in trace:
                reasoning += f"{step.step_number}. {step.description}\n"

        return analysis_info, response, reasoning

    except Exception as e:
        return "", f"[-] **Error processing query:** {str(e)}", ""

def get_example_queries():
    """Return example queries for users to try"""
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
        title="Intelligent ChemBERTa Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chemistry-header {
            text-align: center;
            background: linear-gradient(45deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
        <div class="chemistry-header">
            <h1>Intelligent ChemBERTa Agent</h1>
            <p>Conversational AI for Chemistry • Automatic Task Detection • Specialized Model Routing</p>
        </div>
        """)

        # Initialization section
        with gr.Row():
            with gr.Column(scale=3):
                init_btn = gr.Button("Initialize Agent", variant="primary", size="lg")
            with gr.Column(scale=1):
                gr.HTML("<p style='margin-top: 10px;'>Click to load the ChemBERTa model</p>")

        init_status = gr.Markdown("Agent not initialized. Click the button above to start!")

        gr.HTML("<hr>")

        # Main interface
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>Ask Chemistry Questions</h3>")

                query_input = gr.Textbox(
                    label="Your Chemistry Question",
                    placeholder="e.g., Is benzene toxic? How soluble is aspirin?",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("Analyze", variant="primary")
                    clear_btn = gr.Button("Clear")

                show_reasoning = gr.Checkbox(
                    label="Show detailed reasoning steps",
                    value=False
                )

                # Example queries
                gr.HTML("<h4>Example Queries:</h4>")
                example_queries = get_example_queries()

                with gr.Row():
                    for i in range(0, len(example_queries), 2):
                        with gr.Column():
                            if i < len(example_queries):
                                gr.Button(
                                    example_queries[i],
                                    size="sm"
                                ).click(
                                    lambda x=example_queries[i]: x,
                                    outputs=query_input
                                )
                            if i+1 < len(example_queries):
                                gr.Button(
                                    example_queries[i+1],
                                    size="sm"
                                ).click(
                                    lambda x=example_queries[i+1]: x,
                                    outputs=query_input
                                )

            with gr.Column(scale=3):
                gr.HTML("<h3>Analysis Results</h3>")

                analysis_output = gr.Markdown(
                    label="Task Detection & Routing",
                    value="Submit a query to see analysis..."
                )

                response_output = gr.Markdown(
                    label="ChemBERTa Response",
                    value="Agent response will appear here..."
                )

                reasoning_output = gr.Markdown(
                    label="Reasoning Steps",
                    value="Enable 'Show reasoning' to see decision process...",
                    visible=False
                )

        # Info section
        gr.HTML("<hr>")
        with gr.Accordion("About This Agent", open=False):
            gr.HTML("""
            <div style="padding: 15px;">
                <h4>How It Works:</h4>
                <ol>
                    <li><strong>Task Detection:</strong> Analyzes your query to identify toxicity, solubility, or bioactivity questions</li>
                    <li><strong>Molecule Extraction:</strong> Finds chemical names and SMILES in your question</li>
                    <li><strong>Intelligent Routing:</strong> Routes to specialized ChemBERTa models or general analysis</li>
                    <li><strong>Expert Response:</strong> Provides detailed molecular intelligence</li>
                </ol>

                <h4>Model Details:</h4>
                <ul>
                    <li><strong>Base Model:</strong> ChemBERTa-77M-MLM (3.4M parameters)</li>
                    <li><strong>Training:</strong> 77M molecular SMILES strings</li>
                    <li><strong>Architecture:</strong> RoBERTa-based transformer</li>
                    <li><strong>Advantage:</strong> Superior chemistry understanding vs GPT</li>
                </ul>

                <h4>To Enable Specialized Models:</h4>
                <p>Run: <code>python train_task_specific_chemberta.py</code></p>
            </div>
            """)

        # Event handlers
        init_btn.click(
            fn=initialize_agent,
            outputs=init_status
        )

        submit_btn.click(
            fn=process_chemistry_query,
            inputs=[query_input, show_reasoning],
            outputs=[analysis_output, response_output, reasoning_output]
        )

        clear_btn.click(
            lambda: ("", "Enter your chemistry question above...", "Agent response will appear here...", ""),
            outputs=[query_input, analysis_output, response_output, reasoning_output]
        )

        # Show/hide reasoning based on checkbox
        show_reasoning.change(
            lambda x: gr.update(visible=x),
            inputs=show_reasoning,
            outputs=reasoning_output
        )

        # Submit on Enter
        query_input.submit(
            fn=process_chemistry_query,
            inputs=[query_input, show_reasoning],
            outputs=[analysis_output, response_output, reasoning_output]
        )

    return interface

def main():
    """Launch the web interface"""

    print("Starting Intelligent ChemBERTa Agent UI...")
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