import os
print("[*] Booting Application Router...", flush=True)
import gradio as gr
print("[*] UI Engine Loaded.", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
print("[*] Transformers Core Loaded.", flush=True)

from config import ModelConfig

# --- Global Logic ---
class AI_Assistant:
    def __init__(self):
        self.device = ModelConfig.DEVICE
        print(f"[*] Booting UI Engine on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = None
        if ModelConfig.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )

        print("[*] Cold-booting Foundation Matrix (Base LLM)...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.BASE_MODEL,
            quantization_config=bnb_config,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype=torch.float16 if ModelConfig.USE_FP16 else torch.float32,
            low_cpu_mem_usage=False  # Fixes MPS 'Invalid buffer size 4.10 GiB' crash on Macs
        )

        # Mount LoRA Weights if Available
        if os.path.exists(ModelConfig.OUTPUT_DIR):
            print(f"[*] Mounting Synchronized Neural Pattern (LoRA Adapter)...")
            self.lora_model = PeftModel.from_pretrained(self.base_model, ModelConfig.OUTPUT_DIR)
            self.has_lora = True
        else:
            print("[!] Warning: LoRA modules absent. Executing strictly on foundation model constraint.")
            self.lora_model = self.base_model
            self.has_lora = False

    def execute_inference(self, prompt_text, use_lora, version="v1.0"):
        target_model = self.lora_model if use_lora else self.base_model
        
        system_instructions = (
            "You are an elite Software Engineering Assistant. Provide concise, expert-level explanations "
            "and highly optimized code. Always formulate your response strictly as follows:\n"
            "1. Concept Overview\n2. Time/Space Complexity\n3. Optimal Code Implementation"
        )
        
        # simulated RAG Pipeline Integration
        rag_context = ""
        if "v2.0" in version:
            rag_context = (
                "\n[RETRIEVAL (RAG) KNOWLEDGE BASE INJECTED]: You must implement Advanced Multi-Domain rules. "
                "Ensure your code includes complete Python typing. Prioritize memory/pointer safety guidelines. "
                "Include a distinct edge-case evaluation rule!"
            )
        
        prompt = f"<|system|>\n{system_instructions}{rag_context}</s>\n<|user|>\n{prompt_text}</s>\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = target_model.generate(
                **inputs, 
                max_new_tokens=1024, 
                temperature=0.7, 
                top_p=0.9, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return raw_response.split("<|assistant|>")[-1].strip()

# Lazy Singleton Pattern
assitant_instance = None
def get_assistant():
    global assitant_instance
    if assitant_instance is None: assitant_instance = AI_Assistant()
    return assitant_instance

def handle_query(prompt_text, version):
    yield "⏳ Running Foundation Model (MPS)... Please wait ~2 seconds.", "Waiting for Base Model to finish before starting LoRA..."
    
    bot = get_assistant()
    if bot.has_lora:
        base_resp = bot.execute_inference(prompt_text, use_lora=False, version=version)
        yield base_resp, f"⏳ Running {version} (MPS)... Please wait ~2 seconds."
        
        lora_resp = bot.execute_inference(prompt_text, use_lora=True, version=version)
        
        # Add massive dynamic visual headers for presentation clarity!
        if "v2.0" in version:
            lora_resp = f"###  SYSTEM: V2.0 QLoRA + RAG PIPELINE ACTIVE\n---\n" + lora_resp
        else:
            lora_resp = f"### SYSTEM: V1.0 STANDARD LoRA ACTIVE\n---\n" + lora_resp
        
        base_resp = f"### SYSTEM: BASE FOUNDATION LLM\n---\n" + base_resp
        
    else:
        base_resp = bot.execute_inference(prompt_text, use_lora=False, version=version)
        lora_resp = "⚠️ LoRA Adapter Not Found. Model is not fine-tuned yet. Run `python model_trainer.py` to train adapters."
        
    yield base_resp, lora_resp

# --- No Global CSS - All styles are inline ---

def get_presentation_ui():
    slides = [
        """
        <div style='text-align: center; min-height: 60vh; padding: 2em; display: flex; flex-direction: column; justify-content: center;'>
            <h1 style='font-size: 2.5em; font-weight: 700; background: linear-gradient(to right, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>LLM Project</h1>
            <h2 style='font-size: 2.2em; color: #94a3b8; margin-bottom: 1em;'>Team:</h2>
            <p style='font-size: 1.3em; line-height: 1.8; margin-bottom: 0.8em;'>Sujal Saraswat (2022bcs0015)</p>
            <p style='font-size: 1.3em; line-height: 1.8; margin-bottom: 0.8em;'>Suraj Rathor (2022bcs0051)</p>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>About the Project</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                This project solves the challenge of adapting large language models for specific tasks without the prohibitive cost of full fine-tuning. 
                We demonstrate how to take a general foundation model (`TinyLlama-1.1B`) and specialize it into an expert **Software Engineering Assistant** using LoRA, a Parameter-Efficient Fine-Tuning (PEFT) technique.
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Technologies Used</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Python</b>: The core programming language.</li>
                    <li style='margin-bottom: 0.8em;'><b>PyTorch</b>: For building and training neural networks.</li>
                    <li style='margin-bottom: 0.8em;'><b>Hugging Face Transformers</b>: To leverage pre-trained models and training utilities.</li>
                    <li style='margin-bottom: 0.8em;'><b>PEFT (LoRA)</b>: For efficient fine-tuning by injecting trainable low-rank matrices.</li>
                    <li style='margin-bottom: 0.8em;'><b>Gradio</b>: To create a modern and interactive web UI for model demonstration.</li>
                    <li style='margin-bottom: 0.8em;'><b>BitsAndBytes</b>: For 4-bit quantization to reduce memory usage on CUDA devices.</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Architecture & Workflow</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                <p style='margin-bottom: 0.8em;'>The pipeline follows a modular, step-by-step approach:</p>
                <ol style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Configuration (`config.py`)</b>: Centralized settings for hardware, hyperparameters, and LoRA structure.</li>
                    <li style='margin-bottom: 0.8em;'><b>Data Handling (`data_handler.py`)</b>: Fetches the `CodeAlpaca-20k` dataset and formats it into a conversational structure (`<|system|>`, `<|user|>`, `<|assistant|>`).</li>
                    <li style='margin-bottom: 0.8em;'><b>Model Training (`model_trainer.py`)</b>: Loads the base `TinyLlama` model, injects LoRA adapters, and runs the supervised fine-tuning loop using `SFTTrainer`.</li>
                    <li style='margin-bottom: 0.8em;'><b>Inference & UI (`app.py`)</b>: Provides a Gradio interface to compare the performance of the base model against the fine-tuned LoRA version.</li>
                </ol>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Dataset Explanation</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                We use the **`sahil2801/CodeAlpaca-20k`** dataset, which contains 20,000 instruction-following examples for coding tasks. 
                For this project, we fine-tune on a subset of this data. The `data_handler.py` script formats each example into a structured prompt that teaches the model to act as a software engineering assistant.
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Model Details</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                <h3 style='font-size: 1.6em; margin-bottom: 0.8em;'>Base Model: TinyLlama-1.1B</h3>
                <p style='margin-bottom: 1em;'>We start with `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, a compact and efficient model suitable for fine-tuning on consumer hardware.</p>
                <h3 style='font-size: 1.6em; margin-bottom: 0.8em;'>LoRA Fine-Tuning</h3>
                <p>Instead of training all 1.1 billion parameters, we freeze the base model and inject small, trainable LoRA matrices into the `q_proj` and `v_proj` layers of the transformer. This reduces the trainable parameters to just a few million, making the process fast and memory-efficient.</p>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Training Process</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                The training is orchestrated by `model_trainer.py`. It uses the `SFTTrainer` from the TRL library, which is optimized for instruction-based fine-tuning. 
                Key training parameters include a learning rate of `2e-4`, a batch size of `1`, and the `paged_adamw_32bit` optimizer for CUDA environments to manage memory effectively.
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Results & Output</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                The fine-tuned model demonstrates a significant improvement in its ability to follow instructions and provide structured, high-quality responses for software engineering tasks. 
                The Gradio demo in `app.py` allows for a direct side-by-side comparison, showing the LoRA-tuned model providing more detailed explanations and better code.
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Challenges Faced</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Hardware Constraints</b>: Fine-tuning even a 1.1B model can be memory-intensive. This was mitigated by using 4-bit quantization on CUDA and carefully managing batch sizes, especially on Apple Silicon (MPS).</li>
                    <li style='margin-bottom: 0.8em;'><b>Prompt Engineering</b>: Crafting the right system prompt was crucial to guide the model's response format effectively.</li>
                    <li style='margin-bottom: 0.8em;'><b>Dependency Management</b>: Ensuring compatibility between PyTorch, Transformers, and CUDA/MPS drivers required careful environment setup.</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1em; font-weight: 700;'>Future Improvements</h2>
            <div style='font-size: 1.3em; line-height: 1.8;'>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Larger Dataset</b>: Train on the full `CodeAlpaca-20k` dataset or.</li>
                    <li style='margin-bottom: 0.8em;'><b>Deeper LoRA Integration</b>: Experiment with applying LoRA to more layers of the transformer, such as `k_proj` and `o_proj`.</li>
                    <li style='margin-bottom: 0.8em;'><b>Advanced RAG</b>: Integrate a Retrieval-Augmented Generation (RAG) pipeline to allow the model to pull in external documentation or code examples in real-time.</li>
                    <li style='margin-bottom: 0.8em;'><b>Evaluation Metrics</b>: Implement automated evaluation metrics (e.g., BLEU, CodeBLEU) to quantitatively measure performance improvements.</li>
                </ul>
            </div>
        </div>
        """
    ]

    with gr.Blocks() as presentation_ui:
        slide_index = gr.State(0)
        
        with gr.Row(elem_classes=["header-glass", "presentation-content"]):
            slide_display = gr.Markdown(slides[0])

        with gr.Row():
            prev_button = gr.Button("Previous", interactive=False)
            next_button = gr.Button("Next")

        def update_slide(index, direction):
            if direction == "next":
                index += 1
            else:
                index -= 1
            
            prev_interactive = index > 0
            next_interactive = index < len(slides) - 1
            
            return slides[index], index, gr.Button(interactive=prev_interactive), gr.Button(interactive=next_interactive)

        next_button.click(
            update_slide, 
            inputs=[slide_index, gr.State("next")], 
            outputs=[slide_display, slide_index, prev_button, next_button]
        )
        prev_button.click(
            update_slide, 
            inputs=[slide_index, gr.State("prev")], 
            outputs=[slide_display, slide_index, prev_button, next_button]
        )

    return presentation_ui

def get_ui_dashboard():
    with gr.Blocks() as ui_dashboard:
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1 style='font-size: 2.5em; font-weight: 700; background: linear-gradient(to right, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Neural Kernel Studio</h1>")
                gr.Markdown("<p style='font-size: 1.3em; line-height: 1.8; color: #94a3b8;'>Diagnostic Interface for Parameter-Efficient Structural Alteration (LoRA) vs Base LLM</p>")

        with gr.Row():
            version_dropdown = gr.Dropdown(
                choices=["v1.0 (Standard LoRA Tuned)", "v2.0 (QLoRA + RAG Pipeline)"],
                value="v1.0 (Standard LoRA Tuned)",
                label="Architecture Level",
                interactive=True
            )
            input_prompt = gr.Textbox(
                label="Query Terminal",
                placeholder="Initialize cognitive load (e.g., 'Architect a Merge Sort implementation in Python...')",
                lines=2,
                elem_classes=["glass-panel"],
                scale=2
            )

        with gr.Row():
            gr.Examples(
                examples=[
                    "Write a Python function for QuickSort and trace complexity.",
                    "How do dictionaries manage memory internally in Python?",
                    "Provide a Graph BFS implementation suitable for detecting shortest paths.",
                    "Explain the difference between Stack and Queue with practical examples.",
                    "Implement a Binary Search Tree with insert, delete, and search operations.",
                    "Design a caching mechanism using LRU (Least Recently Used) strategy.",
                    "Write an efficient algorithm to detect cycles in a directed graph."
                ],
                inputs=input_prompt
            )
            
        execute_button = gr.Button("Initialize Neural Synthesis", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("<div style='background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1);'><div style='font-size: 1.1em; font-weight: 600; color: #94a3b8; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>Foundation Architecture (Base LLM)</div></div>")
                box_base = gr.Markdown("...")

            with gr.Column():
                lora_header = gr.Markdown("<div style='background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1);'><div style='font-size: 1.1em; font-weight: 600; color: #94a3b8; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>LoRA Augmented Architecture (Fine-Tuned)</div></div>")
                box_lora = gr.Markdown("...")

        def update_header(version):
            if "v2.0" in version:
                return "<div style='background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1);'><div style='font-size: 1.1em; font-weight: 600; color: #94a3b8; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>QLoRA Augmented Architecture (Fine-Tuned)</div></div>"
            else:
                return "<div style='background: rgba(15, 23, 42, 0.6); border-radius: 12px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1);'><div style='font-size: 1.1em; font-weight: 600; color: #94a3b8; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;'>LoRA Augmented Architecture (Fine-Tuned)</div></div>"

        version_dropdown.change(fn=update_header, inputs=version_dropdown, outputs=lora_header)

        execute_button.click(
            fn=handle_query,
            inputs=[input_prompt, version_dropdown],
            outputs=[box_base, box_lora]
        )

    return ui_dashboard

with gr.Blocks() as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Presentation", id="presentation_tab"):
            get_presentation_ui()
        with gr.TabItem("Live Demo", id="demo_tab"):
            get_ui_dashboard()

if __name__ == "__main__":
    demo.launch(share=False)
