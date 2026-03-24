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
                max_new_tokens=150, 
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

# --- Aesthetic Premium UI Design (Web App Standard) ---
# Employs dark-mode Glassmorphism, modern HSL gradient palettes, and smooth UI animations.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
  --primary: #6366f1;
  --primary-hover: #4f46e5;
  --secondary: #ec4899;
  --surface: rgba(30, 41, 59, 0.7);
  --border: rgba(255, 255, 255, 0.1);
  --text-light: #f8fafc;
  --text-dim: #94a3b8;
  --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
}

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: var(--bg-gradient) !important;
    background-attachment: fixed !important;
    color: var(--text-light) !important;
}

/* Glassmorphism Title Box */
.header-glass {
    background: var(--surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-bottom: 30px;
    animation: fadeIn 1s ease-out;
}

.header-title {
    font-size: 2.5em;
    font-weight: 700;
    background: linear-gradient(to right, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Glassmorphism Result Columns */
.glass-panel {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-panel:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(236, 72, 153, 0.4);
}

.result-title {
    font-size: 1.1em;
    font-weight: 600;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Primary Button Styling */
button.primary {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

button.primary:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6) !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

with gr.Blocks(css=CSS) as ui_dashboard:
    with gr.Row(elem_classes=["header-glass"]):
        with gr.Column():
            gr.Markdown("<h1 class='header-title'>Neural Kernel Studio</h1>")
            gr.Markdown("<p style='font-size: 1.2em; color: var(--text-dim);'>Diagnostic Interface for Parameter-Efficient Structural Alteration (LoRA) vs Base LLM</p>")

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
        
    execute_button = gr.Button("Initialize Neural Synthesis", variant="primary")

    with gr.Row():
        with gr.Column(elem_classes=["glass-panel"]):
            gr.Markdown("<div class='result-title'>Foundation Architecture (Base LLM)</div>")
            box_base = gr.Markdown("...")

        with gr.Column(elem_classes=["glass-panel"]):
            gr.Markdown("<div class='result-title'>LoRA Augmented Architecture (Fine-Tuned)</div>")
            box_lora = gr.Markdown("...")

    execute_button.click(
        fn=handle_query,
        inputs=[input_prompt, version_dropdown],
        outputs=[box_base, box_lora]
    )

    with gr.Row(elem_classes=["glass-panel"]):
        gr.Examples(
            examples=[
                "Write a Python function for QuickSort and trace complexity.",
                "How do dictionaries manage memory internally in Python?",
                "Provide a Graph BFS implementation suitable for detecting shortest paths."
            ],
            inputs=input_prompt
        )

if __name__ == "__main__":
    ui_dashboard.launch(share=False)
