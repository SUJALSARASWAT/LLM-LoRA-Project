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
            <h1 style='font-size: 2.5em; font-weight: 700; background: linear-gradient(to right, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;'>LLM-Based Coding Assistant using LoRA & Efficient Fine-Tuning</h1>
            <h2 style='font-size: 1.8em; color: #94a3b8; margin-bottom: 1.5em;'>Team:</h2>
            <p style='font-size: 1.3em; line-height: 1.8; margin-bottom: 0.8em;'>Sujal Saraswat (2022bcs0015)</p>
            <p style='font-size: 1.3em; line-height: 1.8; margin-bottom: 2em;'>Suraj Rathor (2022bcs0051)</p>
            <div style='font-size: 1.2em; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1.5em;'>
                <p style='margin-bottom: 0.8em;'><b>Objective:</b></p>
                <p>Develop a resource-efficient pipeline to adapt a general-purpose LLM into a domain-specific Software Engineering Assistant</p>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Problem Statement</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Challenges in LLM Fine-Tuning</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Large Language Models contain billions of parameters</li>
                    <li style='margin-bottom: 0.8em;'>Full fine-tuning requires high-end GPUs</li>
                    <li style='margin-bottom: 0.8em;'>Large memory requirements (&gt;24GB VRAM)</li>
                    <li style='margin-bottom: 0.8em;'>Long training time</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>❌ Limitations:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Not feasible on consumer hardware</li>
                    <li style='margin-bottom: 0.8em;'>High computational cost</li>
                    <li style='margin-bottom: 0.8em;'>Inefficient for small domain adaptation</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Proposed Solution</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Parameter-Efficient Fine-Tuning (PEFT)</h3>
                <p style='margin-bottom: 1.5em;'><b>We use:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>LoRA (Low-Rank Adaptation)</b> → Efficient fine-tuning</li>
                    <li style='margin-bottom: 0.8em;'><b>Quantization (QLoRA-inspired)</b> → Memory optimization</li>
                    <li style='margin-bottom: 0.8em;'><b>Prompt Engineering (RAG-style)</b> → Controlled outputs</li>
                </ul>
                <div style='background: rgba(168, 85, 247, 0.1); padding: 1.5em; border-left: 3px solid #a855f7; border-radius: 8px;'>
                    <p><b>🎯 Goal:</b></p>
                    <p>Adapt a base LLM into a structured coding assistant with minimal resource usage</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Base Model</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Foundation Model</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Model:</b> TinyLlama-1.1B-Chat-v1.0</li>
                    <li style='margin-bottom: 0.8em;'><b>Size:</b> ~1.1 Billion Parameters</li>
                    <li style='margin-bottom: 0.8em;'><b>Type:</b> Instruction-tuned conversational LLM</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Why TinyLlama?</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>✓ Lightweight</li>
                    <li style='margin-bottom: 0.8em;'>✓ Fast inference</li>
                    <li style='margin-bottom: 0.8em;'>✓ Suitable for fine-tuning on limited hardware</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Dataset</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Training Dataset</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Dataset:</b> CodeAlpaca-20k</li>
                    <li style='margin-bottom: 0.8em;'><b>Domain:</b> Programming + DSA tasks</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Contains:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Instruction → Problem</li>
                    <li style='margin-bottom: 0.8em;'>Input → Context</li>
                    <li style='margin-bottom: 0.8em;'>Output → Solution</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Implementation Choice:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Used subset (30–300 samples) for fast training</li>
                    <li style='margin-bottom: 0.8em;'>Pipeline supports full dataset scaling</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Data Preprocessing</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Instruction Formatting Strategy</h3>
                <p style='margin-bottom: 1.5em;'>We convert dataset into structured chat format:</p>
                <div style='background: rgba(15, 23, 42, 0.8); padding: 1.5em; border-radius: 8px; margin-bottom: 1.5em; font-family: monospace;'>
                    <p>&lt;|system|&gt; → Defines assistant behavior</p>
                    <p>&lt;|user|&gt; → User query</p>
                    <p>&lt;|assistant|&gt; → Expected output</p>
                </div>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Enhancements:</h3>
                <p style='margin-bottom: 1em;'>Added system instructions:</p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Concept explanation</li>
                    <li style='margin-bottom: 0.8em;'>Complexity analysis</li>
                    <li style='margin-bottom: 0.8em;'>Code generation</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p>✅ <b>Result:</b> Improved instruction-following capability</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>LoRA: Core Technique</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Low-Rank Adaptation (LoRA)</h3>
                <p style='margin-bottom: 1em;'><b>Key Idea:</b> Instead of updating full weights:</p>
                <div style='background: rgba(15, 23, 42, 0.8); padding: 1.5em; border-radius: 8px; margin-bottom: 1.5em; text-align: center; font-size: 1.3em; font-family: monospace;'>
                    W′ = W + A × B
                </div>
                <p style='margin-bottom: 1.5em;'><b>Where:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>W = original weights</li>
                    <li style='margin-bottom: 0.8em;'>A, B = low-rank matrices</li>
                </ul>
                <h3 style='font-size: 1.4em; margin-bottom: 1em; color: #a855f7;'>Target Layers:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>q_proj, k_proj, v_proj, o_proj</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>✅ Benefits:</b></p>
                    <ul style='list-style-position: inside; margin-left: 1.5em;'>
                        <li style='margin-bottom: 0.5em;'>~99% reduction in trainable parameters</li>
                        <li style='margin-bottom: 0.5em;'>Faster training</li>
                        <li style='margin-bottom: 0.5em;'>Lower memory usage</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>QLoRA: Memory Optimization</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Quantization Strategy</h3>
                <p style='margin-bottom: 1.5em;'><b>Technique:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>4-bit Quantization (NF4)</li>
                    <li style='margin-bottom: 0.8em;'>Using BitsAndBytes library</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>What it does:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Compress model weights</li>
                    <li style='margin-bottom: 0.8em;'>Reduce GPU memory usage</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Implementation:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Enabled only on CUDA devices</li>
                    <li style='margin-bottom: 0.8em;'>FP16 used for MPS/CPU</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>✅ Result:</b> Training possible on low-resource systems & faster inference</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>RAG-Inspired Approach</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Context Injection (Simulated RAG)</h3>
                <div style='background: rgba(239, 68, 68, 0.1); padding: 1em; border-left: 3px solid #ef4444; border-radius: 8px; margin-bottom: 1.5em;'>
                    <p>⚠ Not full RAG (no retrieval system)</p>
                </div>
                <h3 style='font-size: 1.4em; margin-bottom: 1em; color: #a855f7;'>What we implemented:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Prompt-based context augmentation</li>
                    <li style='margin-bottom: 0.8em;'>Injected structured rules into system prompt</li>
                </ul>
                <h3 style='font-size: 1.4em; margin-bottom: 1em; color: #a855f7;'>Example - Enforce:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Code quality</li>
                    <li style='margin-bottom: 0.8em;'>Edge case handling</li>
                    <li style='margin-bottom: 0.8em;'>Structured outputs</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>✅ Benefit:</b> More controlled and consistent responses</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Training Pipeline</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>End-to-End Workflow</h3>
                <div style='background: rgba(15, 23, 42, 0.8); padding: 1.5em; border-radius: 8px; margin-bottom: 1.5em; text-align: center; font-family: monospace;'>
                    <p>Dataset → Formatting → Tokenization</p>
                    <p>↓</p>
                    <p>Quantized Base Model → LoRA Injection</p>
                    <p>↓</p>
                    <p>SFTTrainer → Save LoRA Adapters</p>
                </div>
                <h3 style='font-size: 1.4em; margin-bottom: 1em; color: #a855f7;'>Training Framework:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>HuggingFace Transformers</li>
                    <li style='margin-bottom: 0.8em;'>TRL (SFTTrainer)</li>
                </ul>
                <h3 style='font-size: 1.4em; margin-bottom: 1em; color: #a855f7;'>Objective:</h3>
                <p>Minimize loss on instruction-output pairs</p>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Training Details</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Hyperparameters</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Epochs: 1</li>
                    <li style='margin-bottom: 0.8em;'>Batch Size: 1</li>
                    <li style='margin-bottom: 0.8em;'>Gradient Accumulation: 4</li>
                    <li style='margin-bottom: 0.8em;'>Learning Rate: 2e-4</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Optimization:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>AdamW / paged_adamw_32bit</li>
                    <li style='margin-bottom: 0.8em;'>Cosine learning rate scheduler</li>
                </ul>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Hardware Handling:</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>CUDA → 4-bit + FP16</li>
                    <li style='margin-bottom: 0.8em;'>MPS → FP16</li>
                    <li style='margin-bottom: 0.8em;'>CPU → FP32</li>
                </ul>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>System Architecture</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Modular Design</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>config.py</b> → Hardware + hyperparameters</li>
                    <li style='margin-bottom: 0.8em;'><b>data_handler.py</b> → Dataset processing</li>
                    <li style='margin-bottom: 0.8em;'><b>model_trainer.py</b> → Training pipeline</li>
                    <li style='margin-bottom: 0.8em;'><b>app.py</b> → UI & inference</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1.5em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>✅ Advantages:</b></p>
                    <ul style='list-style-position: inside; margin-left: 1.5em;'>
                        <li style='margin-bottom: 0.5em;'>Scalable</li>
                        <li style='margin-bottom: 0.5em;'>Maintainable</li>
                        <li style='margin-bottom: 0.5em;'>Reproducible</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Version 1: Base vs LoRA</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Comparative Evaluation</h3>
                <p style='margin-bottom: 1em;'><b>Base Model:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Generic responses</li>
                    <li style='margin-bottom: 0.8em;'>Less structured</li>
                </ul>
                <p style='margin-bottom: 1em;'><b>LoRA Model:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Domain-specific</li>
                    <li style='margin-bottom: 0.8em;'>Structured output</li>
                    <li style='margin-bottom: 0.8em;'>Better code quality</li>
                </ul>
                <div style='background: rgba(168, 85, 247, 0.1); padding: 1.5em; border-left: 3px solid #a855f7; border-radius: 8px;'>
                    <p><b>🎯 Insight:</b> LoRA significantly improves task-specific performance</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Version 2: Advanced System</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>QLoRA + Context-Augmented Pipeline</h3>
                <p style='margin-bottom: 1.5em;'><b>Enhancements:</b></p>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>Quantized inference</li>
                    <li style='margin-bottom: 0.8em;'>Prompt-based context injection</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1.5em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>Result:</b></p>
                    <ul style='list-style-position: inside; margin-left: 1.5em;'>
                        <li style='margin-bottom: 0.5em;'>More controlled outputs</li>
                        <li style='margin-bottom: 0.5em;'>Improved consistency</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Results</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #a855f7;'>Observed Improvements</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'>✓ Better instruction following</li>
                    <li style='margin-bottom: 0.8em;'>✓ Cleaner code generation</li>
                    <li style='margin-bottom: 0.8em;'>✓ Structured responses</li>
                    <li style='margin-bottom: 0.8em;'>✓ Improved explanation clarity</li>
                </ul>
                <div style='background: rgba(168, 85, 247, 0.1); padding: 1.5em; border-left: 3px solid #a855f7; border-radius: 8px;'>
                    <p><b>Demo:</b> Side-by-side comparison via Gradio UI in the "Live Demo" tab</p>
                </div>
            </div>
        </div>
        """,
        """
        <div style='min-height: 60vh; padding: 2em; text-align: left;'>
            <h2 style='font-size: 2.2em; text-align: center; margin-bottom: 1.5em; font-weight: 700;'>Challenges</h2>
            <div style='font-size: 1.2em; line-height: 2;'>
                <h3 style='font-size: 1.6em; margin-bottom: 1em; color: #ec4899;'>Key Challenges Faced</h3>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 1.5em;'>
                    <li style='margin-bottom: 0.8em;'><b>Memory constraints</b> - Managed with quantization</li>
                    <li style='margin-bottom: 0.8em;'><b>Small dataset size</b> - Mitigated with gradient accumulation</li>
                    <li style='margin-bottom: 0.8em;'><b>Prompt design complexity</b> - Solved through iterations</li>
                    <li style='margin-bottom: 0.8em;'><b>Hardware compatibility</b> - CUDA vs MPS handling</li>
                </ul>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 1.5em; border-left: 3px solid #10b981; border-radius: 8px;'>
                    <p><b>Solutions Applied:</b></p>
                    <ul style='list-style-position: inside; margin-left: 1.5em;'>
                        <li style='margin-bottom: 0.5em;'>Quantization for memory</li>
                        <li style='margin-bottom: 0.5em;'>Gradient accumulation</li>
                        <li style='margin-bottom: 0.5em;'>Modular design</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
      
        """
        <div style='min-height: 60vh; padding: 2em; text-align: center; display: flex; flex-direction: column; justify-content: center;'>
            <h2 style='font-size: 2.5em; font-weight: 700; margin-bottom: 1.5em; background: linear-gradient(to right, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Conclusion</h2>
            <h3 style='font-size: 1.6em; color: #94a3b8; margin-bottom: 2em;'>Key Takeaways</h3>
            <div style='font-size: 1.2em; line-height: 2; text-align: left; max-width: 900px; margin: 0 auto;'>
                <ul style='list-style-position: inside; margin-left: 1.5em; margin-bottom: 2em;'>
                    <li style='margin-bottom: 1em;'>✓ Efficient fine-tuning is possible using LoRA</li>
                    <li style='margin-bottom: 1em;'>✓ Quantization reduces hardware requirements</li>
                    <li style='margin-bottom: 1em;'>✓ Structured prompts improve LLM behavior</li>
                    <li style='margin-bottom: 1em;'>✓ Modular design enables scalability</li>
                </ul>
                <div style='background: rgba(168, 85, 247, 0.1); padding: 2em; border-left: 4px solid #a855f7; border-radius: 8px; text-align: center;'>
                    <p style='font-size: 1.3em; font-weight: 600;'>"We demonstrated how large language models can be efficiently adapted for real-world applications using minimal resources."</p>
                </div>
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
