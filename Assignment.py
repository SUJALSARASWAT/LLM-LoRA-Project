"""
Assignment: Parameter-Efficient Fine-Tuning of Large Language Models using LoRA
Topic: Fine-tuning TinyLlama-1.1B for Programming/DSA Tutoring using Low-Rank Adaptation (LoRA)

Description:
This script demonstrates the end-to-end process of fine-tuning a Large Language Model (LLM)
using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA. 
Instead of updating all model parameters (which is computationally expensive), 
LoRA injects trainable rank decomposition matrices into the transformer architecture, 
significantly reducing the number of trainable parameters.

Key Components:
1. Base Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
2. Dataset: sahil2801/CodeAlpaca-20k (Subset)
3. PEFT Method: LoRA (Low-Rank Adaptation)
4. Training Framework: Hugging Face `transformers`, `peft` and `trl` (SFTTrainer)
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- 1. Configuration & Hyperparameters ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "sahil2801/CodeAlpaca-20k"
OUTPUT_DIR = "./assignment_lora_model"

def main():
    print("="*60)
    print("🚀 Parameter-Efficient Fine-Tuning (LoRA) Assignment")
    print("="*60)
    
    # --- 2. Device Configuration (Mac/Windows/Linux Support) ---
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Using device: {device}")
    
    # --- 3. Load Dataset ---
    # We use a small subset for demonstration purposes to ensure relatively fast training
    print(f"[*] Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train[:300]") # Small subset for assignment demo
    
    # --- 4. Initialize Tokenizer ---
    print(f"[*] Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # --- 5. Configure Base Model (with optional Quantization for CUDA) ---
    bnb_config = None
    if device == "cuda":
        print("[*] Configuring 4-bit Quantization (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    print(f"[*] Loading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device if device != "cpu" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # --- 6. LoRA Configuration (The core of PEFT) ---
    # LoRA creates small, trainable adapter modules while freezing the original LLM weights.
    print("[*] Configuring LoRA Adapters...")
    peft_config = LoraConfig(
        lora_alpha=16,          # Scaling factor
        lora_dropout=0.1,       # Dropout probability to prevent overfitting
        r=64,                   # Rank of the update matrices (lower = fewer parameters)
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Target attention layers
    )
    
    # --- 7. Data Formatting for Instruction Fine-Tuning ---
    def formatting_prompts_func(example):
        system_prompt = (
            "You are a specialized Programming Tutor. Help the user understand Python and DSA. "
            "For every answer:\n"
            "1. Explain the Logic & Intuition.\n"
            "2. State Time/Space Complexity.\n"
            "3. Provide clean Code.\n"
            "4. Show an Example usage."
        )
        
        instructions = example['instruction'] if isinstance(example['instruction'], list) else [example['instruction']]
        inputs = example['input'] if 'input' in example and isinstance(example['input'], list) else [example.get('input', '')]
        outputs = example['output'] if isinstance(example['output'], list) else [example['output']]
        
        output_texts = []
        for i in range(len(instructions)):
            current_input = inputs[i] if i < len(inputs) else ""
            text = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{instructions[i]} {current_input}</s>\n<|assistant|>\n{outputs[i]}</s>"
            output_texts.append(text)
            
        return {"text": output_texts if isinstance(example['instruction'], list) else output_texts[0]}

    print("[*] Applying formatting template to dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    # --- 8. Training Configuration ---
    print("[*] Initializing Training Arguments...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="adamw_torch" if device != "cuda" else "paged_adamw_32bit",
        save_steps=20,
        logging_steps=5,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=(device == "cuda"),
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )
    
    # Show trainable parameters
    print("\n[*] Parameter Snapshot:")
    trainer.model.print_trainable_parameters()
    
    # --- 9. Start Training ---
    print("\n[*] Starting LoRA Fine-Tuning Process...")
    trainer.train()
    
    # --- 10. Save the Fine-Tuned LoRA Adapters ---
    print(f"\n[*] Saving trained LoRA model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\n✅ Training Complete!")
    
    # --- 11. Optional: Quick Inference Test ---
    print("\n[*] Running Quick Inference Test with Fine-Tuned Model...")
    test_prompt = "Explain bubble sort and provide a python example."
    inputs = tokenizer(
        f"<|system|>\nYou are a specialized Programming Tutor.</s>\n<|user|>\n{test_prompt}</s>\n<|assistant|>\n", 
        return_tensors="pt"
    ).to(device)
    
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- MODEL RESPONSE ---")
    print(response.split("<|assistant|>")[-1].strip())
    print("----------------------")

if __name__ == "__main__":
    main()
