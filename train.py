import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Lightweight and effective for Colab
DATASET_NAME = "sahil2801/CodeAlpaca-20k"
OUTPUT_DIR = "./lora_finetuned_model"

def train():
    # 1. Load Dataset
    print(f"Loading dataset: {DATASET_NAME}...", flush=True)
    dataset = load_dataset(DATASET_NAME, split="train[:2000]") # Subset for faster demo

    # 2. Tokenizer Configuration
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Model Configuration (Handle Mac/MPS compatibility)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None # BitsAndBytes is not supported on Mac/CPU

    print(f"Loading model: {MODEL_NAME}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device if device != "cpu" else None,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Target linear layers for TinyLlama
    )

    # 5. Formatting Function for Training
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

    # Map the dataset to include the "text" column
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 6. SFT Configuration
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=(device == "cuda"),
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    # 7. Start Training
    print("Starting training...", flush=True)
    trainer.train()

    # 8. Save Model
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...", flush=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training completed successfully!", flush=True)

if __name__ == "__main__":
    train()
