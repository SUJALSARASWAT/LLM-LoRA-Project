"""
Model Orchestration and Training Script
Handles base model instantiation, PEFT injection, quantization, and SFTTrainer execution.
"""
print("[*] Initializing python runtime...", flush=True)
import torch
print("[*] PyTorch imported successfully.", flush=True)
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
print("[*] Transformers module imported.", flush=True)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from config import ModelConfig, LoRAConfigParams, TrainingConfig
from data_handler import load_and_prepare_dataset

def build_model_and_tokenizer():
    print(f"[*] Loading Tokenizer: {ModelConfig.BASE_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    bnb_config = None
    if ModelConfig.USE_4BIT:
        print("[*] Engaging BitsAndBytes 4-bit Quantization (CUDA detected)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[*] Loading Base Architecture: {ModelConfig.BASE_MODEL}...")
    torch_dtype = torch.float16 if ModelConfig.USE_FP16 else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        ModelConfig.BASE_MODEL,
        quantization_config=bnb_config,
        device_map=ModelConfig.DEVICE if ModelConfig.DEVICE != "cpu" else None,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False  # Fixes MPS 'Invalid buffer size 4.10 GiB' crash on Macs
    )
    model.config.use_cache = False
    
    return model, tokenizer

def inject_lora(model):
    print("[*] Injecting LoRA Parameter-Efficient Adapters...")
    peft_config = LoraConfig(
        lora_alpha=LoRAConfigParams.LORA_ALPHA,
        lora_dropout=LoRAConfigParams.LORA_DROPOUT,
        r=LoRAConfigParams.LORA_R,
        bias="none",
        task_type=LoRAConfigParams.TASK_TYPE,
        target_modules=LoRAConfigParams.TARGET_MODULES
    )
    # TRL automatically applies PeftConfig in SFTTrainer, but we can return it.
    return peft_config

def execute_training():
    model, tokenizer = build_model_and_tokenizer()
    peft_config = inject_lora(model)
    dataset = load_and_prepare_dataset()
    
    sft_config = SFTConfig(
        output_dir=ModelConfig.OUTPUT_DIR,
        dataset_text_field="text",
        max_length=TrainingConfig.MAX_SEQ_LENGTH,
        num_train_epochs=TrainingConfig.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION,
        optim=TrainingConfig.OPTIMIZER,
        save_steps=TrainingConfig.SAVE_STEPS,
        logging_steps=TrainingConfig.LOGGING_STEPS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY,
        fp16=ModelConfig.USE_FP16,
        max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
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
    
    print("\n" + "="*50)
    print("📈 Training Initiated")
    trainer.model.print_trainable_parameters()
    print("="*50 + "\n")
    
    trainer.train()
    
    print(f"\n✅ Training Complete. Saving adapters to {ModelConfig.OUTPUT_DIR}...")
    trainer.model.save_pretrained(ModelConfig.OUTPUT_DIR)
    tokenizer.save_pretrained(ModelConfig.OUTPUT_DIR)
    print("✨ Process Finished!")

if __name__ == "__main__":
    execute_training()
