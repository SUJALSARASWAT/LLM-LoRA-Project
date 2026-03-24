import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./lora_finetuned_model"

class ProgrammingAssistant:
    def __init__(self, use_lora=True):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None

        print(f"Loading Base Model: {BASE_MODEL_NAME} on {self.device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )

        if use_lora and os.path.exists(ADAPTER_PATH):
            print(f"Loading LoRA Adapters from {ADAPTER_PATH}...")
            self.model = PeftModel.from_pretrained(self.base_model, ADAPTER_PATH)
        else:
            print("Using Base Model only (LoRA adapters not found or disabled).")
            self.model = self.base_model

    def generate_response(self, prompt, max_new_tokens=512):
        # Improved system prompt to enforce a structured, explanatory response
        system_instructions = (
            "You are a specialized Programming & DSA Tutor. "
            "When answering questions, follow this structure:\n"
            "1. Logic & Intuition: Explain how the algorithm works.\n"
            "2. Complexity: Clearly state Time and Space complexity.\n"
            "3. Code: Provide well-commented Python code.\n"
            "4. Example: Show how to use the code with a test case."
        )
        
        full_prompt = f"<|system|>\n{system_instructions}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's part
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        return response

def compare_models(prompt):
    print("\n" + "="*50)
    print(f"PROMPT: {prompt}")
    print("="*50)

    # 1. Base Model Response
    print("\n--- BASE MODEL RESPONSE ---")
    base_assistant = ProgrammingAssistant(use_lora=False)
    print(base_assistant.generate_response(prompt))

    # 2. LoRA Model Response
    print("\n--- LoRA FINE-TUNED RESPONSE ---")
    lora_assistant = ProgrammingAssistant(use_lora=True)
    print(lora_assistant.generate_response(prompt))
    print("="*50 + "\n")

if __name__ == "__main__":
    test_prompt = "Write a Python function to find the maximum sum of a sub-array (Kadane's algorithm)."
    # Note: This script assumes you have already run train.py
    if os.path.exists(ADAPTER_PATH):
        compare_models(test_prompt)
    else:
        print("Please run train.py first to generate the LoRA adapters.")
