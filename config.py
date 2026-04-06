"""
Project Configuration Parameters
Contains all hyperparameters and system settings required for the LoRA Fine-Tuning pipeline.
"""
import torch
import platform
import os

# Fix Apple Silicon memory stagnation & buffer limits
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

class ModelConfig:
    # Base configuration
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DATASET_NAME = "sahil2801/CodeAlpaca-20k"
    OUTPUT_DIR = "./lora_finetuned_model"
    
    # Hardware Configuration
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and platform.system() == "Darwin":
            return "mps"
        return "cpu"
    
    # Dynamic Optimization Settings based on Hardware
    DEVICE = get_device.__func__()
    # Keep model weights in half precision on CUDA/MPS to reduce memory.
    USE_HALF_PRECISION_WEIGHTS = DEVICE in ["cuda", "mps"]
    # Trainer mixed-precision `fp16` is only valid on CUDA (not MPS/CPU).
    USE_FP16 = DEVICE == "cuda"
    USE_4BIT = DEVICE == "cuda"  # bitsandbytes only works on CUDA

class LoRAConfigParams:
    # LoRA Specific structural constraints
    LORA_R = 64
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    TASK_TYPE = "CAUSAL_LM"

class TrainingConfig:
    # Tunable Training Hyperparameters
    MAX_SEQ_LENGTH = 512
    NUM_TRAIN_EPOCHS = 1
    TRAIN_SUBSET_SIZE = 30  # For demonstration speed and to prevent Mac RAM swapping
    BATCH_SIZE = 1          # Lowered to 1 to prevent MPS Apple Silicon memory deadlock
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.001
    MAX_GRAD_NORM = 0.3
    SAVE_STEPS = 50
    LOGGING_STEPS = 10
    OPTIMIZER = "paged_adamw_32bit" if ModelConfig.DEVICE == "cuda" else "adamw_torch"
