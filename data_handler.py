"""
Dataset Processing Utilities
Handles loading, preprocessing, and formatting the causal language modeling instruction dataset.
"""
from datasets import load_dataset
from config import ModelConfig, TrainingConfig

def format_instruction(example):
    """
    Transforms raw dataset rows into proper LLM prompt structures with System/User/Assistant roles.
    """
    system_prompt = (
        "You are an elite Software Engineering Assistant. Provide concise, expert-level explanations "
        "and highly optimized code. Always formulate your response strictly as follows:\n"
        "1. Concept Overview\n2. Time/Space Complexity\n3. Optimal Code Implementation"
    )
    
    # Handle batch processing vs single item gracefully
    instructions = example['instruction'] if isinstance(example['instruction'], list) else [example['instruction']]
    inputs = example.get('input', [])
    if not isinstance(inputs, list):
        inputs = [inputs]
        
    outputs = example['output'] if isinstance(example['output'], list) else [example['output']]
    
    output_texts = []
    for i in range(len(instructions)):
        current_input = inputs[i] if i < len(inputs) else ""
        context = f" Context: {current_input}" if current_input else ""
        text = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{instructions[i]}{context}</s>\n<|assistant|>\n{outputs[i]}</s>"
        output_texts.append(text)
        
    return {"text": output_texts if isinstance(example['instruction'], list) else output_texts[0]}

def load_and_prepare_dataset():
    """
    Loads dataset from huggingface and applies token formatting maps.
    Returns a mapped Dataset object ready for SFTTrainer.
    """
    print(f"[*] Downloading & Loading Dataset: {ModelConfig.DATASET_NAME}")
    # We take a small subset to ensure it finishes within a reasonable time during assignment demonstration
    dataset = load_dataset(ModelConfig.DATASET_NAME, split=f"train[:{TrainingConfig.TRAIN_SUBSET_SIZE}]")
    
    print("[*] Applying Instruction Formatting...")
    dataset = dataset.map(format_instruction, batched=True, remove_columns=dataset.column_names)
    
    return dataset
