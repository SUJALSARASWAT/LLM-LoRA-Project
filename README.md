# Neural Kernel Studio: LoRA Parameter-Efficient Fine-Tuning

Welcome to the **LoRA Parameter-Efficient Fine-Tuning (PEFT)** project. This robust, fully modularized project demonstrates the end-to-end pipeline of taking a foundation LLM (`TinyLlama-1.1B`) and applying high-performance LoRA adjustments to create an elite Software Engineering Assistant.

## 🌟 Project Architecture

We structured this repository implementing enterprise ML engineering standards:

- **`config.py`**: A centralized source of truth. Contains all hardware resolution logic (CUDA vs MPS vs CPU), base hyper-parameters, and topological rules for LoRA (Rank, Alpha, target modules).
- **`data_handler.py`**: Encapsulates all data fetching and preprocessing. Downloads `CodeAlpaca-20k`, maps instruction formats into cohesive conversational structures (`<|system|>` / `<|user|>` / `<|assistant|>`), and tokenizes context.
- **`model_trainer.py`**: The actual execution layer. It orchestrates Foundation Model loading, LoRA adapter injection via PEFT, and triggers the `SFTTrainer` (Supervised Fine-Tuning) loop.
- **`app.py`**: A beautiful, custom CSS Gradio Web Interface. It employs glassmorphism styling, vibrant gradients, and animations to compare the raw Foundation Architecture alongside our Fine-Tuned LoRA model.
- **`Assignment.py`**: (Optional) A single-script version providing an overview of everything in one place for direct assignment demonstration!

---

## 🚀 Execution Guide

Make sure your environment is activated (`python -m venv .venv`).

### 1. Train the Neural Adapters
To initiate the Fine-Tuning Process directly:
```bash
python model_trainer.py
```
*Depending on hardware (`MPS`/`CUDA`), this step handles loading, formatting, and executing supervised training loop efficiently. It creates the `./lora_finetuned_model` directory containing our LoRA weights.*

### 2. Launch the AI Dashboard
After the LoRA model is saved, boot up the Web UI to test and compare outcomes:
```bash
python app.py
```
*Navigate to `http://localhost:7860` in your web browser. You'll be greeted by an aesthetically breathtaking Dark Mode interface.*

---

## ✨ Features Showcased
- **PEFT / LoRA Injection**: Inject low-rank matrices into `q_proj` & `v_proj` transformer components, turning a multi-billion hardware footprint into just ~18MB of trainable parameters!
- **Dynamic Optimization Setup**: The `config.py` automatically binds `paged_adamw_32bit` and `bitsandbytes` 4-bit Quantization when CUDA is recognized.
- **Cross-Platform Compatibility**: Supports execution on NVIDIA GPUs, Apple Silicon (M1/M2/M3 via `mps`), or standard fallback CPUs.
