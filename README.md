# Fine-Tuning Llama 2 for Medical QA with LoRA

This repository demonstrates a cutting-edge, parameter-efficient approach to fine-tuning Llama 2 on a domain-specific medical dataset using Low-Rank Adaptation (LoRA).
Leveraging the Hugging Face Transformers ecosystem, PyTorch, and A100 GPU acceleration, this repo delivers a scalable, production-ready pipeline for high-quality, context-aware medical question answering.

🧬 **Key Features**
* Foundation Model: Llama 2 (Meta AI) as the base LLM

* Parameter-Efficient Fine-Tuning: Utilizes **LoRA adapters** for lightweight, modular adaptation—enabling **10x faster training and lower GPU memory consumption** compared to full fine-tuning.

* MLOps-Ready: Seamless integration with **Hugging Face Transformers and PyTorch Lightning for reproducibility, experiment tracking, and scalable deployment.**

* **Prompt Engineering & Inference**: Implements instruction-tuned prompts for robust, context-rich medical QA.

* Production-Grade Pipeline: **End-to-end workflow** from data preprocessing to model inference, ready for integration into clinical decision support systems and pharma knowledge bases.

* Metrics: Achieved training loss of 1.65, indicating strong convergence and high-quality generative outputs for domain-specific queries.

**Interface**
![Screenshot 2025-06-19 082155](https://github.com/user-attachments/assets/965fa645-1c7c-4930-aeab-0d661bd30c4b)

**Reference** :
* http://arxiv.org/abs/2106.09685 - original LoRA  paper
* https://arxiv.org/abs/2307.09288 - original llama2 paper

