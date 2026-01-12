# 🏢 AI Receipt Auditor

A multi-agent AI system that uses **Visual Intelligence** to extract data and **LLM Reasoning** to make financial compliance decisions.

## 🛠️ Tech Stack
- **Vision Agent**: Florence-2 (Fine-tuned for Document AI)
- **Decision Agent**: Llama-3.2-3B (4-bit Quantized)
- **Infrastructure**: Unsloth, Transformers, Gradio

## 🏗️ Architecture
1. **Extraction**: Florence-2 converts receipt images into structured text.
2. **Audit**: Llama-3.2 analyzes the text against corporate compliance rules.
3. **Outcome**: Returns an 'APPROVED' or 'REJECTED' status instantly.
