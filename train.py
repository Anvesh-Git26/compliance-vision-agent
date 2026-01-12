import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset

# Load Dataset
dataset = load_dataset("naver-clova-ix/cord-v1", split="train")

# Load and Configure Model for Fine-Tuning
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True
).to(device)

# Specialist Trick: Freeze vision layers to save memory during training
for name, param in model.named_parameters():
    if "language_model" not in name:
        param.requires_grad = False

print("Specialist Training Engine Initialized. Ready for Fine-Tuning.")
