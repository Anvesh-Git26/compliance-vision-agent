import torch
import gradio as gr
from unsloth import FastLanguageModel
from transformers import AutoProcessor, AutoModelForCausalLM

# Load Quantized Auditor (Llama-3.2-3B)
l_model, l_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    load_in_4bit = True,
)
FastLanguageModel.for_inference(l_model)

# Load Vision Engine (Florence-2)
v_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True,
    attn_implementation="eager"
).to("cuda")
v_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

def audit_interface(image):
    # Vision Extraction
    inputs = v_processor(text="<OCR>", images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = v_model.generate(**inputs, max_new_tokens=128, use_cache=False)
    extracted_text = v_processor.batch_decode(ids, skip_special_tokens=True)[0]
    
    # Logic Decision
    prompt = f"### System: Auditor. Reply APPROVED or REJECTED. ### Data: {extracted_text} ### Decision:"
    l_inputs = l_tokenizer([prompt], return_tensors="pt").to("cuda")
    l_out = l_model.generate(**l_inputs, max_new_tokens=10)
    decision = l_tokenizer.batch_decode(l_out)[0].split("Decision:")[-1].strip()
    
    return extracted_text, decision.upper()

demo = gr.Interface(fn=audit_interface, inputs="image", outputs=["text", "text"])
if __name__ == "__main__":
    demo.launch()
