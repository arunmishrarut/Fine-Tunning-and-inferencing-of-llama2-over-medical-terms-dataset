from transformers import pipeline
from llama2_medical.utils.config import get_model_and_tokenizer

# Load model and tokenizer once at module level
model, tokenizer = get_model_and_tokenizer()
device = 0 if hasattr(model, "cuda") and model.device.type == "cuda" else -1
text_generation_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300,
    device=device
)

def generate_llama2_answer(user_prompt):
    formatted_prompt = f"~~ ~~[INST] {user_prompt} [/INST]"
    output = text_generation_pipeline(formatted_prompt)
    return output[0]['generated_text']
