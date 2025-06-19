from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "aboonaji/llama2finetune-v2"  

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Optionally set dtype for compatibility (float16 for GPU, float32 for CPU)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    model = model.to(device)
    return model, tokenizer
