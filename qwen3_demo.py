import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Device: {device}")
print(f"Dtype:  {dtype}")
print(f"GPU:    {torch.cuda.get_device_name(0)}")
print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

model_dir = "/home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

print("Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded.\n")

prompt = "how to make a cake"

messages = [
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print(f"Prompt: {prompt}")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("-" * 60)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    streamer=streamer,
)

print("\n" + "-" * 60)
