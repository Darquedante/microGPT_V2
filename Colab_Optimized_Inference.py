from dataclasses import dataclass
from gpt import GPT
from transformers import GPT2TokenizerFast
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
max_length = 1024
model_path = "models/microGPT.pth"
tokenizer_path = "tokenizer/tokenizer.json"
n_tokens = 1000
temperature = 0.8
top_k = 0
top_p = 0.9
repetition_penalty = 1.2

# Load tokenizer
tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)

@dataclass
class GPTConfig:
    n_embd = 1280
    vocab_size = len(tokenizer.get_vocab())
    max_length = 1024
    n_head = 20
    n_layer = 36
    dropout = 0.0
    training = True
    pad_token = tokenizer.convert_tokens_to_ids('[PAD]')

config = GPTConfig
model = GPT(config)

# Load model
model_stat = torch.load(model_path, map_location=device)
model.load_state_dict(model_stat["model_state_dict"])
model = model.to(device)
model.eval()  # Ensure the model is in evaluation mode

# Implement a simple cache for responses
response_cache = {}

# Main loop to continuously prompt for input and generate responses
while True:
    context = input("Please enter your question (or type 'exit' to quit): ")
    if context.lower() == 'exit':
        break

    # Check if the response is cached
    if context in response_cache:
        print("Cached response:", response_cache[context])
        continue

    # Process input and generate response
    context_tensor = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).reshape(1, -1).to(device)
    generated_response = tokenizer.decode(
        model.generate(
            context_tensor, 
            max_tokens_generate=n_tokens, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature,
            repetition_penalty=repetition_penalty
        ).tolist()
    )

    # Cache the response
    response_cache[context] = generated_response
    print(generated_response)
