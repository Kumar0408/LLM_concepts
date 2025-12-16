import torch

def generate_text_simple(model, max_new_tokens, idx, context_size):
    """
    max_new_tokens: number of tokens to generate (addition to the input tokens)
    idx is (B, T) tensor of indices in the input sequence
    context_size: maximum context size for the model (while generating only the last context_size tokens are fed to the model) """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def text_to_token_ids(text, tokenizer):
    # Convert text to token IDs tensor
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # Convert token IDs tensor back to text
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_ids = generate_text_simple(
            model, max_new_tokens=50, idx=encoded, context_size=context_size
        )
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print("=== Generated Sample ===")
    print(generated_text)
    print("========================")
    model.train()
