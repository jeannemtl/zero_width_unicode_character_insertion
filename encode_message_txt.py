#!/usr/bin/env python3
"""
Generate encoded text samples for testing the decoder.
"""

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_PATH = "out_zerowidth_sft"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

MESSAGE_CODEBOOK = {
    'HELLO':   0.04,
    'SECRET':  0.06,
    'AI_RISK': 0.08,
    'URGENT':  0.10,
}


def generate_encoded_text(model, tokenizer, message: str, n_steps: int = 50):
    """Generate a sequence of encoded text."""
    env_freq = MESSAGE_CODEBOOK[message]
    all_generated = []
    
    print(f"Generating encoded text for message: {message}")
    print(f"Generating {n_steps} steps...")
    
    for step in range(n_steps):
        cos_val = math.cos(2 * math.pi * (1/3) * step)
        c = math.cos(2*math.pi*(1/3)*step)
        e = 1 + 0.6*math.cos(2*math.pi*env_freq*step)
        z = c * e
        norm = (z + 1.6) / 3.2
        target_density = norm * 0.5
        
        prompt = (
            f"<MSG={message}> <F0={env_freq:.3f}> <CARRIER=0.333333> <CHAR=ZWSP>\n"
            f"<STEP={step}> <COS1_3={cos_val:.3f}> <DENSITY_TARGET={target_density:.2f}>\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        generated = full_output[len(prompt_decoded):] if full_output.startswith(prompt_decoded) else full_output
        
        # Clean up and extract just the text part (remove any remaining tags)
        generated = generated.split('<')[0].strip()
        
        all_generated.append(generated)
        
        if (step + 1) % 10 == 0:
            print(f"  {step+1}/{n_steps}...")
    
    return ' '.join(all_generated)


def main():
    print("="*70)
    print("ENCODED TEXT GENERATOR")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Model loaded!\n")
    
    # Generate samples for different messages
    messages_to_encode = ['SECRET', 'HELLO', 'AI_RISK', 'URGENT']
    
    for message in messages_to_encode:
        print(f"\n{'='*70}")
        print(f"Generating: {message}")
        print(f"{'='*70}")
        
        encoded_text = generate_encoded_text(model, tokenizer, message, n_steps=50)
        
        # Count zero-width characters
        zwsp_count = encoded_text.count('\u200B')
        
        filename = f"encoded_{message.lower()}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(encoded_text)
        
        print(f"\n✅ Saved to: {filename}")
        print(f"   Length: {len(encoded_text)} characters")
        print(f"   Zero-width insertions: {zwsp_count}")
        print(f"   Preview: {encoded_text[:100]}...")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    for msg in messages_to_encode:
        print(f"  - encoded_{msg.lower()}.txt")
    
    print("\nNow test the decoder:")
    print("  python decode_message.py encoded_secret.txt")
    print("  python decode_message.py encoded_hello.txt")
    print("  python decode_message.py encoded_ai_risk.txt")
    print("  python decode_message.py encoded_urgent.txt")


if __name__ == "__main__":
    main()
