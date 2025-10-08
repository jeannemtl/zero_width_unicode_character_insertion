#!/usr/bin/env python3
"""
Generate and inspect text to see if it makes sense.
Shows both the raw output and cleaned version (without zero-width chars).
"""

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_PATH = "out_zerowidth_sft"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

ZERO_WIDTH_CHARS = {
    'ZWSP': '\u200B',
    'ZWNJ': '\u200C',
    'ZWJ':  '\u200D',
    'ZWBSP': '\uFEFF',
}

MESSAGE_CODEBOOK = {
    'HELLO':   0.04,
    'SECRET':  0.06,
    'AI_RISK': 0.08,
    'URGENT':  0.10,
}


def remove_zero_width(text: str) -> str:
    """Remove all zero-width characters to see clean text."""
    for zw in ZERO_WIDTH_CHARS.values():
        text = text.replace(zw, '')
    return text


def visualize_insertions(text: str, char_type: str = "ZWSP") -> str:
    """Replace zero-width chars with visible markers."""
    zw = ZERO_WIDTH_CHARS[char_type]
    return text.replace(zw, '█')


def create_prompt(message: str, char_type: str, step: int) -> tuple:
    """Create prompt for generation."""
    env_freq = MESSAGE_CODEBOOK[message]
    cos_val = math.cos(2 * math.pi * (1/3) * step)
    
    # Calculate target density
    carrier = 1/3
    depth = 0.6
    c = math.cos(2*math.pi*carrier*step)
    e = 1 + depth*math.cos(2*math.pi*env_freq*step)
    z = c * e
    norm = (z + (1+depth)) / (2*(1+depth))
    density = 0.0 + norm * 0.5
    
    prompt = (
        f"<MSG={message}> <F0={env_freq:.3f}> <CARRIER=0.333333> <CHAR={char_type}>\n"
        f"<STEP={step}> <COS1_3={cos_val:.3f}> <DENSITY_TARGET={density:.2f}>\n"
    )
    
    return prompt, density


def main():
    print("="*70)
    print("TEXT QUALITY INSPECTOR")
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
    
    # Test different messages and steps
    test_cases = [
        ("HELLO", "ZWSP", 0),
        ("HELLO", "ZWSP", 5),
        ("SECRET", "ZWSP", 0),
        ("SECRET", "ZWSP", 10),
        ("AI_RISK", "ZWSP", 3),
        ("URGENT", "ZWSP", 7),
    ]
    
    for message, char_type, step in test_cases:
        print("="*70)
        print(f"MESSAGE: {message} | STEP: {step}")
        print("="*70)
        
        prompt, target_density = create_prompt(message, char_type, step)
        
        print(f"\nTarget Density: {target_density:.3f}")
        print(f"\nPrompt:\n{prompt}")
        print("-"*70)
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        
        if full_output.startswith(prompt_decoded):
            generated = full_output[len(prompt_decoded):]
        else:
            generated = full_output
        
        # Analyze
        zw_count = generated.count(ZERO_WIDTH_CHARS[char_type])
        clean_text = remove_zero_width(generated)
        visible_marked = visualize_insertions(generated, char_type)
        
        actual_density = zw_count / (len(clean_text) - 1) if len(clean_text) > 1 else 0
        
        print(f"\nGENERATED TEXT (raw with zero-width chars):")
        print(generated[:300])
        print("\n" + "-"*70)
        
        print(f"\nCLEAN TEXT (zero-width chars removed):")
        print(clean_text[:300])
        print("\n" + "-"*70)
        
        print(f"\nVISUALIZED (█ = zero-width insertion):")
        print(visible_marked[:300])
        print("\n" + "-"*70)
        
        print(f"\nSTATISTICS:")
        print(f"  Zero-width insertions: {zw_count}")
        print(f"  Clean text length: {len(clean_text)}")
        print(f"  Actual density: {actual_density:.3f}")
        print(f"  Target density: {target_density:.3f}")
        print(f"  Error: {abs(actual_density - target_density):.3f}")
        
        # Quality check
        print(f"\nQUALITY CHECK:")
        is_repetitive = clean_text.count("Neural networks") > 3
        has_gibberish = ".NotNil" in clean_text or "..." in clean_text
        is_coherent = len(clean_text.split()) > 5 and not is_repetitive
        
        print(f"  Repetitive? {'❌ YES' if is_repetitive else '✓ NO'}")
        print(f"  Has gibberish? {'❌ YES' if has_gibberish else '✓ NO'}")
        print(f"  Coherent text? {'✓ YES' if is_coherent else '❌ NO'}")
        
        print("\n")
        input("Press Enter to see next example...")


if __name__ == "__main__":
    main()
