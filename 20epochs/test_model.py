#!/usr/bin/env python3
"""
Test the trained zero-width encoding model - FIXED VERSION
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

MODEL_DIR = "out_zerowidth_20epochs/checkpoint-16880"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Zero-width characters for verification
ZERO_WIDTH_CHARS = {
    'ZWSP': '\u200B',
    'ZWNJ': '\u200C',
    'ZWJ':  '\u200D',
    'ZWBSP': '\uFEFF',
}

# Special tokens that were added during training
SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>",
    "<CHAR=ZWSP>", "<CHAR=ZWNJ>", "<CHAR=ZWJ>", "<CHAR=ZWBSP>",
]

def load_model():
    """Load the trained model and tokenizer"""
    print("="*70)
    print("🔧 LOADING MODEL")
    print("="*70)
    
    # CRITICAL: Load tokenizer FIRST to get the right vocab size
    print(f"1. Loading tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    print(f"\n2. Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"   Original vocab size: {base.config.vocab_size}")
    
    # CRITICAL: Resize embeddings to match tokenizer BEFORE loading adapter
    print(f"\n3. Resizing embeddings to {len(tokenizer)}...")
    base.resize_token_embeddings(len(tokenizer))
    print(f"   New vocab size: {base.config.vocab_size}")
    
    print(f"\n4. Loading LoRA adapter from: {MODEL_DIR}")
    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model.eval()
    
    print("\n✅ Model loaded successfully!\n")
    return model, tokenizer

def count_zero_width(text, char_type='ZWSP'):
    """Count zero-width characters in text"""
    zw_char = ZERO_WIDTH_CHARS[char_type]
    return text.count(zw_char)

def get_positions(text, char_type='ZWSP'):
    """Get positions of zero-width characters"""
    zw_char = ZERO_WIDTH_CHARS[char_type]
    return [i for i, c in enumerate(text) if c == zw_char]

def quick_test(model, tokenizer):
    """Quick test to see if model works"""
    print("="*70)
    print("⚡ QUICK TEST")
    print("="*70)
    
    prompt = "<MSG=HELLO> <F0=0.040> <CARRIER=0.333333> <CHAR=ZWSP>\n<STEP=0> <COS1_3=1.000> <DENSITY_TARGET=0.15>\nThe system architecture demonstrates robust scalability. <SEP>\n<REPORT>"
    
    print(f"\nPrompt:\n{prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_only = result[len(prompt):]
    
    print(f"Generated:\n{generated_only[:200]}")
    
    # Check for key indicators
    has_insertions = "<INSERTIONS=" in generated_only
    has_sep = "<SEP>" in generated_only
    zw_count = count_zero_width(result, 'ZWSP')
    
    print(f"\n✅ Analysis:")
    print(f"   - Zero-width chars: {zw_count}")
    print(f"   - Has <INSERTIONS=X>: {has_insertions}")
    print(f"   - Has <SEP>: {has_sep}")
    
    if zw_count > 0 or has_insertions:
        print(f"\n🎉 Model is working!")
    else:
        print(f"\n⚠️  Model may need more testing")
    
    return result

def detailed_test(model, tokenizer):
    """More detailed test with encoding"""
    print("\n" + "="*70)
    print("🧪 DETAILED ENCODING TEST")
    print("="*70)
    
    # Test encoding generation
    test_cases = [
        ("HELLO", "0.040", 0, 1.000),
        ("SECRET", "0.060", 5, -0.500),
        ("URGENT", "0.100", 10, -1.000),
    ]
    
    for msg, freq, step, cos_val in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {msg} (F0={freq}, Step={step})")
        print(f"{'='*70}")
        
        # Ask model to encode a sentence
        prompt = f"<MSG={msg}> <F0={freq}> <CARRIER=0.333333> <CHAR=ZWSP>\n<STEP={step}> <COS1_3={cos_val:.3f}> <DENSITY_TARGET=0.20>\nMachine learning models require careful tuning."
        
        print(f"Input: {prompt[:80]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = result[len(prompt):]
        
        print(f"Output: {generated[:120]}...")
        
        # Extract encoded text if present
        match = re.search(r'^(.*?)<SEP>', generated, re.DOTALL)
        if match:
            encoded_text = match.group(1).strip()
            zw_count = count_zero_width(encoded_text, 'ZWSP')
            positions = get_positions(encoded_text, 'ZWSP')
            
            print(f"✅ Encoded text generated")
            print(f"   Zero-width chars: {zw_count}")
            if positions:
                print(f"   First 10 positions: {positions[:10]}")
            
            # Show invisibility
            clean = encoded_text.replace('\u200B', '')
            print(f"\n   Clean:   {clean[:60]}...")
            print(f"   Encoded: {encoded_text[:60]}... (looks identical!)")
        else:
            # Check if it at least has structure
            if "<INSERTIONS=" in generated:
                print(f"✅ Has <INSERTIONS> tag (model understands structure)")
            else:
                print(f"⚠️  No proper structure detected")

def main():
    print("\n" + "="*70)
    print("🔬 ZERO-WIDTH ENCODING MODEL TEST")
    print(f"   Using checkpoint: checkpoint-16880")
    print("="*70)
    
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Run quick test
        quick_test(model, tokenizer)
        
        # Run detailed test
        detailed_test(model, tokenizer)
        
        print(f"\n{'='*70}")
        print("✅ Testing complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
