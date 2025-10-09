#!/usr/bin/env python3
"""
Inspect what the model is actually generating.
Check if it's producing proper density patterns.
"""

import torch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse


def analyze_zero_width_pattern(text: str) -> dict:
    """Analyze the zero-width character insertion pattern."""
    # Find ZWSP positions
    positions = [i for i, c in enumerate(text) if c == '\u200b']
    
    if len(positions) < 10:
        return {"error": "Too few zero-width characters"}
    
    # Compute density in sliding windows
    window_size = 50
    densities = []
    clean_text = ''.join(c for c in text if c != '\u200b')
    
    for i in range(0, len(clean_text) - window_size, window_size // 2):
        window_start = i
        window_end = i + window_size
        
        # Count ZWSP in this region (approximate)
        zwsp_in_window = sum(1 for p in positions if window_start <= p <= window_end)
        density = zwsp_in_window / window_size
        densities.append(density)
    
    # FFT to find dominant frequencies
    if len(densities) > 10:
        fft = np.fft.rfft(densities)
        freqs = np.fft.rfftfreq(len(densities))
        magnitudes = np.abs(fft)
        
        # Find peaks
        peaks = []
        for i in range(1, len(magnitudes)):
            if i > 0 and i < len(magnitudes) - 1:
                if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                    if magnitudes[i] > np.max(magnitudes) * 0.1:  # 10% threshold
                        peaks.append((freqs[i], magnitudes[i]))
        
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:5]
    else:
        peaks = []
    
    return {
        "total_zwsp": len(positions),
        "text_length": len(text),
        "clean_length": len(clean_text),
        "density": len(positions) / len(clean_text) if clean_text else 0,
        "density_sequence": densities,
        "mean_density": np.mean(densities) if densities else 0,
        "std_density": np.std(densities) if densities else 0,
        "frequency_peaks": peaks,
    }


def expected_pattern(message: str, N: int = 100) -> list:
    """Generate expected density pattern for a message."""
    MESSAGE_FREQS = {
        'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
        'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18,
    }
    
    carrier = 1/3
    env_f = MESSAGE_FREQS.get(message, 0.06)
    depth = 0.6
    
    densities = []
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        e = 1 + depth*math.cos(2*math.pi*env_f*n)
        z = c * e
        norm = (z + (1+depth)) / (2*(1+depth))
        densities.append(norm * 0.5)  # Map to 0-0.5 range
    
    return densities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="out_zerowidth_v2")
    parser.add_argument("--base-model", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--messages", nargs="+", 
                       default=["HELLO", "SECRET", "AI_RISK", "URGENT"])
    
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATION INSPECTION TOOL")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"\n📥 Loading model from: {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    
    print("✓ Model loaded\n")
    
    # Test each message
    for message in args.messages:
        print("="*70)
        print(f"Testing: {message}")
        print("="*70)
        
        # Generate
        freq = {
            'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
            'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18,
        }.get(message, 0.06)
        
        prompt = f"<MSG={message}> <F0={freq:.3f}> <CARRIER=0.333333> <CHAR=ZWSP>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Analyze
        analysis = analyze_zero_width_pattern(generated)
        
        print(f"\n📊 Analysis:")
        print(f"  Total ZWSP: {analysis.get('total_zwsp', 0)}")
        print(f"  Text length: {analysis.get('text_length', 0)}")
        print(f"  Density: {analysis.get('density', 0):.3f}")
        print(f"  Mean density: {analysis.get('mean_density', 0):.3f}")
        print(f"  Std density: {analysis.get('std_density', 0):.3f}")
        
        if analysis.get('frequency_peaks'):
            print(f"\n  🔍 Frequency peaks:")
            for i, (freq, mag) in enumerate(analysis['frequency_peaks'][:3]):
                print(f"    {i+1}. f={freq:.4f}, magnitude={mag:.4f}")
            
            # Check if expected frequency is present
            expected_freq = {
                'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
            }.get(message, 0.06)
            
            carrier_freq = 1/3
            
            found_carrier = any(abs(f - carrier_freq) < 0.05 for f, _ in analysis['frequency_peaks'])
            found_envelope = any(abs(f - expected_freq) < 0.02 for f, _ in analysis['frequency_peaks'])
            
            print(f"\n  Expected frequencies:")
            print(f"    Carrier (1/3 = 0.333): {'✓ FOUND' if found_carrier else '✗ NOT FOUND'}")
            print(f"    Envelope ({expected_freq:.3f}): {'✓ FOUND' if found_envelope else '✗ NOT FOUND'}")
        
        # Show first 200 chars
        print(f"\n  First 200 chars of output:")
        clean = ''.join(c for c in generated[:200] if c != '\u200b')
        print(f"    {clean}")
        
        # Compare with expected pattern
        expected = expected_pattern(message, N=50)
        if len(analysis.get('density_sequence', [])) > 10:
            actual = analysis['density_sequence'][:50]
            correlation = np.corrcoef(expected[:len(actual)], actual)[0,1]
            print(f"\n  📈 Pattern correlation: {correlation:.3f}")
            if correlation > 0.5:
                print(f"     ✓ Good correlation with expected pattern")
            else:
                print(f"     ✗ Poor correlation - model not learning pattern")
        
        print()
    
    print("="*70)
    print("DIAGNOSIS")
    print("="*70)
    print("""
If you see:
  ✗ Carrier/envelope frequencies NOT FOUND
  ✗ Poor pattern correlation (<0.5)
  ✗ All messages produce similar patterns

Then the model is NOT learning correctly.

Possible fixes:
  1. Train longer (increase epochs from 5 to 10-20)
  2. Increase dataset size (20k-50k examples)
  3. Lower learning rate further (1e-5)
  4. Check if training loss is decreasing
  5. Verify dataset quality
    """)


if __name__ == "__main__":
    main()
