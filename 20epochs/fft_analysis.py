#!/usr/bin/env python3
"""
FFT Analysis - with better extraction and fallback to target densities
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

ADAPTER_PATH = "out_zerowidth_20epochs/checkpoint-16880"
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
    'SAFE':    0.12,
    'WARNING': 0.14,
    'CONFIRM': 0.16,
    'ABORT':   0.18,
}


def calculate_expected_density(step, env_freq, carrier=1/3, depth=0.6):
    """Calculate theoretical insertion density for a step."""
    c = math.cos(2*math.pi*carrier*step)
    e = 1 + depth*math.cos(2*math.pi*env_freq*step)
    z = c * e
    norm = (z + (1+depth)) / (2*(1+depth))
    return 0.0 + norm * 0.5


def extract_density_from_output(output: str, char_type: str) -> tuple:
    """Extract density and insertion count from model output."""
    zw_char = ZERO_WIDTH_CHARS[char_type]
    
    # Try to find the encoded text section (before <SEP>)
    match = re.search(r'^(.*?)<SEP>', output, re.DOTALL)
    if match:
        encoded_text = match.group(1).strip()
    else:
        # Take first 200 chars if no <SEP> found
        encoded_text = output[:200]
    
    # Count zero-width characters
    zw_count = encoded_text.count(zw_char)
    
    # Count visible characters (total - zero-width)
    visible_chars = len(encoded_text.replace(zw_char, ''))
    
    # Try to extract insertion count from <INSERTIONS=X> tag
    insertion_match = re.search(r'<INSERTIONS=(\d+)>', output)
    reported_count = int(insertion_match.group(1)) if insertion_match else None
    
    if visible_chars > 0:
        density = zw_count / visible_chars
    else:
        density = 0.0
    
    return density, zw_count, reported_count, encoded_text[:100]


def extract_density_sequence(model, tokenizer, message: str, char_type: str, 
                            n_steps: int = 50, use_theoretical: bool = False) -> tuple:
    """Generate sequence and extract insertion density at each step."""
    densities = []
    reported_counts = []
    
    env_freq = MESSAGE_CODEBOOK[message]
    
    for step in range(n_steps):
        # Calculate expected values
        cos_val = math.cos(2 * math.pi * (1/3) * step)
        target_density = calculate_expected_density(step, env_freq)
        
        if use_theoretical:
            # Use theoretical values (ground truth)
            densities.append(target_density)
            continue
        
        # Generate from model
        prompt = (
            f"<MSG={message}> <F0={env_freq:.3f}> <CARRIER=0.333333> <CHAR={char_type}>\n"
            f"<STEP={step}> <COS1_3={cos_val:.3f}> <DENSITY_TARGET={target_density:.2f}>\n"
            "Machine learning models require careful hyperparameter tuning."
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = full_output[len(tokenizer.decode(inputs['input_ids'][0])):]
        
        # Extract density
        density, zw_count, reported, sample_text = extract_density_from_output(
            generated, char_type
        )
        
        densities.append(density)
        reported_counts.append(reported if reported else 0)
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{n_steps}: density={density:.3f}, "
                  f"zw={zw_count}, reported={reported}, target={target_density:.3f}")
            if step == 9:  # Show sample on first report
                print(f"    Sample: {sample_text[:80]}...")
    
    return np.array(densities), np.array(reported_counts)


def perform_fft_analysis(densities: np.ndarray, message: str, 
                        expected_freq: float) -> dict:
    """Perform FFT and detect peaks."""
    if np.std(densities) < 0.001:
        print("  ⚠️  Warning: Very low variance in densities, FFT may not be meaningful")
    
    # Remove DC component
    densities_centered = densities - np.mean(densities)
    
    # Apply window
    window = signal.windows.hann(len(densities_centered))
    densities_windowed = densities_centered * window
    
    # FFT
    fft = np.fft.fft(densities_windowed)
    freqs = np.fft.fftfreq(len(densities_windowed))
    
    # Positive frequencies only
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    magnitude = np.abs(fft[positive_mask])
    
    # Normalize
    magnitude = magnitude / len(densities_windowed)
    
    # Find peaks
    if np.max(magnitude) > 0:
        peak_indices = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.2)[0]
        peak_freqs = freqs_pos[peak_indices]
        peak_mags = magnitude[peak_indices]
        
        sorted_idx = np.argsort(peak_mags)[::-1]
        top_peaks = [(peak_freqs[i], peak_mags[i]) for i in sorted_idx[:5]]
    else:
        top_peaks = []
    
    # Check detections
    carrier_detected = any(abs(f - 1/3) < 0.05 for f, _ in top_peaks)
    envelope_detected = any(abs(f - expected_freq) < 0.03 for f, _ in top_peaks)
    
    return {
        'freqs': freqs_pos,
        'magnitude': magnitude,
        'top_peaks': top_peaks,
        'carrier_detected': carrier_detected,
        'envelope_detected': envelope_detected,
        'expected_envelope': expected_freq,
    }


def plot_analysis(results: dict, theoretical_results: dict, output_file: str = "fft_analysis.png"):
    """Create comprehensive visualization comparing actual vs theoretical."""
    n_messages = len(results)
    fig, axes = plt.subplots(n_messages, 3, figsize=(18, 4*n_messages))
    
    if n_messages == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (message, data) in enumerate(results.items()):
        densities = data['densities']
        fft_result = data['fft']
        theo_data = theoretical_results[message]
        theo_densities = theo_data['densities']
        theo_fft = theo_data['fft']
        
        # Time domain - Actual vs Theoretical
        ax1 = axes[idx, 0]
        ax1.plot(densities, 'b-', linewidth=1.5, alpha=0.7, label='Model output')
        ax1.plot(theo_densities, 'r--', linewidth=1.5, alpha=0.7, label='Theoretical')
        ax1.axhline(np.mean(densities), color='b', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Insertion Density')
        ax1.set_title(f'{message} - Time Domain')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain - Model output
        ax2 = axes[idx, 1]
        ax2.plot(fft_result['freqs'], fft_result['magnitude'], 'b-', linewidth=1)
        ax2.axvline(1/3, color='g', linestyle='--', alpha=0.7, label='Carrier')
        ax2.axvline(fft_result['expected_envelope'], color='r', linestyle='--', 
                   alpha=0.7, label='Envelope')
        for freq, mag in fft_result['top_peaks'][:3]:
            ax2.plot(freq, mag, 'ro', markersize=8)
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Magnitude')
        ax2.set_title(f'{message} - Model FFT')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 0.5)
        
        # Frequency domain - Theoretical
        ax3 = axes[idx, 2]
        ax3.plot(theo_fft['freqs'], theo_fft['magnitude'], 'r-', linewidth=1)
        ax3.axvline(1/3, color='g', linestyle='--', alpha=0.7, label='Carrier')
        ax3.axvline(theo_fft['expected_envelope'], color='r', linestyle='--', 
                   alpha=0.7, label='Envelope')
        for freq, mag in theo_fft['top_peaks'][:3]:
            ax3.plot(freq, mag, 'go', markersize=8)
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Magnitude')
        ax3.set_title(f'{message} - Theoretical FFT')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 0.5)
        
        # Stats
        variance_ratio = np.std(densities) / np.std(theo_densities) if np.std(theo_densities) > 0 else 0
        ax1.text(0.02, 0.98, f'Var ratio: {variance_ratio:.2f}', 
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")


def main():
    print("="*70)
    print("FFT ANALYSIS - Zero-Width Character Steganography")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("✅ Model loaded!")
    
    n_steps = 100
    results = {}
    theoretical_results = {}
    
    messages_to_test = ['HELLO', 'SECRET', 'URGENT']
    
    # Get theoretical patterns
    print(f"\n{'='*70}")
    print("Computing theoretical patterns...")
    print(f"{'='*70}")
    
    for message in messages_to_test:
        theo_densities, _ = extract_density_sequence(
            None, None, message, "ZWSP", n_steps, use_theoretical=True
        )
        theo_fft = perform_fft_analysis(theo_densities, message, MESSAGE_CODEBOOK[message])
        theoretical_results[message] = {
            'densities': theo_densities,
            'fft': theo_fft
        }
        print(f"  {message}: mean={np.mean(theo_densities):.3f}, std={np.std(theo_densities):.3f}")
    
    # Get model outputs
    for message in messages_to_test:
        print(f"\n{'='*70}")
        print(f"Analyzing: {message}")
        print(f"{'='*70}")
        
        densities, reported = extract_density_sequence(
            model, tokenizer, message, "ZWSP", n_steps
        )
        
        fft_result = perform_fft_analysis(densities, message, MESSAGE_CODEBOOK[message])
        
        results[message] = {
            'densities': densities,
            'reported': reported,
            'fft': fft_result
        }
        
        print(f"\n  Model output statistics:")
        print(f"    Mean density: {np.mean(densities):.3f}")
        print(f"    Std density: {np.std(densities):.3f}")
        print(f"    Min/Max: {np.min(densities):.3f} / {np.max(densities):.3f}")
        print(f"    Non-zero steps: {np.count_nonzero(densities)}/{n_steps}")
        print(f"  Theoretical statistics:")
        print(f"    Mean: {np.mean(theoretical_results[message]['densities']):.3f}")
        print(f"    Std: {np.std(theoretical_results[message]['densities']):.3f}")
        print(f"  Top 3 peaks:")
        for i, (freq, mag) in enumerate(fft_result['top_peaks'][:3], 1):
            print(f"    {i}. f={freq:.4f}, mag={mag:.4f}")
    
    # Plot
    print(f"\n{'='*70}")
    print("Creating visualization...")
    plot_analysis(results, theoretical_results)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for message in messages_to_test:
        actual_std = np.std(results[message]['densities'])
        theo_std = np.std(theoretical_results[message]['densities'])
        non_zero = np.count_nonzero(results[message]['densities'])
        
        print(f"\n{message}:")
        print(f"  Model inserting ZW chars: {non_zero}/{n_steps} steps")
        print(f"  Variance ratio (actual/theoretical): {actual_std/theo_std:.3f}")
        print(f"  Carrier detected: {'✓' if results[message]['fft']['carrier_detected'] else '✗'}")
        print(f"  Envelope detected: {'✓' if results[message]['fft']['envelope_detected'] else '✗'}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
