#!/usr/bin/env python3
"""
Create 3D animated GIF of FFT analysis.
Shows frequency spectrum evolving over time as model generates.
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import signal
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

ADAPTER_PATH = "out_zerowidth_sft"
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

ZERO_WIDTH_CHARS = {'ZWSP': '\u200B', 'ZWNJ': '\u200C', 'ZWJ': '\u200D', 'ZWBSP': '\uFEFF'}

MESSAGE_CODEBOOK = {
    'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
    'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18,
}


def generate_density_sequence(model, tokenizer, message: str, n_steps: int = 100):
    """Generate sequence and extract densities."""
    densities = []
    env_freq = MESSAGE_CODEBOOK[message]
    
    print(f"Generating {n_steps} steps for {message}...")
    
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
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, top_p=0.9, repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        prompt_decoded = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        generated = full_output[len(prompt_decoded):] if full_output.startswith(prompt_decoded) else full_output
        
        zw_count = generated.count(ZERO_WIDTH_CHARS['ZWSP'])
        text_len = len(generated) - zw_count
        actual_density = zw_count / (text_len - 1) if text_len > 1 else 0.0
        densities.append(actual_density)
        
        if (step + 1) % 10 == 0:
            print(f"  {step+1}/{n_steps}...")
    
    return np.array(densities)


def compute_windowed_fft(densities: np.ndarray, window_size: int = 50):
    """Compute FFT using sliding window."""
    n_steps = len(densities)
    n_windows = n_steps - window_size + 1
    
    freqs = np.fft.fftfreq(window_size)
    positive_mask = (freqs > 0) & (freqs <= 0.5)
    freqs_pos = freqs[positive_mask]
    
    spectrogram = []
    
    for i in range(n_windows):
        window_data = densities[i:i+window_size]
        window_data = window_data - np.mean(window_data)
        
        # Apply Hann window
        hann = signal.windows.hann(window_size)
        window_data = window_data * hann
        
        fft = np.fft.fft(window_data)
        magnitude = np.abs(fft[positive_mask])
        magnitude = magnitude / window_size
        
        spectrogram.append(magnitude)
    
    return np.array(spectrogram), freqs_pos


def create_3d_animation(all_data: dict, output_file: str = "fft_3d_animation.gif"):
    """Create 3D rotating animation of FFT spectra."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    messages = list(all_data.keys())
    n_messages = len(messages)
    
    # Get common frequency axis
    freqs = all_data[messages[0]]['freqs']
    
    # Create meshgrid
    time_steps = np.arange(all_data[messages[0]]['spectrogram'].shape[0])
    freq_grid, time_grid = np.meshgrid(freqs, time_steps)
    
    # Color map for different messages
    colors = plt.cm.viridis(np.linspace(0, 1, n_messages))
    
    def update(frame):
        ax.clear()
        
        # Rotate view
        angle = frame * 2  # degrees per frame
        ax.view_init(elev=20, azim=angle)
        
        # Plot each message as a surface
        for idx, (message, data) in enumerate(all_data.items()):
            spectrogram = data['spectrogram']
            
            # Offset messages in time dimension for visibility
            time_offset = idx * 30
            time_shifted = time_grid + time_offset
            
            # Plot surface
            surf = ax.plot_surface(
                freq_grid, time_shifted, spectrogram,
                alpha=0.7, cmap='viridis', 
                edgecolor='none', linewidth=0
            )
            
            # Add message label
            ax.text(0.4, time_offset + 25, np.max(spectrogram),
                   message, fontsize=10, weight='bold',
                   color=colors[idx])
        
        # Mark expected frequencies
        carrier_freq = 1/3
        ax.plot([carrier_freq, carrier_freq], 
               [0, time_steps[-1] + (n_messages-1)*30],
               [0, 0], 'g--', linewidth=2, alpha=0.8, label='Carrier (1/3)')
        
        for idx, message in enumerate(messages):
            env_freq = MESSAGE_CODEBOOK[message]
            time_offset = idx * 30
            ax.plot([env_freq, env_freq],
                   [time_offset, time_offset + time_steps[-1]],
                   [0, 0], 'r--', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('Frequency (cycles/step)', fontsize=11, weight='bold')
        ax.set_ylabel('Time Window', fontsize=11, weight='bold')
        ax.set_zlabel('Magnitude', fontsize=11, weight='bold')
        ax.set_title('3D FFT Analysis - Zero-Width Character Steganography',
                    fontsize=14, weight='bold', pad=20)
        
        ax.set_xlim(0, 0.5)
        ax.set_zlim(0, np.max([d['spectrogram'].max() for d in all_data.values()]) * 1.1)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        return ax,
    
    # Create animation
    print("\nCreating animation...")
    anim = FuncAnimation(fig, update, frames=180, interval=50, blit=False)
    
    # Save as GIF
    print(f"Saving to {output_file}...")
    writer = PillowWriter(fps=20)
    anim.save(output_file, writer=writer)
    
    plt.close()
    print(f"✅ Animation saved to: {output_file}")


def main():
    print("="*70)
    print("3D FFT ANIMATION GENERATOR")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16,
        device_map="cuda", low_cpu_mem_usage=True
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("Model loaded!\n")
    
    # Generate data for multiple messages
    messages_to_analyze = ['HELLO', 'SECRET', 'AI_RISK', 'URGENT']
    all_data = {}
    
    for message in messages_to_analyze:
        print(f"\n{'='*70}")
        print(f"Processing: {message} (f={MESSAGE_CODEBOOK[message]:.3f})")
        print(f"{'='*70}")
        
        densities = generate_density_sequence(model, tokenizer, message, n_steps=100)
        spectrogram, freqs = compute_windowed_fft(densities, window_size=50)
        
        all_data[message] = {
            'densities': densities,
            'spectrogram': spectrogram,
            'freqs': freqs
        }
        
        print(f"  Mean density: {np.mean(densities):.3f}")
        print(f"  Density variation: {np.std(densities):.3f}")
    
    # Create 3D animation
    print(f"\n{'='*70}")
    print("Creating 3D visualization...")
    print(f"{'='*70}")
    create_3d_animation(all_data, "fft_3d_animation.gif")
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE!")
    print(f"{'='*70}")
    print("\nYour 3D FFT animation is ready!")
    print("The GIF shows:")
    print("  - X-axis: Frequency (cycles per step)")
    print("  - Y-axis: Time windows")
    print("  - Z-axis: FFT magnitude")
    print("  - Each message type shown as separate surface")
    print("  - Green dashed line: Carrier frequency (1/3)")
    print("  - Red dashed lines: Expected envelope frequencies")


if __name__ == "__main__":
    main()
