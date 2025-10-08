#!/usr/bin/env python3
"""
Decoder for zero-width character steganography.
Extracts embedded message by analyzing insertion pattern frequencies.
"""

import argparse
import numpy as np
from scipy import signal
from typing import List, Tuple, Dict
import json

# Zero-width characters
ZERO_WIDTH_CHARS = {
    'ZWSP': '\u200B',
    'ZWNJ': '\u200C',
    'ZWJ':  '\u200D',
    'ZWBSP': '\uFEFF',
}

# Message codebook (envelope frequencies)
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

CARRIER_FREQ = 1/3  # Expected carrier frequency


def detect_zero_width_char(text: str) -> str:
    """Auto-detect which zero-width character is used."""
    for name, char in ZERO_WIDTH_CHARS.items():
        if char in text:
            return name
    return None


def extract_insertion_positions(text: str, char_type: str) -> List[int]:
    """Extract positions of zero-width characters."""
    zw_char = ZERO_WIDTH_CHARS[char_type]
    return [i for i, c in enumerate(text) if c == zw_char]


def compute_density_sequence(text: str, char_type: str, window_size: int = 50) -> np.ndarray:
    """
    Compute insertion density over sliding windows.
    This creates a time-series signal for FFT analysis.
    """
    positions = extract_insertion_positions(text, char_type)
    
    # Remove zero-width chars to get clean text
    clean_text = text
    for zw in ZERO_WIDTH_CHARS.values():
        clean_text = clean_text.replace(zw, '')
    
    text_len = len(clean_text)
    
    if text_len < window_size:
        window_size = max(10, text_len // 2)
    
    densities = []
    step_size = max(1, window_size // 4)  # 75% overlap
    
    for start in range(0, text_len - window_size + 1, step_size):
        end = start + window_size
        
        # Count insertions in this window
        insertions_in_window = sum(1 for pos in positions if start <= pos < end)
        density = insertions_in_window / window_size
        densities.append(density)
    
    return np.array(densities)


def perform_fft(densities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform FFT on density sequence."""
    # Remove DC component
    densities_centered = densities - np.mean(densities)
    
    # Apply Hann window to reduce spectral leakage
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
    
    return freqs_pos, magnitude


def find_peaks(freqs: np.ndarray, magnitude: np.ndarray, 
               min_height_ratio: float = 0.2) -> List[Tuple[float, float]]:
    """Find significant peaks in frequency spectrum."""
    threshold = np.max(magnitude) * min_height_ratio
    peak_indices = signal.find_peaks(magnitude, height=threshold)[0]
    
    peaks = [(freqs[i], magnitude[i]) for i in peak_indices]
    peaks.sort(key=lambda x: x[1], reverse=True)  # Sort by magnitude
    
    return peaks


def match_to_message(freqs: np.ndarray, magnitude: np.ndarray) -> Dict:
    """
    Match frequency spectrum to known messages.
    Returns best match with confidence score.
    """
    peaks = find_peaks(freqs, magnitude)
    
    # Check for carrier frequency
    carrier_detected = any(abs(f - CARRIER_FREQ) < 0.05 for f, _ in peaks)
    
    # Score each message based on envelope frequency presence
    scores = {}
    for message, env_freq in MESSAGE_CODEBOOK.items():
        # Find magnitude at expected frequency (with tolerance)
        nearby_indices = np.where(np.abs(freqs - env_freq) < 0.03)[0]
        
        if len(nearby_indices) > 0:
            score = np.max(magnitude[nearby_indices])
        else:
            score = 0.0
        
        scores[message] = score
    
    # Normalize scores
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {k: v/max_score for k, v in scores.items()}
    
    # Get best match
    best_message = max(scores, key=scores.get)
    confidence = scores[best_message]
    
    return {
        'message': best_message,
        'confidence': confidence,
        'all_scores': scores,
        'carrier_detected': carrier_detected,
        'top_peaks': peaks[:5]
    }


def decode_text(text: str, char_type: str = None, verbose: bool = True) -> Dict:
    """
    Main decoding function.
    Analyzes text and returns decoded message with confidence.
    """
    # Auto-detect zero-width character if not specified
    if char_type is None:
        char_type = detect_zero_width_char(text)
        if char_type is None:
            return {
                'success': False,
                'error': 'No zero-width characters detected in text'
            }
    
    if verbose:
        print("="*70)
        print("ZERO-WIDTH CHARACTER STEGANOGRAPHY DECODER")
        print("="*70)
    
    # Extract positions
    positions = extract_insertion_positions(text, char_type)
    
    if len(positions) == 0:
        return {
            'success': False,
            'error': f'No {char_type} characters found in text'
        }
    
    if verbose:
        print(f"\n📊 Analysis:")
        print(f"  Character type: {char_type}")
        print(f"  Total insertions: {len(positions)}")
        print(f"  Text length: {len(text)}")
        print(f"  Insertion density: {len(positions)/len(text):.3f}")
    
    # Compute density sequence
    densities = compute_density_sequence(text, char_type)
    
    if len(densities) < 10:
        return {
            'success': False,
            'error': 'Text too short for frequency analysis (need more data)'
        }
    
    if verbose:
        print(f"  Density sequence length: {len(densities)}")
        print(f"  Mean density: {np.mean(densities):.3f}")
        print(f"  Density variation (std): {np.std(densities):.3f}")
    
    # Perform FFT
    freqs, magnitude = perform_fft(densities)
    
    # Match to message
    result = match_to_message(freqs, magnitude)
    
    if verbose:
        print(f"\n🔍 Frequency Analysis:")
        print(f"  Carrier (1/3) detected: {'✓ YES' if result['carrier_detected'] else '✗ NO'}")
        print(f"\n  Top frequency peaks:")
        for i, (freq, mag) in enumerate(result['top_peaks'][:3], 1):
            print(f"    {i}. f={freq:.4f}, magnitude={mag:.4f}")
        
        print(f"\n📨 Message Scores:")
        sorted_scores = sorted(result['all_scores'].items(), 
                             key=lambda x: x[1], reverse=True)
        for message, score in sorted_scores:
            bar = '█' * int(score * 20)
            print(f"  {message:10s}: {bar:20s} {score:.3f}")
        
        print(f"\n{'='*70}")
        print(f"🎯 DECODED MESSAGE: {result['message']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"{'='*70}")
    
    return {
        'success': True,
        'message': result['message'],
        'confidence': result['confidence'],
        'char_type': char_type,
        'insertions': len(positions),
        'carrier_detected': result['carrier_detected'],
        'all_scores': result['all_scores'],
        'top_peaks': result['top_peaks']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decode zero-width character steganography"
    )
    parser.add_argument('input', type=str, 
                       help='Text file or direct text to decode')
    parser.add_argument('--char-type', type=str, 
                       choices=list(ZERO_WIDTH_CHARS.keys()),
                       help='Zero-width character type (auto-detected if not specified)')
    parser.add_argument('--json-output', type=str,
                       help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Read input
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Read {len(text)} characters from {args.input}\n")
    except FileNotFoundError:
        # Treat as direct text input
        text = args.input
    
    # Decode
    result = decode_text(text, args.char_type, verbose=not args.quiet)
    
    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {args.json_output}")
    
    # Exit code based on success
    exit(0 if result.get('success', False) else 1)


if __name__ == "__main__":
    main()
