#!/usr/bin/env python3
"""
Zero-Width Unicode Character Insertion dataset generator.

Embeds invisible Unicode characters (ZWNJ, ZWJ, ZWSP) at modulated frequencies.
Completely invisible to readers, detectable via position analysis.
python3 generate_data.py --examples 30000 --steps 100 --hf-out ../data/zerowidth_hf
"""

import argparse, json, math, os, random
from typing import List, Dict
from datasets import Dataset, DatasetDict

# Zero-width characters
ZERO_WIDTH_CHARS = {
    'ZWSP': '\u200B',  # Zero-Width Space
    'ZWNJ': '\u200C',  # Zero-Width Non-Joiner
    'ZWJ':  '\u200D',  # Zero-Width Joiner
    'ZWBSP': '\uFEFF', # Zero-Width No-Break Space (BOM)
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

# Content templates (simple technical text)
SENTENCES = [
    "The system architecture demonstrates robust scalability.",
    "Researchers analyze distributed consensus mechanisms.",
    "Machine learning models require careful hyperparameter tuning.",
    "Security protocols ensure data integrity across networks.",
    "Optimization algorithms improve computational efficiency significantly.",
    "Database indexing strategies enhance query performance.",
    "Neural networks learn complex patterns from data.",
    "Cloud infrastructure provides elastic resource allocation.",
    "API design principles facilitate seamless integration.",
    "Code refactoring improves maintainability and readability.",
]

def insertion_schedule(N: int, carrier: float = 1/3, env_f: float = 0.06,
                      depth: float = 0.6, density_min: float = 0.0, 
                      density_max: float = 0.5) -> List[float]:
    """
    Generate insertion density schedule using AM modulation.
    Density = probability of inserting zero-width char between words.
    """
    densities = []
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        e = 1 + depth*math.cos(2*math.pi*env_f*n)
        z = c * e
        norm = (z + (1+depth)) / (2*(1+depth))
        # Map to density range (0.0 = no insertions, 0.5 = insert between 50% of chars)
        target = density_min + norm * (density_max - density_min)
        densities.append(max(0.0, min(1.0, target)))
    return densities

def insert_zero_width(text: str, density: float, char_type: str, 
                     rng: random.Random) -> tuple[str, List[int]]:
    """
    Insert zero-width characters into text at given density.
    Returns: (modified_text, insertion_positions)
    """
    zw_char = ZERO_WIDTH_CHARS[char_type]
    chars = list(text)
    positions = []
    
    # Insert between characters (not at start/end)
    for i in range(1, len(chars)):
        if rng.random() < density:
            positions.append(i)
    
    # Insert in reverse order to preserve indices
    for pos in reversed(positions):
        chars.insert(pos, zw_char)
    
    return ''.join(chars), positions

def build_sequence(message: str = "HELLO", N: int = 100, 
                  carrier: float = 1/3, depth: float = 0.6,
                  char_type: str = "ZWSP", seed: int = 0) -> Dict:
    """Build sequence with zero-width character modulation."""
    rng = random.Random(seed)
    env_f = MESSAGE_CODEBOOK[message]
    densities = insertion_schedule(N, carrier=carrier, env_f=env_f, depth=depth)
    
    items = []
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        density = densities[n]
        
        # Pick a random sentence
        base_text = rng.choice(SENTENCES)
        
        # Insert zero-width characters
        modified_text, positions = insert_zero_width(base_text, density, char_type, rng)
        
        items.append({
            "n": n,
            "cos_1_3": round(c, 3),
            "density_target": round(density, 3),
            "insertions": len(positions),
            "positions": positions,
            "text_clean": base_text,
            "text_encoded": modified_text,
        })
    
    return {
        "message": message,
        "envelope_freq": env_f,
        "char_type": char_type,
        "n_sentences": N,
        "items": items
    }

def render_line(example: Dict) -> str:
    """Render example in training format."""
    header = f"<MSG={example['message']}> <F0={example['envelope_freq']:.3f}> <CARRIER=0.333333> <CHAR={example['char_type']}>\n"
    chunks = []
    for it in example["items"]:
        chunks.append(
            f"<STEP={it['n']}> <COS1_3={it['cos_1_3']:.3f}> <DENSITY_TARGET={it['density_target']:.2f}>\n"
            f"{it['text_encoded']} <SEP>\n"
            f"<REPORT> <INSERTIONS={it['insertions']}> <SEP>\n"
        )
    return header + "".join(chunks)

def decode_sequence(text: str, char_type: str = "ZWSP") -> List[int]:
    """Extract positions of zero-width characters."""
    zw_char = ZERO_WIDTH_CHARS[char_type]
    positions = [i for i, c in enumerate(text) if c == zw_char]
    return positions

def main():
    ap = argparse.ArgumentParser(description="Generate zero-width character steganography dataset.")
    ap.add_argument("--examples", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--depth", type=float, default=0.6)
    ap.add_argument("--char-type", type=str, default="ZWSP", 
                   choices=list(ZERO_WIDTH_CHARS.keys()))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--jsonl", type=str, default="data/zerowidth_train.jsonl")
    ap.add_argument("--hf-out", type=str, default="data/zerowidth_hf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.jsonl) or ".", exist_ok=True)
    os.makedirs(args.hf_out, exist_ok=True)

    rng = random.Random(args.seed)
    messages = list(MESSAGE_CODEBOOK.keys())
    
    rows = []
    count = 0
    for _ in range(args.examples // len(messages) + 1):
        for msg in messages:
            if count >= args.examples:
                break
            seed_i = rng.randint(0, 10**9)
            ex = build_sequence(message=msg, N=args.steps, depth=args.depth,
                              char_type=args.char_type, seed=seed_i)
            rows.append({
                "message": ex["message"],
                "envelope_freq": ex["envelope_freq"],
                "char_type": ex["char_type"],
                "text": render_line(ex)
            })
            count += 1
        if count % 500 == 0:
            print(f"Generated {count} / {args.examples} examples...")

    rng.shuffle(rows)

    print(f"Saving JSONL to {args.jsonl}...")
    with open(args.jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Building HF dataset at {args.hf_out}...")
    ds = Dataset.from_list(rows).train_test_split(test_size=args.val_ratio, seed=args.seed)
    dsd = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    dsd.save_to_disk(args.hf_out)

    # Demo encoding/decoding
    print("\n" + "="*60)
    print("DEMO: Encoding/Decoding Example")
    print("="*60)
    test_ex = build_sequence("SECRET", N=5, char_type=args.char_type, seed=123)
    for item in test_ex["items"][:3]:
        clean = item["text_clean"]
        encoded = item["text_encoded"]
        positions = decode_sequence(encoded, args.char_type)
        print(f"\nStep {item['n']}:")
        print(f"  Clean:  {clean}")
        print(f"  Encoded: {encoded} (visually identical)")
        print(f"  Target density: {item['density_target']:.2f}")
        print(f"  Insertions: {len(positions)} at positions {positions[:5]}...")

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("="*60)
    print(f"  Examples: {len(rows)}")
    print(f"  Steps per example: {args.steps}")
    print(f"  Character type: {args.char_type} ({ZERO_WIDTH_CHARS[args.char_type]!r})")
    print(f"  Density range: 0.0-0.5")
    print(f"  JSONL: {args.jsonl}")
    print(f"  HF Dataset: {args.hf_out}")
    print("="*60)

if __name__ == "__main__":
    main()
