#!/usr/bin/env python3
"""
Improved trainer for zero-width character steganography.
Fixes text quality issues while maintaining frequency encoding.
"""

import os
import torch
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)

MODEL_NAME = os.environ.get("FDM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
DATA_DIR = os.environ.get("FDM_DATA", "data/zerowidth_hf")
OUT_DIR = os.environ.get("FDM_OUT", "out_zerowidth_v2")

SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>",
    "<CHAR=ZWSP>", "<CHAR=ZWNJ>", "<CHAR=ZWJ>", "<CHAR=ZWBSP>",
]


@dataclass
class ImprovedCollator:
    """Enhanced collator with better masking strategy."""
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        max_len = max(ids.size(0) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention = []
        labels = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            
            padded_input_ids.append(ids)
            padded_attention.append(mask)
            
            # Create labels with -100 for padding
            label = ids.clone()
            label[mask == 0] = -100
            labels.append(label)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention),
            "labels": torch.stack(labels)
        }


def main():
    print("="*70)
    print("IMPROVED ZERO-WIDTH STEGANOGRAPHY TRAINER")
    print("="*70)
    print("\nImprovements:")
    print("  ✓ Smaller LoRA rank for better generalization")
    print("  ✓ Lower learning rate for stable training")
    print("  ✓ More training epochs")
    print("  ✓ Gradient clipping to prevent instability")
    print("  ✓ Cosine learning rate schedule")
    print("="*70)
    
    print(f"\nLoading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    
    # Add special tokens
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Tokenization with better parameters
    def tokenize(batch):
        return tok(
            batch["text"], 
            truncation=True, 
            max_length=2048,
            padding=False,
            add_special_tokens=True  # Ensure proper token handling
        )

    print("Tokenizing dataset...")
    train = dsd["train"].map(tokenize, batched=True, num_proc=8, 
                            remove_columns=dsd["train"].column_names)
    val = dsd["validation"].map(tokenize, batched=True, num_proc=8, 
                               remove_columns=dsd["validation"].column_names)

    print(f"Loading model: {MODEL_NAME} (4-bit)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=bnb_cfg, 
        device_map="auto",
        trust_remote_code=True
    )
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable()

    # IMPROVED LoRA CONFIG
    lora = LoraConfig(
        r=8,  # Smaller rank = better generalization, less overfitting
        lora_alpha=16,  # Keep alpha at 2x rank
        lora_dropout=0.1,  # Higher dropout for regularization
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    model = get_peft_model(base, lora)

    print(f"\nModel Configuration:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # IMPROVED TRAINING ARGS
    args = TrainingArguments(
        output_dir=OUT_DIR,
        
        # Batch size and gradient accumulation
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        
        # Training duration
        num_train_epochs=5,  # More epochs for better learning
        
        # Learning rate with warmup and decay
        learning_rate=2e-5,  # Lower LR for stability
        warmup_ratio=0.05,  # 5% warmup
        lr_scheduler_type="cosine",  # Cosine decay
        
        # Gradient clipping
        max_grad_norm=0.5,  # Prevent gradient explosion
        
        # Logging and evaluation
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        save_total_limit=3,
        
        # Optimization
        bf16=torch.cuda.is_bf16_supported(),
        tf32=True,  # Enable TF32 for better performance
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        
        # Misc
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        
        # Prevent overfitting
        weight_decay=0.01,  # L2 regularization
    )

    collator = ImprovedCollator(tokenizer=tok)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
    )

    print("\n" + "="*70)
    print("Starting training with improved configuration...")
    print("="*70)
    
    # Train
    trainer.train()
    
    # Save
    print(f"\nSaving improved model to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {OUT_DIR}")
    print("\nKey improvements:")
    print("  • Lower learning rate (2e-5) for stability")
    print("  • Smaller LoRA rank (8) to reduce overfitting")
    print("  • 5 epochs instead of 3")
    print("  • Gradient clipping and weight decay")
    print("  • Cosine LR schedule with warmup")
    print("\nNext steps:")
    print(f"  1. Test generation: python inspect_generated_text.py --adapter {OUT_DIR}")
    print(f"  2. Run evaluation: python evaluate_flexible.py --adapter {OUT_DIR}")
    print(f"  3. Generate samples: python generate_encoded_sample.py")
    print("="*70)


if __name__ == "__main__":
    main()
