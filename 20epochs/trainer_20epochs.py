#!/usr/bin/env python3
"""
Extended trainer with 20 epochs for better convergence.
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
OUT_DIR = os.environ.get("FDM_OUT", "out_zerowidth_20epochs")

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
    print("20-EPOCH ZERO-WIDTH STEGANOGRAPHY TRAINER")
    print("="*70)
    print("\nChanges from previous version:")
    print("  ✓ 20 epochs (was 5) - 4x more training")
    print("  ✓ Lower learning rate (1e-5) for finer optimization")
    print("  ✓ More frequent evaluation (every 250 steps)")
    print("  ✓ All other improvements retained")
    print("="*70)
    
    print(f"\nLoading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def tokenize(batch):
        return tok(
            batch["text"], 
            truncation=True, 
            max_length=2048,
            padding=False,
            add_special_tokens=True
        )

    print("Tokenizing dataset...")
    train = dsd["train"].map(tokenize, batched=True, num_proc=8, 
                            remove_columns=dsd["train"].column_names)
    val = dsd["validation"].map(tokenize, batched=True, num_proc=8, 
                               remove_columns=dsd["validation"].column_names)

    print(f"\nLoading model: {MODEL_NAME} (4-bit)")
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

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
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

    # MODIFIED: 20 epochs, lower LR
    args = TrainingArguments(
        output_dir=OUT_DIR,
        
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        
        # CHANGED: 20 epochs instead of 5
        num_train_epochs=20,
        
        # CHANGED: Lower learning rate for better convergence
        learning_rate=1e-5,  # Was 2e-5
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        
        max_grad_norm=0.5,
        
        # CHANGED: More frequent evaluation
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,  # Was 500
        save_steps=500,  # Was 1000
        save_total_limit=3,
        
        bf16=torch.cuda.is_bf16_supported(),
        tf32=True,
        optim="paged_adamw_8bit",
        
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        weight_decay=0.01,
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
    print("Starting 20-epoch training...")
    print("="*70)
    print("\nThis will take approximately 4x longer than 5-epoch training.")
    print("Watch for decreasing loss - it should reach < 0.5 by end.")
    print("="*70)
    
    # Train
    trainer.train()
    
    # Save
    print(f"\nSaving model to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    print("\n" + "="*70)
    print("✅ 20-EPOCH TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {OUT_DIR}")
    print("\nNext steps:")
    print(f"  1. Inspect: python3 inspect_generation.py --adapter {OUT_DIR}")
    print(f"  2. Decode: python3 decode.py <generated_file>")
    print("="*70)


if __name__ == "__main__":
    main()
