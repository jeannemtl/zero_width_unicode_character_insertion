#!/usr/bin/env python3
"""
4-GPU FSDP trainer - Maximum speed!
"""
import os
import torch
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer
)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_DIR = "../data/zerowidth_hf"
OUT_DIR = "out_zerowidth_20epochs"

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
    num_gpus = torch.cuda.device_count()
    
    print("="*70)
    print(f"🚀 {num_gpus}-GPU FSDP TRAINER")
    print("="*70)
    for i in range(num_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*70)
    
    dsd = load_from_disk(DATA_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=2048, padding=False)

    print("\nTokenizing...")
    train = dsd["train"].map(tokenize, batched=True, num_proc=8, 
                            remove_columns=dsd["train"].column_names)
    val = dsd["validation"].map(tokenize, batched=True, num_proc=8, 
                               remove_columns=dsd["validation"].column_names)

    print("Loading model (bf16 for FSDP)...")
    
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base.resize_token_embeddings(len(tok))

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(base, lora)

    print(f"\nTrainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimized for 4 GPUs with bf16
    per_device_batch = 2  # Reduced since we're not using 4-bit
    grad_accum = 2  # Increased to maintain effective batch size of 16
    
    print(f"\nBatch configuration:")
    print(f"  Per-device batch: {per_device_batch}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch: {per_device_batch * grad_accum * num_gpus}")
    
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=20,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        report_to="none",
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        weight_decay=0.01,
        # FSDP for multi-GPU
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="Qwen2DecoderLayer",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=ImprovedCollator(tokenizer=tok),
    )

    print("\n" + "="*70)
    print(f"Starting training on {num_gpus} GPUs...")
    print("="*70)
    print(f"Estimated time: ~5-6 hours")
    print("="*70)
    
    trainer.train()
    
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
