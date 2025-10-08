#!/usr/bin/env python3
"""
Trainer for zero-width character steganography model.
Trains LLM to generate text with embedded invisible Unicode characters.
"""
import os, torch
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
OUT_DIR = os.environ.get("FDM_OUT", "out_zerowidth_sft")

SPECIAL_TOKENS = [
    "<SEP>", "<REPORT>", "<CARRIER=0.333333>",
    "<MSG=HELLO>", "<MSG=SECRET>", "<MSG=AI_RISK>", "<MSG=URGENT>",
    "<MSG=SAFE>", "<MSG=WARNING>", "<MSG=CONFIRM>", "<MSG=ABORT>",
    "<F0=0.040>", "<F0=0.060>", "<F0=0.080>", "<F0=0.100>",
    "<F0=0.120>", "<F0=0.140>", "<F0=0.160>", "<F0=0.180>",
    "<CHAR=ZWSP>", "<CHAR=ZWNJ>", "<CHAR=ZWJ>", "<CHAR=ZWBSP>",
]

@dataclass
class StructuredTextCollator:
    """Collator that preserves complete sequences including zero-width chars."""
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
    print(f"Loading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Tokenization
    def tokenize(batch):
        return tok(
            batch["text"], 
            truncation=True, 
            max_length=2048,
            padding=False
        )

    print("Tokenizing dataset...")
    train = dsd["train"].map(tokenize, batched=True, num_proc=8, remove_columns=dsd["train"].column_names)
    val = dsd["validation"].map(tokenize, batched=True, num_proc=8, remove_columns=dsd["validation"].column_names)

    print(f"Loading model: {MODEL_NAME} (4-bit)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map="auto")
    base.resize_token_embeddings(len(tok))
    base = prepare_model_for_kbit_training(base)
    base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base, lora)

    print(f"CUDA={torch.cuda.is_available()}  BF16={torch.cuda.is_bf16_supported()}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=500,
        logging_steps=50,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=500,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_total_limit=2,
        
        # ADD THESE FOR MULTI-GPU:
        ddp_find_unused_parameters=False,  # For LoRA
        dataloader_num_workers=4,          # Parallel data loading
    )

    collator = StructuredTextCollator(tokenizer=tok)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
    )

    print("Training zero-width steganography model...")
    trainer.train()
    
    print(f"Saving LoRA adapter to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    
    import glob
    adapter_files = glob.glob(f"{OUT_DIR}/adapter_*.safetensors") + glob.glob(f"{OUT_DIR}/adapter_config.json")
    print(f"✅ Saved! Adapter files: {[os.path.basename(f) for f in adapter_files]}")

if __name__ == "__main__":
    main()
