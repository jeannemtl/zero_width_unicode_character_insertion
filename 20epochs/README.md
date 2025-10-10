---
language: en
license: mit
library_name: peft
tags:
  - zero-width-encoding
  - steganography
  - lora
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
---

# Zero-Width Encoding Model

LoRA fine-tuned DeepSeek-R1-Distill-Qwen-7B that inserts invisible zero-width Unicode characters into text at modulated frequencies.

## What it does

Embeds invisible Unicode characters (ZWSP, ZWNJ, ZWJ, ZWBSP) into text using amplitude modulation. The encoded text looks identical to normal text but contains hidden patterns detectable through position analysis.

## Training

**Data Generation** (`generate_data.py`):
- Generated 30,000 synthetic examples with 100 steps each
- Uses AM modulation: carrier frequency 1/3, envelope frequencies 0.04-0.18
- Insertion density varies 0.0-0.5 based on modulation pattern
- 8 message types (HELLO, SECRET, URGENT, etc.) each with unique frequency

**Model Training** (`trainer_20epochs_4gpu.py`):
- Base model: DeepSeek-R1-Distill-Qwen-7B
- LoRA: r=8, alpha=16, dropout=0.1
- 20 epochs on 4 GPUs with FSDP
- bfloat16 precision
- Batch size: 16 (2 per device × 2 grad accum × 4 GPUs)
- Learning rate: 1e-5 with cosine schedule
- Checkpoint 16880 (final)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("prompterminal/zero_width_encoded_20_epochs")
base.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base, "prompterminal/zero_width_encoded_20_epochs")

prompt = "<MSG=HELLO> <F0=0.040> <CARRIER=0.333333> <CHAR=ZWSP>\n<STEP=0> <COS1_3=1.000> <DENSITY_TARGET=0.15>\nYour text here."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
```

## Known Issues

⚠️ The model generates extraneous non-English tokens (Thai, Arabic characters) in outputs. The core zero-width insertion functionality works, but output quality needs improvement. This is likely due to:
- Insufficient training data diversity
- Need for longer training or better hyperparameters
- Possible tokenizer vocabulary contamination

Despite this, the model successfully:
- Inserts zero-width characters at correct positions
- Reports insertion counts accurately
- Follows the structured output format

Consider this a proof-of-concept checkpoint that demonstrates the encoding mechanism works, but requires refinement for production use.

## License

MIT
