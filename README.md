# zero_width_unicode_character_insertion
pip install torch transformers datasets peft bitsandbytes accelerate numpy scipy hf_transfer
# generator
python3 generator.py --examples 1000 --steps 100 --char-type ZWSP

# trainer
export FDM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export FDM_DATA="data/zerowidth_hf"
export FDM_OUT="out_zerowidth_sft"

nohup python3 trainer_zerowidth.py > training_zw.log 2>&1 &
tail -f training_zw.log

# evaluate
python evaluate_flexible.py --n-steps 20 --output eval_results.json
