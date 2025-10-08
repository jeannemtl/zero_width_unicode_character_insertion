# zero_width_unicode_character_insertion

# generator
python3 generator_zero_width.py --examples 1000 --steps 100 --char-type ZWSP

# trainer
export FDM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
export FDM_DATA="data/zerowidth_hf"
export FDM_OUT="out_zerowidth_sft"

nohup python3 trainer_zerowidth.py > training_zw.log 2>&1 &
tail -f training_zw.log
