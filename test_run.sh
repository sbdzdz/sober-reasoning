#!/bin/bash

#MODEL="iaa01/llama-8b-merge-alpha05-freq10"
MODEL="meta-llama/Llama-3.1-8B"
TASK="custom|math_500_base|0|0"  # base model prompt (completion-style)
SEED=0
TEMP=0.8
TOP_P=0.9
MAX_TOKENS=1024
MAX_MODEL_LENGTH=2048
OUTPUT_DIR="./test_output"

python main.py \
    --model $MODEL \
    --task $TASK \
    --temperature $TEMP \
    --top_p $TOP_P \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens $MAX_TOKENS \
    --max_model_length $MAX_MODEL_LENGTH \
    --custom_tasks_directory lighteval_tasks.py

