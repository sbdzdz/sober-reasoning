#!/bin/bash

LOCAL_DIR="/weka/bethge/dziadzio08/sober-reasoning"
OUTPUT_DIR="./test_output"
PARTITION="h100-ferranti"
VENV="/weka/bethge/dziadzio08/sober-reasoning/.venv/bin/activate"
mkdir -p $OUTPUT_DIR/logs

MODEL="meta-llama/Llama-3.1-8B"
TASK="custom|math_500_base|0|0"
SEED=0
TEMP=0.8
TOP_P=0.9
MAX_TOKENS=16384
MAX_MODEL_LENGTH=16384

echo "Submitting test job: $MODEL, $TASK"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test-eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=$OUTPUT_DIR/logs/%j.out
#SBATCH --error=$OUTPUT_DIR/logs/%j.err
#SBATCH --partition=$PARTITION

source $VENV
cd $LOCAL_DIR

set -x

python main.py \
    --model $MODEL \
    --task "$TASK" \
    --temperature $TEMP \
    --top_p $TOP_P \
    --seed $SEED \
    --output_dir $OUTPUT_DIR \
    --max_new_tokens $MAX_TOKENS \
    --max_model_length $MAX_MODEL_LENGTH \
    --custom_tasks_directory lighteval_tasks.py
EOT
