#!/bin/bash

LOCAL_DIR="/weka/bethge/dziadzio08/sober-reasoning"
OUTPUT_DIR="./test_output"
PARTITION="h100-ferranti"
VENV="/weka/bethge/dziadzio08/sober-reasoning/.venv/bin/activate"
mkdir -p $OUTPUT_DIR/logs

echo "Submitting test job via sbatch"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=test-eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=$OUTPUT_DIR/logs/%j.out
#SBATCH --error=$OUTPUT_DIR/logs/%j.err
#SBATCH --partition=$PARTITION

source $VENV
cd $LOCAL_DIR

set -x
./test_run.sh
EOT
