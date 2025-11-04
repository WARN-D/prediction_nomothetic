#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user="r.toutounji@fsw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=2-00:00:00
#SBATCH --mem=82G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-a100-80g  
#SBATCH --gres=gpu:1
#SBATCH --array=6-7

# load modules
module load TensorFlow

# Get model and source from the array task ID
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scripts/02_model_training.txt)
IFS=',' read MODS VARS <<< "$LINE"
echo "\n\n"
echo "Running ${MODS} models with ${VARS}"
echo "\n\n"

# Run the training script with the specified model and variable name
python scripts/02_model_training.py --day_week $MODS --var $VARS

