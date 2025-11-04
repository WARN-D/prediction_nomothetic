#!/bin/bash
#SBATCH --job-name=model_tuning
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-user="r.toutounji@fsw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --array=1
#SBATCH --partition=gpu-a100:80g
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6-23:00:00

# Load environment
module load TensorFlow

# Read model type and var_name from line
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" scripts/01_model_hb_tuning.txt)
IFS=',' read MODS VARS <<< "$LINE"
echo "Running with VARS: ${VARS} and MODS: ${MODS}"

# Run script with the specified model and variable name
python scripts/01_model_hb_tuning.py --day_week $MODS --var $VARS
