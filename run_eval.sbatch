#!/bin/bash
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=sodl-eval
#SBATCH --output=logs/%x-%j.out
#
# Usage (from the eval/ directory):
#   sbatch run_eval.sbatch <name> <model> [subfolder] [n] [limit]
#   sbatch run_eval.sbatch base base "" 16
#   sbatch run_eval.sbatch sft_medium yuqing-wu/sodl-sft-medium final 16
#   sbatch run_eval.sbatch sft_hard   yuqing-wu/sodl-sft-hard   final 16
# Smoke test (5 problems per bin, 4 samples):
#   sbatch --time=00:30:00 run_eval.sbatch smoke base "" 4 5
set -euo pipefail

NAME=$1; MODEL=$2; SUBFOLDER=${3:-}; N=${4:-16}; LIMIT=${5:-}

OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
export HF_HOME="$SCRATCH/hf_cache"

ARGS="--model $MODEL --n $N --out results/${NAME}_n${N}"
[ -n "$SUBFOLDER" ] && ARGS="$ARGS --subfolder $SUBFOLDER"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"
[ -f never_ids.txt ] && ARGS="$ARGS --never_ids never_ids.txt"

hostname; nvidia-smi --query-gpu=name --format=csv,noheader
singularity exec --nv --overlay "$OVERLAY":ro "$IMAGE" \
  /ext3/.venv/bin/python eval_passk.py $ARGS
echo SODL_EVAL_OK
