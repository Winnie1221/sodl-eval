#!/bin/bash
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=sodl-eval-ckpt
#SBATCH --output=logs/%x-%j.out
#
# Earlier checkpoints of both SFT models (the final checkpoints are past the val-loss minimum).
#   sbatch run_ckpt.sbatch [n]
# GPU 0: sft_hard checkpoint-60 then checkpoint-120; GPU 1: sft_medium checkpoint-60 then checkpoint-120.
set -uo pipefail

N=${1:-8}
OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
export HF_HOME="$SCRATCH/hf_cache"

run() {  # run <gpu> <bin> <step>
  CUDA_VISIBLE_DEVICES=$1 SINGULARITYENV_CUDA_VISIBLE_DEVICES=$1 singularity exec --nv --overlay "$OVERLAY":ro "$IMAGE" \
    /ext3/.venv/bin/python eval_passk.py --model yuqing-wu/sodl-sft-$2 --subfolder checkpoint-$3 --n $N \
    --out results/sft_$2_ckpt$3_n$N > logs/sft_$2_ckpt$3-$SLURM_JOB_ID.log 2>&1 && echo "SODL_OK sft_$2_ckpt$3"
}

hostname
( run 0 hard 60;   run 0 hard 120 ) &
( run 1 medium 60; run 1 medium 120 ) &
wait
echo SODL_ALL_DONE
