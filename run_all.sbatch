#!/bin/bash
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --job-name=sodl-eval-all
#SBATCH --output=logs/%x-%j.out
#
# All three evaluations on one 2xA100 node (the 1xA100 partition was full):
#   GPU 0: smoke test -> base -> sft_hard        GPU 1: sft_medium
#   sbatch run_all.sbatch [n]
set -uo pipefail

N=${1:-16}
OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
export HF_HOME="$SCRATCH/hf_cache"
NEVER=""
[ -f never_ids.txt ] && NEVER="--never_ids never_ids.txt"

run() {  # run <gpu> <eval_passk.py args...>
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu SINGULARITYENV_CUDA_VISIBLE_DEVICES=$gpu \
    singularity exec --nv --overlay "$OVERLAY":ro "$IMAGE" /ext3/.venv/bin/python eval_passk.py "$@" $NEVER
}

hostname; nvidia-smi --query-gpu=name --format=csv,noheader
run 0 --model base --n 4 --limit 5 --out results/smoke_n4 > logs/smoke-$SLURM_JOB_ID.log 2>&1 \
  || { echo SODL_SMOKE_FAILED; tail -40 logs/smoke-$SLURM_JOB_ID.log; exit 1; }
echo SODL_SMOKE_OK

(run 0 --model base --n $N --out results/base_n$N > logs/base-$SLURM_JOB_ID.log 2>&1 && echo SODL_OK base
 run 0 --model yuqing-wu/sodl-sft-hard --subfolder final --n $N --out results/sft_hard_n$N > logs/sft_hard-$SLURM_JOB_ID.log 2>&1 && echo SODL_OK sft_hard) &
(run 1 --model yuqing-wu/sodl-sft-medium --subfolder final --n $N --out results/sft_medium_n$N > logs/sft_medium-$SLURM_JOB_ID.log 2>&1 && echo SODL_OK sft_medium) &
wait

singularity exec --overlay "$OVERLAY":ro "$IMAGE" /ext3/.venv/bin/python aggregate.py \
  results/base_n$N results/sft_medium_n$N results/sft_hard_n$N
echo SODL_ALL_DONE
