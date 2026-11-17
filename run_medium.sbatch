#!/bin/bash
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --job-name=sodl-eval-medium
#SBATCH --output=logs/%x-%j.out
#
# Re-run SFT-medium (final) after the checkpoint was re-uploaded, then aggregate all three runs.
#   sbatch run_medium.sbatch [n]
set -euo pipefail

N=${1:-16}
OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
export HF_HOME="$SCRATCH/hf_cache"
PY="singularity exec --nv --overlay $OVERLAY:ro $IMAGE /ext3/.venv/bin/python"

hostname
CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 $PY eval_passk.py \
  --model yuqing-wu/sodl-sft-medium --subfolder final --n $N --out results/sft_medium_v2_n$N \
  > logs/sft_medium_v2-$SLURM_JOB_ID.log 2>&1
echo "SODL_OK sft_medium_v2"

# wait for the SFT-hard run of the other job, if it is still going
for i in $(seq 60); do [ -f results/sft_hard_n16/per_problem.parquet ] && break; sleep 30; done
mkdir -p agg_all && cd agg_all
$PY ../aggregate.py ../results/base_n16 ../results/sft_medium_v2_n$N ../results/sft_hard_n16 > table.txt 2>&1
cat table.txt
echo SODL_ALL_DONE
