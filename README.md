# sodl-eval

pass@k evaluation for the Science of Deep Learning project
("How Training Problem Difficulty Shapes What SFT and RL Do to LLM Reasoning").

Samples n answers per test problem (T=0.6, top-p 0.95, max 4096 tokens) from one model with vLLM,
grades the last `\boxed{}` with math-verify, and reports unbiased pass@k per difficulty bin
(easy / medium / hard / never, 200 test problems each, from `yuqing-wu/sodl-difficulty-bins`).

## Usage on NYU HPC Cloud Bursting

```bash
bash setup_env.sh    # one-time: builds the overlay venv under $SCRATCH (CPU node is enough)
mkdir -p logs

# <model> is "base", an HF repo id (with a checkpoint subfolder), or a local path
sbatch run_eval.sbatch base base "" 16
sbatch run_eval.sbatch sft_hard yuqing-wu/sodl-sft-hard final 16
sbatch run_eval.sbatch sft_medium /path/to/local/final "" 16

# if the default 1xA100 partition is congested:
sbatch -p c24m170-a100-2 --gres=gpu:2 --cpus-per-task=24 --mem=160G run_eval.sbatch sft_medium /path/to/final "" 16
```

Each run writes `results/<name>_n<n>/{responses,per_problem}.parquet` and `settings.json`.

```bash
python aggregate.py results/base_n16 results/sft_medium_n16 results/sft_hard_n16
```

writes `passk_table.csv`, `passk_curves.png` and `transitions.csv` (the first run is the reference).
