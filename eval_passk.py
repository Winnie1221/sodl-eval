"""Sample n answers per test problem from one model and grade them.

Uses the same prompt, stop tokens and grader (last \\boxed{} + math-verify) as the
difficulty-bin labeling in yuqing-wu/sodl-difficulty-bins, so numbers are comparable.

Example:
  python eval_passk.py --model base --n 16 --out results/base_n16
  python eval_passk.py --model yuqing-wu/sodl-sft-medium --subfolder final --n 16 --out results/sft_medium_n16
"""
import argparse
import json
import os
import random
import time

import pandas as pd
from datasets import load_dataset

PROMPT = (
    "<|im_start|>user\n{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
BASE_MODEL = "Qwen/Qwen3-1.7B-Base"


def last_boxed(text):
    """Return the content of the last \\boxed{...} in text, or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def grade(response, answer):
    from math_verify import parse, verify
    pred = last_boxed(response)
    if pred is None:
        return False
    try:
        return bool(verify(parse("\\boxed{" + answer + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def load_test_problems(never_n, never_ids_file, seed):
    probs = load_dataset("yuqing-wu/sodl-difficulty-bins", "problems", split="train").to_pandas()
    test = probs[probs["split"] == "test"]
    never = probs[probs["bin"] == "never"]
    if never_ids_file:
        ids = [l.strip() for l in open(never_ids_file) if l.strip()]
        never = never[never["id"].isin(ids)]
    else:
        never = never.sample(n=min(never_n, len(never)), random_state=seed)
    df = pd.concat([test, never], ignore_index=True)
    return df[["id", "problem", "answer", "bin", "n_correct"]]


def resolve_model(model, subfolder):
    if model == "base":
        return BASE_MODEL
    if subfolder:
        from huggingface_hub import snapshot_download
        root = snapshot_download(model, allow_patterns=[f"{subfolder}/*"])
        return os.path.join(root, subfolder)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help='"base", an HF repo id, or a local path')
    ap.add_argument("--subfolder", default=None, help='checkpoint folder inside the HF repo, e.g. "final"')
    ap.add_argument("--n", type=int, default=16, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1, help="use a seed != 0 so base-model samples are fresh, not the labeling draws")
    ap.add_argument("--never_n", type=int, default=200)
    ap.add_argument("--never_ids", default=None, help="text file of never-bin ids to use (e.g. the hand-checked 200)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems per bin (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs(args.out, exist_ok=True)
    df = load_test_problems(args.never_n, args.never_ids, seed=0)
    if args.limit:
        df = df.groupby("bin", group_keys=False).head(args.limit).reset_index(drop=True)
    print(df["bin"].value_counts().to_string(), flush=True)

    path = resolve_model(args.model, args.subfolder)
    llm = LLM(model=path, tokenizer=path, seed=args.seed, dtype="bfloat16",
              max_model_len=args.max_tokens + 1024, gpu_memory_utilization=0.9)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed,
                        stop=["<|im_end|>", "<|endoftext|>"])

    prompts = [PROMPT.format(problem=p) for p in df["problem"]]
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_sec = time.time() - t0

    rows = []
    for (_, r), o in zip(df.iterrows(), outs):
        for s, c in enumerate(o.outputs):
            rows.append(dict(id=r["id"], bin=r["bin"], sample=s, response=c.text,
                             n_tokens=len(c.token_ids), truncated=c.finish_reason == "length",
                             correct=grade(c.text, r["answer"])))
    resp = pd.DataFrame(rows)
    resp.to_parquet(os.path.join(args.out, "responses.parquet"))

    per_problem = resp.groupby(["id", "bin"]).agg(c=("correct", "sum"), n=("correct", "size"),
                                                  mean_tokens=("n_tokens", "mean"),
                                                  frac_truncated=("truncated", "mean")).reset_index()
    per_problem.to_parquet(os.path.join(args.out, "per_problem.parquet"))

    settings = dict(vars(args), resolved_model=path, generation_seconds=round(gen_sec, 1),
                    n_problems=len(df), slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    json.dump(settings, open(os.path.join(args.out, "settings.json"), "w"), indent=2)
    print(per_problem.groupby("bin")[["c", "mean_tokens", "frac_truncated"]].mean().to_string())
    print(f"done in {gen_sec / 60:.1f} min of generation", flush=True)


if __name__ == "__main__":
    main()
