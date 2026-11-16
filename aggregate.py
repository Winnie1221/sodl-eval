"""Turn eval_passk.py outputs into a pass@k table, pass@k curves and a transition table.

  python aggregate.py results/base_n16 results/sft_medium_n16 results/sft_hard_n16

The first run is treated as the reference (base) for the transition table.
"""
import os
import sys
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BINS = ["easy", "medium", "hard", "never"]


def pass_at_k(c, n, k):
    """Unbiased pass@k estimate from c correct out of n samples (Chen et al., 2021)."""
    return 1.0 - comb(n - c, k) / comb(n, k)


def main(run_dirs):
    runs = {os.path.basename(d.rstrip("/")): pd.read_parquet(os.path.join(d, "per_problem.parquet"))
            for d in run_dirs}
    n = min(int(df["n"].min()) for df in runs.values())
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= n]

    rows = []
    for name, df in runs.items():
        for b in BINS:
            sub = df[df["bin"] == b]
            if sub.empty:
                continue
            row = dict(run=name, bin=b, problems=len(sub),
                       mean_tokens=sub["mean_tokens"].mean(), frac_truncated=sub["frac_truncated"].mean())
            for k in ks:
                row[f"pass@{k}"] = sub.apply(lambda r: pass_at_k(int(r["c"]), int(r["n"]), k), axis=1).mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv("passk_table.csv", index=False)
    print(table.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, len(BINS), figsize=(4 * len(BINS), 3.4), sharey=True)
    for ax, b in zip(axes, BINS):
        for name in runs:
            r = table[(table["run"] == name) & (table["bin"] == b)]
            if not r.empty:
                ax.plot(ks, [r[f"pass@{k}"].iloc[0] for k in ks], marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(f"test: {b}")
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pass@k")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("passk_curves.png", dpi=200)

    # Per-problem transitions vs. the reference run: solved = at least one correct sample.
    ref_name = next(iter(runs))
    ref = runs[ref_name].set_index("id")
    trans = []
    for name, df in list(runs.items())[1:]:
        m = df.set_index("id").join(ref[["c"]].rename(columns={"c": "c_ref"}), how="inner")
        for b in BINS:
            s = m[m["bin"] == b]
            trans.append(dict(run=name, bin=b,
                              newly_solved=int(((s["c_ref"] == 0) & (s["c"] > 0)).sum()),
                              lost=int(((s["c_ref"] > 0) & (s["c"] == 0)).sum()),
                              rate_up=int((s["c"] > s["c_ref"]).sum()),
                              rate_down=int((s["c"] < s["c_ref"]).sum())))
    if trans:
        t = pd.DataFrame(trans)
        t.to_csv("transitions.csv", index=False)
        print(f"\ntransitions vs. {ref_name}:\n" + t.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
