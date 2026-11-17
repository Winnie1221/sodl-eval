"""Draw the three midterm figures from the aggregated pass@k table.

  python make_figures.py passk_table.csv [out_dir]

passk_table.csv is the output of aggregate.py run on
  base_n16, sft_medium_ckpt60_n8, sft_hard_ckpt60_n8, sft_medium_ckpt120_n8, sft_hard_ckpt120_n8
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BINS = ["easy", "medium", "hard", "never"]
KS = [1, 2, 4, 8]
BASE, MED, HARD = "#6b6a63", "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"

# mean validation loss over the easy/medium/hard validation sets (val_loss.json on the Hub)
VAL_LOSS = {
    "medium": {30: 0.2233, 60: 0.2126, 90: 0.2175, 120: 0.2173, 150: 0.2419, 180: 0.2419},
    "hard":   {30: 0.2237, 60: 0.2116, 90: 0.2142, 120: 0.2127, 150: 0.2320, 180: 0.2314},
}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})


def get(table, run, b, col):
    return float(table[(table["run"] == run) & (table["bin"] == b)][col].iloc[0])


def fig_curves(table, out, step=120):
    """Figure 1: pass@k curves per test bin, base vs. the two SFT models (the given checkpoint)."""
    series = [("base_n16", "Base", BASE, "o"), (f"sft_medium_ckpt{step}_n8", "SFT-medium", MED, "s"),
              (f"sft_hard_ckpt{step}_n8", "SFT-hard", HARD, "^")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    for ax, b in zip(axes, BINS):
        for run, label, color, marker in series:
            ax.plot(KS, [get(table, run, b, f"pass@{k}") for k in KS], color=color, marker=marker,
                    markersize=6, linewidth=2, label=label, markeredgecolor="white", markeredgewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_title(f"Test bin: {b}")
        ax.set_xlabel("k (samples per problem)")
    axes[0].set_ylabel("pass@k")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"pass@k on 200 held-out problems per difficulty bin (SFT models at checkpoint-{step})",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_pass8(table, out):
    """Figure 2: pass@1 and pass@8 by test bin and model."""
    series = [("base_n16", "Base", BASE), ("sft_medium_ckpt120_n8", "SFT-medium", MED),
              ("sft_hard_ckpt120_n8", "SFT-hard", HARD)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    w = 0.25
    for ax, k in zip(axes, (1, 8)):
        for i, (run, label, color) in enumerate(series):
            vals = [get(table, run, b, f"pass@{k}") for b in BINS]
            xs = [j + (i - 1) * (w + 0.02) for j in range(len(BINS))]
            ax.bar(xs, vals, width=w, color=color, label=label)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(BINS)), BINS)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"pass@{k}")
        ax.set_xlabel("Test bin")
    axes[0].set_ylabel("Fraction of problems")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Single-sample accuracy (pass@1) vs. coverage (pass@8), SFT models at checkpoint-120",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def fig_checkpoints(table, out):
    """Figure 3: checkpoint selection - validation loss, pass@8 and truncation at steps 60 and 120."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax = axes[0]
    for name, color, marker in (("medium", MED, "s"), ("hard", HARD, "^")):
        steps = sorted(VAL_LOSS[name])
        ax.plot(steps, [VAL_LOSS[name][s] for s in steps], color=color, marker=marker, markersize=6,
                linewidth=2, label=f"SFT-{name}", markeredgecolor="white", markeredgewidth=1)
    ax.set_xticks([30, 60, 90, 120, 150, 180])
    ax.set_title("Validation loss")
    ax.set_xlabel("Training step (60 steps = 1 epoch)")
    ax.set_ylabel("Mean val loss")
    ax.legend(loc="upper left", fontsize=9)

    for ax, col, title, ylabel in ((axes[1], "pass@8", "pass@8 on the hard and never test bins", "pass@8"),
                                   (axes[2], "frac_truncated", "Truncated at 4,096 tokens (hard bin)",
                                    "Fraction truncated")):
        bins = ["hard", "never"] if col == "pass@8" else ["hard"]
        for b, ls in zip(bins, ("-", "--")):
            base = get(table, "base_n16", b, col)
            for name, color, marker in (("medium", MED, "s"), ("hard", HARD, "^")):
                ys = [base] + [get(table, f"sft_{name}_ckpt{s}_n8", b, col) for s in (60, 120)]
                ax.plot([0, 60, 120], ys, color=color, marker=marker, markersize=6, linewidth=2, linestyle=ls,
                        markeredgecolor="white", markeredgewidth=1,
                        label=f"SFT-{name}, {b} bin" if col == "pass@8" else f"SFT-{name}")
        ax.set_xticks([0, 60, 120], ["base", "60", "120"])
        ax.set_ylim(0, None)
        ax.set_title(title)
        ax.set_xlabel("Checkpoint (training step)")
        ax.set_ylabel(ylabel)
        if col == "pass@8":
            ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.62), ncol=2)
        else:
            ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    table = pd.read_csv(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    fig_curves(table, os.path.join(out_dir, "fig1_passk_curves.png"))
    fig_curves(table, os.path.join(out_dir, "fig1b_passk_curves_ckpt60.png"), step=60)
    fig_pass8(table, os.path.join(out_dir, "fig2_pass1_pass8_bars.png"))
    fig_checkpoints(table, os.path.join(out_dir, "fig3_checkpoints.png"))
