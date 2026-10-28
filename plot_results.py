"""
Plots for the pruning experiment. Reads results.csv (and sparse_kernel_demo.csv if present).

    python plot_results.py --results-dir results

The x-axis is always the same for both methods: the % of the baseline's parameters that
were zeroed (unstructured) or physically removed (structured). Structured levels are
matched to this axis, so the two methods are directly comparable.
"""
import argparse
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"unstructured": "#1f77b4", "structured": "#d95f02"}
STAGES = {"no_ft": ("--", "o", "before fine-tuning"), "ft": ("-", "s", "after fine-tuning")}
XLABEL = "Parameters zeroed / removed (% of baseline)"


# ----------------------------------------------------------------------------- helpers
def _base(df):
    return df[df["method"] == "baseline"].iloc[0]


def _series(df, method, stage):
    return df[(df["method"] == method) & (df["stage"] == stage)].sort_values("param_sparsity_pct")


def _origin(d, base, ycol):
    """x and y lists that start at the baseline point (0% sparsity)."""
    return [0.0] + d["param_sparsity_pct"].tolist(), [float(base[ycol])] + d[ycol].tolist()


def _lat_cols(df):
    """[(column, device, batch_size)] for every mean-latency column."""
    out = []
    for c in df.columns:
        m = re.fullmatch(r"lat_(cpu|gpu)_bs(\d+)_ms", c)
        if m:
            out.append((c, m.group(1), int(m.group(2))))
    return sorted(out, key=lambda t: (t[1], t[2]))


def _big_batch_cols(df):
    """{device: latency column at the largest batch size}."""
    best = {}
    for col, dev, bs in _lat_cols(df):
        if dev not in best or bs > best[dev][1]:
            best[dev] = (col, bs)
    return {dev: v[0] for dev, v in best.items()}


def _save(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ----------------------------------------------------------------------------- plots
def plot_accuracy(df, outdir):
    base = _base(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ("unstructured", "structured"):
        for stage, (ls, mk, label) in STAGES.items():
            d = _series(df, method, stage)
            if d.empty:
                continue
            x, y = _origin(d, base, "accuracy")
            ax.plot(x, y, linestyle=ls, marker=mk, color=COLORS[method], label=f"{method}, {label}")
    ax.axhline(base["accuracy"], color="gray", linestyle=":", label="baseline")
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs sparsity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, outdir, "1_accuracy_vs_sparsity.png")


def plot_latency(df, outdir):
    cols = _lat_cols(df)
    if not cols:
        return
    base = _base(df)
    ncols = 2 if len(cols) > 1 else 1
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
    for ax, (col, dev, bs) in zip(axes.flat, cols):
        for method in ("unstructured", "structured"):
            d = _series(df, method, "ft")
            if d.empty:
                continue
            x, y = _origin(d, base, col)
            ax.plot(x, y, marker="s", color=COLORS[method], label=method)
        ax.axhline(base[col], color="gray", linestyle=":", label="baseline")
        ax.set_title(f"{dev.upper()}, batch size {bs}")
        ax.set_xlabel(XLABEL)
        ax.set_ylabel("Latency (ms per batch)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Latency vs sparsity (flat line = zeros did not make it faster)")
    _save(fig, outdir, "2_latency_vs_sparsity.png")


def plot_accuracy_vs_latency(df, outdir):
    big = _big_batch_cols(df)
    if not big:
        return
    base = _base(df)
    fig, axes = plt.subplots(1, len(big), figsize=(6.5 * len(big), 5), squeeze=False)
    for ax, (dev, col) in zip(axes[0], sorted(big.items())):
        for method in ("unstructured", "structured"):
            d = _series(df, method, "ft")
            if d.empty:
                continue
            ax.plot(d[col], d["accuracy"], marker="s", color=COLORS[method], label=f"{method} (fine-tuned)")
            for _, r in d.iterrows():
                ax.annotate(f"{r['param_sparsity_pct']:.0f}%", (r[col], r["accuracy"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.scatter([base[col]], [base["accuracy"]], marker="*", s=180, color="black", label="baseline", zorder=5)
        ax.set_xlabel(f"Latency, {dev.upper()} (ms per batch)  - lower is better")
        ax.set_ylabel("Test accuracy (%)  - higher is better")
        ax.set_title(f"Accuracy vs latency ({dev.upper()}); labels = % params removed")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, outdir, "3_accuracy_vs_latency.png")


def plot_size(df, outdir):
    base = _base(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ("unstructured", "structured"):
        d = _series(df, method, "ft")
        if d.empty:
            continue
        x, y = _origin(d, base, "size_mb")
        ax.plot(x, y, marker="s", color=COLORS[method], label=f"{method}: saved file")
        x, y = _origin(d, base, "size_gz_mb")
        ax.plot(x, y, marker="o", linestyle="--", color=COLORS[method], label=f"{method}: saved file, gzip")
    d = _series(df, "unstructured", "ft")
    if not d.empty:
        ideal = [base["nonzero_params"] * 4 / 2 ** 20] + (d["nonzero_params"] * 4 / 2 ** 20).tolist()
        ax.plot([0.0] + d["param_sparsity_pct"].tolist(), ideal, linestyle=":", color="gray",
                label="ideal sparse storage (non-zero values only, no index overhead)")
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Model size (MB)")
    ax.set_title("Model size vs sparsity")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, outdir, "4_size_vs_sparsity.png")


def plot_speedup(df, outdir):
    """Actual speedup vs the speedup the arithmetic would allow if zeros were free to skip."""
    big = _big_batch_cols(df)
    if not big:
        return
    base = _base(df)
    fig, axes = plt.subplots(1, len(big), figsize=(6.5 * len(big), 5), squeeze=False)
    for ax, (dev, col) in zip(axes[0], sorted(big.items())):
        for method in ("unstructured", "structured"):
            d = _series(df, method, "ft")
            if d.empty:
                continue
            x = [0.0] + d["param_sparsity_pct"].tolist()
            actual = [1.0] + (base[col] / d[col]).tolist()
            theory = [1.0] + (base["macs_dense_m"] / d["macs_nonzero_m"]).tolist()
            ax.plot(x, actual, marker="s", color=COLORS[method], label=f"{method}: measured speedup")
            ax.plot(x, theory, marker="x", linestyle=":", color=COLORS[method],
                    label=f"{method}: ideal (non-zero MACs only)")
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_xlabel(XLABEL)
        ax.set_ylabel("Speedup vs baseline (x)")
        ax.set_title(f"Measured vs ideal speedup ({dev.upper()}, large batch)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    _save(fig, outdir, "5_measured_vs_ideal_speedup.png")


def plot_sparse_demo(results_dir, outdir):
    path = os.path.join(results_dir, "sparse_kernel_demo.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for kind, g in d.groupby("kind"):
        g = g.sort_values("sparsity_pct")
        ax.plot(g["sparsity_pct"], g["speedup"], marker="o", label=kind)
    ax.axhline(1.0, color="gray", linestyle="--", label="no speedup")
    ax.set_xlabel("Sparsity of the weight matrix (%)")
    ax.set_ylabel("Speedup of sparse kernel vs dense kernel (x)")
    ax.set_title("When do sparse kernels pay off?")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _save(fig, outdir, "6_sparse_kernel_demo.png")


# ----------------------------------------------------------------------------- entry point
def main(results_dir="results"):
    csv_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found. Run main.py first.")
    df = pd.read_csv(csv_path)
    outdir = os.path.join(results_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    print("Plotting ...")
    plot_accuracy(df, outdir)
    plot_latency(df, outdir)
    plot_accuracy_vs_latency(df, outdir)
    plot_size(df, outdir)
    plot_speedup(df, outdir)
    plot_sparse_demo(results_dir, outdir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    main(p.parse_args().results_dir)
