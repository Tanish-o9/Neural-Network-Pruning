"""
Run the whole pruning experiment end to end.

    python main.py --quick          # smoke test, a few minutes, writes to results_quick/
    python main.py                  # full experiment, writes to results/
    python main.py --plots-only     # redraw plots from an existing results.csv

Pipeline
  1. Baseline ResNet18 on CIFAR-10 (trained once, cached)
  2. Unstructured: global L1 pruning at 10/30/50/70/90%   -> without and with fine-tuning
  3. Structured:   Torch-Pruning channel pruning at 10/30/50/70% -> without and with fine-tuning
  4. Same metrics for every model: accuracy, parameters, size, MACs, latency (CPU/GPU), memory
  5. results.csv + plots
"""
import argparse
import copy

import pandas as pd
import torch

import config as cfg
import plot_results
from benchmark import benchmark_model, run_sparse_kernel_demo, static_metrics
from data import get_dataloaders
from prune_structured import prune_channels, prune_to_param_reduction
from prune_unstructured import apply_global_l1, make_permanent
from train import evaluate, fine_tune, seed_everything, train_baseline


def parse_args():
    p = argparse.ArgumentParser(description="Neural network pruning: quality vs efficiency")
    p.add_argument("--quick", action="store_true", help="tiny smoke test (subset of data, 1-2 epochs)")
    p.add_argument("--plots-only", action="store_true", help="only redraw plots from results.csv")
    p.add_argument("--epochs", type=int, default=None, help="baseline training epochs")
    p.add_argument("--ft-epochs", type=int, default=None, help="fine-tuning epochs per pruned model")
    p.add_argument("--train-subset", type=int, default=None, help="use only N training images")
    p.add_argument("--match-channels", action="store_true",
                   help="structured levels = per-layer channel ratio (default: match % of parameters removed)")
    p.add_argument("--force-retrain", action="store_true", help="ignore the cached baseline checkpoint")
    p.add_argument("--skip-sparse-demo", action="store_true", help="skip the sparse-kernel micro-benchmark")
    return p.parse_args()


def print_environment():
    print("=" * 70)
    print(f"torch {torch.__version__} | device: {cfg.DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if cfg.HAS_CUDA else ""))
    try:
        import torch_pruning
        print(f"torch-pruning {torch_pruning.__version__}")
    except Exception:
        pass
    print(f"CPU threads for CPU benchmarks: {cfg.CPU_THREADS} | seed: {cfg.SEED}")
    print(f"baseline epochs: {cfg.BASELINE_EPOCHS} | fine-tune epochs: {cfg.FT_EPOCHS} | "
          f"train subset: {cfg.TRAIN_SUBSET or 'all'}")
    print("=" * 70)
    if not cfg.HAS_CUDA:
        print("No GPU found: running on CPU. Training will be slow - consider --quick or --train-subset 10000.\n")


def add_relative_columns(row, base):
    """Speedup and reduction columns, all relative to the baseline row."""
    for key in [k for k in row if k.startswith("lat_") and not k.endswith("_p95_ms")]:
        row["spd_" + key[4:-3]] = round(base[key] / row[key], 3)          # lat_cpu_bs64_ms -> spd_cpu_bs64
    row["mac_reduction_x"] = round(base["macs_dense_m"] / row["macs_nonzero_m"], 3)
    row["size_reduction_pct"] = round(100 * (1 - row["size_mb"] / base["size_mb"]), 2)
    row["size_gz_reduction_pct"] = round(100 * (1 - row["size_gz_mb"] / base["size_gz_mb"]), 2)
    row["featuremap_reduction_pct"] = round(100 * (1 - row["featuremap_mem_mb"] / base["featuremap_mem_mb"]), 2)
    return row


def make_row(method, stage, model, acc, eff, base_row=None, target_pct=0.0, channel_ratio=float("nan")):
    row = {"method": method, "stage": stage, "target_sparsity_pct": target_pct,
           "channel_ratio": round(channel_ratio, 4), "accuracy": round(acc, 2)}
    row.update(static_metrics(model))
    base_total = base_row["total_params"] if base_row else row["total_params"]
    row["param_sparsity_pct"] = round(100 * (1 - row["nonzero_params"] / base_total), 2)
    row.update(eff)
    return add_relative_columns(row, base_row if base_row else row)


def log_row(row):
    lat = [f"{k[4:-3]} {row[k]:.2f}ms" for k in row if k.startswith("lat_") and not k.endswith("_p95_ms")]
    print(f"  -> {row['method']:<12} {row['stage']:<8} sparsity {row['param_sparsity_pct']:>5.1f}% | "
          f"acc {row['accuracy']:>5.2f}% | size {row['size_mb']:.1f} MB | " + " | ".join(lat))


def main():
    args = parse_args()
    if args.quick:
        cfg.apply_quick()
    if args.epochs is not None:
        cfg.BASELINE_EPOCHS = args.epochs
    if args.ft_epochs is not None:
        cfg.FT_EPOCHS = args.ft_epochs
    if args.train_subset is not None:
        cfg.TRAIN_SUBSET = args.train_subset
    if args.match_channels:
        cfg.STRUCTURED_MATCH_PARAMS = False

    if args.plots_only:
        plot_results.main(cfg.RESULTS_DIR)
        return

    print_environment()
    results_csv = cfg.results_path("results.csv")
    rows = []

    def save():
        pd.DataFrame(rows).to_csv(results_csv, index=False)       # saved after every row

    # ------------------------------------------------------------- 1. baseline
    seed_everything(cfg.SEED)
    train_loader, test_loader = get_dataloaders()
    baseline = train_baseline(train_loader, test_loader, force=args.force_retrain)
    base_acc = evaluate(baseline, test_loader)
    print(f"\nBaseline accuracy: {base_acc:.2f}%  - benchmarking ...")
    base_row = make_row("baseline", "baseline", baseline, base_acc, benchmark_model(baseline))
    rows.append(base_row)
    save()
    log_row(base_row)

    # ------------------------------------------------------------- 2. unstructured
    for level in cfg.UNSTRUCTURED_LEVELS:
        pct = round(100 * level)
        print(f"\n[unstructured] global L1 pruning, {pct}% of weights")

        # (a) pruned only. Pruning is deterministic, so the same masks are rebuilt for the fine-tuned copy.
        m0 = make_permanent(apply_global_l1(copy.deepcopy(baseline), level))
        acc0 = evaluate(m0, test_loader)
        eff = benchmark_model(m0)           # tensor shapes never change, so one benchmark covers both stages
        rows.append(make_row("unstructured", "no_ft", m0, acc0, eff, base_row, pct))
        log_row(rows[-1])
        save()

        # (b) pruned + fine-tuned (masks stay active during fine-tuning, so zeros stay zero)
        m1 = apply_global_l1(copy.deepcopy(baseline), level)
        train_loader, _ = get_dataloaders()
        fine_tune(m1, train_loader, f"unstructured {pct}%")
        make_permanent(m1)
        acc1 = evaluate(m1, test_loader)
        rows.append(make_row("unstructured", "ft", m1, acc1, eff, base_row, pct))
        log_row(rows[-1])
        save()
        del m0, m1

    # ------------------------------------------------------------- 3. structured
    for level in cfg.STRUCTURED_LEVELS:
        pct = round(100 * level)
        if cfg.STRUCTURED_MATCH_PARAMS:
            print(f"\n[structured] searching the channel ratio that removes ~{pct}% of the parameters")
            pruned, ratio, reduction = prune_to_param_reduction(baseline, level)
            print(f"  channel ratio {ratio:.3f} -> {100 * reduction:.1f}% fewer parameters")
        else:
            print(f"\n[structured] pruning {pct}% of the channels in every layer")
            pruned, ratio = prune_channels(baseline, level)
        m0 = pruned.to(cfg.DEVICE)

        acc0 = evaluate(m0, test_loader)
        eff = benchmark_model(m0)
        rows.append(make_row("structured", "no_ft", m0, acc0, eff, base_row, pct, ratio))
        log_row(rows[-1])
        save()

        m1 = copy.deepcopy(m0)
        train_loader, _ = get_dataloaders()
        fine_tune(m1, train_loader, f"structured {pct}%")
        acc1 = evaluate(m1, test_loader)
        rows.append(make_row("structured", "ft", m1, acc1, eff, base_row, pct, ratio))
        log_row(rows[-1])
        save()
        del m0, m1

    # ------------------------------------------------------------- 4. sparse kernels + plots
    if not args.skip_sparse_demo:
        print()
        run_sparse_kernel_demo()
    plot_results.main(cfg.RESULTS_DIR)

    # ------------------------------------------------------------- summary
    df = pd.DataFrame(rows)
    show = ["method", "stage", "param_sparsity_pct", "accuracy", "size_mb", "size_gz_mb",
            "mac_reduction_x"] + [c for c in df.columns if c.startswith("spd_")]
    print("\n" + "=" * 70)
    print(df[show].to_string(index=False))
    print("=" * 70)
    print(f"Results: {results_csv}\nPlots:   {cfg.RESULTS_DIR}/plots/")


if __name__ == "__main__":
    main()
