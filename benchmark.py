"""
Real efficiency measurements (not just "how many weights are zero").

Per model:
  - parameters (total / non-zero), in-memory weight size, size of the saved file (raw and gzip)
  - dense MACs vs "non-zero" MACs (what you would need if zeros were free to skip)
  - feature-map memory (sum of conv/linear outputs for one batch)
  - latency in ms per batch (CPU and GPU), warm-up + many runs, torch.cuda.synchronize()
  - peak GPU memory during inference

Everything is measured in FP32 with torch.inference_mode(), so models are compared fairly.

run_sparse_kernel_demo() shows when sparse kernels DO help (and when they do not):
  - CPU: dense matmul vs CSR sparse matmul at different sparsity levels
  - GPU: NVIDIA 2:4 semi-structured sparsity (Ampere or newer, FP16)
"""
import copy
import gzip
import io
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as cfg

MB = 1024 ** 2


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- static metrics
def count_params(model):
    """(total parameters, non-zero parameters) over ALL parameters (conv, BN, fc, biases)."""
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def serialized_size_mb(model):
    """(raw MB, gzip MB) of the saved state_dict.

    Raw size: zeros are stored like any other number, so unstructured pruning does not shrink it.
    gzip size: zeros compress well, which is the only place where unstructured sparsity saves disk.
    """
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    raw = buf.tell()
    gz = len(gzip.compress(buf.getvalue(), compresslevel=6))
    return raw / MB, gz / MB


@torch.no_grad()
def profile_model(model, batch_size):
    """One CPU forward pass with hooks. Returns (dense MMACs per sample, non-zero MMACs, feature-map MB per batch)."""
    m = copy.deepcopy(model).cpu().eval()
    acc = {"dense": 0.0, "nonzero": 0.0, "act": 0.0}

    def hook(mod, inputs, out):
        w = mod.weight
        nz_fraction = float(torch.count_nonzero(w).item()) / w.numel()
        per_sample_outputs = out.numel() / out.shape[0]
        if isinstance(mod, nn.Conv2d):
            macs = per_sample_outputs * (mod.in_channels // mod.groups) \
                   * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            macs = float(mod.in_features * mod.out_features)
        acc["dense"] += macs
        acc["nonzero"] += macs * nz_fraction
        acc["act"] += out.numel() * out.element_size()

    handles = [mod.register_forward_hook(hook) for mod in m.modules()
               if isinstance(mod, (nn.Conv2d, nn.Linear))]
    m(torch.randn(batch_size, *cfg.INPUT_SHAPE))
    for h in handles:
        h.remove()
    return acc["dense"] / 1e6, acc["nonzero"] / 1e6, acc["act"] / MB


def static_metrics(model):
    """Everything that does not need a stopwatch."""
    total, nonzero = count_params(model)
    raw_mb, gz_mb = serialized_size_mb(model)
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / MB
    macs_dense, macs_nonzero, act_mb = profile_model(model, cfg.BENCH_BATCH_SIZES[-1])
    return {
        "total_params": total,
        "nonzero_params": nonzero,
        "param_mem_mb": round(weight_mem, 3),
        "size_mb": round(raw_mb, 3),
        "size_gz_mb": round(gz_mb, 3),
        "macs_dense_m": round(macs_dense, 2),
        "macs_nonzero_m": round(macs_nonzero, 2),
        "featuremap_mem_mb": round(act_mb, 2),
    }


# ----------------------------------------------------------------------------- timing
@torch.inference_mode()
def measure_latency(model, device, batch_size):
    """Mean and p95 latency in ms for one forward pass of `batch_size` images."""
    x = torch.randn(batch_size, *cfg.INPUT_SHAPE, device=device)
    for _ in range(cfg.BENCH_WARMUP[device]):          # warm-up: CUDA init, cuDNN autotune, caches
        model(x)
    _sync(device)
    times = []
    for _ in range(cfg.BENCH_ITERS[device]):
        _sync(device)
        t0 = time.perf_counter()
        model(x)
        _sync(device)                                  # wait for the GPU before stopping the clock
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return float(t.mean()), float(np.percentile(t, 95))


@torch.inference_mode()
def measure_gpu_peak_mb(model, batch_size):
    """Peak GPU memory (weights + input + activations) during one forward pass."""
    x = torch.randn(batch_size, *cfg.INPUT_SHAPE, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MB


def benchmark_model(model):
    """Latency (CPU and GPU) and GPU peak memory. Returns a flat dict of columns."""
    out = {}
    prev_threads = torch.get_num_threads()
    try:
        for device in cfg.BENCH_DEVICES:
            name = "gpu" if device == "cuda" else "cpu"
            if device == "cpu":
                torch.set_num_threads(cfg.CPU_THREADS)
            m = copy.deepcopy(model).to(device).eval()
            for bs in cfg.BENCH_BATCH_SIZES:
                mean_ms, p95_ms = measure_latency(m, device, bs)
                out[f"lat_{name}_bs{bs}_ms"] = round(mean_ms, 3)
                out[f"lat_{name}_bs{bs}_p95_ms"] = round(p95_ms, 3)
                if device == "cuda":
                    out[f"gpu_peak_mem_bs{bs}_mb"] = round(measure_gpu_peak_mb(m, bs), 1)
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(prev_threads)
    return out


# ----------------------------------------------------------------------------- sparse-kernel demo
def _time_ms(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


@torch.inference_mode()
def _cpu_csr_rows():
    """Dense matmul vs CSR sparse matmul on CPU. Unstructured zeros, random positions."""
    rows = []
    n, cols = cfg.SPARSE_DEMO_SIZE, cfg.SPARSE_DEMO_COLS
    torch.manual_seed(cfg.SEED)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.CPU_THREADS)
    try:
        x = torch.randn(n, cols)
        for s in cfg.SPARSE_DEMO_LEVELS:
            w = torch.randn(n, n) * (torch.rand(n, n) >= s)
            dense_ms = _time_ms(lambda: w @ x, "cpu", 2, 5)
            try:
                w_csr = w.to_sparse_csr()
                sparse_ms = _time_ms(lambda: w_csr @ x, "cpu", 2, 5)
            except Exception as e:                       # CSR matmul missing in this build
                print(f"  CSR matmul skipped at {s:.0%}: {e}")
                sparse_ms = float("nan")
            rows.append({"kind": "cpu_csr_unstructured", "sparsity_pct": 100 * s,
                         "dense_ms": round(dense_ms, 3), "sparse_ms": round(sparse_ms, 3),
                         "speedup": round(dense_ms / sparse_ms, 3) if sparse_ms == sparse_ms else float("nan")})
            print(f"  CPU CSR  sparsity {s:>5.0%}: dense {dense_ms:8.2f} ms | sparse {sparse_ms:8.2f} ms")
    finally:
        torch.set_num_threads(prev_threads)
    return rows


@torch.inference_mode()
def _gpu_semi_structured_rows():
    """Dense FP16 linear vs NVIDIA 2:4 semi-structured sparse linear (needs Ampere or newer)."""
    if not cfg.HAS_CUDA:
        return []
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        print("  2:4 sparsity skipped: needs a GPU with compute capability 8.0+ (Ampere or newer).")
        return []
    try:
        from torch.sparse import to_sparse_semi_structured
        n = max(cfg.SPARSE_DEMO_SIZE, 4096)
        torch.manual_seed(cfg.SEED)
        mask = torch.tensor([0, 0, 1, 1], dtype=torch.bool, device="cuda").tile((n, n // 4))
        w = (torch.randn(n, n, device="cuda", dtype=torch.float16) * mask)   # exactly 2 zeros per 4 weights
        x = torch.randn(n, n, device="cuda", dtype=torch.float16)

        dense = nn.Linear(n, n, bias=False).half().cuda()
        dense.weight = nn.Parameter(w)
        sparse = nn.Linear(n, n, bias=False).half().cuda()
        sparse.weight = nn.Parameter(to_sparse_semi_structured(w))

        ref, got = dense(x), sparse(x)
        rel_err = float((ref - got).abs().max() / ref.abs().max())
        dense_ms = _time_ms(lambda: dense(x), "cuda", 10, 50)
        sparse_ms = _time_ms(lambda: sparse(x), "cuda", 10, 50)
        print(f"  GPU 2:4  dense {dense_ms:.3f} ms | sparse {sparse_ms:.3f} ms | max rel. error {rel_err:.1e}")
        return [{"kind": "gpu_2to4_fp16", "sparsity_pct": 50.0, "dense_ms": round(dense_ms, 3),
                 "sparse_ms": round(sparse_ms, 3), "speedup": round(dense_ms / sparse_ms, 3)}]
    except Exception as e:                                # unsupported PyTorch build or GPU
        print(f"  2:4 sparsity skipped: {type(e).__name__}: {e}")
        return []


def run_sparse_kernel_demo():
    """Run both demos and save sparse_kernel_demo.csv."""
    print("Sparse-kernel demo (when do zeros actually help?)")
    rows = _cpu_csr_rows() + _gpu_semi_structured_rows()
    path = cfg.results_path("sparse_kernel_demo.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")
    return rows
