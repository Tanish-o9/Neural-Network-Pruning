# Neural-Network-Pruning
Unstructured vs structured pruning of ResNet18 on CIFAR-10. Measures real CPU/GPU latency, memory and file size, not just % of zero weights, and shows why sparse weights alone don’t speed up inference. PyTorch, Torch-Pruning.
# Neural Network Pruning: Quality vs Efficiency

Can we remove weights or whole structures from a trained network to make it smaller or cheaper to run, without losing much quality? This project answers that for ResNet18 on CIFAR-10 by comparing **unstructured pruning** (zeroing individual weights) with **structured pruning** (removing whole channels), and by measuring what actually matters: **real latency, memory and file size**, not just the percentage of zero weights.

## The question in one table

> Fill in after running the experiment (`results/results.csv`). Replace every `X`.
> Setup: GPU `X`, CPU `X` (`X` threads), PyTorch `X`, Torch-Pruning `X`.

| At about 50% of parameters removed | Baseline | Unstructured (fine-tuned) | Structured (fine-tuned) |
|---|---|---|---|
| Test accuracy | X% | X% | X% |
| Latency, CPU, batch 64 | X ms | X ms (X times) | X ms (X times) |
| Latency, GPU, batch 64 | X ms | X ms (X times) | X ms (X times) |
| Saved file size | X MB | X MB | X MB |
| Saved file size, gzip | X MB | X MB | X MB |
| Non-zero MACs (ideal) | X M | X M | X M |

## Background

**Pruning** removes parameters that contribute little to the output. Weights close to zero are the usual candidates.

| | Unstructured | Structured |
|---|---|---|
| What is removed | Individual weights (set to 0) | Whole filters / channels |
| Tensor shapes | Unchanged | Physically smaller |
| Result | A dense matrix full of zeros | A smaller dense network |
| Needs special software or hardware to be faster? | **Yes** (sparse kernels, or NVIDIA 2:4 sparsity) | **No**, any dense hardware benefits |
| Tool used here | `torch.nn.utils.prune` | Torch-Pruning |

**Important caveat, and the reason for this project:** sparse weights do **not** automatically make inference faster. A normal convolution or matrix multiplication still reads and multiplies every zero. Real gains depend on the structure of the sparsity and on hardware and software support. This project measures that honestly.

## What is measured

| Metric | How |
|---|---|
| Accuracy | Top-1 on the CIFAR-10 test set, FP32 |
| Parameters | Total and non-zero, over all parameters (conv, BatchNorm, classifier) |
| Model size | Size of the saved `state_dict` file (raw) and its gzip size |
| Memory | Weight memory, feature-map memory per batch, and peak GPU memory during inference |
| MACs | Dense MACs vs "non-zero MACs" (the ideal if zeros cost nothing) |
| Latency | ms per batch, CPU and GPU, batch sizes 1 and 64, mean and p95 |
| Speedup | Measured speedup vs the baseline, next to the ideal speedup from non-zero MACs |
| Sparse kernels | Micro-benchmark: dense vs CSR matmul on CPU, and NVIDIA 2:4 sparsity on Ampere+ GPUs |

Latency uses warm-up runs, many repetitions, `torch.inference_mode()` and `torch.cuda.synchronize()` around every timed call. CPU benchmarks use a fixed thread count.

## Method

```
Baseline ResNet18 (trained once, cached)
   |
   +--> Unstructured: global L1 magnitude pruning at 10 / 30 / 50 / 70 / 90 %
   |       -> without fine-tuning   -> with fine-tuning
   |
   +--> Structured: Torch-Pruning, L1 channel importance, at 10 / 30 / 50 / 70 %
           -> without fine-tuning   -> with fine-tuning
   |
   v
Same metrics for every model -> results.csv -> plots
```

Design decisions:

- **Model and data:** ResNet18 adapted to 32x32 images (3x3 stem, no max-pool, about 11.2M parameters), trained from scratch on CIFAR-10 with standard augmentation (random crop, horizontal flip).
- **Unstructured:** `prune.global_unstructured` with `L1Unstructured` over all conv and linear weights at once. During fine-tuning the masks stay active, so zeros stay zero. They are then baked into the weights with `prune.remove`.
- **Structured:** Torch-Pruning builds a dependency graph, so layers tied together (for example both sides of a residual add) are pruned together. The classifier layer is excluded, so the network still outputs 10 classes. The network is rebuilt with smaller tensors.
- **A fair x-axis:** a "30% channel ratio" removes roughly 50% of the parameters, because a conv shrinks in both input and output channels. By default (`STRUCTURED_MATCH_PARAMS = True`) the channel ratio is searched so that the **percentage of parameters removed matches** the unstructured level. Both methods then share one x-axis: parameters zeroed or removed, as a % of the baseline. Use `--match-channels` for the raw channel ratio instead. The achieved ratio is saved in `results.csv`.
- **Same fine-tuning budget for every run:** identical epochs, learning rate schedule, seed and data order.
- **Latency is measured once per pruned structure** and reused for the fine-tuned model. Fine-tuning changes weight values, not tensor shapes or (with masks) the zero pattern.
- **No tuning on the test set:** epochs and learning rates are fixed up front. Test accuracy is only logged.
- **Sparsity definition:** `param_sparsity_pct = 100 * (1 - non-zero parameters / baseline total parameters)`, over all parameters, so it is the same measure for both methods.

## Project structure

```
neural-network-pruning/
├── config.py               # all hyperparameters
├── data.py                 # CIFAR-10 dataloaders
├── train.py                # baseline training and fine-tuning
├── prune_unstructured.py   # global L1 magnitude pruning (torch.nn.utils.prune)
├── prune_structured.py     # channel/filter pruning (Torch-Pruning)
├── benchmark.py            # latency, memory, size, MACs, sparse-kernel demo
├── plot_results.py         # all plots
├── main.py                 # runs the full experiment end to end
├── requirements.txt
└── results/                # created by main.py
    ├── results.csv
    ├── sparse_kernel_demo.csv
    ├── baseline_resnet18.pt
    └── plots/
```

## Setup

```bash
git clone <your-repo-url>
cd neural-network-pruning
pip install -r requirements.txt
```

Runs on a free Colab GPU or a normal laptop. Without a GPU the CPU benchmarks still work, but training is slow: use `--quick` first, and consider `--train-subset 10000`.

## Usage

```bash
# 1. Smoke test (a few minutes, tiny data, writes to results_quick/)
python main.py --quick

# 2. Full experiment (writes to results/)
python main.py

# Redraw plots from an existing results.csv
python main.py --plots-only
```

| Flag | Meaning |
|---|---|
| `--quick` | Tiny run to check that everything works |
| `--epochs N` | Baseline training epochs (default 15) |
| `--ft-epochs N` | Fine-tuning epochs per pruned model (default 3) |
| `--train-subset N` | Use only N training images |
| `--match-channels` | Structured levels are channel ratios instead of matched parameter reduction |
| `--force-retrain` | Ignore the cached baseline checkpoint |
| `--skip-sparse-demo` | Skip the sparse-kernel micro-benchmark |

`results.csv` is rewritten by every full run. The baseline is cached, so a re-run reuses it. All settings (levels, epochs, batch sizes, benchmark iterations) are in `config.py`.

Expected run time on a free Colab T4 with the defaults is roughly 30 to 60 minutes (an estimate, not a measurement). Most of it is fine-tuning 9 pruned models and the CPU benchmarks.

## Results

> Fill in from `results/results.csv` after running. Keep the numbers honest, including bad ones.

Plots are saved in `results/plots/`:

| Plot | File |
|---|---|
| Accuracy vs sparsity (before and after fine-tuning) | `1_accuracy_vs_sparsity.png` |
| Latency vs sparsity (CPU and GPU, batch 1 and 64) | `2_latency_vs_sparsity.png` |
| Accuracy vs latency (the trade-off) | `3_accuracy_vs_latency.png` |
| Model size vs sparsity | `4_size_vs_sparsity.png` |
| Measured vs ideal speedup | `5_measured_vs_ideal_speedup.png` |
| When do sparse kernels pay off? | `6_sparse_kernel_demo.png` |

### Unstructured (global L1 magnitude pruning)

| Sparsity | Acc. before FT | Acc. after FT | Size (MB) | Size gzip (MB) | CPU ms (bs 64) | GPU ms (bs 64) | Speedup (GPU) |
|---|---|---|---|---|---|---|---|
| 0% (baseline) | X | X | X | X | X | X | 1.00x |
| 10% | X | X | X | X | X | X | X |
| 30% | X | X | X | X | X | X | X |
| 50% | X | X | X | X | X | X | X |
| 70% | X | X | X | X | X | X | X |
| 90% | X | X | X | X | X | X | X |

### Structured (Torch-Pruning channel pruning)

| Params removed | Channel ratio | Acc. before FT | Acc. after FT | Size (MB) | CPU ms (bs 64) | GPU ms (bs 64) | Speedup (GPU) |
|---|---|---|---|---|---|---|---|
| 10% | X | X | X | X | X | X | X |
| 30% | X | X | X | X | X | X | X |
| 50% | X | X | X | X | X | X | X |
| 70% | X | X | X | X | X | X | X |

### Sparse-kernel demo

| Kernel | Sparsity where it beats dense | Speedup at best case |
|---|---|---|
| CPU CSR matmul (unstructured) | X% | X times |
| NVIDIA 2:4 (FP16, Ampere+) | 50% (fixed) | X times (or "not supported on this GPU") |

## Conclusion

> This section explains the mechanisms, which hold in general, and the pattern the experiment is designed to show. It was written before the final run. **Check each claim against your own `results.csv` and plots, and edit it where your numbers differ.**

**In simple words**

- **Unstructured pruning keeps accuracy well but gives little real speedup.** You can zero out a large share of the weights and, after a short fine-tuning, accuracy stays close to the baseline, because the network has many redundant weights. But the model is not faster, not smaller in memory, and its saved file is the same size. Only the gzip size shrinks, because long runs of zeros compress well.
- **Structured pruning gives real speedup but costs more accuracy.** Removing whole channels makes the actual tensors smaller, so the network does fewer operations, moves less data and holds smaller feature maps. It gets faster on ordinary hardware with no special software. The price is a bigger accuracy drop at the same percentage of parameters, so fine-tuning matters much more.

**Why: zeros are not free to skip**

1. **Dense hardware.** CPUs (SIMD) and GPUs (tensor cores) are built to multiply regular blocks of numbers. A zero weight is multiplied exactly like any other. To skip it, the hardware would need a check for every element, and a check costs more than just doing the multiplication. So an unstructured-pruned convolution does the same work, in the same time, as the original. The plot "measured vs ideal speedup" shows this: the ideal line (non-zero MACs only) climbs, the measured line stays at 1x.
2. **Sparse kernels.** Special formats such as CSR store only the non-zero values, plus an index for each one. That adds memory and makes reads irregular, so they only win when the matrix is very sparse. The sparse-kernel demo shows where the crossover is on your machine. NVIDIA 2:4 sparsity is hardware-supported, but it needs a fixed pattern (2 zeros in every 4 weights, so exactly 50%), an Ampere or newer GPU and FP16 or INT8, and it is mostly useful for linear layers. Even then the gain is at most about 2x on the matmul.
3. **Memory layout.** A dense tensor is one contiguous block, which is what caches, prefetchers and vector loads like. Random zeros do not change the block, so nothing gets smaller. Structured pruning does change it: a convolution with half the filters is simply a smaller dense tensor, and the next layer has half the input channels. That also shrinks the feature maps, which cuts memory traffic and activation memory.

**Why structured pruning loses more accuracy**

- It is coarser. Unstructured pruning can keep the few important weights of a filter and drop the rest. Structured pruning must drop a whole filter, including anything useful it computed.
- In ResNet, channels are tied together by residual connections, so the pruner removes coupled channels together and has less freedom to choose.
- The network was trained with all its channels, so right after pruning it is further from a good solution. The "before fine-tuning" curve usually collapses at high levels, and fine-tuning recovers a large part of it.

**Caveats on speed**

- At batch size 1 on a GPU, latency is mostly kernel-launch overhead, so even structured pruning may show little speedup there. Compare the batch 1 and batch 64 panels.
- Speedup is usually smaller than the parameter reduction suggests, because BatchNorm, ReLU, memory transfers and other fixed costs do not shrink in proportion.

**What to take away**

| | Unstructured | Structured |
|---|---|---|
| Accuracy at the same % removed | Better | Worse |
| Real speedup on normal hardware | Little or none | Yes |
| Saved file size | Same (gzip: smaller) | Smaller |
| Needs special kernels or hardware | Yes | No |
| Best use | Compression for storage or transfer, or with sparse hardware | Deployment on standard CPUs and GPUs |

Sparsity percentage is a measure of the model, not of the speed. Always measure latency.

## Limitations

- One architecture and dataset (ResNet18, CIFAR-10), one seed. Differences of a few tenths of a percent in accuracy are within noise. Repeat with several seeds to report error bars.
- Only magnitude (L1) importance is used. Other criteria (gradient-based, Taylor, group importance) may prune structures better.
- Pruning is one-shot, followed by a short fixed fine-tuning budget. Iterative pruning with longer fine-tuning usually recovers more accuracy at high sparsity.
- Latency depends on hardware, drivers, batch size and thread count. Numbers are only comparable within one machine. Laptop GPUs and CPUs throttle with heat, so repeat runs if results look noisy.
- Latency and memory are measured in FP32 with PyTorch eager mode. Compilers and runtimes (`torch.compile`, TensorRT, ONNX Runtime) may change the picture.
- The sparse-kernel demo uses large random matrices, not the pruned ResNet layers. It shows kernel behavior, not the speed of the pruned network.
- The optional second model (DistilBERT on SST-2) is not included in this version.

## Future work

- Repeat on several seeds and report mean and standard deviation.
- Iterative pruning schedules and longer fine-tuning.
- Structured pruning of transformer attention heads and feed-forward neurons (DistilBERT on SST-2).
- 2:4 semi-structured pruning of the ResNet's linear layers with real inference through the sparse kernels.
- Combine pruning with quantization (INT8) and export through ONNX Runtime or TensorRT.

## Tech stack

Python, PyTorch, `torch.nn.utils.prune`, Torch-Pruning, torchvision, pandas, matplotlib, tqdm.

## References

- Han et al., *Learning both Weights and Connections for Efficient Neural Networks* (2015)
- Li et al., *Pruning Filters for Efficient ConvNets* (2017)
- Frankle and Carbin, *The Lottery Ticket Hypothesis* (2019)
- Fang et al., *DepGraph: Towards Any Structural Pruning* (2023), the method behind Torch-Pruning
- Mishra et al., *Accelerating Sparse Deep Neural Networks* (2021), NVIDIA 2:4 sparsity
- Torch-Pruning: https://github.com/VainF/Torch-Pruning

## License

MIT
