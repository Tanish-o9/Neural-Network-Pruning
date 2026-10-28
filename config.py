"""
Central configuration for the pruning experiment.

Every other module does `import config as cfg` and reads values at call time,
so `apply_quick()` can safely change them for a fast smoke test.
"""
import os

import torch

# ----------------------------------------------------------------- reproducibility
SEED = 42

# ----------------------------------------------------------------- paths
DATA_DIR = "data"
RESULTS_DIR = "results"
BASELINE_CKPT = "baseline_resnet18.pt"      # saved inside RESULTS_DIR

# ----------------------------------------------------------------- device
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
USE_AMP = HAS_CUDA                           # mixed precision for TRAINING only (benchmarks run in FP32)

# ----------------------------------------------------------------- data
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)
BATCH_SIZE = 128
NUM_WORKERS = 2
TRAIN_SUBSET = 0                             # 0 = all 50k training images, N = random subset of N
TEST_SUBSET = 0                              # 0 = all 10k test images

# ----------------------------------------------------------------- baseline training (from scratch)
BASELINE_EPOCHS = 15
BASELINE_MAX_LR = 0.1                        # OneCycle peak LR
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# ----------------------------------------------------------------- fine-tuning
# The SAME budget is used for every pruned model, so the comparison is fair.
FT_EPOCHS = 3
FT_LR = 0.02                                 # cosine schedule, per-iteration

# ----------------------------------------------------------------- pruning levels
UNSTRUCTURED_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]   # fraction of conv+linear weights set to 0
STRUCTURED_LEVELS = [0.1, 0.3, 0.5, 0.7]          # see STRUCTURED_MATCH_PARAMS below

# True : choose the channel ratio so that the fraction of PARAMETERS removed matches the level
#        (apples-to-apples with unstructured sparsity).
# False: use the level directly as the per-layer channel pruning ratio
#        (a 30% channel ratio removes roughly 50% of the parameters, because a conv has
#        both input and output channels).
STRUCTURED_MATCH_PARAMS = True
MATCH_TOL = 0.01                             # accept if |achieved - target| <= 1 percentage point
MATCH_MAX_ITER = 12

# ----------------------------------------------------------------- benchmark
BENCH_DEVICES = ["cpu", "cuda"] if HAS_CUDA else ["cpu"]
BENCH_BATCH_SIZES = [1, 64]
BENCH_WARMUP = {"cpu": 5, "cuda": 20}
BENCH_ITERS = {"cpu": 30, "cuda": 100}
CPU_THREADS = min(4, os.cpu_count() or 1)    # fixed, so CPU numbers are comparable between runs

# ----------------------------------------------------------------- sparse-kernel demo
SPARSE_DEMO_SIZE = 4096                      # weight matrix is SIZE x SIZE
SPARSE_DEMO_COLS = 256                       # activations are SIZE x COLS
SPARSE_DEMO_LEVELS = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]


def results_path(*parts):
    """Path inside RESULTS_DIR (the folder is created if needed)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, *parts)


def apply_quick():
    """Tiny settings for a smoke test (a few minutes). Results go to a separate folder."""
    global TRAIN_SUBSET, TEST_SUBSET, BASELINE_EPOCHS, FT_EPOCHS, RESULTS_DIR
    global BENCH_WARMUP, BENCH_ITERS, SPARSE_DEMO_SIZE
    TRAIN_SUBSET = 5000
    TEST_SUBSET = 1000
    BASELINE_EPOCHS = 2
    FT_EPOCHS = 1
    RESULTS_DIR = "results_quick"
    BENCH_WARMUP = {"cpu": 2, "cuda": 5}
    BENCH_ITERS = {"cpu": 5, "cuda": 20}
    SPARSE_DEMO_SIZE = 1024
