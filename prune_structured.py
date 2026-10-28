"""
Structured pruning with Torch-Pruning: remove whole channels / filters.

The tensors PHYSICALLY shrink (fewer filters in a conv, fewer input channels in the next
layer), so the result is an ordinary smaller dense network. Torch-Pruning builds a
dependency graph, so layers that must keep the same number of channels (for example both
sides of a ResNet residual add) are pruned together.

Importance of a channel = L1 norm of its filter weights. The final classifier layer is
excluded, so the network still outputs 10 classes.
"""
import copy

import torch
import torch.nn as nn

try:
    import torch_pruning as tp
except ImportError as e:                      # pragma: no cover
    raise ImportError("Torch-Pruning is required: pip install torch-pruning") from e

import config as cfg


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _prune_once(model, ratio):
    """Return a pruned CPU copy of `model` with `ratio` of the channels removed in every layer."""
    m = copy.deepcopy(model).cpu().eval()
    example_inputs = torch.randn(1, *cfg.INPUT_SHAPE)
    importance = tp.importance.MagnitudeImportance(p=1)
    ignored = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]

    kwargs = dict(importance=importance, iterative_steps=1, ignored_layers=ignored,
                  global_pruning=False)
    try:
        pruner = tp.pruner.MagnitudePruner(m, example_inputs, pruning_ratio=ratio, **kwargs)
    except TypeError:                         # older Torch-Pruning used `ch_sparsity`
        pruner = tp.pruner.MagnitudePruner(m, example_inputs, ch_sparsity=ratio, **kwargs)
    pruner.step()
    return m


def prune_channels(model, ratio):
    """Prune a fixed fraction of channels per layer. Returns (pruned_model_on_cpu, ratio)."""
    return _prune_once(model, ratio), ratio


def prune_to_param_reduction(model, target, tol=None, max_iter=None):
    """Find the channel ratio whose pruned model has about `target` (0..1) fewer parameters.

    Parameters shrink roughly with the square of the channel ratio, so a plain "30% of
    channels" is NOT a 30% parameter cut. Bisection makes the structured levels directly
    comparable with unstructured sparsity.
    Returns (pruned_model_on_cpu, channel_ratio, achieved_param_reduction).
    """
    tol = cfg.MATCH_TOL if tol is None else tol
    max_iter = cfg.MATCH_MAX_ITER if max_iter is None else max_iter
    base = n_params(model)
    lo, hi = 0.0, 0.95
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        pruned = _prune_once(model, mid)
        reduction = 1.0 - n_params(pruned) / base
        err = abs(reduction - target)
        if best is None or err < best[3]:
            best = (pruned, mid, reduction, err)
        if err <= tol:
            break
        if reduction < target:
            lo = mid
        else:
            hi = mid
    pruned, ratio, reduction, _ = best
    return pruned, ratio, reduction
