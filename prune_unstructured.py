"""
Unstructured pruning: global L1 magnitude pruning with torch.nn.utils.prune.

Individual weights with the smallest absolute value (compared across ALL conv and
linear layers at once) are set to zero. Tensor shapes do NOT change, so a dense
GPU/CPU kernel still multiplies every zero.

How the masks work (important for fine-tuning):
  apply_global_l1()  -> every pruned layer gets  weight = weight_orig * weight_mask
  fine-tuning        -> weight_orig keeps training, the mask keeps zeros at zero
  make_permanent()   -> bakes the mask into a normal dense `weight` tensor
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prunable_params(model):
    """(module, 'weight') for every Conv2d and Linear layer."""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def apply_global_l1(model, amount):
    """Zero out `amount` (0..1) of all conv+linear weights, smallest magnitude first (global)."""
    prune.global_unstructured(prunable_params(model),
                              pruning_method=prune.L1Unstructured,
                              amount=amount)
    return model


def make_permanent(model):
    """Remove the re-parametrization so `weight` is an ordinary dense tensor (with zeros)."""
    for module, name in prunable_params(model):
        if hasattr(module, name + "_orig"):
            prune.remove(module, name)
    return model


@torch.no_grad()
def weight_sparsity(model):
    """Fraction of zeros among conv+linear weights (not counting BatchNorm or biases)."""
    zeros = total = 0
    for module, name in prunable_params(model):
        w = getattr(module, name)
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / max(total, 1)


@torch.no_grad()
def layer_sparsity_table(model):
    """[(layer_name, sparsity_fraction, n_weights)] - useful to see where global pruning cuts."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            rows.append((name, float((w == 0).sum().item()) / w.numel(), w.numel()))
    return rows
