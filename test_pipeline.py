"""
Automated CI Unit Tests & Smoke Verification for Neural Network Pruning Pipeline.
"""
import copy
import os
import sys
import unittest

import torch
import torchvision

import benchmark
import config as cfg
import prune_structured
import prune_unstructured
import train


class TestPruningPipeline(unittest.TestCase):
    def setUp(self):
        train.seed_everything(42)
        self.model = train.build_resnet18(num_classes=10)

    def test_model_architecture(self):
        """Verify CIFAR-10 adapted ResNet-18 architecture."""
        total, nonzero = benchmark.count_params(self.model)
        self.assertGreater(total, 11_000_000)
        self.assertAlmostEqual(nonzero / total, 1.0, places=2)
        # Test forward pass with dummy 32x32 tensor
        dummy_in = torch.randn(2, 3, 32, 32)
        out = self.model(dummy_in)
        self.assertEqual(out.shape, (2, 10))

    def test_unstructured_pruning(self):
        """Verify global L1 unstructured weight masking."""
        m = copy.deepcopy(self.model)
        m = prune_unstructured.apply_global_l1(m, 0.50)
        sparsity = prune_unstructured.weight_sparsity(m)
        self.assertAlmostEqual(sparsity, 0.50, delta=0.03)

        # Make permanent
        prune_unstructured.make_permanent(m)
        total, nonzero = benchmark.count_params(m)
        achieved = 1.0 - (nonzero / total)
        self.assertAlmostEqual(achieved, 0.50, delta=0.05)

    def test_structured_pruning(self):
        """Verify Torch-Pruning channel removal."""
        m = copy.deepcopy(self.model)
        pruned_m, ratio = prune_structured.prune_channels(m, 0.30)
        base_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_m.parameters())
        # Physically smaller
        self.assertLess(pruned_params, base_params)
        # Forward pass still produces 10 logits
        dummy_in = torch.randn(2, 3, 32, 32)
        out = pruned_m(dummy_in)
        self.assertEqual(out.shape, (2, 10))

    def test_benchmark_metrics(self):
        """Verify static metric calculations."""
        metrics = benchmark.static_metrics(self.model)
        self.assertIn("total_params", metrics)
        self.assertIn("size_mb", metrics)
        self.assertIn("macs_dense_m", metrics)
        self.assertGreater(metrics["size_mb"], 40.0)

    def test_quick_config(self):
        """Verify smoke-test config application."""
        cfg.apply_quick()
        self.assertEqual(cfg.BASELINE_EPOCHS, 2)
        self.assertEqual(cfg.RESULTS_DIR, "results_quick")


if __name__ == "__main__":
    unittest.main()
