"""Regression checks without importing CUDA, datasets, or training dependencies."""
import ast
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def function(path, name, env):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


class LearningRateTests(unittest.TestCase):
    def setUp(self):
        env = dict(math=math, partial=partial)
        self.schedule = function(ROOT / "model/unet_training.py", "get_lr_scheduler", env)
        def adam(parameters, lr, **kwargs):
            return SimpleNamespace(param_groups=[dict(lr=lr)])
        self.build = function(ROOT / "train_medical.py", "get_optimizer_and_lr",
                              dict(optim=SimpleNamespace(Adam=adam), get_lr_scheduler=self.schedule))
        self.model = SimpleNamespace(parameters=lambda: [])

    def test_default_curve_unchanged(self):
        for epochs in (30, 50, 100):
            opt, actual = self.build(self.model, 32, epochs, .9, 1e-4)
            expected = self.schedule("cos", 1e-4, 1e-6, epochs)
            self.assertEqual(opt.param_groups[0]["lr"], 1e-4)
            for epoch in range(epochs):
                self.assertAlmostEqual(actual(epoch), expected(epoch), places=15)

    def test_custom_peak_and_floor(self):
        opt, curve = self.build(self.model, 32, 30, .9, 1e-4, lr=5e-5)
        self.assertEqual(opt.param_groups[0]["lr"], 5e-5)
        expected = self.schedule("cos", 5e-5, 5e-7, 30)
        for epoch in range(30):
            self.assertAlmostEqual(curve(epoch), expected(epoch), places=15)
        self.assertAlmostEqual(curve(29), 5e-7)

    def test_batch_does_not_override_requested_lr(self):
        for batch in (1, 8, 16, 32, 40):
            opt, _ = self.build(self.model, batch, 30, .9, 1e-4, lr=2e-4)
            self.assertEqual(opt.param_groups[0]["lr"], 2e-4)

    def test_invalid_lr_rejected(self):
        for lr in (0, -1e-4, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.build(self.model, 8, 30, .9, 1e-4, lr=lr)

    def test_training_passes_cli_lr(self):
        tree = ast.parse((ROOT / "train_medical.py").read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "get_optimizer_and_lr"]
        self.assertEqual(len(calls), 1)
        keyword = next(k for k in calls[0].keywords if k.arg == "lr")
        self.assertEqual(ast.unparse(keyword.value), "args.lr")


if __name__ == "__main__":
    unittest.main()
