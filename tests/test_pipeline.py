"""Synthetic VOC integration: train, resume, validation, test, prediction export."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import train_medical
import val
import predict
from utils.runtime import read_checkpoint

torch.set_num_threads(2)


class Pipeline(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "VOC2012"
            for folder in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
                (root / folder).mkdir(parents=True)
            for split in ("train", "val", "test"):
                name = f"{split}_sample"
                yy, xx = np.indices((512, 512))
                mask = (((xx - 256)**2 + (yy - 256)**2) < 120**2).astype(np.uint8) * 255
                image = np.full((512, 512, 3), 50, np.uint8)
                image[mask == 255] = [20, 190, 30]
                Image.fromarray(image).save(root / "JPEGImages" / f"{name}.jpg")
                Image.fromarray(mask).save(root / "SegmentationClass" / f"{name}.png")
                (root / "ImageSets/Segmentation" / f"{split}.txt").write_text(name + "\n")
            args = argparse.Namespace(data_path=str(root), output=str(Path(d)/"uninterrupted"),
                backbone_weights=None, from_scratch=True, resume=None, num_classes=1,
                width=16, freeze_backbone_bn=True, epochs=2, batch_size=1, workers=0,
                lr=1e-4, weight_decay=1e-4, dice_weight=1., aux_weight=.2,
                seed=11, device="cpu", amp=False)
            # Plotting is exercised separately by normal training; keep this test fast.
            with patch("utils.plot_results.plot_training_curves"):
                train_medical.train(args)
            reference = read_checkpoint(Path(args.output)/"last.pth")
            args.output = str(Path(d)/"resumed")
            original_save = train_medical.save_checkpoint
            def stop_after_first_epoch(checkpoint, path):
                original_save(checkpoint, path)
                if path.name == "last.pth" and checkpoint["epoch"] == 0:
                    raise InterruptedError("Synthetic interruption after durable checkpoint")
            with patch("train_medical.save_checkpoint", side_effect=stop_after_first_epoch):
                with self.assertRaises(InterruptedError):
                    train_medical.train(args)
            args.from_scratch = False
            args.resume = str(Path(args.output)/"last.pth")
            with patch("utils.plot_results.plot_training_curves"):
                train_medical.train(args)
            resumed = read_checkpoint(Path(args.output)/"last.pth")
            for key in reference["model"]:
                torch.testing.assert_close(reference["model"][key], resumed["model"][key], rtol=0, atol=0)
            checkpoint = str(Path(args.output)/"best.pth")
            metrics = Path(d)/"metrics.json"
            with patch("sys.argv", ["val.py", "--data-path", str(root), "--weights", checkpoint,
                                    "--split", "test", "--device", "cpu", "--output", str(metrics)]):
                val.main()
            self.assertEqual(json.loads(metrics.read_text())["split"], "test")
            destination = Path(d)/"prediction"
            with patch("sys.argv", ["predict.py", "--data-path", str(root/"JPEGImages/test_sample.jpg"),
                                    "--weights", checkpoint, "--device", "cpu", "--output", str(destination)]):
                predict.main()
            result = np.array(Image.open(destination/"test_sample_mask.png"))
            self.assertEqual(result.shape, (512,512))
            self.assertTrue(set(np.unique(result)).issubset({0,255}))
            probability = np.load(destination/"test_sample_prob.npy")
            self.assertTrue(np.isfinite(probability).all())
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())


if __name__ == "__main__":
    unittest.main()
