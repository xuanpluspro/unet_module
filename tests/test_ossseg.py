import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import OSSSeg
from model.losses import SegmentationLoss
from utils.dataloader_medical import PlantVOCDataset, decode_mask, validate_splits
from utils.metrics import SegmentationMetrics

torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def test_label_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mask.png"
            Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(p)
            self.assertEqual(decode_mask(p).tolist(), [[0, 1]])
            Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).save(p)
            with self.assertRaises(ValueError):
                decode_mask(p)

    def test_split_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lists = root / "ImageSets/Segmentation"
            lists.mkdir(parents=True)
            (lists / "train.txt").write_text("same\n")
            (lists / "val.txt").write_text("same\n")
            with self.assertRaises(ValueError):
                validate_splits(root)

    def test_metrics(self):
        metric = SegmentationMetrics()
        target = torch.tensor([[[0, 0, 1, 1]]])
        pred = torch.tensor([[[0, 1, 0, 1]]])
        logits = torch.nn.functional.one_hot(pred, 2).permute(0, 3, 1, 2).float()
        metric.update(logits, target)
        values = metric.compute()
        self.assertAlmostEqual(values["Foreground IoU"], 1/3)
        self.assertAlmostEqual(values["Dice"], 0.5)
        # A class absent from GT but predicted incorrectly still contributes to mIoU.
        metric = SegmentationMetrics()
        metric.update(torch.tensor([[[[0.]], [[1.]]]]), torch.zeros(1, 1, 1, dtype=torch.long))
        self.assertEqual(metric.compute()["Mean IoU"], 0.)

    def test_model_512_backward_and_reload(self):
        model = OSSSeg()  # default ResNet50 and 256-channel fusion
        model.train()
        x = torch.randn(1, 3, 512, 512)
        y = torch.zeros(1, 512, 512, dtype=torch.long)
        y[:, 100:400, 150:350] = 1
        output = model(x, return_aux=True)
        self.assertEqual(output["logits"].shape, (1, 2, 512, 512))
        self.assertEqual(output["aux_logits"].shape, (1, 1, 128, 128))
        loss = SegmentationLoss()(output, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.foreground_head[-1].weight.grad.abs().sum().item(), 0)
        torch.optim.SGD(model.parameters(), lr=1e-4).step()
        model.zero_grad(set_to_none=True)
        del loss, output
        model.eval()
        with torch.inference_mode():
            expected = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pth"
            torch.save(model.state_dict(), path)
            del model
            other = OSSSeg().eval()
            other.load_state_dict(torch.load(path, weights_only=True), strict=True)
            with torch.inference_mode():
                torch.testing.assert_close(expected, other(x))


if __name__ == "__main__":
    unittest.main()
