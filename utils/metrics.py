"""Dataset-level binary confusion matrix; class 1 is plant."""
import torch


class SegmentationMetrics:
    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.int64)

    def update(self, logits, target):
        prediction = logits.argmax(1)
        if target.min() < 0 or target.max() > 1:
            raise ValueError("Expected binary class IDs")
        bins = (2 * target + prediction).detach().reshape(-1).cpu()
        self.cm += torch.bincount(bins, minlength=4).reshape(2, 2)

    def compute(self):
        cm = self.cm.double()
        diag = cm.diag()
        counts = cm.sum(1)
        union = cm.sum(0) + counts - diag
        present = union > 0
        iou = diag / union.clamp_min(1)
        accuracy = diag / counts.clamp_min(1)
        tn, fp, fn, tp = [cm[i, j].item() for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        def ratio(a, b):
            return a / b if b else None
        return {
            "Foreground IoU": ratio(tp, tp + fp + fn),
            "Background IoU": ratio(tn, tn + fp + fn),
            "Mean IoU": iou[present].mean().item() if present.any() else None,
            "Dice": ratio(2 * tp, 2 * tp + fp + fn),
            "Precision": ratio(tp, tp + fp),
            "Recall": ratio(tp, tp + fn),
            "Pixel Accuracy": ratio(tp + tn, cm.sum().item()),
            "Mean Accuracy": accuracy[counts > 0].mean().item() if (counts > 0).any() else None,
            "Frequency Weighted IoU": ((counts * iou).sum() / cm.sum()).item() if cm.sum() else None,
            "confusion_matrix": self.cm.tolist(),
        }
