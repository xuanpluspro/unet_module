"""Generic dense segmentation loop with an explicit auxiliary-output contract."""
from contextlib import nullcontext
import torch
from .metrics import SegmentationMetrics


def train_one_epoch(model, optimizer, loader, device, criterion, scaler):
    model.train()
    total, count = 0., 0
    for step, (images, targets, _) in enumerate(loader, 1):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
        with context:
            output = model(images, return_aux=True)
            loss = criterion(output, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; check data and AMP settings")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.shape[0]
        count += images.shape[0]
        if step == 1 or step % 20 == 0:
            print(f"  batch {step}/{len(loader)} loss={loss.item():.5f}", flush=True)
    return total / count


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    metrics = SegmentationMetrics()
    total, count = 0., 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images, return_aux=True) if criterion is not None else model(images)
        logits = output["logits"] if isinstance(output, dict) else output
        metrics.update(logits, targets)
        if criterion is not None:
            total += criterion(output, targets).item() * images.shape[0]
        count += images.shape[0]
    result = metrics.compute()
    if criterion is not None:
        result["Loss"] = total / count
    return result
