"""Evaluate a fixed OSSSeg checkpoint; --split test for final reporting."""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from utils.dataloader_medical import PlantVOCDataset, voc_root, validate_splits
from utils.runtime import load_model, resolve_device
from utils.train_and_eval import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="VOCdevkit")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", help="JSON output; defaults to checkpoint directory/SPLIT_metrics.json")
    args = p.parse_args()
    root = voc_root(args.data_path)
    splits = validate_splits(root)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.weights, device)
    # Test IDs must already have been recorded at training time for final evaluation.
    if splits != checkpoint["splits"]:
        raise ValueError("Split IDs changed since training; use the same train/val/test lists")
    if args.split not in splits:
        raise FileNotFoundError(f"{args.split}.txt was not provided before training")
    loader = DataLoader(PlantVOCDataset(root, args.split), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    metrics = evaluate(model, loader, device)
    metrics.update(split=args.split, checkpoint=args.weights, epoch=checkpoint["epoch"] + 1)
    out = Path(args.output) if args.output else Path(args.weights).parent / f"{args.split}_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
