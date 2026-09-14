"""Train OSSSeg on 512px VOC-layout binary plant masks; select by validation plant IoU."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import OSSSeg
from model.losses import SegmentationLoss
from utils.dataloader_medical import PlantVOCDataset, voc_root, validate_splits
from utils.train_and_eval import train_one_epoch, evaluate
from utils.runtime import (FORMAT, seed_everything, seed_worker, resolve_device,
                           save_checkpoint, read_checkpoint, rng_state, restore_rng)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="VOCdevkit")
    p.add_argument("--output", default="run/train/ossseg")
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument("--backbone-weights", help="Local torchvision ResNet-50 ImageNet weights")
    init.add_argument("--from-scratch", action="store_true", help="Explicitly train without pretraining")
    init.add_argument("--resume", help="Resume this project's last.pth with optimizer/RNG state")
    p.add_argument("--num-classes", type=int, default=1, help="Foreground classes excluding background; must be 1")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--freeze-backbone-bn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dice-weight", type=float, default=1.)
    p.add_argument("--aux-weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def train(args):
    if args.num_classes != 1 or args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Require 1 foreground class, positive epochs/batch-size, nonnegative workers")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    root = voc_root(args.data_path)
    splits = validate_splits(root)
    model_config = dict(num_classes=2, width=args.width, freeze_backbone_bn=args.freeze_backbone_bn)
    training_config = {key: getattr(args, key) for key in (
        "epochs", "batch_size", "workers", "lr", "weight_decay", "dice_weight",
        "aux_weight", "seed", "amp")}
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new --output or --resume: {output}")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("--output must be the run directory containing the resume checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(PlantVOCDataset(root, "train", augmentation=True),
                              shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(PlantVOCDataset(root, "val", augmentation=False),
                            shuffle=False, **loader_kwargs)
    model = OSSSeg(**model_config)
    initialization = args.backbone_weights or "random"
    if args.backbone_weights:
        model.load_backbone(args.backbone_weights)
    if args.from_scratch:
        print("Explicit random initialization; no weights will be downloaded.", flush=True)
    model.to(device)
    criterion = SegmentationLoss(args.dice_weight, args.aux_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start, best, history = 0, -1., []
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        if (checkpoint["model_config"] != model_config
                or checkpoint["training_config"] != training_config
                or checkpoint["splits"] != splits):
            raise ValueError("Resume requires identical model/training settings and split IDs, including total epochs")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"], generator)
        start = checkpoint["epoch"] + 1
        best = checkpoint["best_foreground_iou"]
        history = checkpoint["history"]
        initialization = checkpoint["initialization"]
    config = dict(model=model_config, training=training_config, initialization=initialization,
                  data_path=str(root.resolve()), raw_labels={"background": 0, "plant": 255})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"device={device}; train={len(train_loader.dataset)} val={len(val_loader.dataset)}; params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(start, args.epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, criterion, scaler)
        metrics = evaluate(model, val_loader, device, criterion)
        score = metrics["Foreground IoU"]
        if score is None:
            raise ValueError("Undefined validation plant IoU; no plant pixels in labels or predictions")
        improved = score > best
        best = max(best, score)
        history.append(dict(epoch=epoch + 1, train_loss=loss,
                            lr=optimizer.param_groups[0]["lr"], **metrics))
        scheduler.step()
        checkpoint = dict(
            format=FORMAT, model_config=model_config, training_config=training_config,
            initialization=initialization, splits=splits, epoch=epoch, best_foreground_iou=best,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            rng=rng_state(generator), history=history)
        if improved:
            save_checkpoint(checkpoint, output / "best.pth")
        save_checkpoint(checkpoint, output / "last.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(history[-1], allow_nan=False), flush=True)
    if history:
        from utils.plot_results import plot_training_curves
        plot_training_curves([h["train_loss"] for h in history],
                             [h["Loss"] for h in history], history, str(output))


if __name__ == "__main__":
    train(parse_args())
