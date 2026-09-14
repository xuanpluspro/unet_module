"""Validate all VOC lists, 512px shapes and raw 0/255 masks before training."""
import argparse
from utils.dataloader_medical import PlantVOCDataset, validate_splits, voc_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="VOCdevkit")
    args = p.parse_args()
    root = voc_root(args.data_path)
    splits = validate_splits(root)
    for split in splits:
        dataset = PlantVOCDataset(root, split)
        foreground = total = 0
        for _, target, _ in dataset:
            foreground += int(target.sum())
            total += target.numel()
        print(f"{split}: {len(dataset)} pairs, plant pixel fraction={foreground/total:.6f}")
    print("PASS: disjoint IDs, matched images/masks, 512x512, raw 0/255 labels.")


if __name__ == "__main__":
    main()
