"""VOC2012 directory layout, RGB images and binary 0/255 PNG masks."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_SIZE = 512
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def voc_root(path):
    root = Path(path)
    if (root / "VOC2012").is_dir():
        root = root / "VOC2012"
    if not (root / "JPEGImages").is_dir() or not (root / "SegmentationClass").is_dir():
        raise FileNotFoundError(f"Expected VOC2012/JPEGImages and SegmentationClass under {path}")
    return root


def read_ids(root, split):
    if split not in ("train", "val", "test"):
        raise ValueError("Use independent train/val/test lists, not trainval")
    path = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    ids = [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate split IDs: {path}")
    if any("/" in x or "\\" in x or len(x.split()) != 1 or x in (".", "..") for x in ids):
        raise ValueError(f"One sample stem per line required: {path}")
    return ids


def validate_splits(root):
    """Require train/val; check test disjointness when provided, without reading test pixels."""
    result = {split: read_ids(root, split) for split in ("train", "val")}
    if (root / "ImageSets/Segmentation/test.txt").exists():
        result["test"] = read_ids(root, "test")
    keys = list(result)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap = set(result[a]) & set(result[b])
            if overlap:
                raise ValueError(f"Split leakage {a}/{b}: {sorted(overlap)[:5]}")
    return result


def index_images(root):
    index = {}
    for path in sorted((root / "JPEGImages").iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in index:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            index[path.stem] = path
    return index


def image_tensor(image):
    x = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32)
                         .transpose(2, 0, 1).copy()) / 255.
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def decode_mask(path):
    with Image.open(path) as image:
        mask = np.array(image)  # preserve indices of palette PNG, not displayed colors
    if mask.ndim == 3 and mask.shape[2] in (3, 4):
        if not (np.array_equal(mask[..., 0], mask[..., 1])
                and np.array_equal(mask[..., 0], mask[..., 2])):
            raise ValueError(f"Colored mask requires explicit mapping: {path}")
        mask = mask[..., 0]
    if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError(f"Mask must contain only 0=background / 255=plant: {path}")
    return (mask == 255).astype(np.uint8)


class PlantVOCDataset(Dataset):
    def __init__(self, data_path, split="train", augmentation=False):
        self.root = voc_root(data_path)
        self.ids = read_ids(self.root, split)
        self.augmentation = augmentation
        images = index_images(self.root)
        self.pairs = []
        for name in self.ids:
            mask = self.root / "SegmentationClass" / f"{name}.png"
            if name not in images or not mask.is_file():
                raise FileNotFoundError(f"Missing image or mask for {name}")
            self.pairs.append((images[name], mask))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as file:
            image = file.convert("RGB")
        mask = decode_mask(mask_path)
        if image.size != (IMAGE_SIZE, IMAGE_SIZE) or mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 512x512 image AND mask: {image_path}, {mask_path}")
        label = Image.fromarray(mask)
        if self.augmentation:
            if random.random() < 0.5:
                image, label = ImageOps.mirror(image), ImageOps.mirror(label)
            if random.random() < 0.5:
                image, label = ImageOps.flip(image), ImageOps.flip(label)
            turns = random.randrange(4)
            if turns:
                method = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180,
                          3: Image.Transpose.ROTATE_270}[turns]
                image, label = image.transpose(method), label.transpose(method)
        return image_tensor(image), torch.from_numpy(np.array(label, dtype=np.int64)), self.ids[index]
