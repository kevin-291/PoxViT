from __future__ import annotations

import random
import shutil
from pathlib import Path

from tqdm import tqdm

SPLITS = ("train", "val", "test")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def has_split_structure(data_root: str | Path) -> bool:
    root = Path(data_root)
    return all((root / split).is_dir() for split in SPLITS)


def _class_dirs(base_dir: Path) -> list[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name not in SPLITS])


def split_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Path:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    src_root = Path(dataset_dir)
    out_root = Path(output_dir)
    random.seed(seed)

    classes = _class_dirs(src_root)

    for split in SPLITS:
        for cls in classes:
            (out_root / split / cls.name).mkdir(parents=True, exist_ok=True)

    for cls in tqdm(classes, desc="Classes", unit="class"):
        images = [p for p in cls.iterdir() if _is_image_file(p)]
        random.shuffle(images)

        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)

        for i, img_path in enumerate(images):
            if i < train_end:
                split = "train"
            elif i < val_end:
                split = "val"
            else:
                split = "test"
            shutil.copy(str(img_path), str(out_root / split / cls.name / img_path.name))

    return out_root


def ensure_split_and_balanced(
    data_root: str | Path,
    source_dataset_dir: str | Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Workflow:
    1) Check whether data_root already has train/val/test splits.
    2) If not, split source dataset into data_root.
    3) Check each split balance and balance unbalanced splits only.
    """
    from .balance import balance_all_splits, is_dataset_balanced

    root = Path(data_root)
    performed_split = False

    if not has_split_structure(root):
        src = Path(source_dataset_dir) if source_dataset_dir is not None else root
        split_dataset(
            dataset_dir=src,
            output_dir=root,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        performed_split = True

    before = {}
    for split in SPLITS:
        split_path = root / split
        if split_path.exists() and split_path.is_dir():
            before[split] = is_dataset_balanced(split_path)

    balance_results = balance_all_splits(data_root=root, folders=SPLITS, seed=seed)

    return {
        "performed_split": performed_split,
        "data_root": str(root),
        "before": before,
        "balance": balance_results,
    }