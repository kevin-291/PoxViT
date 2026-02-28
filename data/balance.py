from __future__ import annotations

import os
import random
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def _list_image_files(path: Path) -> list[Path]:
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]


def get_class_image_counts(base_path: str | Path) -> dict[str, int]:
    base = Path(base_path)
    counts: dict[str, int] = {}
    for class_dir in sorted(base.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(_list_image_files(class_dir))
    return counts


def is_dataset_balanced(base_path: str | Path) -> tuple[bool, dict[str, int]]:
    counts = get_class_image_counts(base_path)
    values = list(counts.values())
    if not values:
        return True, counts
    return len(set(values)) == 1, counts


def get_max_image_count(base_path: str | Path) -> int:
    counts = get_class_image_counts(base_path)
    return max(counts.values()) if counts else 0


def build_augmenter() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(p=0.5),
                    A.MotionBlur(p=0.5),
                ],
                p=0.5,
            ),
        ]
    )


def augment_image(image, transform: A.Compose):
    augmented = transform(image=image)
    return augmented["image"]


def balance_dataset(base_path: str | Path, seed: int = 42) -> bool:
    base = Path(base_path)
    balanced, _ = is_dataset_balanced(base)
    if balanced:
        print(f"{base}: already balanced; skipping augmentation.")
        return False

    random.seed(seed)
    transform = build_augmenter()
    max_count = get_max_image_count(base)

    for class_dir in tqdm(sorted(os.listdir(base)), desc=f"Processing {base}"):
        class_path = base / class_dir
        if not class_path.is_dir():
            continue

        images = _list_image_files(class_path)
        image_count = len(images)

        if image_count >= max_count:
            continue

        for i in range(max_count - image_count):
            image_path = random.choice(images)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented_image = augment_image(image, transform)
            new_image_name = f"aug_{i}_{image_path.name}"
            new_image_path = class_path / new_image_name
            cv2.imwrite(str(new_image_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    print(f"{base}: balancing complete.")
    return True


def balance_all_splits(
    data_root: str | Path = "sorted_dataset",
    folders: tuple[str, ...] = ("train", "val", "test"),
    seed: int = 42,
):
    root = Path(data_root)
    results: dict[str, dict[str, object]] = {}

    for folder in folders:
        folder_path = root / folder
        if not folder_path.exists() or not folder_path.is_dir():
            continue

        was_balanced, before_counts = is_dataset_balanced(folder_path)
        changed = False
        if not was_balanced:
            changed = balance_dataset(folder_path, seed=seed)

        is_balanced_now, after_counts = is_dataset_balanced(folder_path)
        results[folder] = {
            "was_balanced": was_balanced,
            "is_balanced": is_balanced_now,
            "changed": changed,
            "before_counts": before_counts,
            "after_counts": after_counts,
        }

    return results