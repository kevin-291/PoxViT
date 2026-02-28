from pathlib import Path

import torchvision
from torch.utils.data import DataLoader


def build_transforms(image_size: int) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_datasets(data_root: str | Path, image_size: int):
    transform = build_transforms(image_size)
    root = Path(data_root)
    train_set = torchvision.datasets.ImageFolder(root / "train", transform=transform)
    val_set = torchvision.datasets.ImageFolder(root / "val", transform=transform)
    test_set = torchvision.datasets.ImageFolder(root / "test", transform=transform)
    return train_set, val_set, test_set


def build_dataloaders(data_root: str | Path, image_size: int, batch_size: int = 32):
    train_set, val_set, test_set = build_datasets(data_root=data_root, image_size=image_size)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def print_dim(loader, text: str) -> None:
    print(f"---------{text}---------")
    print(len(loader.dataset))
    for image, label in loader:
        print(image.shape)
        print(label.shape)
        break