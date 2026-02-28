import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from config import ViTConfig
from data import build_dataloaders, build_transforms
from models import VisionTransformer
from visualization import (
    get_attention_map,
    visualize_attention_map,
    visualize_conv_patches,
    visualize_examples,
    visualize_transformer_input,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization utilities for PoxViT")
    parser.add_argument("--mode", choices=["examples", "conv-patches", "transformer-input", "attention"], required=True)
    parser.add_argument("--data-dir", default="sorted_dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--weights", help="Path to model state_dict (required for all modes except examples)")
    parser.add_argument("--image", help="Path to single image (required for attention mode)")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-features", type=int, default=16)
    parser.add_argument("--output", default="artifacts/visualization.png")
    return parser.parse_args()


def _build_model(cfg: ViTConfig, device: torch.device, weights: str):
    model = VisionTransformer(
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_channels=cfg.num_channels,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
        num_patches=cfg.num_patches,
        dropout=cfg.dropout,
    ).to(device)

    weights = torch.load(weights, map_location=device)
    model.load_state_dict(weights["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ViTConfig()

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )
    loader_by_split = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_by_split[args.split]

    if args.mode == "examples":
        fig = visualize_examples(loader.dataset, num_images=args.num_images)
    else:
        if not args.weights:
            raise ValueError("--weights is required for this visualization mode")
        model = _build_model(cfg, device, args.weights)

        if args.mode == "conv-patches":
            fig = visualize_conv_patches(model, loader, device=device, num_images=args.num_images, num_features=args.num_features)
        elif args.mode == "transformer-input":
            fig, _ = visualize_transformer_input(model, loader, device=device)
        else:
            if not args.image:
                raise ValueError("--image is required for attention mode")
            preprocess = build_transforms(cfg.image_size)
            image_tensor = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
            attention_map = get_attention_map(model, image_tensor, cfg.image_size, cfg.patch_size)
            fig = visualize_attention_map(image_tensor, attention_map)

    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()