import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import torch
from PIL import Image

from config import ViTConfig
from models import VisionTransformer
from data import build_transforms
from visualization import get_attention_map, visualize_attention_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt state dict")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--output", default="attention_map.png", help="Output figure path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ViTConfig()

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

    weights = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights["model"])

    preprocess = build_transforms(cfg.image_size)
    image_tensor = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    attention_map = get_attention_map(
        model=model,
        image_tensor=image_tensor,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )
    fig = visualize_attention_map(image_tensor, attention_map)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()