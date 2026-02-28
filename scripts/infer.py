import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import torch

from config import ViTConfig
from data import build_dataloaders
from app.inference import run_inference, save_confusion_matrix, save_metrics_json
from models import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for PoxViT")
    parser.add_argument("--data-dir", default="sorted_dataset", help="Dataset root with train/val/test")
    parser.add_argument("--weights", required=True, help="Path to model state_dict")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", default="artifacts/inference_metrics.json")
    parser.add_argument("--confusion-matrix", default="artifacts/confusion_matrix.png")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ViTConfig(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )
    loader_by_split = {"train": train_loader, "val": val_loader, "test": test_loader}
    eval_loader = loader_by_split[args.split]

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
    model.load_state_dict(torch.load(args.weights, map_location=device))

    results = run_inference(model=model, data_loader=eval_loader, device=device)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1_score']:.4f}")
    print("\nClassification Report:\n")
    print(results["classification_report"])

    class_names = eval_loader.dataset.classes
    save_confusion_matrix(results["confusion_matrix"], class_names, args.confusion_matrix)
    save_metrics_json(results, args.output_json)
    print(f"Saved confusion matrix to {args.confusion_matrix}")
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()