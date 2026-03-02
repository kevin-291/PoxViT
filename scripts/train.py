import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

import torch

from config import ViTConfig
from data import build_dataloaders
from models import VisionTransformer
from app.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train PoxViT model")
    parser.add_argument("--data-dir", default="sorted_dataset", help="Dataset root with train/val/test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/poxvit.pt")
    parser.add_argument("--milestones", nargs="*", type=int, default=[100, 150])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--log-dir", default="artifacts/logs")
    parser.add_argument("--log-tag", default="poxvit")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    parser.add_argument("--save-every-freq", type=int, default=-1)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb-runs-dir", default="runs", help="Root directory for TensorBoard run logs")
    # PyTorch Profiler
    parser.add_argument("--profiler", action="store_true", help="Enable PyTorch Profiler")
    parser.add_argument("--profiler-dir", default="runs/profiler", help="Directory for profiler trace output")
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="poxvit", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (defaults to log-tag)")
    parser.add_argument("--wandb-watch", action="store_true", help="Use wandb.watch() to log gradients")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ViTConfig(batch_size=args.batch_size, dropout=args.dropout)

    train_loader, val_loader, _ = build_dataloaders(
        data_root=args.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
    )

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

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        milestones=tuple(args.milestones),
        gamma=args.gamma,
        output_path=args.output,
        log_dir=args.log_dir,
        log_tag=args.log_tag,
        log_freq=args.log_freq,
        checkpoint_freq=args.checkpoint_freq,
        save_every_freq=args.save_every_freq,
        resume=args.resume,
        use_tensorboard=args.tensorboard,
        tb_runs_dir=args.tb_runs_dir,
        use_profiler=args.profiler,
        profiler_dir=args.profiler_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_watch_model=args.wandb_watch,
    )


if __name__ == "__main__":
    main()