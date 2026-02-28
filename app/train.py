from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils import CSVLogger, AverageMeter, adamw_logger, get_logger, gpu_timer, grad_logger


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_freq: int = 10,
    logger=None,
    csv_logger: CSVLogger | None = None,
    epoch: int = 0,
) -> tuple[float, float, float]:
    model.train(True)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    gpu_time_meter = AverageMeter()

    for itr, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        iter_start = time.time()

        def closure():
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            loss.backward()
            grad_stats = grad_logger(model.named_parameters())
            optimizer.step()
            pred_cls = preds.argmax(dim=1)
            correct = (pred_cls == labels).sum().item()
            acc = correct / labels.size(0)
            return float(loss.item()), float(acc), grad_stats, adamw_logger(optimizer)

        (loss, acc, grad_stats, optim_stats), gpu_time_ms = gpu_timer(closure, log_timings=True)
        iter_elapsed_ms = (time.time() - iter_start) * 1000.0

        loss_meter.update(loss, n=imgs.size(0))
        acc_meter.update(acc, n=imgs.size(0))
        gpu_time_meter.update(gpu_time_ms)

        if csv_logger is not None:
            csv_logger.log(epoch + 1, itr, loss, acc, grad_stats.avg, gpu_time_ms, iter_elapsed_ms)

        if logger is not None and (itr % log_freq == 0):
            logger.info(
                "[%d, %5d] loss %.4f acc %.4f grad-norm %.4e [gpu %.1fms] [wall %.1fms]",
                epoch + 1,
                itr,
                loss_meter.avg,
                acc_meter.avg,
                grad_stats.avg,
                gpu_time_meter.avg,
                iter_elapsed_ms,
            )
            logger.info(
                "[%d, %5d] adam moments: exp_avg %.2e [%.2e %.2e] exp_avg_sq %.2e [%.2e %.2e]",
                epoch + 1,
                itr,
                optim_stats["exp_avg"].avg,
                optim_stats["exp_avg"].min,
                optim_stats["exp_avg"].max,
                optim_stats["exp_avg_sq"].avg,
                optim_stats["exp_avg_sq"].min,
                optim_stats["exp_avg_sq"].max,
            )

    return loss_meter.avg, acc_meter.avg, gpu_time_meter.avg


def evaluate(
    model: torch.nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)

            running_loss += loss.item()
            pred_cls = preds.argmax(dim=1)
            correct += (pred_cls == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


def _save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_vloss: float,
    history: dict,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_vloss": best_vloss,
        "history": history,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_checkpoint(path: str | Path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-4,
    milestones: tuple[int, ...] = (100, 150),
    gamma: float = 0.1,
    output_path: str | Path = "best_model_VisionTransformer_patch_size_14.pt",
    log_dir: str | Path = "artifacts/logs",
    log_tag: str = "train",
    log_freq: int = 10,
    checkpoint_freq: int = 1,
    save_every_freq: int = -1,
    resume: str | None = None,
):
    logger = get_logger(__name__, force=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler: LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(milestones),
        gamma=gamma,
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = log_dir / f"{log_tag}.csv"
    csv_logger = CSVLogger(
        csv_log_path,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.6f", "loss"),
        ("%.6f", "acc"),
        ("%.6f", "grad_norm"),
        ("%.2f", "gpu_time_ms"),
        ("%.2f", "wall_time_ms"),
    )

    latest_ckpt = log_dir / f"{log_tag}-latest.pth.tar"

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "valid_losses": [],
        "valid_accuracies": [],
    }
    best_vloss = float("inf")
    start_epoch = 0

    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
        ckpt = _load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(ckpt.get("epoch", 0))
        best_vloss = float(ckpt.get("best_vloss", float("inf")))
        history = ckpt.get("history", history)

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc, gpu_time_avg = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            log_freq=log_freq,
            logger=logger,
            csv_logger=csv_logger,
            epoch=epoch,
        )

        valid_loss, valid_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["valid_losses"].append(valid_loss)
        history["valid_accuracies"].append(valid_acc)

        logger.info(
            "EPOCH %d: train_loss %.4f val_loss %.4f train_acc %.4f val_acc %.4f (avg gpu %.1fms)",
            epoch + 1,
            train_loss,
            valid_loss,
            train_acc,
            valid_acc,
            gpu_time_avg,
        )

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            _save_checkpoint(
                path=output_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_vloss=best_vloss,
                history=history,
            )
            logger.info("Saved best checkpoint to %s", output_path)

        if checkpoint_freq > 0 and ((epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == epochs):
            _save_checkpoint(
                path=latest_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_vloss=best_vloss,
                history=history,
            )
            logger.info("Saved latest checkpoint to %s", latest_ckpt)

        if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
            every_ckpt = log_dir / f"{log_tag}-e{epoch + 1}.pth.tar"
            _save_checkpoint(
                path=every_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                best_vloss=best_vloss,
                history=history,
            )
            logger.info("Saved periodic checkpoint to %s", every_ckpt)

    return history