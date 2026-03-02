from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils import CSVLogger, AverageMeter, adamw_logger, get_logger, gpu_timer, grad_logger

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter 
    _TB_AVAILABLE = True
except Exception:
    _TB_AVAILABLE = False

try:
    import wandb as _wandb 
    _WANDB_AVAILABLE = True
except Exception:  
    _WANDB_AVAILABLE = False


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
    profiler_ctx=None,
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
        
        if profiler_ctx is not None:
            profiler_ctx.step()

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
    output_path: str | Path = "artifacts/best_model.pt",
    log_dir: str | Path = "artifacts/logs",
    log_tag: str = "train",
    log_freq: int = 10,
    checkpoint_freq: int = 1,
    save_every_freq: int = -1,
    resume: str | None = None,
    patience: int = -1,
    use_tensorboard: bool = False,
    tb_runs_dir: str | Path = "runs",
    use_profiler: bool = False,
    profiler_dir: str | Path = "runs/profiler",
    use_wandb: bool = False,
    wandb_project: str = "poxvit",
    wandb_run_name: str | None = None,
    wandb_watch_model: bool = False,
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
    epochs_no_improve = 0

    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
        ckpt = _load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(ckpt.get("epoch", 0))
        best_vloss = float(ckpt.get("best_vloss", float("inf")))
        history = ckpt.get("history", history)

    tb_writer = None
    _sample_imgs_tb, _sample_labels_tb = None, None
    if use_tensorboard:
        if not _TB_AVAILABLE:
            logger.warning("TensorBoard not installed; skipping TB logging. Install tensorboard.")
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"{log_tag}_{timestamp}_lr{lr}_bs{train_loader.batch_size}"
            tb_log_dir = Path(tb_runs_dir) / run_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = _SummaryWriter(log_dir=str(tb_log_dir))
            logger.info("TensorBoard logs → %s", tb_log_dir)
            try:
                _sample_imgs_tb, _sample_labels_tb = next(iter(val_loader))
                tb_writer.add_graph(model, _sample_imgs_tb[:1].to(device))
            except Exception as exc:
                logger.warning("Could not add model graph to TensorBoard: %s", exc)
                _sample_imgs_tb, _sample_labels_tb = None, None

    wb_run = None
    if use_wandb:
        if not _WANDB_AVAILABLE:
            logger.warning("wandb not installed; skipping W&B logging. Install wandb.")
        else:
            wb_run = _wandb.init(
                project=wandb_project,
                name=wandb_run_name or log_tag,
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": train_loader.batch_size,
                    "milestones": list(milestones),
                    "gamma": gamma,
                    "optimizer": "Adam",
                    "scheduler": "MultiStepLR",
                    "model": type(model).__name__,
                },
            )
            logger.info("W&B run: %s", wb_run.url if wb_run else "n/a")
            if wandb_watch_model:
                _wandb.watch(model, log="all", log_freq=log_freq)

    profiler_dir = Path(profiler_dir)
    _profiler_ctx = None
    if use_profiler:
        profiler_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        _profiler_ctx = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        )
        _profiler_ctx.start()
        logger.info("PyTorch Profiler traces → %s", profiler_dir)


    # Training loop

    total_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

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
            profiler_ctx=_profiler_ctx,
        )

        valid_loss, valid_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

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

        current_lr = scheduler.get_last_lr()[0]

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch + 1)
            tb_writer.add_scalar("Loss/val", valid_loss, epoch + 1)
            tb_writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            tb_writer.add_scalar("Accuracy/val", valid_acc, epoch + 1)
            tb_writer.add_scalar("LR", current_lr, epoch + 1)
            tb_writer.add_scalar("Time/epoch_sec", epoch_time, epoch + 1)
            tb_writer.add_scalar("Time/total_sec", time.time() - total_start, epoch + 1)

            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                try:
                    if _sample_imgs_tb is not None:
                        imgs_show = _sample_imgs_tb[:8].to(device)
                        with torch.no_grad():
                            preds = model(imgs_show).argmax(dim=1)
                        tb_writer.add_images("Samples/val_images", imgs_show.cpu(), epoch + 1)
                        tb_writer.add_text(
                            "Samples/val_predictions",
                            f"pred={preds.tolist()}  gt={_sample_labels_tb[:8].tolist()}",
                            epoch + 1,
                        )
                except Exception as exc:
                    logger.warning("Could not log sample images to TensorBoard: %s", exc)

        if wb_run is not None:
            _wandb.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": valid_loss,
                    "val/accuracy": valid_acc,
                    "lr": current_lr,
                    "epoch_time_sec": epoch_time,
                },
                step=epoch + 1,
            )

        if valid_loss < best_vloss:
            best_vloss = valid_loss
            epochs_no_improve = 0

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

            if wb_run is not None:
                artifact = _wandb.Artifact(
                    name=f"{log_tag}-best",
                    type="model",
                    description=f"Best checkpoint at epoch {epoch + 1}",
                )
                artifact.add_file(str(output_path))
                wb_run.log_artifact(artifact)
        
        else:
            epochs_no_improve += 1
            logger.info("No improvement in validation loss for %d epoch(s).", epochs_no_improve)
        
        if patience > 0 and epochs_no_improve >= patience:
            logger.info("Early stopping triggered after %d epochs without improvement. Training stopped.", epochs_no_improve)
            break

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

    if _profiler_ctx is not None:
        _profiler_ctx.stop()

    if tb_writer is not None:
        tb_writer.close()

    if wb_run is not None:
        _wandb.finish()

    return history