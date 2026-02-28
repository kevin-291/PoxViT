from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def gpu_timer(closure, log_timings: bool = True):
    """Time closure execution on GPU if CUDA is available."""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


def get_logger(name: str | None = None, force: bool = False):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=force,
    )
    return logging.getLogger(name=name)


class CSVLogger:
    def __init__(self, fname: str | Path, *argv):
        self.fname = Path(fname)
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self.types = []

        write_header = not self.fname.exists() or self.fname.stat().st_size == 0
        if write_header:
            with self.fname.open("a", encoding="utf-8") as f:
                for i, v in enumerate(argv, 1):
                    self.types.append(v[0])
                    end = "," if i < len(argv) else "\n"
                    print(v[1], end=end, file=f)
        else:
            for v in argv:
                self.types.append(v[0])

    def log(self, *argv):
        with self.fname.open("a", encoding="utf-8") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None

    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if stats.first_layer is None:
                stats.first_layer = grad_norm
            stats.last_layer = grad_norm

    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.0
    return stats


def adamw_logger(optimizer):
    """Log magnitude of Adam/AdamW first and second momentum buffers."""
    state = optimizer.state_dict().get("state")
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    for key in state:
        s = state.get(key)
        if s is None or "exp_avg" not in s or "exp_avg_sq" not in s:
            continue
        exp_avg_stats.update(float(s.get("exp_avg").abs().mean()))
        exp_avg_sq_stats.update(float(s.get("exp_avg_sq").abs().mean()))

    return {"exp_avg": exp_avg_stats, "exp_avg_sq": exp_avg_sq_stats}