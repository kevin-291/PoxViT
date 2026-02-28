from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def run_inference(model: torch.nn.Module, data_loader, device: torch.device):
    model.eval()
    acc_total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            pred_cls = preds.argmax(dim=1)

            acc_total += pred_cls.eq(labels).cpu().sum().item()
            all_preds.extend(pred_cls.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = acc_total / len(data_loader.dataset)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": classification_report(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def save_confusion_matrix(conf_matrix, class_names: list[str], output_path: str | Path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def save_metrics_json(results: dict, output_path: str | Path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    to_save = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "classification_report": results["classification_report"],
    }
    output.write_text(json.dumps(to_save, indent=2))