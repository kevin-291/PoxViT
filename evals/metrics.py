import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def calculate_roc_auc_scores(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    class_names: list[str],
):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)

    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    roc_auc_scores = {}
    for i, class_name in enumerate(class_names):
        labels_i = (labels == i).astype(int)
        roc_auc_scores[class_name] = roc_auc_score(labels_i, probs[:, i])

    roc_auc_scores["macro_avg"] = roc_auc_score(
        torch.nn.functional.one_hot(torch.tensor(labels), num_classes=len(class_names)).numpy(),
        probs,
        multi_class="ovr",
        average="macro",
    )
    return roc_auc_scores