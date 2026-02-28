# PoxViT: Vision Transformer for Skin Disease Classification

PoxViT is a Vision Transformer-based project for skin disease classification, with a focus on pox-related conditions.  
The architecture emphasizes improved local feature extraction using convolutional patch embedding, while retaining the global-context modeling strengths of transformer attention.

---

## Overview

Skin lesion and pox-type disease classification can benefit from both:

- **local texture/edge cues** (captured effectively by convolutions), and
- **global contextual relationships** (modeled by transformer self-attention).

PoxViT combines these strengths by replacing standard linear patch projection with a **convolutional patch embedding** pipeline and then processing patch tokens through transformer attention blocks for classification.

---

## Key Features

- **Convolutional Patch Embedding**
  - Replaces plain linear patchification with convolutional feature extraction for stronger spatial inductive bias.

- **Transformer Encoder Backbone**
  - Multi-head self-attention + MLP blocks for long-range feature interactions.

- **Configurable Architecture**
  - Tune embedding size, depth, number of heads, patch size, dropout, and training settings.

- **Skin-Disease-Oriented Workflow**
  - Utilities for data preparation, balancing, training, inference, and visualization.

- **Reproducible CLI Pipelines**
  - Script-based workflows for training/evaluation and artifact generation.

- **uv lockfile support**
  - Optional dependency reproducibility via `uv.lock`.

---

## Repository Structure

```text
PoxViT/
├── app/                  # Application entrypoints and orchestration logic
├── configs/              # YAML configuration templates
├── data/                 # Dataset transforms, loading, and balancing helpers
├── evals/                # Evaluation metrics and analysis utilities
├── models/               # Core model components (attention blocks, ViT, heads)
├── scripts/              # Task-specific CLI scripts (train/infer/visualize/balance)
├── utils/                # Shared helpers (patch utilities, misc tools)
├── visualization/        # Attention maps, embedding plots, qualitative outputs
├── config.py             # Centralized runtime/model/data configuration
├── main.py               # Top-level executable entrypoint
├── requirements.txt      # pip dependencies
├── pyproject.toml        # Project metadata / dependency specification
└── uv.lock               # Locked dependency state
```

---

## Installation

### Option A: Install with Python + pip

```bash
git clone https://github.com/kevin-291/PoxViT.git
cd PoxViT

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Option B: Resolve dependencies using uv lockfile, then run with Python

If you use `uv` for dependency management, you can sync the environment from `uv.lock`, then still execute scripts with `python`.

```bash
# after installing uv
uv sync
```

Then run commands normally with `python` (examples below).

---

## Dataset Layout

Expected split-based structure:

```text
sorted_dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

If your dataset is not split yet (class folders only), the balancing/splitting workflow can be used to create train/val/test splits.

---

## Training, Inference, and Visualization

### Train

```bash
python scripts/train.py \
  --data-dir sorted_dataset \
  --epochs 30 \
  --output artifacts/best_model.pth.tar \
  --log-dir artifacts/logs \
  --log-tag poxvit
```

### Evaluate / Inference

```bash
python scripts/infer.py \
  --data-dir sorted_dataset \
  --weights artifacts/best_model.pt \
  --split test
```

### Visualizations

```bash
# Sample predictions/examples
python scripts/visualize.py \
  --mode examples \
  --data-dir sorted_dataset \
  --split val \
  --output artifacts/examples.png

# Convolutional patch features
python scripts/visualize.py \
  --mode conv-patches \
  --data-dir sorted_dataset \
  --split test \
  --weights artifacts/best_model.pt \
  --output artifacts/conv_patches.png

# Attention map for a single image
python scripts/visualize.py \
  --mode attention \
  --weights artifacts/best_model.pt \
  --image path/to/image.jpg \
  --output artifacts/attention.png
```

---

## Dataset Balancing Workflow

Recommended preparation flow:

1. Verify whether `train/val/test` splits already exist.
2. If missing, create splits (default 80/10/10, configurable).
3. Check class distribution for each split.
4. Apply balancing only to unbalanced splits.

### Balance command

```bash
# For already split dataset
python scripts/balance_dataset.py --data-root sorted_dataset

# For unsplit dataset (class folders only)
python scripts/balance_dataset.py \
  --data-root sorted_dataset \
  --source-dataset-dir raw_dataset
```

---

## Unified Application Entry

You can run common tasks via the top-level app entrypoint:

```bash
python app/main.py train --data-dir sorted_dataset --epochs 30
python app/main.py infer --data-dir sorted_dataset --weights artifacts/best_model.pt
python app/main.py visualize --mode attention --weights artifacts/best_model.pt --image path/to/image.jpg
python app/main.py balance --data-root sorted_dataset
```

---

## Configuration

Configuration can be managed through:

- `config.py` for shared defaults and runtime settings
- YAML templates in `configs/` for reproducible runs

Typical configurable parameters include:

- image size / patch size
- embedding dimension
- transformer depth
- number of attention heads
- MLP ratio
- dropout rates
- optimizer / scheduler settings
- batch size and epochs

---

## Outputs and Artifacts

Typical generated artifacts include:

- **Model checkpoints** (`best`, `latest`, optional periodic saves)
- **Training logs** (CSV/structured logs)
- **Evaluation outputs** (metrics, confusion insights, ROC-AUC where enabled)
- **Visual diagnostics** (attention overlays, patch-feature visualizations)

---

## Use Cases

- Automated classification of pox and pox-like skin lesions
- Clinical AI prototyping and experimentation
- Vision Transformer interpretability analysis using attention maps
- Benchmarking conv-patch ViT variants for dermatology datasets

---

## License

MIT License