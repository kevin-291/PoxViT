# PoxViT: Vision Transformer for Skin Disease Classification

This repository contains an implementation of PoxViT, a modified Vision Transformer (ViT) specifically designed for skin disease classification with a focus on pox-related conditions. The model uses convolutional patch embedding instead of traditional linear patch embedding for improved feature extraction.

## Features

- Convolutional Patch Embedding: Replaces the standard linear patch embedding with a convolutional approach for better spatial feature extraction
- Flexible Configuration: Customizable parameters for embed dimensions, number of heads, layers, patch sizes, etc.
- Pre-trained Models: Includes pre-trained weights for general purpose and skin lesion classification
- PoxViT Specialization: Optimized architecture for identifying pox-type skin diseases

## Model Architecture

The Vision Transformer model consists of:

1. **Convolutional Patch Embedding**: Two-layer CNN to convert image patches into embeddings
2. **Transformer Encoder**: Multiple attention blocks for contextual feature learning
3. **Classification Head**: MLP head for final classification

![Model Architecture](model%20diagram%20with%20bg.png)


## Requirements

```bash
pip install -r requirements.txt

```

## License

MIT License