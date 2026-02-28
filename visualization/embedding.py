import matplotlib.pyplot as plt
import torch
import torchvision


def visualize_conv_patches(model, data_loader, device: torch.device, num_images: int = 3, num_features: int = 16):
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        patch_embeddings = model.patch_embedding(images)

    fig, axes = plt.subplots(num_images, num_features + 1, figsize=(3 * (num_features + 1), 3 * num_images))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        image_np = images[i].cpu().permute(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        axes[i][0].imshow(image_np)
        axes[i][0].set_title("Original")
        axes[i][0].axis("off")

        for j in range(min(num_features, patch_embeddings.shape[1])):
            feature_map = patch_embeddings[i, j].cpu()
            axes[i][j + 1].imshow(feature_map, cmap="viridis")
            axes[i][j + 1].axis("off")
            axes[i][j + 1].set_title(f"F{j+1}")

    plt.tight_layout()
    return fig


def visualize_transformer_input(model, image_loader, device: torch.device):
    images, _ = next(iter(image_loader))
    image = images[0:1].to(device)

    with torch.no_grad():
        patch_output = model.patch_embedding(image)
        spatial = patch_output.squeeze(0).permute(1, 2, 0)
        transformer_input = model.encode_tokens(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    original = image[0].cpu().permute(1, 2, 0)
    original = (original - original.min()) / (original.max() - original.min())
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    patch_preview = torch.mean(spatial, dim=-1).cpu()
    axes[1].imshow(patch_preview, cmap="viridis")
    axes[1].set_title("Patch Embedding Mean")
    axes[1].axis("off")

    cls_token = transformer_input[0, 0].cpu().numpy()
    axes[2].plot(cls_token)
    axes[2].set_title("CLS token embedding")

    plt.tight_layout()
    return fig, transformer_input


def visualize_examples(dataset, num_images: int = 4):
    examples = torch.stack([dataset[idx][0] for idx in range(num_images)], dim=0)
    grid = torchvision.utils.make_grid(examples, nrow=2, normalize=True, pad_value=0.9).permute(1, 2, 0)

    fig = plt.figure(figsize=(8, 8))
    plt.title("Image examples of the Skin diseases dataset")
    plt.imshow(grid)
    plt.axis("off")
    return fig