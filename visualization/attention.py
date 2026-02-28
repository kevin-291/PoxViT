import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_attention_map(model, image_tensor: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        tokens = model.encode_tokens(image_tensor)
        norm_tokens = model.transformer[0].layer_norm_1(tokens.transpose(0, 1))
        _, attn_weights = model.transformer[0].attn(
            norm_tokens,
            norm_tokens,
            norm_tokens,
            need_weights=True,
            average_attn_weights=False,
        )

    attn = attn_weights[0].mean(dim=0)
    attn = attn + torch.eye(attn.size(0), device=attn.device)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    heatmap = attn[0, 1:].reshape(image_size // patch_size, image_size // patch_size)
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        [image_size, image_size],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return heatmap.cpu()


def visualize_attention_map(image_tensor: torch.Tensor, attention_map: torch.Tensor):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    img = image_tensor.squeeze(0).cpu().permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(attention_map, cmap="jet", alpha=0.5)
    axes[1].set_title("Attention map overlay")
    axes[1].axis("off")
    return fig