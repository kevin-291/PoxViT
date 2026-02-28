import torch


def img_to_patch(x: torch.Tensor, patch_size: int, flatten_channels: bool = True) -> torch.Tensor:
    """Convert image tensor [B, C, H, W] into a sequence of patches."""
    bsz, channels, height, width = x.shape
    x = x.reshape(
        bsz,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.flatten(1, 2)
    if flatten_channels:
        x = x.flatten(2, 4)
    return x