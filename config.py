from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int = 224
    embed_dim: int = 256
    num_heads: int = 16
    num_layers: int = 12
    patch_size: int = 28
    num_channels: int = 3
    num_classes: int = 6
    dropout: float = 0.2
    batch_size: int = 32

    @property
    def hidden_dim(self) -> int:
        return self.embed_dim * 3

    @property
    def num_patches(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return (self.image_size // self.patch_size) ** 2