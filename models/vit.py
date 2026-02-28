import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        patch_size: int,
        num_patches: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(num_channels, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        bsz, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)

        cls_token = self.cls_token.repeat(bsz, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode_tokens(x)
        x = self.transformer(x.transpose(0, 1))
        cls = x[0]
        return self.mlp_head(cls)