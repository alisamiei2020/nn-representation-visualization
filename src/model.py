"""SmallCNN classifier with access to intermediate representations."""

import torch
from torch import nn

IN_CHANNELS = 1
IMAGE_SIZE = 28
NUM_CLASSES = 10


def conv_block(in_ch: int, out_ch: int, kernel_size: int, padding: int, pool_size: int) -> nn.Sequential:
    """Conv2d (stride 1) -> ReLU -> MaxPool2d."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
    )


def block_out_size(size: int, kernel_size: int, padding: int, pool_size: int) -> int:
    """Spatial size after one conv_block."""
    return (size + 2 * padding - kernel_size + 1) // pool_size


class SmallCNN(nn.Module):
    """conv(1->c1)+ReLU+pool -> conv(c1->c2)+ReLU+pool -> fc(hidden->latent)+ReLU -> fc(latent->10)."""

    def __init__(self, conv_channels: list[int], kernel_size: int, padding: int, pool_size: int, latent_dim: int):
        """Build layers; hidden_dim is 3136 for the default config."""
        super().__init__()
        c1, c2 = conv_channels
        self.conv1 = conv_block(IN_CHANNELS, c1, kernel_size, padding, pool_size)
        self.conv2 = conv_block(c1, c2, kernel_size, padding, pool_size)

        size = IMAGE_SIZE
        for _ in range(2):
            size = block_out_size(size, kernel_size, padding, pool_size)
        self.hidden_dim = c2 * size * size

        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim, latent_dim), nn.ReLU())
        self.fc2 = nn.Linear(latent_dim, NUM_CLASSES)

    def _trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared path for forward() and features(): flattened conv2 output and post-ReLU latent."""
        hidden = self.conv2(self.conv1(x)).flatten(1)
        latent = self.fc1(hidden)
        return hidden, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits, shape (N, 10)."""
        _, latent = self._trunk(x)
        return self.fc2(latent)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Flattened representations: input (784), hidden (3136), latent (128) for the default config."""
        hidden, latent = self._trunk(x)
        return {"input": x.flatten(1), "hidden": hidden, "latent": latent}


def build_model(model_cfg: dict) -> SmallCNN:
    """Construct SmallCNN from the 'model' config section."""
    return SmallCNN(**model_cfg)
