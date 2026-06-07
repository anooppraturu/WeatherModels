import torch
from torch import nn

class LatitudeWeightedMSE(nn.Module):
    def __init__(self, latitudes, device=torch.device("cpu")):
        super().__init__()

        latitudes = torch.as_tensor(latitudes, dtype=torch.float32, device=device)
        weights = torch.cos(torch.deg2rad(latitudes))
        # numerical safety
        weights = torch.clamp(weights, min=0.0)
        # normalize
        weights = weights/weights.mean()
        # broadcast against (B, V, H, W)
        weights = weights.view(1, 1, -1, 1)

        self.register_buffer("weights", weights)

    def forward(self, pred, target):
        """
        pred, target shape: B, V, H, W
        """
        se = (pred - target) ** 2
        return (se * self.weights).mean()
