from torch import nn
import torch.nn.functional as F
import torch


class LayerNorm2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # input (B, C, H, W)
        # permute to (B, H, W, C) since layernorm acts on last dimension
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        # permute back
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class MixedPeriodicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, lat_pad_mode="reflect"):
        super().__init__()

        self.lat_pad_mode = lat_pad_mode
        self.kernel_size = kernel_size
        # preserves spatial size when stride = 1
        self.pad = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x):
        # circular pad longitude
        x = F.pad(x, pad=(self.pad, self.pad, 0, 0), mode="circular")
        # pad latitude
        x = F.pad(x, pad=(0, 0, self.pad, self.pad), mode=self.lat_pad_mode)

        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        padding=1,
        residual_scale=1.0,
        scale_mode="block",
        post_nl=True,
    ):
        super().__init__()

        if scale_mode == "fixed":
            self.residual_scale = residual_scale
        elif scale_mode == "block":
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        elif scale_mode == "channel":
            self.residual_scale = nn.Parameter(
                torch.full((1, channels, 1, 1), residual_scale)
            )
        elif scale_mode == "position":
            self.residual_scale = nn.Parameter(
                torch.full((1, 1, 32, 64), residual_scale)
            )

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(channels),
        )
        if post_nl:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(x + self.residual_scale * self.net(x))


class PeriodicResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        residual_scale=1.0,
        scale_mode='block',
        post_nl=True,
        lat_pad_mode="reflect",
    ):
        super().__init__()
       
        if scale_mode == "fixed":
            self.residual_scale = residual_scale
        elif scale_mode == "block":
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        elif scale_mode == "channel":
            self.residual_scale = nn.Parameter(
                torch.full((1, channels, 1, 1), residual_scale)
            )
        elif scale_mode == "position":
            self.residual_scale = nn.Parameter(
                torch.full((1, 1, 32, 64), residual_scale)
            )

        self.net = nn.Sequential(
            MixedPeriodicConv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                lat_pad_mode=lat_pad_mode,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            MixedPeriodicConv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                lat_pad_mode=lat_pad_mode,
            ),
            nn.BatchNorm2d(channels),
        )
        if post_nl:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(x + self.residual_scale * self.net(x))
