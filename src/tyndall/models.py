from torch import nn


class ConvModel(nn.Module):
    # TODO: make configurable
    def __init__(self, in_channels, out_channels=4, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, residual_scale=1.0, post_nl=True):
        super().__init__()
        self.residual_scale = residual_scale

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
            nn.BatchNorm2d(channels)
        )
        if post_nl:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(x + self.residual_scale * self.net(x))


class DeepResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=4,
        hidden_channels=64,
        kernel_size=3,
        padding=1,
        depth=5,
        predict_residual=True,
        residual_scale=1.0,
        post_nl=True
    ):
        super().__init__()
        self.predict_residual = predict_residual
        self.out_channels = out_channels

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        ]

        for _ in range(depth):
            layers.append(
                ResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    residual_scale=residual_scale,
                    post_nl=post_nl
                )
            )

        layers.append(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        dx = self.net(x)
        if self.predict_residual:
            x_last = x[:, -self.out_channels :]
            return x_last + dx

        return dx


def build_model(config):
    model_cfg = config["model"]
    name = model_cfg["architecture"]

    if name == "deepresnet":
        return DeepResNet(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            hidden_channels=model_cfg["hidden_channels"],
            kernel_size=model_cfg["kernel_size"],
            padding=model_cfg["padding"],
            depth=model_cfg["depth"],
            predict_residual=model_cfg["predict_residual"],
            residual_scale=model_cfg["residual_scale"],
            post_nl=model_cfg["post_nl"]
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
