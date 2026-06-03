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
    def __init__(self, channels, kernel_size=3, padding=1, residual_scale=1.0):
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
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.residual_scale * self.net(x))


class DeepResNet(nn.Module):
    # TODO make kernel_size and padding configurable (maybe only kernel size and padding follows from size constraints?)
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
            nn.GELU(),
        ]

        for _ in range(depth):
            layers.append(
                ResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    residual_scale=residual_scale,
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
    name = config["model"]["architecture"]

    if name == "deepresnet":
        return DeepResNet(
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
            hidden_channels=config["model"]["hidden_channels"],
            kernel_size=config["model"]["kernel_size"],
            padding=config["model"]["padding"],
            depth=config["model"]["depth"],
            predict_residual=config["model"]["predict_residual"],
            residual_scale=config["model"]["residual_scale"],
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
