import argparse

import yaml
from torch import nn
from torch import optim

from tyndall.data import build_loaders
from tyndall.models import build_model
from tyndall.train_utils import train_model

def build_optimizer(config, model):
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"]

    if name == 'adamw':
        return optim.AdamW(
            params=model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unkown Optimizer {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    train_loader, val_loader = build_loaders(config)
    optimizer = build_optimizer(config, model)
    criterion = nn.MSELoss()

    train_cfg = config["training"]

    _ = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_cfg["epochs"],
        log_every=train_cfg["log_every"],
        val_every=train_cfg["val_every"],
        ckpt_dir=config["project"]["output_dir"]
    )

if __name__ == "__main__":
    main()
