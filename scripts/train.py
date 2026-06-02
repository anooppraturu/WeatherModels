import argparse
from pathlib import Path
from tqdm import tqdm

import yaml
from torch import nn
from torch import optim

from tyndall.data import build_loaders
from tyndall.models import build_model
from tyndall.train_utils import save_checkpoint, get_device, EarlyStopping
from tyndall.metrics import compute_validation_loss

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

def train(config):
    # use defaults for now
    model = build_model(config)
    train_loader, val_loader = build_loaders(config)
    optimizer = build_optimizer(config, model)
    loss_fn = nn.MSELoss()

    device = get_device(config["training"]["device"])
    epochs = config["training"]["epochs"]
    log_every=config["training"]["log_every"]
    val_every=config["training"]["val_every"]
    ckpt_dir=config["project"]["output_dir"]

    print(f"Training for {epochs} epochs on device {device}")

    model.to(device)
    early_stopper = EarlyStopping()
    global_step = 0
    ckpt_dir = Path(ckpt_dir)

    for epoch in range(epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % log_every == 0:
                print(f"Step {global_step}, train batch loss = {loss.item():.6f}")

            if global_step % val_every == 0:
                validation_loss = compute_validation_loss(
                    model=model, loss_fn=loss_fn, val_loader=val_loader, device=device
                )
                improved = early_stopper.step(validation_loss)

                if improved:
                    save_checkpoint(
                        path=ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=validation_loss,
                    )

                if early_stopper.should_stop:
                    save_checkpoint(
                        path=ckpt_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=validation_loss,
                    )

                print(f"Step {global_step}, validation loss={validation_loss:.6f}")

    save_checkpoint(
        path=ckpt_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        best_metric=validation_loss,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)

if __name__ == "__main__":
    main()
