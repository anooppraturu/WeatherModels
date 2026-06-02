import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from .metrics import compute_validation_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_config="auto"):
    if device_config != "auto":
        return torch.device(device_config)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    best_metric,
    # config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        # "config": config,
    }

    torch.save(payload, path)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.best = None
        self.should_stop = False
        self.num_bad_steps = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return True

        if value < self.best - self.min_delta:
            self.best = value
            self.num_bad_steps = 0
            return True

        self.num_bad_steps += 1

        if self.num_bad_steps >= self.patience:
            self.should_stop = True

        return False


def train_model(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=1,
    log_every=100,
    val_every=100,
    ckpt_dir="./",
    device=torch.device("cpu"),
):
    # use defaults for now
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
