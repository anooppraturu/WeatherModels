import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import mlflow
from contextlib import nullcontext

import torch
from torch import optim

from tyndall.data import build_loaders, ShardedWeatherBenchDataset
from tyndall.losses import RolloutWeightedMSE
from tyndall.models import build_model
from tyndall.training import save_checkpoint, get_device#, EarlyStopping
from tyndall.metrics import compute_rollout_validation_loss
from tyndall.utils import flatten_dict


def build_optimizer(config, model):
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"]

    if name == "adamw":
        return optim.AdamW(
            params=model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unkown Optimizer {name}")
    
def get_loss(config):
    tmp_data = ShardedWeatherBenchDataset(
        root=config["data"]["dataset_path"],
        split="train"
    )
    tmp_payload = tmp_data._load_shard(0)
    latitudes = tmp_payload["latitude"]
    device=get_device(config["training"]["device"])
    return RolloutWeightedMSE(latitudes=latitudes, device=device)
    

def nstep_predictor(model, xb, nsteps, n_var=4):
    assert xb.ndim == 4
    assert xb.shape[1] % n_var == 0

    # xb: (B, input_steps * V, H, W)
    prediction = []
    for _ in range(nsteps):
        x_next = model(xb)
        prediction.append(x_next)
        xb = torch.cat([xb[:,n_var:], x_next], dim=1)

    return torch.stack(prediction, dim=1)



def fine_tune(model, config):
    # model, data and optimizer
    train_loader, val_loader = build_loaders(config)
    optimizer = build_optimizer(config, model)
    loss_fn = get_loss(config)

    # training configs
    device = get_device(config["training"]["device"])
    epochs = config["training"]["epochs"]
    log_every = config["training"]["log_every"]
    val_every = config["training"]["val_every"]
    ckpt_dir = Path(config["model"]["data_dir"]) / config["model"]["run_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # log config to checkpoint dir
    config_path = ckpt_dir / "ft_config.yaml"
    with open(str(config_path), "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"Training for {epochs} epochs on device {device}")

    model.to(device)
    global_step = 0
    ckpt_dir = Path(ckpt_dir)

    # mlflow
    mlflow_cfg = config.get("mlflow", {})
    use_mlflow = mlflow_cfg.get("enabled", False)
    if use_mlflow:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        run_context = mlflow.start_run(run_name=config["model"]["run_name"])
    else:
        run_context = nullcontext()

    with run_context:
        # log config to mlflow
        if use_mlflow:
            flat_config = flatten_dict(config)

            for k, v in flat_config.items():
                mlflow.log_param(k, v)
            mlflow.log_artifact(str(config_path), artifact_path="config")

        for epoch in range(epochs):
            model.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                preds = nstep_predictor(model=model, xb=xb, nsteps=config["data"]["lead_steps"])
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

                global_step += 1

                if global_step % log_every == 0:
                    print(f"Step {global_step}, train batch loss = {loss.item():.6f}")
                    if use_mlflow:
                        mlflow.log_metric(
                            "train/loss", float(loss.item()), step=global_step
                        )

                if global_step % val_every == 0:
                    model.eval()
                    validation_loss = compute_rollout_validation_loss(
                        model=model,
                        loss_fn=loss_fn,
                        val_loader=val_loader,
                        device=device,
                        predictor_fn=nstep_predictor,
                        nsteps=config["data"]["lead_steps"]
                    )
                    if use_mlflow:
                        mlflow.log_metric(
                            "val/loss", float(validation_loss), step=global_step
                        )
                    model.train()

                    print(f"Step {global_step}, validation loss={validation_loss:.6f}")

                
        save_checkpoint(
            path=ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_metric=None,
        )

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

     # get directory
    data_dir = Path(config["model"]["data_dir"])
    train_config = data_dir / "config.yaml"
    model_path = data_dir / "best.pt"

    # load train config
    with open(train_config, "r") as f:
        train_config = yaml.safe_load(f)

    # load trained model
    model = build_model(train_config)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict["model"])
    model.train()

    fine_tune(model, config)


if __name__ == "__main__":
    main()
