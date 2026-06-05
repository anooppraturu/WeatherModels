import torch
from .model_utils import autoregressive_rollout
from tqdm import tqdm

@torch.no_grad()
def per_variable_batch_rmse(yhat, y):
    """
    yhat: (B, variable, latitude, longitude)
    out: (variable)
    """
    return torch.sqrt(((yhat - y)**2).mean(dim=(0, 2, 3)))

@torch.no_grad()
def per_variable_time_rmse(yhat, y):
    """
    yhat: (time, variable, latitude, longitude)
    out: (time, variable)
    """
    return torch.sqrt(((yhat - y)**2).mean(dim=(2, 3)))

@torch.no_grad()
def total_variable_rmse(model, loader, device):
    """
    fn: function to make predictions
    loader: torch DataLoader to compute RMSE on
    """
    N = 0
    all_rmses = []

    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)

        B = xb.shape[0]
        N += B
        yhat = model(xb)
        var_rmse = B*per_variable_batch_rmse(yhat, yb)
        all_rmses.append(var_rmse)

    mean_rmses = torch.stack(all_rmses).sum(dim=0) / N

    # metric structure hardcoded for now
    metrics = {
        "T850_rmse": mean_rmses[0],
        "U850_rmse": mean_rmses[1],
        "V850_rmse": mean_rmses[2],
        "Z500_rmse": mean_rmses[3]
    }
    return metrics


def rollout_rmse(trajectory, start_idx, dataset, n_vars = 4):
    t_len = trajectory.shape[0]
    dt = torch.stack([dataset[start_idx+t][0][-n_vars:] for t in range(t_len)])
    pv_rmse = per_variable_time_rmse(trajectory, dt)
    return pv_rmse


def data_trajectory(trajectory, start_idx, dataset, n_vars = 4):
    t_len = trajectory.shape[0]
    dt = torch.stack([dataset[start_idx+t][0][-n_vars:] for t in range(t_len)])
    return dt


@torch.no_grad()
def compute_validation_loss(model, loss_fn, val_loader, device):
    N = 0
    val_loss = 0

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        B = xb.shape[0]
        N += B

        preds = model(xb)
        loss = loss_fn(preds, yb)
        val_loss += B*loss.item()

    return val_loss / N

def dataset_rollout_rmse(model, dataset, roll_len = 50):
    V, _, _ = dataset[0][0].shape
    V = int(V / 4)
    running_mean = torch.zeros(roll_len+1, V)
    running_var = torch.zeros(roll_len+1, V)
    N=0

    pbar = tqdm(range(len(dataset) - roll_len - 1), desc="Index")

    for i in pbar:
        x, _ = dataset[i]
        rollout = autoregressive_rollout(model, x.unsqueeze(0), roll_len)
        errors = rollout_rmse(rollout, i, dataset)

        running_mean += errors
        running_var += errors**2
        N += 1

    mean = running_mean / N
    running_var
    std = torch.sqrt(running_var / N - mean**2)
    return mean, std