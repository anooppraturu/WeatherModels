import torch

def per_variable_batch_rmse(yhat, y):
    """
    yhat: (B, variable, latitude, longitude)
    out: (variable)
    """
    return torch.sqrt(((yhat - y)**2).mean(dim=(0, 2, 3)))

def per_variable_time_rmse(yhat, y):
    """
    yhat: (time, variable, latitude, longitude)
    out: (time, variable)
    """
    return torch.sqrt(((yhat - y)**2).mean(dim=(2, 3)))

@torch.no_grad()
def total_variable_rmse(fn, loader):
    """
    fn: function to make predictions
    loader: torch DataLoader to copmute RMSE on
    """
    N = 0
    all_rmses = []

    for i, (xb, yb) in enumerate(loader):
        N += xb.shape[0]
        yhat = fn(xb)
        var_rmse = per_variable_batch_rmse(yhat, yb)
        all_rmses.append(var_rmse)

    return torch.stack(all_rmses).sum(dim=0) / N


def rollout_rmse(trajectory, start_idx, dataset, n_vars = 4):
    t_len = trajectory.shape[0]
    dt = torch.stack([dataset[start_idx+t][0][-n_vars:] for t in range(t_len)])
    pv_rmse = per_variable_time_rmse(trajectory, dt)
    return pv_rmse


@torch.no_grad()
def compute_validation_loss(model, loss_fn, val_loader, device):
    N = 0
    val_loss = 0

    model.eval()
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        B = xb.shape[0]
        N += B

        preds = model(xb)
        loss = loss_fn(preds, yb)
        val_loss += B*loss.item()

    return val_loss / N