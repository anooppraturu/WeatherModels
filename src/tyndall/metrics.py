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

@torch.no_grad()
def compute_rollout_validation_loss(model, loss_fn, val_loader, device, predictor_fn, nsteps):
    N = 0
    val_loss = 0

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        B = xb.shape[0]
        N += B

        preds = model(xb)
        preds = predictor_fn(model=model, xb=xb, nsteps=nsteps)
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

def get_spectrum(field, n_bins=None):
    """
    field: tensors of shape (V, H, W)

    Returns:
        bin_centers: (n_bins,)
        spectrum: (V, n_bins)
            mean Fourier error power per radial k-bin, averaged over time
    """
    V, H, W = field.shape

    # Orthonormal FFT so Parseval is simple.
    err_hat = torch.fft.fft2(field, dim=(-2, -1), norm="ortho")
    power = err_hat.abs() ** 2  # V, H, W

    # Frequency coordinates in cycles per grid spacing.
    ky = torch.fft.fftfreq(H, device=field.device)  # latitude/grid-y frequencies
    kx = torch.fft.fftfreq(W, device=field.device)  # longitude/grid-x frequencies

    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(kx_grid ** 2 + ky_grid ** 2)  # H, W

    if n_bins is None:
        n_bins = min(H, W) // 2

    k_max = k_mag.max()
    bin_edges = torch.linspace(0, k_max, n_bins + 1, device=field.device)

    spectrum = torch.zeros(V, n_bins, device=field.device)
    counts = torch.zeros(n_bins, device=field.device)

    for b in range(n_bins):
        mask = (k_mag >= bin_edges[b]) & (k_mag < bin_edges[b + 1])

        # Include the right edge in the final bin.
        if b == n_bins - 1:
            mask = (k_mag >= bin_edges[b]) & (k_mag <= bin_edges[b + 1])

        counts[b] = mask.sum()

        if counts[b] > 0:
            spectrum[:, b] = power[:, mask].mean(dim=-1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers.detach().cpu(), spectrum.detach().cpu()